"""Yunshu External Prefill — prefill outside BatchGenerator for memory preflight,
chunked progress tracking, and mid-prefill abort.

Currently, mlx-lm's BatchGenerator manages its own KV cache internally, so we
cannot directly inject external KV cache into it. The external prefill path's
value TODAY is:
  1. Memory preflight: estimate if a prompt fits before committing GPU memory
  2. Chunked progress tracking: on_progress callback per chunk for dashboards
  3. Mid-prefill abort: interrupt long prefills between chunks
  4. Prefix cache integration: report cached_tokens from KVCacheManager

Future work: direct KV cache injection into BatchGenerator (requires MLX-level
changes to BatchGenerator's internal cache management).

Architecture:
  Scheduler._schedule_waiting() → ExternalPrefiller.prefill_chunked()
    → memory preflight check (raises PrefillMemoryExceededError if OOM)
    → chunked model forward passes (building KV cache externally)
    → on_progress callback per chunk
    → mid-prefill abort check between chunks
  Then BatchGenerator.insert() proceeds as normal for the decode phase.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# MLX is optional for testability — all MLX calls go through _run_model_step
# which can be mocked in tests.
try:
    import mlx.core as mx
    _HAS_MLX = True
except ImportError:
    mx = None  # type: ignore
    _HAS_MLX = False


@dataclass
class PrefillResult:
    """Result of an external prefill operation.

    Attributes:
        token_ids: The token IDs that were prefilled.
        num_tokens: Count of prefilled tokens.
        kv_cache: Resulting KV cache state (if applicable). Currently None
                  since BatchGenerator manages its own cache internally.
        cached_tokens: Tokens that hit prefix cache (from KVCacheManager).
        duration_s: Wall-clock prefill duration in seconds.
    """

    token_ids: list[int]
    num_tokens: int
    kv_cache: Any | None = None
    cached_tokens: int = 0
    duration_s: float = 0.0


class ExternalPrefiller:
    """Runs model prefill outside BatchGenerator.

    Provides:
    - Memory preflight checking before committing GPU resources
    - Chunked prefill with progress callbacks
    - Mid-prefill abort support
    - Prefix cache hit reporting

    Usage:
        prefiller = ExternalPrefiller(model, tokenizer, executor)
        result = prefiller.prefill_chunked(
            token_ids=[1, 2, 3, ...],
            chunk_size=2048,
            on_progress=lambda done, total: print(f"{done}/{total}"),
            request_id="req-abc123",
            pending_aborts=set(),
        )
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._executor = executor

    def prefill(
        self,
        token_ids: list[int],
        cached_prefix_len: int = 0,
    ) -> PrefillResult:
        """Run prefill on the full token sequence outside BatchGenerator.

        Creates a fresh KV cache, processes tokens through the model to build
        KV cache state, and returns the result.

        Args:
            token_ids: The prompt token IDs to prefill.
            cached_prefix_len: Number of tokens from the start that are already
                cached (from prefix cache). These are still included in the
                forward pass for correctness but counted as cached_tokens.

        Returns:
            PrefillResult with KV cache state and timing information.
        """
        if not token_ids:
            return PrefillResult(
                token_ids=[],
                num_tokens=0,
                kv_cache=None,
                cached_tokens=0,
                duration_s=0.0,
            )

        t0 = time.monotonic()

        kv_cache = self._create_kv_cache()
        self._run_model_step(token_ids, kv_cache)

        elapsed = time.monotonic() - t0

        return PrefillResult(
            token_ids=token_ids,
            num_tokens=len(token_ids),
            kv_cache=kv_cache,
            cached_tokens=cached_prefix_len,
            duration_s=elapsed,
        )

    def prefill_chunked(
        self,
        token_ids: list[int],
        chunk_size: int = 2048,
        on_progress: Callable[[int, int], None] | None = None,
        request_id: str | None = None,
        pending_aborts: set[str] | None = None,
        memory_monitor: Any | None = None,
    ) -> PrefillResult:
        """Run chunked prefill with progress tracking and abort support.

        Splits token_ids into chunks and processes each one, building KV cache
        incrementally. Between chunks:
        - Calls on_progress(completed, total) for progress tracking
        - Checks pending_aborts for mid-prefill abort
        - Checks memory pressure via memory_monitor

        Args:
            token_ids: The prompt token IDs to prefill.
            chunk_size: Number of tokens per chunk. Default 2048.
            on_progress: Callback(completed_tokens, total_tokens) called after
                each chunk.
            request_id: Request ID for abort checking and logging.
            pending_aborts: Set of request IDs to abort. Checked between chunks.
            memory_monitor: MemoryMonitor instance for preflight check.

        Returns:
            PrefillResult with KV cache state and timing information.

        Raises:
            PrefillAbortedError: If request is aborted mid-prefill.
            PrefillMemoryExceededError: If memory preflight check fails.
        """
        if not token_ids:
            return PrefillResult(
                token_ids=[],
                num_tokens=0,
                kv_cache=None,
                cached_tokens=0,
                duration_s=0.0,
            )

        total = len(token_ids)

        # ── Memory preflight check ──
        if memory_monitor is not None:
            self._preflight_memory_check(memory_monitor, total, chunk_size, request_id)

        t0 = time.monotonic()
        kv_cache = self._create_kv_cache()
        completed = 0

        for chunk_start in range(0, total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total)
            chunk = token_ids[chunk_start:chunk_end]

            # ── Mid-prefill abort check ──
            if request_id is not None and pending_aborts is not None:
                if check_abort(request_id, pending_aborts):
                    logger.info(
                        f"Prefill aborted for request {request_id} "
                        f"at {completed}/{total} tokens"
                    )
                    raise PrefillAbortedError(
                        request_id=request_id,
                        completed_tokens=completed,
                        total_tokens=total,
                    )

            # ── Memory pressure check between chunks ──
            if memory_monitor is not None and chunk_start > 0:
                if memory_monitor.is_under_pressure(threshold_pct=90.0):
                    logger.warning(
                        f"Memory pressure during prefill of {request_id}: "
                        f"at {completed}/{total} tokens, aborting"
                    )
                    raise PrefillAbortedError(
                        request_id=request_id,
                        completed_tokens=completed,
                        total_tokens=total,
                    )

            # Process this chunk through the model
            self._run_model_step(chunk, kv_cache)
            completed = chunk_end

            # Progress callback
            if on_progress is not None:
                try:
                    on_progress(completed, total)
                except Exception:
                    logger.debug("on_progress callback error", exc_info=True)

        elapsed = time.monotonic() - t0

        return PrefillResult(
            token_ids=token_ids,
            num_tokens=total,
            kv_cache=kv_cache,
            cached_tokens=0,
            duration_s=elapsed,
        )

    def _run_model_step(self, token_ids: list[int], kv_cache: Any | None) -> Any:
        """Run one forward pass through the model with the given tokens.

        This method encapsulates all MLX-specific logic so it can be mocked
        in tests without requiring a real MLX installation.

        Args:
            token_ids: Token IDs for this step.
            kv_cache: KV cache state (may be None).

        Returns:
            Model output (logits).
        """
        if not _HAS_MLX or mx is None:
            # MLX not available — no-op (for testing without GPU)
            return None

        input_ids = mx.array(token_ids).reshape(1, -1)
        stream = self._get_stream()

        with mx.stream(stream):
            output = self._model(input_ids, cache=kv_cache)
            mx.eval(output)

        return output

    def _create_kv_cache(self) -> Any:
        """Create a fresh KV cache for the model.

        Uses the model's cache structure if available, otherwise returns None.
        """
        if not _HAS_MLX or mx is None:
            return None

        if hasattr(self._model, 'make_cache'):
            return self._model.make_cache()
        # Try mlx-lm's create_kv_cache utility
        try:
            from mlx_lm.models.cache import make_prompt_cache as create_kv_cache
            return create_kv_cache(self._model)
        except (ImportError, AttributeError):
            pass
        return None

    def _get_stream(self) -> Any:
        """Get the MLX generation stream for GPU work."""
        try:
            import sys
            gen_mod = sys.modules.get("mlx_lm.generate")
            if gen_mod is not None and hasattr(gen_mod, "generation_stream"):
                return gen_mod.generation_stream
        except Exception:
            pass
        return None

    def _preflight_memory_check(
        self,
        memory_monitor: Any,
        total_tokens: int,
        chunk_size: int,
        request_id: str | None = None,
    ) -> None:
        """Check if there's enough memory for the prefill.

        Raises PrefillMemoryExceededError if estimated peak exceeds available.
        """
        info = memory_monitor.get_memory_info()
        estimated_peak = memory_monitor.estimate_prefill_peak_bytes(
            total_tokens, chunk_size,
        )

        if estimated_peak > 0 and estimated_peak > info.available_bytes:
            from .exceptions import PrefillMemoryExceededError
            raise PrefillMemoryExceededError(
                message=(
                    f"Prefill would require ~{estimated_peak} bytes "
                    f"but only {info.available_bytes} available"
                ),
                request_id=request_id,
                estimated_bytes=estimated_peak,
                limit_bytes=info.available_bytes,
            )


class PrefillAbortedError(Exception):
    """Raised when a prefill is aborted mid-chunk."""

    def __init__(
        self,
        request_id: str | None = None,
        completed_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        self.request_id = request_id
        self.completed_tokens = completed_tokens
        self.total_tokens = total_tokens
        msg = f"Prefill aborted for {request_id} at {completed_tokens}/{total_tokens} tokens"
        super().__init__(msg)


def check_abort(request_id: str, pending_aborts: set[str]) -> bool:
    """Check if a request should be aborted during chunked prefill.

    Thread-safe: CPython GIL guarantees set membership check atomicity.

    Args:
        request_id: The request ID to check.
        pending_aborts: Set of request IDs pending abort.

    Returns:
        True if the request should be aborted.
    """
    return request_id in pending_aborts
