from __future__ import annotations

"""Yunshu External Prefill — prefill outside BatchGenerator for memory preflight,
chunked progress tracking, mid-prefill abort, and remote KV transfer.

Currently, mlx-lm's BatchGenerator manages its own KV cache internally, so we
cannot directly inject external KV cache into it. The external prefill path's
value TODAY is:
  1. Memory preflight: estimate if a prompt fits before committing GPU memory
  2. Chunked progress tracking: on_progress callback per chunk for dashboards
  3. Mid-prefill abort: interrupt long prefills between chunks
  4. Prefix cache integration: report cached_tokens from KVCacheManager
  5. Remote KV transfer: send prefilled KV blocks to decode nodes
     (enabled via YUNSHU_KV_TRANSFER=1)
  6. Disaggregated prefill server/client (YUNSHU_EXTERNAL_PREFILL=1)
     for offloading prefill to a dedicated node

Future work: direct KV cache injection into BatchGenerator (requires MLX-level
changes to BatchGenerator's internal cache management).

Architecture:
  Scheduler._schedule_waiting() → ExternalPrefiller.prefill_chunked()
    → memory preflight check (raises PrefillMemoryExceededError if OOM)
    → chunked model forward passes (building KV cache externally)
    → on_progress callback per chunk
    → mid-prefill abort check between chunks
    → optional: transfer_prefill_result() sends KV blocks to remote node
  Then BatchGenerator.insert() proceeds as normal for the decode phase.

  ExternalPrefillServer / ExternalPrefillClient:
    When YUNSHU_EXTERNAL_PREFILL=1 and YUNSHU_PREFILL_ROLE=server,
    ExternalPrefillServer listens for TCP connections and runs prefills
    on behalf of remote decode nodes.

    When YUNSHU_EXTERNAL_PREFILL=1 and YUNSHU_PREFILL_ROLE=client,
    ExternalPrefillClient.prefill_remote() sends token IDs to the
    remote prefill server and receives the resulting KV cache.
"""

import asyncio
import json
import logging
import os
import struct
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

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
    # last-position logits from the prefill forward = the logits that
    # predict the FIRST generated token. Captured so the disaggregated decode can
    # reuse the prefilled KV (cache offset = len(token_ids)) and decode forward
    # WITHOUT re-prefilling. None when MLX/model unavailable.
    last_logits: Any | None = None
    # serialized KV blocks carried over the wire from a REMOTE prefill so the
    # decode node can RECONSTRUCT the cache (make_prompt_cache + load_kv_blocks_into_cache)
    # and reuse it via generate_with_kv — instead of the old behaviour where the wire
    # dropped KV (kv_cache=None) and decode silently re-prefilled the whole prompt. The
    # serialize→reconstruct→reuse round-trip is verified lossless.
    kv_blocks: Any | None = None  # list[KVBlockData] on the decode side


# ── ExternalPrefillConfig ─────────────────────────────────────────────


@dataclass
class ExternalPrefillConfig:
    """Configuration for external prefill server/client.

    Attributes:
        server_host: Host address for the prefill server.
        server_port: Port for the prefill server.
        max_connections: Maximum concurrent connections.
        chunk_size: Default chunk size for prefill operations.
        timeout_seconds: Timeout for prefill operations.
        retry_attempts: Number of retry attempts for client requests.
        compression: Compression algorithm ('none', 'lz4', 'zstd').
    """

    server_host: str = "0.0.0.0"
    server_port: int = 7891
    max_connections: int = 8
    chunk_size: int = 2048
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    compression: str = "none"

    @classmethod
    def from_env(cls) -> ExternalPrefillConfig:
        """Create config from environment variables."""
        return cls(
            server_host=os.environ.get("YUNSHU_PREFILL_HOST", "0.0.0.0"),
            server_port=int(os.environ.get("YUNSHU_PREFILL_PORT", "7891")),
            max_connections=int(os.environ.get("YUNSHU_PREFILL_MAX_CONN", "8")),
            chunk_size=int(os.environ.get("YUNSHU_PREFILL_CHUNK_SIZE", "2048")),
            timeout_seconds=float(os.environ.get("YUNSHU_PREFILL_TIMEOUT", "30.0")),
            retry_attempts=int(os.environ.get("YUNSHU_PREFILL_RETRIES", "3")),
            compression=os.environ.get("YUNSHU_PREFILL_COMPRESSION", "none"),
        )


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
        self._kv_transfer_client = None

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
        _out = self._run_model_step(token_ids, kv_cache)

        # keep the last-position logits (first generated-token logits)
        # so the decode side can reuse this KV cache instead of re-prefilling.
        _last_logits = None
        if _out is not None:
            try:
                _last_logits = _out[:, -1, :]
                if mx is not None:
                    mx.eval(_last_logits)
            except Exception:
                _last_logits = None

        elapsed = time.monotonic() - t0

        return PrefillResult(
            token_ids=token_ids,
            num_tokens=len(token_ids),
            kv_cache=kv_cache,
            cached_tokens=cached_prefix_len,
            duration_s=elapsed,
            last_logits=_last_logits,
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
        _out = None  # last forward output (for last_logits capture)

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
            _out = self._run_model_step(chunk, kv_cache)
            completed = chunk_end

            # Progress callback
            if on_progress is not None:
                try:
                    on_progress(completed, total)
                except Exception:
                    logger.debug("on_progress callback error", exc_info=True)

        # keep the FINAL chunk's last-position logits (first generated
        # token) so disaggregated decode can reuse the prefilled KV.
        _last_logits = None
        if _out is not None:
            try:
                _last_logits = _out[:, -1, :]
                if mx is not None:
                    mx.eval(_last_logits)
            except Exception:
                _last_logits = None

        elapsed = time.monotonic() - t0

        return PrefillResult(
            token_ids=token_ids,
            num_tokens=total,
            kv_cache=kv_cache,
            cached_tokens=0,
            duration_s=elapsed,
            last_logits=_last_logits,
        )

    def transfer_prefill_result(
        self,
        result: PrefillResult,
        request_id: str | None = None,
        model_name: str = "",
        layer_count: int = 0,
    ) -> Any:
        """Transfer prefilled KV blocks to a remote decode node.

        Only active when YUNSHU_KV_TRANSFER=1 is set. Extracts KV blocks
        from the PrefillResult's kv_cache and sends them via KVTransferClient.

        This enables disaggregated prefill: the prefill node sends the
        completed KV state to the decode node, which can then skip prefill
        and immediately begin decode.

        Args:
            result: The PrefillResult from a completed prefill operation.
            request_id: Request ID for tracking.
            model_name: Model name for cache compatibility.
            layer_count: Number of KV layers.

        Returns:
            KVTransferResult if transfer was attempted, None if disabled.
        """
        if os.environ.get("YUNSHU_KV_TRANSFER", "0") != "1":
            return None

        if result.kv_cache is None:
            logger.debug(
                "Skipping KV transfer for %s: no KV cache in PrefillResult",
                request_id,
            )
            return None

        try:
            from .kv_transfer import (
                KVTransferClient,
                KVTransferConfig,
                extract_kv_blocks_from_cache,
            )

            config = KVTransferConfig.from_env()
            blocks = extract_kv_blocks_from_cache(
                result.kv_cache,
                result.token_ids,
                block_size=config.block_size,
            )

            if not blocks:
                logger.debug(
                    "No KV blocks extracted for %s",
                    request_id,
                )
                return None

            if self._kv_transfer_client is None:
                self._kv_transfer_client = KVTransferClient(config)
            client = self._kv_transfer_client
            transfer_result = client.send_blocks_sync(
                blocks=blocks,
                request_id=request_id,
                model_name=model_name,
                total_tokens=result.num_tokens,
                layer_count=layer_count,
            )

            if transfer_result.status.value == "completed":
                logger.info(
                    "KV transfer sent: %d blocks for %s (%d bytes, %.1f ms)",
                    transfer_result.blocks_transferred,
                    request_id,
                    transfer_result.bytes_transferred,
                    transfer_result.duration_seconds * 1000,
                )
            else:
                logger.warning(
                    "KV transfer failed for %s: %s",
                    request_id,
                    transfer_result.error,
                )

            return transfer_result

        except Exception:
            logger.warning(
                "KV transfer error for %s",
                request_id,
                exc_info=True,
            )
            return None

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
        if not _HAS_MLX or mx is None or self._model is None:
            # MLX not available or no model loaded — no-op (for testing without GPU)
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
        if not _HAS_MLX or mx is None or self._model is None:
            return None

        if hasattr(self._model, "make_cache"):
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
            logger.debug("MLX generation stream lookup failed", exc_info=True)
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
            total_tokens,
            chunk_size,
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


# ── Wire protocol helpers ─────────────────────────────────────────────

# Message format:
# 4 bytes  — magic (b"YPFL")
# 4 bytes  — header length (big-endian uint32)
# N bytes  — JSON header (UTF-8)
# M bytes  — payload (KV cache data, may be empty)

_WIRE_MAGIC = b"YPFL"


def _encode_message(header: dict, payload: bytes = b"") -> bytes:
    """Encode a prefill message into wire format."""
    header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")
    header_len = len(header_bytes)
    return _WIRE_MAGIC + struct.pack(">I", header_len) + header_bytes + payload


def _decode_message(data: bytes) -> tuple[dict, bytes]:
    """Decode a prefill message from wire format.

    Returns (header_dict, payload_bytes).

    Raises ValueError if magic bytes don't match.
    """
    if len(data) < 12:
        raise ValueError("Message too short")
    magic = data[:4]
    if magic != _WIRE_MAGIC:
        raise ValueError(f"Invalid magic: {magic!r}")
    header_len = struct.unpack(">I", data[4:8])[0]
    header_bytes = data[8 : 8 + header_len]
    header = json.loads(header_bytes.decode("utf-8"))
    payload = data[8 + header_len :]
    return header, payload


async def _read_message(reader: asyncio.StreamReader) -> tuple[dict, bytes]:
    """Read a framed message from an asyncio StreamReader."""
    magic = await reader.readexactly(4)
    if magic != _WIRE_MAGIC:
        raise ValueError(f"Invalid magic: {magic!r}")
    header_len_bytes = await reader.readexactly(4)
    header_len = struct.unpack(">I", header_len_bytes)[0]
    header_bytes = await reader.readexactly(header_len)
    header = json.loads(header_bytes.decode("utf-8"))
    payload_len = header.get("payload_len", 0)
    payload = b""
    if payload_len > 0:
        payload = await reader.readexactly(payload_len)
    return header, payload


def _serialize_kv_blocks(blocks: list) -> bytes:
    """Frame a list[KVBlockData] into self-describing bytes for the prefill wire.

    Layout: [n_blocks:u32] then per block [block_hash:i64][token_count:u32][n_layers:u32]
    then per layer [layer_idx:u32][len:u32][layer_bytes]. layer_bytes is exactly what
    load_kv_blocks_into_cache reads back (K||V self-describing tensors)."""
    out = [struct.pack(">I", len(blocks))]
    for b in blocks:
        ld = b.layer_data
        out.append(struct.pack(">QII", int(b.block_hash), int(b.token_count), len(ld)))
        for layer_idx, lbytes in ld.items():
            out.append(struct.pack(">II", int(layer_idx), len(lbytes)))
            out.append(lbytes)
    return b"".join(out)


def _deserialize_kv_blocks(data: bytes) -> list:
    """Inverse of _serialize_kv_blocks → list[KVBlockData] ready for load_kv_blocks_into_cache."""
    from .kv_transfer import KVBlockData

    blocks: list = []
    if not data:
        return blocks
    off = 0
    (n_blocks,) = struct.unpack_from(">I", data, off)
    off += 4
    for _ in range(n_blocks):
        block_hash, token_count, n_layers = struct.unpack_from(">QII", data, off)
        off += 16
        layer_data: dict[int, bytes] = {}
        for _ in range(n_layers):
            layer_idx, ln = struct.unpack_from(">II", data, off)
            off += 8
            layer_data[layer_idx] = data[off : off + ln]
            off += ln
        blocks.append(
            KVBlockData(
                block_hash=block_hash, token_count=token_count, layer_data=layer_data
            )
        )
    return blocks


def _serialize_prefill_result(result: PrefillResult) -> bytes:
    """Serialize a PrefillResult into bytes for wire transfer.

    The payload CARRIES the KV cache (serialized blocks) + the first-token
    logits when present, so a remote decode node can reconstruct the cache and REUSE it
    instead of re-prefilling. Payload layout (each section length-prefixed):
      [len:u32][token_bytes] [len:u32][logits_bytes] [len:u32][kv_blocks_bytes]
    The old format shipped only token_bytes and forced a re-prefill (the disagg no-op).
    """
    token_bytes = (
        struct.pack(f">{len(result.token_ids)}I", *result.token_ids)
        if result.token_ids
        else b""
    )
    kv_bytes = b""
    logits_bytes = b""
    if result.kv_cache is not None and result.token_ids:
        try:
            from .kv_transfer import _tensor_to_bytes, extract_kv_blocks_from_cache

            blocks = extract_kv_blocks_from_cache(result.kv_cache, result.token_ids)
            kv_bytes = _serialize_kv_blocks(blocks)
            if result.last_logits is not None:
                logits_bytes = _tensor_to_bytes(result.last_logits)
        except Exception:
            logger.debug(
                "KV serialization for prefill wire failed; decode will re-prefill",
                exc_info=True,
            )
            kv_bytes = b""
            logits_bytes = b""
    payload = (
        struct.pack(">I", len(token_bytes))
        + token_bytes
        + struct.pack(">I", len(logits_bytes))
        + logits_bytes
        + struct.pack(">I", len(kv_bytes))
        + kv_bytes
    )
    header = {
        "num_tokens": result.num_tokens,
        "cached_tokens": result.cached_tokens,
        "duration_s": result.duration_s,
        "payload_len": len(payload),
        "has_kv_cache": bool(kv_bytes),
    }
    return _encode_message(header, payload)


def _deserialize_prefill_result(data: bytes) -> PrefillResult:
    """Deserialize a PrefillResult from wire bytes.

    Parse the length-prefixed payload back into token_ids + (when present) the
    serialized KV blocks and first-token logits, so the decode node can reconstruct the
    cache and reuse it. kv_cache stays None here (rebuilding a live cache needs the model,
    which lives on the decode node) — the decode endpoint does the reconstruction.
    """
    header, payload = _decode_message(data)
    n_tokens = header.get("num_tokens", 0)
    token_ids: list[int] = []
    kv_blocks = None
    last_logits = None
    try:
        off = 0
        (tlen,) = struct.unpack_from(">I", payload, off)
        off += 4
        tbytes = payload[off : off + tlen]
        off += tlen
        if tbytes and n_tokens > 0:
            token_ids = list(struct.unpack(f">{n_tokens}I", tbytes[: n_tokens * 4]))
        (llen,) = struct.unpack_from(">I", payload, off)
        off += 4
        lbytes = payload[off : off + llen]
        off += llen
        (klen,) = struct.unpack_from(">I", payload, off)
        off += 4
        kbytes = payload[off : off + klen]
        off += klen
        if kbytes:
            kv_blocks = _deserialize_kv_blocks(kbytes)
            if lbytes:
                from .kv_transfer import _bytes_to_tensor

                last_logits = _bytes_to_tensor(lbytes)
    except (struct.error, IndexError):
        # Legacy/short payload (bare token_bytes) — fall back to token-only parse.
        if payload and n_tokens > 0:
            token_ids = list(struct.unpack(f">{n_tokens}I", payload[: n_tokens * 4]))
    return PrefillResult(
        token_ids=token_ids,
        num_tokens=n_tokens,
        kv_cache=None,  # reconstructed on the decode node (needs the model)
        cached_tokens=header.get("cached_tokens", 0),
        duration_s=header.get("duration_s", 0.0),
        last_logits=last_logits,
        kv_blocks=kv_blocks,
    )


# ── ExternalPrefillServer ─────────────────────────────────────────────


class ExternalPrefillServer:
    """TCP server that receives prefill requests from remote nodes.

    Listens for incoming prefill requests, runs the model prefill,
    and returns the resulting KV cache data to the client.

    Usage:
        config = ExternalPrefillConfig(server_port=7891)
        server = ExternalPrefillServer(model, tokenizer, config)
        await server.serve(host="0.0.0.0", port=7891)

    Wire protocol:
      Client sends:  {"type": "prefill", "num_tokens": N, "chunk_size": C, ...}
                     + binary payload (token IDs as int32 array)
      Server sends:  {"type": "result", "num_tokens": N, ...}
                     + binary payload (token IDs as int32 array)
      Server sends:  {"type": "error", "message": "..."} on failure
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: ExternalPrefillConfig | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._config = config or ExternalPrefillConfig()
        self._prefiller = ExternalPrefiller(model, tokenizer)

        # Server state
        self._server: asyncio.Server | None = None
        self._running = False
        self._active_connections: int = 0
        self._lock = asyncio.Lock()

        # Stats
        self._requests_served: int = 0
        self._total_prefill_time_s: float = 0.0
        self._total_bytes_transferred: int = 0
        self._errors: int = 0

    async def serve(self, host: str | None = None, port: int | None = None) -> None:
        """Start the TCP prefill server."""
        h = host or self._config.server_host
        p = port or self._config.server_port
        self._running = True

        self._server = await asyncio.start_server(
            self._handle_connection,
            host=h,
            port=p,
        )
        logger.info(f"ExternalPrefillServer listening on {h}:{p}")

        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        """Stop the prefill server gracefully."""
        self._running = False
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        logger.info("ExternalPrefillServer stopped")

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single client connection."""
        async with self._lock:
            reject = self._active_connections >= self._config.max_connections
            if not reject:
                self._active_connections += 1
        if reject:
            writer.close()
            await writer.wait_closed()
            logger.warning("Rejected connection: max_connections reached")
            return

        try:
            while self._running:
                try:
                    header, payload = await asyncio.wait_for(
                        _read_message(reader),
                        timeout=self._config.timeout_seconds,
                    )
                except TimeoutError:
                    break
                except (asyncio.IncompleteReadError, ConnectionError):
                    break

                response = await self._handle_prefill(header, payload)

                try:
                    writer.write(response)
                    await writer.drain()
                except (ConnectionError, OSError):
                    break
        finally:
            async with self._lock:
                self._active_connections -= 1
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                logger.debug("writer close failed", exc_info=True)

    async def handle_prefill(
        self,
        request_header: dict,
        request_payload: bytes,
    ) -> bytes:
        """Handle a single prefill request.

        Args:
            request_header: JSON header with 'num_tokens', 'chunk_size', etc.
            request_payload: Binary payload with token IDs (int32 array).

        Returns:
            Wire-encoded response (PrefillResult or error).
        """
        return await self._handle_prefill(request_header, request_payload)

    async def _handle_prefill(
        self,
        header: dict,
        payload: bytes,
    ) -> bytes:
        """Process a prefill request and return serialized response."""
        header.get("type", "prefill")
        num_tokens = header.get("num_tokens", 0)
        chunk_size = header.get("chunk_size", self._config.chunk_size)

        # Decode token IDs from payload
        token_ids: list[int] = []
        if payload and num_tokens > 0:
            try:
                token_ids = list(
                    struct.unpack(f">{num_tokens}I", payload[: num_tokens * 4])
                )
            except struct.error as e:
                self._errors += 1
                return _encode_message(
                    {
                        "type": "error",
                        "message": f"Invalid payload: {e}",
                    }
                )

        if not token_ids:
            result = PrefillResult(token_ids=[], num_tokens=0)
            self._requests_served += 1
            return _serialize_prefill_result(result)

        try:
            prefill_coro = asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self._prefiller.prefill_chunked(
                    token_ids=token_ids,
                    chunk_size=chunk_size,
                ),
            )
            result = await asyncio.wait_for(
                prefill_coro,
                timeout=self._config.timeout_seconds,
            )
            self._requests_served += 1
            self._total_prefill_time_s += result.duration_s
            self._total_bytes_transferred += len(payload)
            return _serialize_prefill_result(result)

        except TimeoutError:
            self._errors += 1
            return _encode_message(
                {
                    "type": "error",
                    "message": f"Prefill timed out after {self._config.timeout_seconds}s",
                }
            )
        except PrefillAbortedError as e:
            self._errors += 1
            return _encode_message(
                {
                    "type": "error",
                    "message": f"Prefill aborted: {e}",
                    "completed_tokens": e.completed_tokens,
                    "total_tokens": e.total_tokens,
                }
            )
        except Exception as e:
            self._errors += 1
            logger.warning("Prefill server error: %s", e, exc_info=True)
            return _encode_message(
                {
                    "type": "error",
                    "message": str(e),
                }
            )

    def get_stats(self) -> dict:
        """Return server statistics."""
        avg_time = (
            self._total_prefill_time_s / self._requests_served
            if self._requests_served > 0
            else 0.0
        )
        return {
            "running": self._running,
            "requests_served": self._requests_served,
            "avg_prefill_time_s": round(avg_time, 4),
            "bytes_transferred": self._total_bytes_transferred,
            "active_connections": self._active_connections,
            "errors": self._errors,
        }


# ── ExternalPrefillClient ─────────────────────────────────────────────


class ExternalPrefillClient:
    """TCP client that sends prefill requests to remote prefill nodes.

    Connects to a remote ExternalPrefillServer, sends token IDs for
    prefill, and receives the resulting KV cache.

    Features:
    - Automatic retry with exponential backoff
    - Connection health checking
    - Stats tracking

    Usage:
        config = ExternalPrefillConfig(server_host="10.0.0.1", server_port=7891)
        client = ExternalPrefillClient(config)
        result = await client.prefill_remote(token_ids=[1, 2, 3, ...])
    """

    def __init__(self, config: ExternalPrefillConfig | None = None) -> None:
        self._config = config or ExternalPrefillConfig()

        # Stats
        self._requests_sent: int = 0
        self._total_latency_s: float = 0.0
        self._successes: int = 0
        self._failures: int = 0

    async def prefill_remote(
        self,
        token_ids: list[int],
        model_config: dict | None = None,
        chunk_size: int | None = None,
    ) -> PrefillResult:
        """Send a prefill request to the remote server and return the result.

        Args:
            token_ids: The prompt token IDs to prefill.
            model_config: Optional model configuration for the server.
            chunk_size: Override chunk size for this request.

        Returns:
            PrefillResult from the remote prefill.

        Raises:
            ConnectionError: If unable to connect after retries.
            RuntimeError: If the server returns an error.
        """
        cs = chunk_size or self._config.chunk_size
        n_tokens = len(token_ids)

        # Encode token IDs as binary payload
        payload = struct.pack(f">{n_tokens}I", *token_ids) if token_ids else b""

        request_header = {
            "type": "prefill",
            "num_tokens": n_tokens,
            "chunk_size": cs,
        }
        if model_config:
            request_header["model_config"] = model_config

        request_header["payload_len"] = len(payload)
        request_data = _encode_message(request_header, payload)

        last_error: Exception | None = None
        t_start = time.monotonic()
        for attempt in range(self._config.retry_attempts):
            try:
                result = await self._send_request(request_data)
                self._requests_sent += 1
                self._successes += 1
                self._total_latency_s += time.monotonic() - t_start
                return result
            except Exception as e:
                last_error = e
                self._failures += 1
                if attempt < self._config.retry_attempts - 1:
                    backoff = 0.1 * (2**attempt)
                    logger.info(
                        "Prefill client retry %d/%d after %.1fs: %s",
                        attempt + 1,
                        self._config.retry_attempts,
                        backoff,
                        e,
                    )
                    await asyncio.sleep(backoff)

        self._requests_sent += 1
        raise ConnectionError(
            f"Failed after {self._config.retry_attempts} attempts: {last_error}"
        )

    async def _send_request(self, data: bytes) -> PrefillResult:
        """Send a request to the server and return the response."""
        reader, writer = await asyncio.open_connection(
            self._config.server_host,
            self._config.server_port,
        )

        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                writer.write(data)
                await writer.drain()

                # Read response
                magic = await reader.readexactly(4)
                if magic != _WIRE_MAGIC:
                    raise ValueError(f"Invalid response magic: {magic!r}")
                header_len_bytes = await reader.readexactly(4)
                header_len = struct.unpack(">I", header_len_bytes)[0]
                header_bytes = await reader.readexactly(header_len)
                header = json.loads(header_bytes.decode("utf-8"))
                payload_len = header.get("payload_len", 0)
                payload = b""
                if payload_len > 0:
                    payload = await reader.readexactly(payload_len)

                # Check for error response
                if header.get("type") == "error":
                    msg = header.get("message", "Unknown server error")
                    raise RuntimeError(f"Remote prefill error: {msg}")

                # Deserialize PrefillResult
                full_data = magic + header_len_bytes + header_bytes + payload
                return _deserialize_prefill_result(full_data)

        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                logger.debug("writer close failed", exc_info=True)

    async def health_check(self) -> bool:
        """Check if the remote prefill server is reachable.

        Returns True if the server is reachable, False otherwise.
        """
        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                reader, writer = await asyncio.open_connection(
                    self._config.server_host,
                    self._config.server_port,
                )
            writer.close()
            await writer.wait_closed()
            return True
        except (TimeoutError, ConnectionError, OSError):
            return False

    def get_stats(self) -> dict:
        """Return client statistics."""
        avg_latency = (
            self._total_latency_s / self._requests_sent
            if self._requests_sent > 0
            else 0.0
        )
        success_rate = (
            self._successes / self._requests_sent if self._requests_sent > 0 else 0.0
        )
        return {
            "requests_sent": self._requests_sent,
            "avg_latency_s": round(avg_latency, 4),
            "success_rate": round(success_rate, 4),
            "successes": self._successes,
            "failures": self._failures,
        }


# ── EngineCore wiring helper ──────────────────────────────────────────


def get_prefill_role() -> str | None:
    """Return the prefill role from environment: 'server', 'client', or None."""
    if os.environ.get("YUNSHU_EXTERNAL_PREFILL", "0") != "1":
        return None
    return os.environ.get("YUNSHU_PREFILL_ROLE", None)


def get_external_prefill_stats() -> dict:
    """Return external prefill stats from EngineCore singleton, if active.

    This is called from the monitoring endpoint to expose prefill server/client
    stats. Returns {"active": False} when not enabled.
    """
    try:
        from .batched_engine import get_batched_engine

        engine = get_batched_engine()
        if engine is None:
            return {"active": False}
        core = getattr(engine, "_engine_core", None)
        if core is None:
            return {"active": False}
        server = getattr(core, "_prefill_server", None)
        client = getattr(core, "_prefill_client", None)
        result: dict = {"active": True, "role": get_prefill_role()}
        if server is not None:
            result["server"] = server.get_stats()
        if client is not None:
            result["client"] = client.get_stats()
        return result
    except Exception:
        logger.debug("get_external_prefill_stats failed", exc_info=True)
        return {"active": False}
