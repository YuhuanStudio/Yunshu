from __future__ import annotations

"""Yunshu MemoryGuard — proactive memory admission control.

Prevents OOM by rejecting requests that would exceed available memory:
- preflight_check: estimate prompt + decode KV + prefill peak before accepting
- generation_guard: concurrent request limit under memory pressure
- get_recommended_max_tokens: compute max tokens that fit in remaining memory

Architecture:
  EngineCore.add_request()
    → MemoryGuard.preflight_check(num_prompt_tokens, max_tokens)
    → reject immediately with finish_reason="memory_exceeded" if insufficient memory

  BatchedEngine.generate() / stream_generate()
    → MemoryGuard preflight before submitting to EngineCore
    → return GenerationOutput with finish_reason="memory_limit" on rejection

Integration:
  - KVEvictionPredictor (kv_optimizations.py) can be attached for prediction-based
    eviction decisions. Import via:
      from .kv_optimizations import KVEvictionPredictor
"""

import logging
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .memory_monitor import MemoryMonitor

logger = logging.getLogger(__name__)

# Safety margin: keep 10% of available memory as headroom
_SAFETY_MARGIN_PCT = 0.10

# Default maximum concurrent requests when model info is unknown
_DEFAULT_MAX_CONCURRENT = 64


class MemoryGuard:
    """Proactive memory admission controller for LLM inference.

    Works alongside MemoryMonitor (which tracks actual memory) to provide
    pre-admission checks that reject requests before they cause OOM.
    """

    def __init__(
        self,
        memory_monitor: MemoryMonitor,
        max_concurrent_requests: int = _DEFAULT_MAX_CONCURRENT,
        safety_margin_pct: float = _SAFETY_MARGIN_PCT,
    ) -> None:
        self._monitor = memory_monitor
        self._max_concurrent = max_concurrent_requests
        if safety_margin_pct < 0:
            logger.warning(
                f"safety_margin_pct={safety_margin_pct} is negative, "
                f"clamping to 0.0"
            )
            safety_margin_pct = 0.0
        elif safety_margin_pct > 1.0:
            logger.warning(
                f"safety_margin_pct={safety_margin_pct} exceeds 1.0, "
                f"clamping to 0.5"
            )
            safety_margin_pct = 0.5
        self._safety_margin_pct = safety_margin_pct

        # Stats tracking (atomic via _stats_lock)
        self._stats_lock = threading.Lock()
        self._total_checks: int = 0
        self._total_rejections: int = 0
        self._preflight_rejections: int = 0
        self._concurrent_rejections: int = 0

        # Optional eviction predictor for prediction-based eviction (kv_optimizations)
        self._eviction_predictor: Any | None = None

    def preflight_check(
        self,
        num_prompt_tokens: int,
        max_tokens: int,
    ) -> tuple[bool, str]:
        """Check if a request can fit in available memory.

        Estimates total memory needed:
        1. Prompt KV bytes: memory for storing the prompt's KV cache
        2. Decode KV bytes: memory for max_tokens worth of generation KV cache
        3. Prefill peak bytes: worst-case memory during prefill computation

        Compares against available memory minus safety margin.

        Returns:
            (ok, reason) — ok=True if request can proceed,
            ok=False with reason if it should be rejected.
        """
        with self._stats_lock:
            self._total_checks += 1

        info = self._monitor.get_memory_info()
        available = info.available_bytes
        safety_headroom = int(available * self._safety_margin_pct)
        usable = available - safety_headroom

        if usable <= 0:
            with self._stats_lock:
                self._total_rejections += 1
                self._preflight_rejections += 1
            reason = (
                f"No usable memory available: "
                f"available={info.available_bytes}, "
                f"safety_margin={safety_headroom}"
            )
            logger.warning(f"Preflight rejected: {reason}")
            return (False, reason)

        # 1. Prompt KV memory (included in prefill_peak, not added separately)
        prompt_kv = self._monitor.estimate_prompt_kv_bytes(num_prompt_tokens)

        # 2. Decode KV memory (max_tokens worth of generation)
        decode_kv = self._monitor.estimate_prompt_kv_bytes(max_tokens)

        # 3. Prefill peak memory (prompt processing spike — includes prompt_kv)
        prefill_peak = self._monitor.estimate_prefill_peak_bytes(
            total_prompt_tokens=num_prompt_tokens,
            chunk_size=min(num_prompt_tokens, 2048),
        )

        # Total = decode_kv + prefill_peak (prefill_peak already includes prompt_kv)
        total_estimated = decode_kv + prefill_peak

        if total_estimated > usable:
            with self._stats_lock:
                self._total_rejections += 1
                self._preflight_rejections += 1
            reason = (
                f"Insufficient memory: estimated={total_estimated}, "
                f"usable={usable}, "
                f"prompt_kv={prompt_kv}, decode_kv={decode_kv}, "
                f"prefill_peak={prefill_peak}"
            )
            logger.warning(
                f"Preflight rejected ({num_prompt_tokens} prompt + "
                f"{max_tokens} max_tokens): {reason}"
            )
            return (False, reason)

        logger.debug(
            f"Preflight OK: {num_prompt_tokens} prompt + {max_tokens} max_tokens, "
            f"estimated={total_estimated}, usable={usable}"
        )
        return (True, "")

    def generation_guard(self, num_active_requests: int) -> bool:
        """Check if we can accept more concurrent requests.

        Returns False if:
        - Number of active requests exceeds max_concurrent_requests
        - Memory is under pressure (>90% utilization)
        """
        with self._stats_lock:
            self._total_checks += 1

        # Hard limit on concurrent requests
        if num_active_requests >= self._max_concurrent:
            with self._stats_lock:
                self._total_rejections += 1
                self._concurrent_rejections += 1
            logger.warning(
                f"Generation guard rejected: "
                f"{num_active_requests} >= {self._max_concurrent} concurrent"
            )
            return False

        # Memory pressure check
        if self._monitor.is_under_pressure(threshold_pct=90.0):
            with self._stats_lock:
                self._total_rejections += 1
                self._concurrent_rejections += 1
            info = self._monitor.get_memory_info()
            logger.warning(
                f"Generation guard rejected: memory pressure "
                f"{info.utilization_pct:.1f}%"
            )
            return False

        return True

    def get_recommended_max_tokens(self, num_prompt_tokens: int) -> int:
        """Compute the maximum generation tokens that fit in remaining memory.

        Given the prompt length, calculates how many decode tokens can
        be generated without exceeding available memory. Returns 0 if
        even the prompt cannot fit.
        """
        info = self._monitor.get_memory_info()
        available = info.available_bytes
        safety_headroom = int(available * self._safety_margin_pct)
        usable = available - safety_headroom

        if usable <= 0:
            return 0

        # Prompt KV cost
        prompt_kv = self._monitor.estimate_prompt_kv_bytes(num_prompt_tokens)
        if prompt_kv == 0:
            # Model info not set — return a sensible default
            return 256

        # Prefill peak cost
        prefill_peak = self._monitor.estimate_prefill_peak_bytes(
            total_prompt_tokens=num_prompt_tokens,
            chunk_size=min(num_prompt_tokens, 2048),
        )

        remaining = usable - prompt_kv - prefill_peak
        if remaining <= 0:
            return 0

        # Per-token decode cost
        per_token_kv = self._monitor.estimate_prompt_kv_bytes(1)
        if per_token_kv == 0:
            return 256

        max_tokens = remaining // per_token_kv
        return max(0, min(max_tokens, 32768))  # Cap at 32K tokens

    def get_stats(self) -> dict:
        """Return memory guard statistics."""
        monitor_stats = self._monitor.get_stats()
        with self._stats_lock:
            total_checks = self._total_checks
            total_rejections = self._total_rejections
            preflight_rejections = self._preflight_rejections
            concurrent_rejections = self._concurrent_rejections
        return {
            "total_checks": total_checks,
            "total_rejections": total_rejections,
            "preflight_rejections": preflight_rejections,
            "concurrent_rejections": concurrent_rejections,
            "max_concurrent_requests": self._max_concurrent,
            "safety_margin_pct": self._safety_margin_pct,
            "rejection_rate": (
                total_rejections / total_checks * 100
                if total_checks > 0
                else 0.0
            ),
            "memory": monitor_stats,
        }

    def set_eviction_predictor(self, predictor: Any) -> None:
        """Set a KVEvictionPredictor for prediction-based eviction decisions.

        When attached, the predictor is consulted before evicting blocks,
        allowing more intelligent retention of blocks predicted to be needed.

        Args:
            predictor: A KVEvictionPredictor instance from kv_optimizations.
        """
        self._eviction_predictor = predictor

    def should_evict_block(self, block_id: str) -> bool:
        """Check if a block should be evicted.

        Uses the eviction predictor if available, otherwise returns True
        (allow eviction by default).

        Args:
            block_id: The block ID to evaluate.

        Returns:
            True if the block should be evicted.
        """
        if self._eviction_predictor is not None:
            return self._eviction_predictor.should_evict(block_id)
        return True
