from __future__ import annotations

"""Yunshu Memory-Aware Request Scheduler.

Tracks GPU memory budget per request and provides admission control:
- Before adding a new request, estimates its KV cache memory needs
- Reserves memory for admitted requests, releases on completion
- Pauses admissions under high memory pressure (integrates with MemoryGuard)
- Provides memory budget stats for monitoring

Architecture:
  Request lifecycle:
    1. estimate_kv_memory() — calculate bytes needed for prompt + decode
    2. can_admit_request() — check if estimated memory fits in budget
    3. reserve_memory() — reserve bytes for the request
    4. [request runs...]
    5. release_memory() — return reserved bytes on completion

  Pressure integration:
    - When utilization exceeds pressure_threshold_pct, new admissions are paused
    - When utilization drops below (pressure_threshold_pct - hysteresis_pct),
      admissions resume
    - This prevents thrashing between admit/pause states

Reference:
  - vLLM Scheduler policy (block-manager-v2 + policy.py)
  - Orca iteration-level scheduling (Yu et al., OSDI 2022)
"""

import logging
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────────────────


@dataclass
class MemoryBudget:
    """Current memory budget state."""

    total_bytes: int
    reserved_bytes: int
    used_bytes: int  # reserved_bytes + overhead
    available_bytes: int
    utilization_pct: float

    @property
    def free_bytes(self) -> int:
        return max(0, self.total_bytes - self.used_bytes)


@dataclass
class RequestMemoryEntry:
    """Per-request memory reservation tracking."""

    request_id: str
    reserved_bytes: int
    num_tokens: int
    timestamp: float
    model_name: str = ""


@dataclass
class SchedulerStats:
    """Memory-aware scheduler statistics."""

    total_admissions: int = 0
    total_rejections: int = 0
    total_releases: int = 0
    total_bytes_reserved: int = 0
    total_bytes_released: int = 0
    peak_reserved_bytes: int = 0
    current_reserved_bytes: int = 0
    active_requests: int = 0
    pressure_pauses: int = 0
    pressure_resumes: int = 0
    is_paused: bool = False


# ── Constants ─────────────────────────────────────────────────────────────────

# Default memory budget: 4 GB
_DEFAULT_BUDGET = 4 * 1024 ** 3

# Default model parameters for estimation
_DEFAULT_NUM_LAYERS = 32
_DEFAULT_NUM_KV_HEADS = 8
_DEFAULT_HEAD_DIM = 128
_DEFAULT_DTYPE_SIZE = 2  # float16

# Pressure thresholds
_DEFAULT_PRESSURE_THRESHOLD = 85.0  # Pause at 85% utilization
_DEFAULT_HYSTERESIS = 5.0  # Resume at 80% (85% - 5%)

# Safety margin: never use more than this fraction of budget
_SAFETY_MARGIN = 0.95


# ═══════════════════════════════════════════════════════════════════════════════
# MemoryAwareScheduler
# ═══════════════════════════════════════════════════════════════════════════════


class MemoryAwareScheduler:
    """Memory-aware request scheduler with admission control.

    Tracks GPU memory budget per request. Before adding a new request,
    estimates its KV cache memory needs and checks if memory is available.
    When pressure is high, pauses admissions until memory is freed.

    Thread-safe: all state mutations are protected by a lock.
    """

    def __init__(
        self,
        total_budget_bytes: int = _DEFAULT_BUDGET,
        pressure_threshold_pct: float = _DEFAULT_PRESSURE_THRESHOLD,
        hysteresis_pct: float = _DEFAULT_HYSTERESIS,
        safety_margin: float = _SAFETY_MARGIN,
    ) -> None:
        self._total_budget = total_budget_bytes
        self._pressure_threshold = pressure_threshold_pct
        self._hysteresis = hysteresis_pct
        self._safety_margin = min(1.0, max(0.5, safety_margin))

        # Model parameters for estimation (set via set_model_config)
        self._num_layers: int = _DEFAULT_NUM_LAYERS
        self._num_kv_heads: int = _DEFAULT_NUM_KV_HEADS
        self._head_dim: int = _DEFAULT_HEAD_DIM
        self._dtype_size: int = _DEFAULT_DTYPE_SIZE

        # Per-request reservations
        self._reservations: dict[str, RequestMemoryEntry] = {}

        # State
        self._lock = threading.Lock()
        self._is_paused: bool = False
        self._paused_since: float | None = None

        # Stats
        self._stats = SchedulerStats()

    def set_model_config(
        self,
        num_layers: int | None = None,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        dtype_size: int | None = None,
    ) -> None:
        """Set model architecture parameters for memory estimation.

        Should be called once after model loading.
        """
        if num_layers is not None:
            self._num_layers = num_layers
        if num_kv_heads is not None:
            self._num_kv_heads = num_kv_heads
        if head_dim is not None:
            self._head_dim = head_dim
        if dtype_size is not None:
            self._dtype_size = dtype_size

        per_token = self._per_token_kv_bytes()
        logger.info(
            f"MemoryAwareScheduler model config: "
            f"{self._num_layers}L, {self._num_kv_heads} KV heads, "
            f"{self._head_dim}d, {self._dtype_size}B dtype. "
            f"Per-token KV: {per_token} bytes"
        )

    def estimate_kv_memory(
        self,
        num_tokens: int,
        max_tokens: int = 0,
        model_config: dict | None = None,
    ) -> int:
        """Estimate KV cache memory needed for a request.

        Args:
            num_tokens: Number of prompt tokens.
            max_tokens: Maximum generation tokens (default 0 = prompt only).
            model_config: Optional override for model parameters.

        Returns:
            Estimated bytes needed for KV cache.
        """
        if model_config is not None:
            layers = model_config.get("num_layers", self._num_layers)
            kv_heads = model_config.get("num_kv_heads", self._num_kv_heads)
            head_dim = model_config.get("head_dim", self._head_dim)
            dtype_size = model_config.get("dtype_size", self._dtype_size)
            per_token = layers * kv_heads * head_dim * dtype_size * 2
        else:
            per_token = self._per_token_kv_bytes()

        total_tokens = num_tokens + max_tokens
        return total_tokens * per_token

    def can_admit_request(
        self,
        estimated_memory: int,
    ) -> tuple[bool, str]:
        """Check if a request can be admitted given memory constraints.

        Checks:
        1. Is the scheduler paused due to memory pressure?
        2. Would this request exceed the safety margin?
        3. Is there enough free memory?

        Args:
            estimated_memory: Bytes estimated for this request
                (from estimate_kv_memory).

        Returns:
            (can_admit, reason) — can_admit=True if the request can proceed.
        """
        with self._lock:
            budget = self._get_budget()

            # Check pressure pause
            if self._is_paused:
                # Check if we can resume
                if budget.utilization_pct <= (self._pressure_threshold - self._hysteresis):
                    self._is_paused = False
                    self._paused_since = None
                    self._stats.pressure_resumes += 1
                    self._stats.is_paused = False
                    logger.info(
                        f"MemoryAwareScheduler resumed admissions "
                        f"(utilization={budget.utilization_pct:.1f}%)"
                    )
                else:
                    reason = (
                        f"Scheduler paused: utilization={budget.utilization_pct:.1f}% "
                        f"exceeds threshold={self._pressure_threshold}%"
                    )
                    return (False, reason)

            # Check safety margin
            usable = int(self._total_budget * self._safety_margin)
            if budget.used_bytes + estimated_memory > usable:
                reason = (
                    f"Would exceed safety margin: "
                    f"used={budget.used_bytes} + requested={estimated_memory} "
                    f"> usable={usable} "
                    f"(budget={self._total_budget} * {self._safety_margin})"
                )
                return (False, reason)

            # Check if adding this would push into pressure zone
            projected_util = (
                (budget.used_bytes + estimated_memory) / self._total_budget * 100
            )
            if projected_util >= self._pressure_threshold:
                # Admit this one but pause further admissions
                self._is_paused = True
                self._paused_since = time.monotonic()
                self._stats.pressure_pauses += 1
                self._stats.is_paused = True
                logger.warning(
                    f"MemoryAwareScheduler pausing admissions after this request "
                    f"(projected utilization={projected_util:.1f}%)"
                )

            return (True, "")

    def reserve_memory(
        self,
        request_id: str,
        num_bytes: int,
        num_tokens: int = 0,
        model_name: str = "",
    ) -> bool:
        """Reserve memory for a request.

        Must be called after can_admit_request() returns True.
        If insufficient memory, returns False.

        Args:
            request_id: Unique request identifier.
            num_bytes: Bytes to reserve.
            num_tokens: Number of tokens for this reservation.
            model_name: Model name for tracking.

        Returns:
            True if reservation succeeded, False if insufficient memory.
        """
        with self._lock:
            budget = self._get_budget()

            if num_bytes > budget.available_bytes:
                logger.warning(
                    f"Cannot reserve {num_bytes} bytes for {request_id}: "
                    f"only {budget.available_bytes} available"
                )
                self._stats.total_rejections += 1
                # Unwind pressure-pause if can_admit_request set it but
                # reserve failed — prevents permanent pause when memory
                # pressure was triggered by the admission check itself.
                if self._is_paused:
                    fresh_budget = self._get_budget()
                    if fresh_budget.utilization_pct <= (self._pressure_threshold - self._hysteresis):
                        self._is_paused = False
                        self._paused_since = None
                        self._stats.pressure_resumes += 1
                        self._stats.is_paused = False
                return False

            entry = RequestMemoryEntry(
                request_id=request_id,
                reserved_bytes=num_bytes,
                num_tokens=num_tokens,
                timestamp=time.monotonic(),
                model_name=model_name,
            )
            self._reservations[request_id] = entry

            self._stats.total_admissions += 1
            self._stats.total_bytes_reserved += num_bytes
            self._stats.current_reserved_bytes += num_bytes
            self._stats.active_requests = len(self._reservations)
            self._stats.peak_reserved_bytes = max(
                self._stats.peak_reserved_bytes,
                self._stats.current_reserved_bytes,
            )

            logger.debug(
                f"Reserved {num_bytes} bytes for request {request_id} "
                f"({num_tokens} tokens)"
            )
            return True

    def release_memory(self, request_id: str) -> int:
        """Release reserved memory for a completed request.

        Args:
            request_id: The request to release.

        Returns:
            Number of bytes released (0 if request not found).
        """
        with self._lock:
            entry = self._reservations.pop(request_id, None)
            if entry is None:
                return 0

            released = entry.reserved_bytes
            self._stats.total_releases += 1
            self._stats.total_bytes_released += released
            self._stats.current_reserved_bytes -= released
            self._stats.active_requests = len(self._reservations)

            # Check if we can resume from pressure pause
            if self._is_paused:
                budget = self._get_budget()
                if budget.utilization_pct <= (self._pressure_threshold - self._hysteresis):
                    self._is_paused = False
                    self._paused_since = None
                    self._stats.pressure_resumes += 1
                    self._stats.is_paused = False
                    logger.info(
                        f"MemoryAwareScheduler resumed admissions after release "
                        f"(utilization={budget.utilization_pct:.1f}%)"
                    )

            logger.debug(
                f"Released {released} bytes for request {request_id}"
            )
            return released

    def get_memory_budget(self) -> MemoryBudget:
        """Return current memory budget state.

        Returns:
            MemoryBudget with total/used/available bytes and utilization.
        """
        with self._lock:
            return self._get_budget()

    def get_stats(self) -> SchedulerStats:
        """Return scheduler statistics.

        Returns a snapshot of admissions, rejections, memory utilization,
        and pressure state.
        """
        with self._lock:
            return SchedulerStats(
                total_admissions=self._stats.total_admissions,
                total_rejections=self._stats.total_rejections,
                total_releases=self._stats.total_releases,
                total_bytes_reserved=self._stats.total_bytes_reserved,
                total_bytes_released=self._stats.total_bytes_released,
                peak_reserved_bytes=self._stats.peak_reserved_bytes,
                current_reserved_bytes=self._stats.current_reserved_bytes,
                active_requests=self._stats.active_requests,
                pressure_pauses=self._stats.pressure_pauses,
                pressure_resumes=self._stats.pressure_resumes,
                is_paused=self._is_paused,
            )

    @property
    def is_paused(self) -> bool:
        """Whether admissions are currently paused due to memory pressure."""
        with self._lock:
            return self._is_paused

    @property
    def total_budget(self) -> int:
        return self._total_budget

    # ── Private ────────────────────────────────────────────────────────────

    def _per_token_kv_bytes(self) -> int:
        """Calculate per-token KV cache memory in bytes.

        Per layer: keys + values, each (1, kv_heads, 1, head_dim).
        Total = num_layers * 2 * kv_heads * head_dim * dtype_size.
        """
        return self._num_layers * 2 * self._num_kv_heads * self._head_dim * self._dtype_size

    def _get_budget(self) -> MemoryBudget:
        """Compute current memory budget (caller must hold lock)."""
        reserved = sum(
            entry.reserved_bytes for entry in self._reservations.values()
        )
        usable = int(self._total_budget * self._safety_margin)
        used = min(reserved, usable)
        available = max(0, usable - used)
        util = (used / self._total_budget * 100) if self._total_budget > 0 else 0.0

        return MemoryBudget(
            total_bytes=self._total_budget,
            reserved_bytes=reserved,
            used_bytes=used,
            available_bytes=available,
            utilization_pct=util,
        )
