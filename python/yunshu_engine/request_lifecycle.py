from __future__ import annotations
"""Request lifecycle orchestrator — unified request state machine.

Manages the full lifecycle of an inference request:
  QUEUED → PREFILLING → DECODING → FINISHED
               ↘ REJECTED (memory/schedule)
  FINISHED → RETRYING → QUEUED (fault recovery)

Provides:
  - State transitions with validation
  - Timeout tracking (TTFT, total)
  - Retry coordination with backoff
  - Request-level metrics collection
  - Concurrency control with adaptive limits
"""

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)


class RequestPhase(Enum):
    QUEUED = auto()
    PREFILLING = auto()
    DECODING = auto()
    FINISHED = auto()
    REJECTED = auto()
    RETRYING = auto()
    ABORTED = auto()


@dataclass
class RequestLifecycleState:
    """Tracks a single request through its lifecycle."""
    request_id: str
    phase: RequestPhase = RequestPhase.QUEUED
    # Timing
    created_at: float = field(default_factory=time.monotonic)
    last_active_at: float = 0.0  # Updated on retry; used for timeout check
    queued_at: float = 0.0
    prefill_start: float = 0.0
    prefill_end: float = 0.0
    decode_start: float = 0.0
    decode_end: float = 0.0
    finished_at: float = 0.0
    # Token counts
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Retry
    retry_count: int = 0
    max_retries: int = 3
    last_error: str = ""
    # Priority + metadata
    priority: int = 0
    model: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def queue_time_ms(self) -> float | None:
        if self.queued_at and self.prefill_start:
            return (self.prefill_start - self.queued_at) * 1000
        if self.queued_at:
            return (time.monotonic() - self.queued_at) * 1000
        return None

    @property
    def ttft_ms(self) -> float | None:
        if self.prefill_start and self.decode_start:
            return (self.decode_start - self.prefill_start) * 1000
        if self.prefill_start:
            return (time.monotonic() - self.prefill_start) * 1000
        return None

    @property
    def total_time_ms(self) -> float | None:
        if self.finished_at and self.created_at:
            return (self.finished_at - self.created_at) * 1000
        return None

    @property
    def decode_time_ms(self) -> float | None:
        if self.decode_start and self.decode_end:
            return (self.decode_end - self.decode_start) * 1000
        return None

    @property
    def throughput_tps(self) -> float | None:
        dt = self.decode_time_ms
        if dt and dt > 0 and self.completion_tokens > 0:
            return self.completion_tokens / (dt / 1000.0)
        return None

    def transition(self, new_phase: RequestPhase) -> bool:
        """Attempt a state transition. Returns True if valid."""
        valid_transitions = {
            RequestPhase.QUEUED: {RequestPhase.PREFILLING, RequestPhase.REJECTED, RequestPhase.ABORTED},
            RequestPhase.PREFILLING: {RequestPhase.DECODING, RequestPhase.REJECTED, RequestPhase.ABORTED},
            RequestPhase.DECODING: {RequestPhase.FINISHED, RequestPhase.ABORTED, RequestPhase.RETRYING},
            RequestPhase.REJECTED: {RequestPhase.RETRYING, RequestPhase.FINISHED},
            RequestPhase.RETRYING: {RequestPhase.QUEUED, RequestPhase.FINISHED},
            RequestPhase.FINISHED: set(),
            RequestPhase.ABORTED: set(),
        }
        if new_phase not in valid_transitions.get(self.phase, set()):
            return False

        now = time.monotonic()
        if new_phase == RequestPhase.PREFILLING:
            self.prefill_start = now
        elif new_phase == RequestPhase.DECODING:
            self.decode_start = now
            self.prefill_end = now
        elif new_phase == RequestPhase.FINISHED:
            self.finished_at = now
            self.decode_end = now
        elif new_phase == RequestPhase.QUEUED:
            self.queued_at = now
        elif new_phase == RequestPhase.RETRYING:
            self.retry_count += 1

        self.phase = new_phase
        return True


class AdaptiveConcurrencyController:
    """Adaptive concurrency limiter with AIMD (Additive Increase Multiplicative Decrease).

    Starts at initial concurrency, increases by 1 every success window,
    decreases by half on SLO violation. Provides smooth backpressure
    without oscillation.
    """

    def __init__(
        self,
        initial: int = 8,
        minimum: int = 1,
        maximum: int = 128,
        increase_window: float = 5.0,
        decrease_factor: float = 0.5,
        slo_ttft_ms: float = 500.0,
        slo_total_ms: float = 10000.0,
    ) -> None:
        self._current = initial
        self._minimum = minimum
        self._maximum = maximum
        self._increase_window = increase_window
        self._decrease_factor = decrease_factor
        self._slo_ttft = slo_ttft_ms
        self._slo_total = slo_total_ms
        self._last_increase_time = time.monotonic()
        self._slo_violations = 0
        self._total_adjustments = 0

    @classmethod
    def from_env(cls) -> AdaptiveConcurrencyController:
        # YUNSHU_MAX_CONCURRENT caps the maximum if set (CLI --max-concurrent)
        max_concurrent_env = os.environ.get("YUNSHU_MAX_CONCURRENT")
        max_concurrent_cap = int(max_concurrent_env) if max_concurrent_env else None
        maximum = int(os.environ.get("YUNSHU_CONCURRENCY_MAX", "128"))
        if max_concurrent_cap is not None and max_concurrent_cap > 0:
            maximum = min(maximum, max_concurrent_cap)
        initial = int(os.environ.get("YUNSHU_CONCURRENCY_INITIAL", "8"))
        # Clamp initial to maximum
        initial = min(initial, maximum)
        return cls(
            initial=initial,
            minimum=int(os.environ.get("YUNSHU_CONCURRENCY_MIN", "1")),
            maximum=maximum,
            slo_ttft_ms=float(os.environ.get("YUNSHU_SLO_TTFT_MS", "500.0")),
            slo_total_ms=float(os.environ.get("YUNSHU_SLO_TOTAL_MS", "10000.0")),
        )

    @property
    def current_limit(self) -> int:
        return self._current

    def report_success(self, state: RequestLifecycleState) -> None:
        """Report a successful request completion."""
        ttft = state.ttft_ms
        total = state.total_time_ms

        if ttft is not None and ttft > self._slo_ttft:
            self._slo_violations += 1
            self._decrease()
            return

        if total is not None and total > self._slo_total:
            self._slo_violations += 1
            self._decrease()
            return

        # No SLO violation — try to increase
        now = time.monotonic()
        if now - self._last_increase_time >= self._increase_window:
            if self._current < self._maximum:
                self._current = min(self._current + 1, self._maximum)
                self._last_increase_time = now
                self._total_adjustments += 1

    def report_failure(self, error: str = "") -> None:
        """Report a request failure (OOM, timeout, etc.)."""
        self._decrease()

    def _decrease(self) -> None:
        new_val = max(self._minimum, int(self._current * self._decrease_factor))
        if new_val != self._current:
            self._current = new_val
            self._total_adjustments += 1

    def get_stats(self) -> dict:
        return {
            "current_limit": self._current,
            "minimum": self._minimum,
            "maximum": self._maximum,
            "slo_violations": self._slo_violations,
            "total_adjustments": self._total_adjustments,
            "slo_ttft_ms": self._slo_ttft,
            "slo_total_ms": self._slo_total,
        }


class RequestLifecycleOrchestrator:
    """Orchestrates request lifecycle across the inference pipeline.

    Coordinates:
    - Request state tracking (QUEUED → PREFILLING → DECODING → FINISHED)
    - Adaptive concurrency control
    - Retry coordination with exponential backoff
    - Request-level metrics collection
    - Timeout management

    Integration points:
    - EngineCore.add_request() → orchestrator.on_request_added()
    - Scheduler.step() → orchestrator.on_step_output()
    - EngineCore._cleanup_request() → orchestrator.on_request_finished()
    """

    def __init__(
        self,
        concurrency_controller: AdaptiveConcurrencyController | None = None,
        max_pending: int = 256,
        default_timeout_ms: float = 30000.0,
    ) -> None:
        self._concurrency = concurrency_controller or AdaptiveConcurrencyController.from_env()
        self._max_pending = max_pending
        self._default_timeout = default_timeout_ms

        self._states: dict[str, RequestLifecycleState] = {}
        self._active_count = 0
        self._pending_queue: list[str] = []

        # Metrics
        self._total_requests = 0
        self._total_completed = 0
        self._total_rejected = 0
        self._total_retried = 0
        self._total_timeouts = 0
        self._phase_transitions: dict[tuple[RequestPhase, RequestPhase], int] = defaultdict(int)

        # Per-model counters
        self._model_counts: dict[str, dict] = defaultdict(
            lambda: {"total": 0, "completed": 0, "rejected": 0}
        )

    def on_request_added(
        self,
        request_id: str,
        prompt_tokens: int = 0,
        priority: int = 0,
        model: str = "",
        max_retries: int = 3,
        metadata: dict | None = None,
    ) -> RequestLifecycleState:
        """Called when a new request is added to the system."""
        state = RequestLifecycleState(
            request_id=request_id,
            priority=priority,
            model=model,
            prompt_tokens=prompt_tokens,
            max_retries=max_retries,
            metadata=metadata or {},
        )
        state.queued_at = time.monotonic()
        state.last_active_at = state.created_at
        self._states[request_id] = state
        self._total_requests += 1
        self._model_counts[model]["total"] += 1

        # Check concurrency limit
        if self._active_count >= self._concurrency.current_limit:
            # Check if we can queue it
            if len(self._pending_queue) >= self._max_pending:
                state.transition(RequestPhase.REJECTED)
                state.last_error = "concurrency_limit_exceeded"
                self._total_rejected += 1
                return state

            self._pending_queue.append(request_id)
            return state

        # Auto-start prefill for immediate requests
        self.on_prefill_start(request_id)
        return state

    def on_prefill_start(self, request_id: str) -> bool:
        """Mark request as entering prefill phase."""
        state = self._states.get(request_id)
        if state is None:
            return False
        if state.transition(RequestPhase.PREFILLING):
            self._active_count += 1
            self._phase_transitions[(RequestPhase.QUEUED, RequestPhase.PREFILLING)] += 1
            return True
        return False

    def on_decode_start(self, request_id: str) -> bool:
        """Mark request as entering decode phase."""
        state = self._states.get(request_id)
        if state is None:
            return False
        if state.transition(RequestPhase.DECODING):
            self._phase_transitions[(RequestPhase.PREFILLING, RequestPhase.DECODING)] += 1
            return True
        return False

    def on_request_finished(
        self,
        request_id: str,
        completion_tokens: int = 0,
        finish_reason: str = "stop",
    ) -> RequestLifecycleState | None:
        """Mark request as finished and report to concurrency controller."""
        state = self._states.get(request_id)
        if state is None:
            return None

        # Decrement if the request was in an active phase. Use max(0, ...)
        # guard to prevent drift when scheduler bypasses the orchestrator
        # and directly mutates req.status (dual state machines can diverge).
        was_active = state.phase in (RequestPhase.PREFILLING, RequestPhase.DECODING)
        state.completion_tokens = completion_tokens
        state.transition(RequestPhase.FINISHED)
        self._total_completed += 1
        if was_active or self._active_count > 0:
            self._active_count = max(0, self._active_count - 1)
        self._model_counts[state.model]["completed"] += 1

        # Report to concurrency controller
        self._concurrency.report_success(state)

        # Remove from tracking (keep stats)
        del self._states[request_id]

        # Try to promote a pending request
        if self._pending_queue:
            next_id = self._pending_queue.pop(0)
            next_state = self._states.get(next_id)
            if next_state:
                self.on_prefill_start(next_id)

        return state

    def on_request_failed(
        self,
        request_id: str,
        error: str = "",
        retryable: bool = True,
    ) -> RequestLifecycleState | None:
        """Handle request failure with optional retry."""
        state = self._states.get(request_id)
        if state is None:
            return None

        state.last_error = error

        if retryable and state.retry_count < state.max_retries:
            # Go to REJECTED first (valid from any active phase)
            was_active = state.phase in (RequestPhase.PREFILLING, RequestPhase.DECODING)
            if state.phase not in (RequestPhase.FINISHED, RequestPhase.ABORTED, RequestPhase.REJECTED):
                state.transition(RequestPhase.REJECTED)
            state.transition(RequestPhase.RETRYING)
            state.transition(RequestPhase.QUEUED)
            # Reset timeout baseline so retried requests get a fresh timeout window
            state.last_active_at = time.monotonic()
            if was_active:
                self._active_count = max(0, self._active_count - 1)
            self._total_retried += 1
            # Re-queue the request so it can be promoted when a slot opens
            self._pending_queue.append(request_id)
            return state

        # Terminal failure — go to REJECTED then FINISHED
        was_active = state.phase in (RequestPhase.PREFILLING, RequestPhase.DECODING)
        if state.phase not in (RequestPhase.FINISHED, RequestPhase.ABORTED, RequestPhase.REJECTED):
            state.transition(RequestPhase.REJECTED)
        if state.phase != RequestPhase.FINISHED:
            state.transition(RequestPhase.FINISHED)
        self._total_rejected += 1
        if was_active:
            self._active_count = max(0, self._active_count - 1)
        self._model_counts[state.model]["rejected"] += 1
        self._concurrency.report_failure(error)
        # Remove from pending queue if present
        try:
            self._pending_queue.remove(request_id)
        except ValueError:
            pass
        del self._states[request_id]
        return state

    def on_request_aborted(self, request_id: str) -> None:
        """Handle request abortion."""
        state = self._states.get(request_id)
        if state is None:
            return
        # Only decrement active_count if the request was in an active phase
        # (PREFILLING or DECODING). Requests still in QUEUED/REJECTED/RETRYING
        # were never counted as active.
        was_active = state.phase in (RequestPhase.PREFILLING, RequestPhase.DECODING)
        state.transition(RequestPhase.ABORTED)
        if was_active:
            self._active_count = max(0, self._active_count - 1)
        # Remove from pending queue if present (request was queued but never promoted)
        try:
            self._pending_queue.remove(request_id)
        except ValueError:
            pass
        del self._states[request_id]

    def check_timeouts(self) -> list[str]:
        """Check for timed-out requests and abort them.

        Uses last_active_at (updated on retry) so retried requests get
        a fresh timeout window instead of being measured from creation.
        """
        now = time.monotonic()
        timeout_s = self._default_timeout / 1000.0
        timed_out = []
        for rid, state in list(self._states.items()):
            if state.phase in (RequestPhase.FINISHED, RequestPhase.ABORTED):
                continue
            # Use last_active_at for timeout: set at creation and reset on retry
            reference_time = state.last_active_at or state.created_at
            if now - reference_time > timeout_s:
                self.on_request_failed(rid, error="timeout", retryable=False)
                self._total_timeouts += 1
                timed_out.append(rid)
        return timed_out

    def get_state(self, request_id: str) -> RequestLifecycleState | None:
        return self._states.get(request_id)

    @property
    def active_count(self) -> int:
        return self._active_count

    @property
    def pending_count(self) -> int:
        return len(self._pending_queue)

    def get_stats(self) -> dict:
        phase_counts = defaultdict(int)
        for state in self._states.values():
            phase_counts[state.phase.name] += 1

        return {
            "active_requests": self._active_count,
            "pending_requests": len(self._pending_queue),
            "total_requests": self._total_requests,
            "total_completed": self._total_completed,
            "total_rejected": self._total_rejected,
            "total_retried": self._total_retried,
            "total_timeouts": self._total_timeouts,
            "phase_distribution": dict(phase_counts),
            "concurrency": self._concurrency.get_stats(),
            "per_model": {k: dict(v) for k, v in self._model_counts.items() if k},
        }

    def get_latency_percentiles(self) -> dict:
        """Calculate latency percentiles from recent finished requests.

        Since we delete states on finish, this returns aggregate stats
        based on the concurrency controller's observations.
        """
        return {
            "slo_ttft_ms": self._concurrency._slo_ttft,
            "slo_total_ms": self._concurrency._slo_total,
            "slo_violation_rate": (
                self._total_rejected / max(self._total_requests, 1)
            ),
            "completion_rate": (
                self._total_completed / max(self._total_requests, 1)
            ),
        }
