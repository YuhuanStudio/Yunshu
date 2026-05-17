from __future__ import annotations
"""Yunshu Inference Checkpoint/Restore — fault recovery for long-running tasks.

Studied from oMLX's checkpointing and vLLM's preemption recovery:
- InferenceCheckpoint: full request state capture (tokens, sampling, grammar, thinking)
- FaultRecoveryManager: multi-strategy error recovery (retry/truncate/fallback/graceful)
- ProgressEstimator: token-speed-based ETA prediction for variable-speed generation

Thread safety:
  - InferenceCheckpoint uses a threading.Lock for save/load/delete
  - FaultRecoveryManager is lock-free (strategy selection is stateless per call)
  - ProgressEstimator uses no shared mutable state beyond stats counters
"""

import copy
import enum
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Checkpoint State ──


@dataclass
class InferenceState:
    """Full inference state snapshot for a single request.

    Captures everything needed to resume generation from an arbitrary point.
    """

    request_id: str
    # Token state
    generated_tokens: list[int] = field(default_factory=list)
    output_text: str = ""
    position: int = 0  # current sequence position (prompt + generated)

    # Sampling state
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: int = 256
    repetition_penalty: float = 1.0
    seed: int | None = None

    # Grammar / structured output state
    grammar_state: dict[str, Any] = field(default_factory=dict)

    # Thinking / reasoning state
    thinking_active: bool = False
    thinking_budget: int | None = None
    reasoning_state: dict[str, Any] = field(default_factory=dict)

    # Model metadata
    model_name: str = ""

    # Timing
    timestamp: float = field(default_factory=time.monotonic)

    # Error context (only for fault recovery, not clean checkpoints)
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (deep copy to prevent aliasing)."""
        return copy.deepcopy(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InferenceState:
        """Deserialize from dict.

        Raises ValueError if required fields are missing or have wrong types.
        """
        if not isinstance(data, dict):
            raise ValueError(f"Checkpoint data must be a dict, got {type(data).__name__}")
        # Validate required field
        if "request_id" not in data:
            raise ValueError("Checkpoint data missing required field: request_id")
        if not isinstance(data["request_id"], str):
            raise ValueError(f"request_id must be str, got {type(data['request_id']).__name__}")
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        try:
            return cls(**filtered)
        except TypeError as e:
            raise ValueError(f"Invalid checkpoint data: {e}") from e


class AutoCheckpointPolicy(enum.Enum):
    """When to trigger automatic checkpoints."""

    DISABLED = "disabled"  # no auto-checkpoint
    EVERY_N_TOKENS = "every_n_tokens"  # every N generated tokens
    EVERY_N_SECONDS = "every_n_seconds"  # every N seconds of generation


# ── InferenceCheckpoint ──


class InferenceCheckpoint:
    """Manages inference state snapshots for checkpoint/restore.

    Stores checkpoints in-memory (ordered dict) keyed by request_id.
    Supports auto-checkpoint at configurable token intervals.
    Thread-safe via internal lock.
    """

    def __init__(
        self,
        max_checkpoints: int = 1024,
        auto_checkpoint_interval: int = 100,
        auto_checkpoint_policy: AutoCheckpointPolicy = AutoCheckpointPolicy.EVERY_N_TOKENS,
    ):
        self._checkpoints: OrderedDict[str, InferenceState] = OrderedDict()
        self._lock = threading.Lock()
        self._max_checkpoints = max_checkpoints
        self._auto_interval = auto_checkpoint_interval
        self._auto_policy = auto_checkpoint_policy

        # Stats
        self._saves = 0
        self._loads = 0
        self._auto_checkpoints = 0
        self._deletes = 0

    @property
    def auto_interval(self) -> int:
        return self._auto_interval

    @property
    def auto_policy(self) -> AutoCheckpointPolicy:
        return self._auto_policy

    def save(self, request_id: str, state: InferenceState) -> None:
        """Save a checkpoint for the given request.

        Evicts the oldest checkpoint if capacity is exceeded.
        """
        with self._lock:
            if (
                len(self._checkpoints) >= self._max_checkpoints
                and request_id not in self._checkpoints
            ):
                self._checkpoints.popitem(last=False)
            self._checkpoints[request_id] = state
            self._saves += 1
        logger.debug(f"Checkpoint saved: {request_id} (position={state.position})")

    def load(self, request_id: str) -> InferenceState | None:
        """Load a checkpoint by request_id. Returns None if not found."""
        with self._lock:
            state = self._checkpoints.get(request_id)
            if state is not None:
                self._loads += 1
                # Return a deep copy so caller cannot corrupt stored state
                return copy.deepcopy(state)
        return None

    def list_checkpoints(self) -> list[str]:
        """Return all checkpointed request_ids in insertion order."""
        with self._lock:
            return list(self._checkpoints.keys())

    def delete(self, request_id: str) -> bool:
        """Remove a checkpoint. Returns True if found and deleted."""
        with self._lock:
            if request_id in self._checkpoints:
                del self._checkpoints[request_id]
                self._deletes += 1
                return True
        return False

    def should_auto_checkpoint(
        self, request_id: str, tokens_generated: int, elapsed_s: float = 0.0
    ) -> bool:
        """Check whether an auto-checkpoint should be triggered.

        Args:
            request_id: request to check.
            tokens_generated: total tokens generated so far for this request.
            elapsed_s: seconds since last checkpoint (for time-based policy).
                       If 0, will be computed from stored checkpoint timestamp.

        Returns:
            True if auto-checkpoint should fire.
        """
        if self._auto_policy == AutoCheckpointPolicy.DISABLED:
            return False
        if self._auto_policy == AutoCheckpointPolicy.EVERY_N_TOKENS:
            if tokens_generated <= 0:
                return False
            # Only checkpoint if tokens_generated has crossed an interval
            # boundary since last checkpoint
            with self._lock:
                existing = self._checkpoints.get(request_id)
                prev_tokens = len(existing.generated_tokens) if existing else 0
            prev_interval = prev_tokens // self._auto_interval
            curr_interval = tokens_generated // self._auto_interval
            return curr_interval > prev_interval
        if self._auto_policy == AutoCheckpointPolicy.EVERY_N_SECONDS:
            with self._lock:
                existing = self._checkpoints.get(request_id)
                if existing is None:
                    return True
                # Compute elapsed from stored checkpoint timestamp if not given
                if elapsed_s <= 0:
                    elapsed_s = time.monotonic() - existing.timestamp
            return elapsed_s >= self._auto_interval
        return False

    def auto_checkpoint(self, state: InferenceState) -> bool:
        """Perform an auto-checkpoint if the policy says to.

        Returns True if checkpoint was saved.
        """
        tokens = len(state.generated_tokens)
        if self._auto_policy == AutoCheckpointPolicy.EVERY_N_SECONDS:
            if self.should_auto_checkpoint(state.request_id, tokens):
                self.save(state.request_id, state)
                with self._lock:
                    self._auto_checkpoints += 1
                return True
            return False
        if self.should_auto_checkpoint(state.request_id, tokens):
            self.save(state.request_id, state)
            with self._lock:
                self._auto_checkpoints += 1
            return True
        return False

    def get_stats(self) -> dict[str, int]:
        """Return checkpoint statistics."""
        with self._lock:
            return {
                "checkpoints_saved": self._saves,
                "checkpoints_loaded": self._loads,
                "checkpoints_deleted": self._deletes,
                "auto_checkpoints": self._auto_checkpoints,
                "current_count": len(self._checkpoints),
                "max_capacity": self._max_checkpoints,
            }

    def clear(self) -> int:
        """Clear all checkpoints. Returns number cleared."""
        with self._lock:
            count = len(self._checkpoints)
            self._checkpoints.clear()
            return count


# ── Fault Recovery ──


class RecoveryStrategy(enum.Enum):
    """Recovery strategies for failed inference requests."""

    RETRY = "retry"
    TRUNCATE = "truncate"
    FALLBACK_MODEL = "fallback_model"
    GRACEFUL_ERROR = "graceful_error"


@dataclass
class RecoveryResult:
    """Outcome of a fault recovery attempt."""

    strategy: RecoveryStrategy
    success: bool
    restored_state: InferenceState | None = None
    partial_output: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FaultRecoveryManager:
    """Multi-strategy fault recovery for inference errors.

    When an inference error occurs (OOM, timeout, crash, cache corruption),
    the manager selects and executes a recovery strategy from a configurable
    priority list.

    Recovery strategies:
      retry: restart generation from last checkpoint
      truncate: reduce max_tokens and retry from checkpoint
      fallback_model: switch to smaller model (caller supplies model map)
      graceful_error: return partial output with error metadata
    """

    DEFAULT_STRATEGY_PRIORITY = [
        RecoveryStrategy.RETRY,
        RecoveryStrategy.TRUNCATE,
        RecoveryStrategy.FALLBACK_MODEL,
        RecoveryStrategy.GRACEFUL_ERROR,
    ]

    def __init__(
        self,
        checkpoint_manager: InferenceCheckpoint,
        strategy_priority: list[RecoveryStrategy] | None = None,
        max_retries: int = 3,
        truncate_ratio: float = 0.5,
        fallback_models: dict[str, str] | None = None,
    ):
        self._checkpoints = checkpoint_manager
        self._strategy_priority = strategy_priority or list(
            self.DEFAULT_STRATEGY_PRIORITY
        )
        self._max_retries = max_retries
        self._truncate_ratio = truncate_ratio
        self._fallback_models = fallback_models or {}

        # Per-request retry count
        self._retry_counts: dict[str, int] = {}
        self._lock = threading.Lock()

        # Stats
        self._errors_handled = 0
        self._recoveries_success = 0
        self._recoveries_failed = 0
        self._strategy_stats: dict[str, dict[str, int]] = {
            s.value: {"attempts": 0, "successes": 0} for s in RecoveryStrategy
        }

    def configure(
        self,
        strategy_priority: list[RecoveryStrategy] | None = None,
        max_retries: int | None = None,
        truncate_ratio: float | None = None,
        fallback_models: dict[str, str] | None = None,
    ) -> None:
        """Update recovery configuration."""
        if strategy_priority is not None:
            self._strategy_priority = strategy_priority
        if max_retries is not None:
            self._max_retries = max_retries
        if truncate_ratio is not None:
            self._truncate_ratio = truncate_ratio
        if fallback_models is not None:
            self._fallback_models = fallback_models

    def handle_error(
        self,
        request_id: str,
        error: Exception,
        current_state: InferenceState | None = None,
    ) -> RecoveryResult:
        """Select and execute a recovery strategy for the given error.

        Tries strategies in priority order until one succeeds.

        Args:
            request_id: the failed request.
            error: the exception that occurred.
            current_state: current inference state (may be partially updated).

        Returns:
            RecoveryResult with the outcome of the best available recovery.
        """
        with self._lock:
            self._errors_handled += 1

        # Save current state as checkpoint if provided (for future recovery)
        if current_state is not None:
            current_state.last_error = str(error)
            self._checkpoints.save(request_id, current_state)

        # Try each strategy in priority order
        for strategy in self._strategy_priority:
            result = self._execute_strategy(request_id, error, strategy)
            if result.success:
                with self._lock:
                    self._recoveries_success += 1
                return result

        # All strategies exhausted
        with self._lock:
            self._recoveries_failed += 1
        return RecoveryResult(
            strategy=RecoveryStrategy.GRACEFUL_ERROR,
            success=False,
            error_message=f"All recovery strategies exhausted for {request_id}: {error}",
        )

    def _execute_strategy(
        self,
        request_id: str,
        error: Exception,
        strategy: RecoveryStrategy,
    ) -> RecoveryResult:
        """Execute a single recovery strategy."""
        with self._lock:
            self._strategy_stats[strategy.value]["attempts"] += 1

        try:
            if strategy == RecoveryStrategy.RETRY:
                return self._strategy_retry(request_id, error)
            elif strategy == RecoveryStrategy.TRUNCATE:
                return self._strategy_truncate(request_id, error)
            elif strategy == RecoveryStrategy.FALLBACK_MODEL:
                return self._strategy_fallback(request_id, error)
            elif strategy == RecoveryStrategy.GRACEFUL_ERROR:
                return self._strategy_graceful_error(request_id, error)
        except Exception as e:
            logger.warning(f"Recovery strategy {strategy.value} failed: {e}")
            return RecoveryResult(
                strategy=strategy,
                success=False,
                error_message=str(e),
            )

        return RecoveryResult(
            strategy=strategy,
            success=False,
            error_message="Unknown strategy",
        )

    def _strategy_retry(
        self, request_id: str, error: Exception
    ) -> RecoveryResult:
        """Retry from last checkpoint."""
        with self._lock:
            retries = self._retry_counts.get(request_id, 0)
            if retries >= self._max_retries:
                return RecoveryResult(
                    strategy=RecoveryStrategy.RETRY,
                    success=False,
                    error_message=f"Max retries ({self._max_retries}) exceeded",
                )
            self._retry_counts[request_id] = retries + 1

        state = self._checkpoints.load(request_id)
        if state is None:
            return RecoveryResult(
                strategy=RecoveryStrategy.RETRY,
                success=False,
                error_message="No checkpoint available for retry",
            )

        with self._lock:
            self._strategy_stats[RecoveryStrategy.RETRY.value]["successes"] += 1
            retry_count = self._retry_counts.get(request_id, 0)
        return RecoveryResult(
            strategy=RecoveryStrategy.RETRY,
            success=True,
            restored_state=state,
            metadata={"retry_count": retry_count},
        )

    def _strategy_truncate(
        self, request_id: str, error: Exception
    ) -> RecoveryResult:
        """Reduce max_tokens and retry from checkpoint."""
        state = self._checkpoints.load(request_id)
        if state is None:
            return RecoveryResult(
                strategy=RecoveryStrategy.TRUNCATE,
                success=False,
                error_message="No checkpoint available for truncate",
            )

        original_max_tokens = state.max_tokens
        remaining = state.max_tokens - len(state.generated_tokens)
        new_remaining = max(1, int(remaining * self._truncate_ratio))
        state.max_tokens = len(state.generated_tokens) + new_remaining

        with self._lock:
            self._strategy_stats[RecoveryStrategy.TRUNCATE.value]["successes"] += 1
        return RecoveryResult(
            strategy=RecoveryStrategy.TRUNCATE,
            success=True,
            restored_state=state,
            metadata={
                "original_max_tokens": original_max_tokens,
                "truncated_max_tokens": state.max_tokens,
                "truncate_ratio": self._truncate_ratio,
            },
        )

    def _strategy_fallback(
        self, request_id: str, error: Exception
    ) -> RecoveryResult:
        """Switch to a smaller fallback model."""
        state = self._checkpoints.load(request_id)
        if state is None:
            return RecoveryResult(
                strategy=RecoveryStrategy.FALLBACK_MODEL,
                success=False,
                error_message="No checkpoint available for fallback",
            )

        current_model = state.model_name
        fallback = self._fallback_models.get(current_model)
        if fallback is None:
            return RecoveryResult(
                strategy=RecoveryStrategy.FALLBACK_MODEL,
                success=False,
                error_message=f"No fallback model configured for '{current_model}'",
            )

        state.model_name = fallback
        with self._lock:
            self._strategy_stats[RecoveryStrategy.FALLBACK_MODEL.value][
                "successes"
            ] += 1
        return RecoveryResult(
            strategy=RecoveryStrategy.FALLBACK_MODEL,
            success=True,
            restored_state=state,
            metadata={
                "original_model": current_model,
                "fallback_model": fallback,
            },
        )

    def _strategy_graceful_error(
        self, request_id: str, error: Exception
    ) -> RecoveryResult:
        """Return partial output with error metadata."""
        state = self._checkpoints.load(request_id)
        partial = state.output_text if state is not None else ""

        with self._lock:
            self._strategy_stats[RecoveryStrategy.GRACEFUL_ERROR.value][
                "successes"
            ] += 1
        return RecoveryResult(
            strategy=RecoveryStrategy.GRACEFUL_ERROR,
            success=True,
            partial_output=partial,
            error_message=str(error),
            metadata={
                "tokens_generated": len(state.generated_tokens) if state else 0,
                "position": state.position if state else 0,
            },
        )

    def reset_retry_count(self, request_id: str) -> None:
        """Reset per-request retry counter (call after successful recovery)."""
        with self._lock:
            self._retry_counts.pop(request_id, None)

    def get_stats(self) -> dict[str, Any]:
        """Return fault recovery statistics."""
        with self._lock:
            total = self._errors_handled
            success_rate = (
                self._recoveries_success / total if total > 0 else 0.0
            )
            return {
                "errors_handled": self._errors_handled,
                "recoveries_success": self._recoveries_success,
                "recoveries_failed": self._recoveries_failed,
                "success_rate": success_rate,
                "strategy_stats": dict(self._strategy_stats),
            }


# ── Progress Estimator ──


@dataclass
class _ProgressRecord:
    """Internal progress tracking record."""

    request_id: str
    total_generated: int = 0
    total_elapsed_ms: float = 0.0
    max_tokens: int = 256
    # Sliding window for speed estimation
    last_update_time: float = 0.0
    last_update_tokens: int = 0
    current_speed_tps: float = 0.0  # tokens per second
    # History for accuracy tracking
    estimates_made: int = 0
    estimate_errors_ms: list[float] = field(default_factory=list)


@dataclass
class ProgressInfo:
    """Progress estimation result."""

    request_id: str
    progress_pct: float  # 0.0 to 1.0
    tokens_generated: int
    tokens_remaining: int
    estimated_remaining_ms: float  # -1 if unknown
    speed_tps: float  # tokens per second, 0 if unknown


class ProgressEstimator:
    """Estimates generation progress and remaining time.

    Uses a sliding window of token generation speed to predict completion.
    Handles variable-speed generation (spec decode bursts, preemption pauses)
    by tracking recent speed rather than overall average.
    """

    def __init__(
        self,
        speed_window_s: float = 5.0,
        min_samples: int = 2,
        max_history: int = 1000,
    ):
        self._records: dict[str, _ProgressRecord] = {}
        self._speed_window_s = speed_window_s
        self._min_samples = min_samples
        self._max_history = max_history

        # Stats
        self._estimates_made = 0
        self._total_error_ms: float = 0.0
        self._completed_estimates: int = 0

    def register(
        self, request_id: str, max_tokens: int = 256
    ) -> None:
        """Register a new request for progress tracking."""
        now = time.monotonic()
        self._records[request_id] = _ProgressRecord(
            request_id=request_id,
            max_tokens=max_tokens,
            last_update_time=now,
            last_update_tokens=0,
        )
        # Evict oldest if over capacity
        if len(self._records) > self._max_history:
            oldest_key = next(iter(self._records))
            del self._records[oldest_key]

    def update(
        self,
        request_id: str,
        tokens_generated: int,
        elapsed_ms: float,
    ) -> None:
        """Update progress for a request.

        Args:
            request_id: the request.
            tokens_generated: total tokens generated so far.
            elapsed_ms: wall-clock ms since generation started.
        """
        rec = self._records.get(request_id)
        if rec is None:
            return

        now = time.monotonic()
        dt = now - rec.last_update_time
        d_tokens = tokens_generated - rec.last_update_tokens

        # Update sliding-window speed estimate
        if dt > 0 and d_tokens > 0:
            window_speed = d_tokens / dt  # tokens/s
            # Exponential moving average (EMA) for smoothness
            if rec.current_speed_tps > 0:
                alpha = min(1.0, dt / self._speed_window_s)
                rec.current_speed_tps = (
                    alpha * window_speed
                    + (1 - alpha) * rec.current_speed_tps
                )
            else:
                rec.current_speed_tps = window_speed

        rec.total_generated = tokens_generated
        rec.total_elapsed_ms = elapsed_ms
        rec.last_update_time = now
        rec.last_update_tokens = tokens_generated

    def estimate_remaining(self, request_id: str) -> ProgressInfo | None:
        """Estimate remaining generation time.

        Returns None if request not tracked.
        Returns estimated_remaining_ms=-1 if speed unknown.
        """
        rec = self._records.get(request_id)
        if rec is None:
            return None

        remaining = max(0, rec.max_tokens - rec.total_generated)
        progress_pct = (
            rec.total_generated / rec.max_tokens if rec.max_tokens > 0 else 1.0
        )
        progress_pct = min(1.0, progress_pct)

        if rec.current_speed_tps > 0 and remaining > 0:
            estimated_remaining_ms = (remaining / rec.current_speed_tps) * 1000.0
        elif remaining == 0:
            estimated_remaining_ms = 0.0
        else:
            estimated_remaining_ms = -1.0

        rec.estimates_made += 1
        self._estimates_made += 1

        return ProgressInfo(
            request_id=request_id,
            progress_pct=progress_pct,
            tokens_generated=rec.total_generated,
            tokens_remaining=remaining,
            estimated_remaining_ms=estimated_remaining_ms,
            speed_tps=rec.current_speed_tps,
        )

    def get_progress(self, request_id: str) -> float:
        """Get progress percentage (0.0 to 1.0) for a request.

        Returns 0.0 if request not tracked.
        """
        rec = self._records.get(request_id)
        if rec is None:
            return 0.0
        if rec.max_tokens <= 0:
            return 1.0
        return min(1.0, rec.total_generated / rec.max_tokens)

    def record_completion(
        self,
        request_id: str,
        actual_remaining_ms: float,
        estimated_remaining_ms: float,
    ) -> None:
        """Record actual completion for accuracy tracking.

        Called when a request finishes to compare estimate vs actual.
        """
        error_ms = abs(actual_remaining_ms - estimated_remaining_ms)
        self._total_error_ms += error_ms
        self._completed_estimates += 1

    def unregister(self, request_id: str) -> None:
        """Remove a request from tracking."""
        self._records.pop(request_id, None)

    def get_stats(self) -> dict[str, Any]:
        """Return progress estimator statistics."""
        avg_error_ms = (
            self._total_error_ms / self._completed_estimates
            if self._completed_estimates > 0
            else 0.0
        )
        return {
            "estimates_made": self._estimates_made,
            "active_requests": len(self._records),
            "completed_estimates": self._completed_estimates,
            "average_error_ms": avg_error_ms,
        }

    def clear(self) -> None:
        """Clear all tracking state."""
        self._records.clear()
        self._estimates_made = 0
        self._total_error_ms = 0.0
        self._completed_estimates = 0
