from __future__ import annotations

"""Yunshu Auto-Tuning Engine — adaptive configuration based on observed performance.

Four components:
  - PerformanceProfiler: continuous profiling of inference metrics
  - AutoTuner: hill-climbing parameter tuning with bounded exploration
  - AdaptiveBatchSizer: dynamic batch size based on memory / queue / SLO
  - SLOMonitor: Service Level Objective compliance tracking

Integration points:
  - Engine calls profiler.record_step() after each generation step
  - Scheduler calls adaptive_batch_sizer.compute_optimal_batch() per scheduling
  - SLOMonitor.check_slo() drives AutoTuner.apply_tuning() on violations
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class BottleneckType(StrEnum):
    """Identified performance bottleneck."""
    MEMORY = "memory"
    COMPUTE = "compute"
    IO = "io"
    NONE = "none"


@dataclass
class StepMetrics:
    """Per-step performance metrics recorded by the engine."""

    throughput_tok_s: float = 0.0
    ttft_ms: float = 0.0
    itl_ms: float = 0.0
    tps_per_request: float = 0.0
    gpu_memory_util: float = 0.0
    gpu_memory_active_bytes: int = 0
    gpu_memory_total_bytes: int = 0
    batch_size: int = 0
    tokens_generated: int = 0
    wall_time_ms: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class SLOConfig:
    """Service Level Objective thresholds."""

    ttft_ms: float = 200.0
    itl_ms: float = 50.0
    throughput_tok_s: float = 20.0


@dataclass
class TunableParams:
    """Tunable system parameters with bounds."""

    batch_size: int = 8
    batch_size_min: int = 1
    batch_size_max: int = 64

    prefill_chunk_size: int = 2048
    prefill_chunk_size_min: int = 128
    prefill_chunk_size_max: int = 4096

    kv_quantization_bits: int = 16
    kv_quantization_bits_options: tuple[int, ...] = (4, 8, 16)

    spec_draft_length: int = 4
    spec_draft_length_min: int = 1
    spec_draft_length_max: int = 8

    num_parallel_requests: int = 8
    num_parallel_requests_min: int = 1
    num_parallel_requests_max: int = 64

    def clamp(self) -> TunableParams:
        """Clamp all parameters to their bounds, return self."""
        self.batch_size = max(self.batch_size_min,
                              min(self.batch_size_max, self.batch_size))
        self.prefill_chunk_size = max(self.prefill_chunk_size_min,
                                     min(self.prefill_chunk_size_max,
                                         self.prefill_chunk_size))
        if self.kv_quantization_bits not in self.kv_quantization_bits_options:
            # Snap to nearest valid option
            self.kv_quantization_bits = min(
                self.kv_quantization_bits_options,
                key=lambda x: abs(x - self.kv_quantization_bits),
            )
        self.spec_draft_length = max(
            self.spec_draft_length_min,
            min(self.spec_draft_length_max, self.spec_draft_length),
        )
        self.num_parallel_requests = max(
            self.num_parallel_requests_min,
            min(self.num_parallel_requests_max, self.num_parallel_requests),
        )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "prefill_chunk_size": self.prefill_chunk_size,
            "kv_quantization_bits": self.kv_quantization_bits,
            "spec_draft_length": self.spec_draft_length,
            "num_parallel_requests": self.num_parallel_requests,
        }


@dataclass
class TuningDecision:
    """Record of a single tuning decision."""

    timestamp: float
    param_name: str
    old_value: Any
    new_value: Any
    reason: str
    before_metrics: StepMetrics | None = None
    after_metrics: StepMetrics | None = None
    improvement: float = 0.0  # positive = better
    is_regression: bool = False
    reverted: bool = False


# ---------------------------------------------------------------------------
# 1. PerformanceProfiler
# ---------------------------------------------------------------------------

class PerformanceProfiler:
    """Continuously profiles inference performance.

    Collects per-step metrics, maintains a sliding window, and identifies
    bottlenecks (memory, compute, IO).
    """

    def __init__(
        self,
        window_size: int = 200,
        bottleneck_memory_threshold: float = 0.85,
        bottleneck_compute_threshold: float = 0.7,
    ) -> None:
        self._window_size = window_size
        self._bottleneck_memory_threshold = bottleneck_memory_threshold
        self._bottleneck_compute_threshold = bottleneck_compute_threshold
        self._history: deque[StepMetrics] = deque(maxlen=window_size)
        self._bottleneck_counts: dict[BottleneckType, int] = {
            b: 0 for b in BottleneckType
        }
        self._lock = threading.RLock()
        self._profiling = False
        self._profile_start: float = 0.0
        self._total_steps: int = 0

    # -- Lifecycle --

    def start_profiling(self) -> None:
        """Begin periodic profiling."""
        with self._lock:
            self._profiling = True
            self._profile_start = time.monotonic()
        logger.info("Performance profiling started")

    def stop_profiling(self) -> None:
        """Stop profiling."""
        with self._lock:
            self._profiling = False
        logger.info("Performance profiling stopped")

    @property
    def is_profiling(self) -> bool:
        return self._profiling

    # -- Recording --

    def record_step(self, metrics: StepMetrics) -> None:
        """Record per-step metrics."""
        with self._lock:
            self._history.append(metrics)
            self._total_steps += 1

            # Classify bottleneck for this step
            bottleneck = self._classify_step(metrics)
            self._bottleneck_counts[bottleneck] += 1

    def _classify_step(self, m: StepMetrics) -> BottleneckType:
        """Classify bottleneck for a single step."""
        if m.gpu_memory_util > self._bottleneck_memory_threshold:
            return BottleneckType.MEMORY
        # Compute-bound: low throughput relative to memory availability
        if (m.gpu_memory_util > 0.4
                and m.throughput_tok_s > 0
                and m.throughput_tok_s < 30
                and m.wall_time_ms > 0
                and m.tokens_generated > 0):
            return BottleneckType.COMPUTE
        # IO-bound: meaningful memory usage (>10%) but very low throughput.
        # Using a threshold > 0 to avoid misclassifying idle servers where
        # gpu_memory_util is near zero (e.g. 0.001) as IO-bound.
        if m.gpu_memory_util > 0.1 and m.throughput_tok_s > 0 and m.throughput_tok_s < 10:
            return BottleneckType.IO
        return BottleneckType.NONE

    # -- Analysis --

    def get_bottleneck(self) -> BottleneckType:
        """Identify current bottleneck from recent history."""
        with self._lock:
            if not self._history:
                return BottleneckType.NONE

            # Count bottlenecks in recent window
            counts: dict[BottleneckType, int] = {b: 0 for b in BottleneckType}
            for m in self._history:
                b = self._classify_step(m)
                counts[b] += 1

            # Return the most frequent non-NONE bottleneck
            best = BottleneckType.NONE
            best_count = 0
            for bt, cnt in counts.items():
                if bt == BottleneckType.NONE:
                    continue
                if cnt > best_count:
                    best_count = cnt
                    best = bt

            # Only report bottleneck if > 50% of recent steps are classified
            total = len(self._history)
            if best_count > total * 0.5:
                return best
            return BottleneckType.NONE

    def get_recommendations(self) -> list[dict[str, Any]]:
        """Suggest configuration changes based on current bottleneck."""
        bottleneck = self.get_bottleneck()
        recs: list[dict[str, Any]] = []

        if bottleneck == BottleneckType.MEMORY:
            recs.append({
                "action": "decrease",
                "param": "batch_size",
                "reason": "GPU memory utilization above threshold",
            })
            recs.append({
                "action": "decrease",
                "param": "kv_quantization_bits",
                "reason": "Reduce KV cache memory via quantization",
            })
            recs.append({
                "action": "decrease",
                "param": "num_parallel_requests",
                "reason": "Reduce concurrent memory usage",
            })

        elif bottleneck == BottleneckType.COMPUTE:
            recs.append({
                "action": "decrease",
                "param": "batch_size",
                "reason": "Compute-bound: smaller batches reduce per-step latency",
            })
            recs.append({
                "action": "increase",
                "param": "spec_draft_length",
                "reason": "Speculative decoding can bypass compute bottleneck",
            })

        elif bottleneck == BottleneckType.IO:
            recs.append({
                "action": "increase",
                "param": "prefill_chunk_size",
                "reason": "Larger chunks amortize IO overhead",
            })
            recs.append({
                "action": "increase",
                "param": "batch_size",
                "reason": "Batch more requests to hide IO latency",
            })

        return recs

    def get_latest_metrics(self) -> StepMetrics | None:
        """Return the most recent StepMetrics snapshot (for regression detection)."""
        with self._lock:
            return self._history[-1] if self._history else None

    def get_avg_metrics(self) -> dict[str, float]:
        """Get averaged metrics over the sliding window."""
        with self._lock:
            if not self._history:
                return {
                    "throughput_tok_s": 0.0,
                    "ttft_ms": 0.0,
                    "itl_ms": 0.0,
                    "gpu_memory_util": 0.0,
                }
            n = len(self._history)
            return {
                "throughput_tok_s": sum(m.throughput_tok_s for m in self._history) / n,
                "ttft_ms": sum(m.ttft_ms for m in self._history) / n,
                "itl_ms": sum(m.itl_ms for m in self._history) / n,
                "gpu_memory_util": sum(m.gpu_memory_util for m in self._history) / n,
            }

    # -- Stats --

    def get_stats(self) -> dict[str, Any]:
        """Return profiling statistics."""
        with self._lock:
            total = self._total_steps
            # Compute distribution from the sliding window (not cumulative
            # _bottleneck_counts which are misleading — they span all time
            # while get_bottleneck() only looks at the recent window).
            window_dist: dict[str, int] = {b.value: 0 for b in BottleneckType}
            for m in self._history:
                window_dist[self._classify_step(m).value] += 1
            return {
                "profiling": self._profiling,
                "total_steps": total,
                "window_size": self._window_size,
                "window_filled": len(self._history),
                "bottleneck_distribution": window_dist,
                "current_bottleneck": self.get_bottleneck().value,
                "avg_metrics": self.get_avg_metrics(),
            }


# ---------------------------------------------------------------------------
# 2. SLOMonitor
# ---------------------------------------------------------------------------

class SLOMonitor:
    """Monitors Service Level Objectives and triggers auto-tuning on violation."""

    # Minimum seconds between successive auto-tuning callback triggers
    # for the SAME metric.  Without this, once the violation rate exceeds
    # 30 % every subsequent check_slo() re-fires the callback.
    _CALLBACK_COOLDOWN_S: float = 30.0

    def __init__(self, config: SLOConfig | None = None) -> None:
        self._config = config or SLOConfig()
        self._check_results: dict[str, deque[bool]] = {
            "ttft": deque(maxlen=100), "itl": deque(maxlen=100), "throughput": deque(maxlen=100)
        }
        self._recent_violations: deque[dict[str, Any]] = deque(maxlen=100)
        self._auto_tuning_triggers: int = 0
        self._auto_tuning_callback: Any | None = None
        # Per-metric monotonic timestamp of last callback fire
        self._last_callback_time: dict[str, float] = {}
        self._lock = threading.RLock()

    @property
    def config(self) -> SLOConfig:
        return self._config

    def set_auto_tuning_callback(self, callback: Any) -> None:
        """Set callback invoked when SLOs are persistently violated.

        Signature: callback(violations: list[dict]) -> None
        """
        self._auto_tuning_callback = callback

    def check_slo(self, metric: str, value: float) -> bool:
        """Check if a metric value meets its SLO.

        Args:
            metric: One of 'ttft', 'itl', or 'throughput'.
            value: The observed metric value.

        Returns:
            True if SLO is met, False if violated.
        """
        callback_to_fire = None
        callback_violations = None

        with self._lock:
            if metric not in self._check_results:
                logger.warning("Unknown SLO metric: %s", metric)
                return True

            if metric == "ttft":
                met = value <= self._config.ttft_ms
            elif metric == "itl":
                met = value <= self._config.itl_ms
            elif metric == "throughput":
                met = value >= self._config.throughput_tok_s
            else:
                met = True

            self._check_results[metric].append(met)

            if not met:
                violation = {
                    "metric": metric,
                    "value": value,
                    "threshold": (
                        self._config.ttft_ms if metric == "ttft"
                        else self._config.itl_ms if metric == "itl"
                        else self._config.throughput_tok_s
                    ),
                    "timestamp": time.time(),
                }
                self._recent_violations.append(violation)

                results = self._check_results[metric]
                checks = len(results)
                violations = sum(1 for r in results if not r)
                now = time.monotonic()
                last_fire = self._last_callback_time.get(metric, 0.0)
                if (checks >= 5
                        and violations / checks > 0.3
                        and self._auto_tuning_callback is not None
                        and (now - last_fire) >= self._CALLBACK_COOLDOWN_S):
                    self._auto_tuning_triggers += 1
                    self._last_callback_time[metric] = now
                    callback_to_fire = self._auto_tuning_callback
                    callback_violations = list(self._recent_violations)

        # Invoke callback outside the lock to avoid deadlock
        if callback_to_fire is not None:
            try:
                callback_to_fire(callback_violations)
            except Exception:
                logger.warning("Auto-tuning callback failed", exc_info=True)

        return met

    def get_slo_compliance(self) -> dict[str, float]:
        """Return compliance percentage per SLO."""
        with self._lock:
            result: dict[str, float] = {}
            for metric, results in self._check_results.items():
                if not results:
                    result[metric] = 100.0
                else:
                    met = sum(1 for r in results if r)
                    result[metric] = round(met / len(results) * 100.0, 2)
            return result

    def get_violations(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent SLO violations (most recent first)."""
        with self._lock:
            violations = list(self._recent_violations)
            violations.reverse()
            return violations[:limit]

    def get_stats(self) -> dict[str, Any]:
        """Return SLO monitoring statistics."""
        with self._lock:
            check_counts = {m: len(r) for m, r in self._check_results.items()}
            violation_counts = {
                m: sum(1 for v in r if not v)
                for m, r in self._check_results.items()
            }
            return {
                "config": {
                    "ttft_ms": self._config.ttft_ms,
                    "itl_ms": self._config.itl_ms,
                    "throughput_tok_s": self._config.throughput_tok_s,
                },
                "compliance_pct": self.get_slo_compliance(),
                "check_counts": check_counts,
                "violation_counts": violation_counts,
                "recent_violation_count": len(self._recent_violations),
                "auto_tuning_triggers": self._auto_tuning_triggers,
            }

    def reset(self) -> None:
        """Reset SLO monitoring state."""
        with self._lock:
            for k in self._check_results:
                self._check_results[k].clear()
            self._recent_violations.clear()
            self._auto_tuning_triggers = 0
            self._last_callback_time.clear()


# ---------------------------------------------------------------------------
# 3. AdaptiveBatchSizer
# ---------------------------------------------------------------------------

class AdaptiveBatchSizer:
    """Dynamically adjusts batch size based on system conditions.

    Monitors GPU memory usage, request queue depth, and latency SLO
    to compute optimal batch size.
    """

    def __init__(
        self,
        min_batch: int = 1,
        max_batch: int = 64,
        memory_threshold: float = 0.85,
        latency_multiplier: float = 2.0,
        scale_up_factor: float = 1.5,
        scale_down_factor: float = 0.7,
    ) -> None:
        self._min_batch = min_batch
        self._max_batch = max_batch
        self._memory_threshold = memory_threshold
        self._latency_multiplier = latency_multiplier
        self._scale_up_factor = scale_up_factor
        self._scale_down_factor = scale_down_factor
        self._current_batch: int = min_batch
        self._history: deque[dict[str, Any]] = deque(maxlen=500)
        self._adjustment_count: int = 0
        # Sliding-window SLO compliance — avoids stale cumulative counters
        # that make the compliance rate meaningless on long-running servers.
        self._slo_window: deque[bool] = deque(maxlen=200)
        self._lock = threading.RLock()

    @property
    def current_batch_size(self) -> int:
        return self._current_batch

    def compute_optimal_batch(
        self,
        queue_depth: int,
        memory_available: float,
        slo_latency_ms: float,
        current_latency_ms: float = 0.0,
    ) -> int:
        """Compute optimal batch size.

        Args:
            queue_depth: Number of pending requests.
            memory_available: Fraction of GPU memory available (0.0 - 1.0).
            slo_latency_ms: SLO latency target in milliseconds.
            current_latency_ms: Current observed latency in milliseconds.

        Returns:
            Optimal batch size, clamped to [min_batch, max_batch].
        """
        with self._lock:
            memory_util = 1.0 - memory_available
            old_batch = self._current_batch
            batch = old_batch

            # Memory pressure: scale down
            if memory_util > self._memory_threshold or (current_latency_ms > 0
                  and current_latency_ms > slo_latency_ms * self._latency_multiplier):
                batch = max(
                    self._min_batch,
                    int(batch * self._scale_down_factor),
                )

            # Within SLO and comfortable memory: scale up
            else:
                latency_ok = (
                    current_latency_ms <= 0
                    or current_latency_ms <= slo_latency_ms
                )
                memory_ok = memory_util < self._memory_threshold * 0.75
                queue_ok = queue_depth > batch

                if latency_ok and memory_ok and queue_ok:
                    batch = min(
                        self._max_batch,
                        max(batch + 1, int(batch * self._scale_up_factor)),
                    )

            # Clamp to [min_batch, max_batch] only.  Do not clamp to
            # queue_depth — the scale-up condition already checks
            # queue_ok, and clamping here prevents proactive scaling.
            batch = max(self._min_batch, min(batch, self._max_batch))

            # Track SLO compliance (only when latency data is available)
            if current_latency_ms > 0:
                self._slo_window.append(current_latency_ms <= slo_latency_ms)

            if batch != old_batch:
                self._adjustment_count += 1

            self._current_batch = batch
            self._history.append({
                "timestamp": time.time(),
                "queue_depth": queue_depth,
                "memory_available": memory_available,
                "slo_latency_ms": slo_latency_ms,
                "current_latency_ms": current_latency_ms,
                "batch_size": batch,
                "adjusted": batch != old_batch,
            })

            return batch

    def get_stats(self) -> dict[str, Any]:
        """Return batch sizer statistics."""
        with self._lock:
            if self._slo_window:
                slo_rate = (
                    sum(1 for met in self._slo_window if met)
                    / len(self._slo_window) * 100.0
                )
            else:
                slo_rate = 100.0
            return {
                "current_batch_size": self._current_batch,
                "adjustment_count": self._adjustment_count,
                "slo_compliance_rate": round(slo_rate, 2),
                "slo_checks": len(self._slo_window),
                "history_size": len(self._history),
                "config": {
                    "min_batch": self._min_batch,
                    "max_batch": self._max_batch,
                    "memory_threshold": self._memory_threshold,
                },
            }


# ---------------------------------------------------------------------------
# 4. AutoTuner
# ---------------------------------------------------------------------------

class AutoTuner:
    """Automatically adjusts system configuration based on profiler recommendations.

    Uses hill-climbing with bounded exploration:
    - Moves in the direction suggested by the profiler
    - Evaluates the impact of each change
    - Reverts changes that cause regressions
    """

    # Step sizes for each parameter
    _STEPS: dict[str, Any] = {
        "batch_size": 2,
        "prefill_chunk_size": 256,
        "kv_quantization_bits": 2,
        "spec_draft_length": 1,
        "num_parallel_requests": 2,
    }

    # Valid values for parameters that have discrete allowed values
    _VALID_VALUES: dict[str, list[int]] = {
        "kv_quantization_bits": [2, 4, 8, 16],
    }

    def __init__(
        self,
        params: TunableParams | None = None,
        profiler: PerformanceProfiler | None = None,
        slo_monitor: SLOMonitor | None = None,
        regression_threshold: float = -0.1,
        min_tuning_interval: float = 30.0,
    ) -> None:
        self._params = params or TunableParams()
        self._params.clamp()
        self._profiler = profiler or PerformanceProfiler()
        self._slo_monitor = slo_monitor or SLOMonitor()
        self._regression_threshold = regression_threshold
        # Bounded history — prevents unbounded memory growth on long-running servers.
        self._history: deque[TuningDecision] = deque(maxlen=1000)
        self._total_tunings: int = 0
        self._improvements: int = 0
        self._regressions: int = 0
        self._last_tuning_time: float = 0.0
        self._min_tuning_interval = min_tuning_interval
        self._lock = threading.RLock()

    @property
    def params(self) -> TunableParams:
        return self._params

    @property
    def profiler(self) -> PerformanceProfiler:
        return self._profiler

    @property
    def slo_monitor(self) -> SLOMonitor:
        return self._slo_monitor

    def apply_tuning(
        self,
        param_name: str,
        direction: str,
        reason: str = "",
        before_metrics: StepMetrics | None = None,
    ) -> TuningDecision:
        """Apply a single tuning change in the given direction.

        Args:
            param_name: Name of the parameter to adjust.
            direction: 'increase' or 'decrease'.
            reason: Human-readable reason for the change.
            before_metrics: Metrics snapshot before the change.

        Returns:
            TuningDecision record.
        """
        with self._lock:
            # Rate limit: don't tune more frequently than _min_tuning_interval
            now_mono = time.monotonic()
            if now_mono - self._last_tuning_time < self._min_tuning_interval:
                return TuningDecision(
                    timestamp=time.time(),
                    param_name=param_name,
                    old_value=getattr(self._params, param_name, None),
                    new_value=getattr(self._params, param_name, None),
                    reason="Rate-limited: tuning cooldown active",
                )

            old_value = getattr(self._params, param_name, None)
            if old_value is None:
                logger.warning("Unknown parameter: %s", param_name)
                return TuningDecision(
                    timestamp=time.time(),
                    param_name=param_name,
                    old_value=None,
                    new_value=None,
                    reason=f"Unknown parameter: {param_name}",
                )

            step = self._STEPS.get(param_name, 1)
            valid = self._VALID_VALUES.get(param_name)

            if valid:
                # Discrete parameter: snap to next valid value
                try:
                    idx = valid.index(old_value)
                except ValueError:
                    idx = 0
                if direction == "increase":
                    new_value = valid[min(idx + 1, len(valid) - 1)]
                elif direction == "decrease":
                    new_value = valid[max(idx - 1, 0)]
                else:
                    new_value = old_value
            elif direction == "increase":
                new_value = old_value + step
            elif direction == "decrease":
                new_value = old_value - step
            else:
                logger.warning("Unknown direction: %s", direction)
                return TuningDecision(
                    timestamp=time.time(),
                    param_name=param_name,
                    old_value=old_value,
                    new_value=old_value,
                    reason=f"Unknown direction: {direction}",
                )

            setattr(self._params, param_name, new_value)
            self._params.clamp()
            actual_new = getattr(self._params, param_name)

            decision = TuningDecision(
                timestamp=time.time(),
                param_name=param_name,
                old_value=old_value,
                new_value=actual_new,
                reason=reason,
                before_metrics=before_metrics,
            )

            self._history.append(decision)
            self._total_tunings += 1
            self._last_tuning_time = time.monotonic()

            logger.info(
                "AutoTuner: %s %s %s -> %s (%s)",
                direction,
                param_name,
                old_value,
                actual_new,
                reason,
            )

            return decision

    def evaluate_tuning(
        self,
        decision: TuningDecision,
        after_metrics: StepMetrics,
    ) -> float:
        """Evaluate the impact of a tuning decision.

        Compares throughput before and after the change. A positive
        improvement means the tuning was beneficial.

        Args:
            decision: The tuning decision to evaluate.
            after_metrics: Metrics snapshot after the change.

        Returns:
            Improvement ratio (positive = better, negative = regression).
        """
        if decision.before_metrics is None:
            return 0.0

        before_throughput = decision.before_metrics.throughput_tok_s
        after_throughput = after_metrics.throughput_tok_s

        if before_throughput <= 0:
            improvement = 1.0 if after_throughput > 0 else 0.0
        else:
            improvement = (after_throughput - before_throughput) / before_throughput

        with self._lock:
            # Mutate shared decision inside lock so get_tuning_history()
            # readers never see a partially-updated TuningDecision.
            decision.after_metrics = after_metrics
            decision.improvement = improvement

            if improvement < self._regression_threshold:
                decision.is_regression = True
                self._regressions += 1
                logger.warning(
                    "AutoTuner regression detected: %s %s->%s, improvement=%.2f%% — reverting",
                    decision.param_name,
                    decision.old_value,
                    decision.new_value,
                    improvement * 100,
                )
                # Revert the parameter to its pre-tuning value
                setattr(self._params, decision.param_name, decision.old_value)
                self._params.clamp()
                decision.reverted = True
            elif improvement > 0:
                self._improvements += 1

        return improvement

    def auto_tune_from_profiler(self) -> list[TuningDecision]:
        """Apply recommendations from the profiler.

        Note: apply_tuning() enforces a cooldown period between changes to the
        same parameter. When multiple recommendations target the same param (or
        when a recent change is still in cooldown), only the first recommendation
        is applied; subsequent ones are silently skipped. This is intentional —
        rapid oscillation of tuning parameters degrades inference stability.

        Returns:
            List of tuning decisions made (may be fewer than recommendations).
        """
        recommendations = self._profiler.get_recommendations()
        before_metrics = self._profiler.get_latest_metrics()
        decisions: list[TuningDecision] = []

        for i, rec in enumerate(recommendations):
            d = self.apply_tuning(
                param_name=rec["param"],
                direction=rec["action"],
                reason=rec["reason"],
                before_metrics=before_metrics,
            )
            decisions.append(d)
            if d.reverted:
                logger.debug(
                    "AutoTuner: recommendation %d/%d skipped (cooldown/reverted) "
                    "param=%s direction=%s",
                    i + 1, len(recommendations), rec["param"], rec["action"],
                )

        if len(decisions) < len(recommendations):
            logger.debug(
                "AutoTuner: applied %d/%d profiler recommendations (remainder blocked by cooldown)",
                len(decisions), len(recommendations),
            )

        return decisions

    def auto_tune(self) -> list[TuningDecision]:
        """Run one auto-tuning cycle using profiler recommendations.

        Gets bottleneck analysis from the profiler and applies the
        first recommended change. Returns the list of decisions made.
        """
        recommendations = self._profiler.get_recommendations()
        if not recommendations:
            return []

        # Capture current metrics BEFORE applying any change so that
        # evaluate_tuning() can compute improvement ratios.
        before_metrics = self._profiler.get_latest_metrics()

        decisions: list[TuningDecision] = []
        # Apply only the top recommendation per cycle
        rec = recommendations[0]
        param_name = rec.get("param", "")
        direction = rec.get("action", "")
        reason = rec.get("reason", "")

        if param_name and direction in ("increase", "decrease"):
            decision = self.apply_tuning(
                param_name=param_name,
                direction=direction,
                reason=reason,
                before_metrics=before_metrics,
            )
            decisions.append(decision)
        return decisions

    def get_tuning_history(self) -> list[dict[str, Any]]:
        """Return all tuning decisions and outcomes."""
        with self._lock:
            return [
                {
                    "timestamp": d.timestamp,
                    "param_name": d.param_name,
                    "old_value": d.old_value,
                    "new_value": d.new_value,
                    "reason": d.reason,
                    "improvement": d.improvement,
                    "is_regression": d.is_regression,
                }
                for d in self._history
            ]

    def get_stats(self) -> dict[str, Any]:
        """Return auto-tuner statistics."""
        with self._lock:
            return {
                "current_params": self._params.to_dict(),
                "total_tunings": self._total_tunings,
                "improvements": self._improvements,
                "regressions": self._regressions,
                "history_size": len(self._history),
                "profiler_stats": self._profiler.get_stats(),
                "slo_stats": self._slo_monitor.get_stats(),
            }
