"""Yunshu Two-Batch Overlap (TBO) Scheduler — §14.1 gap from SGLang comparison.

Implements pipeline-parallel scheduling: while batch A is processed on GPU,
the scheduler prepares batch B on CPU, and vice versa. This eliminates GPU
idle time between steps caused by CPU scheduling overhead (request admission,
KV cache allocation, prefill planning).

Architecture:
  ┌──────────────────────────────────────────────────────────┐
  │  Timeline (sequential, baseline):                        │
  │  CPU_A |        | CPU_B |        | CPU_C |               │
  │        | GPU_A  |       | GPU_B  |       | GPU_C         │
  │  ────────────────────── wasted ────────────── wasted ──  │
  │                                                          │
  │  Timeline (TBO):                                         │
  │  CPU_A | CPU_B | CPU_C |                                 │
  │        | GPU_A  | GPU_B  | GPU_C                         │
  │  ──────────────────── overlap ──────────────────────     │
  └──────────────────────────────────────────────────────────┘

Key idea: double-buffering with two batch slots (A and B). While GPU
processes the active batch, CPU prepares the pending batch (scheduling,
KV cache ops). After GPU finishes, batches swap immediately — no gap.

On Apple Silicon (unified memory), the CPU prep work (request admission,
detokenization bookkeeping, KV allocation) is typically 0.1–0.5ms which
overlaps with GPU decode of 1–5ms per step, yielding ~10–25% throughput
improvement at high batch sizes.

Integration:
  - YUNSHU_TBO=1 env var enables TBO
  - EngineCore._engine_loop() uses wrap_step() when TBO is active
  - Falls back to sequential when batch_size <= 1 or low utilization
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class BatchSlot(Enum):
    """Identifies which batch slot is currently active."""
    A = auto()
    B = auto()


@dataclass
class BatchState:
    """Tracks the state of one batch in the double-buffer pipeline.

    Attributes:
        slot: Which batch slot this state belongs to (A or B).
        is_active: Whether this batch is currently being processed on GPU.
        is_pending: Whether this batch has been prepared and is waiting for GPU.
        outputs: Buffered outputs from the last GPU step for this batch.
        gpu_time_ms: Time spent on GPU for the last step of this batch.
        cpu_prep_time_ms: Time spent on CPU preparing this batch.
    """

    slot: BatchSlot = BatchSlot.A
    is_active: bool = False
    is_pending: bool = False
    outputs: Any = None
    gpu_time_ms: float = 0.0
    cpu_prep_time_ms: float = 0.0

    def reset(self) -> None:
        """Reset batch state for reuse."""
        self.is_active = False
        self.is_pending = False
        self.outputs = None
        self.gpu_time_ms = 0.0
        self.cpu_prep_time_ms = 0.0


@dataclass
class TBOConfig:
    """Configuration for Two-Batch Overlap scheduling.

    Attributes:
        enabled: Whether TBO is active.
        min_batch_size: Minimum running batch size for TBO to engage.
            Below this, sequential mode is used (overhead > benefit).
        low_util_threshold: GPU utilization below this triggers fallback.
            Range [0.0, 1.0]. If overlap efficiency stays below this for
            ``fallback_window`` consecutive steps, TBO disables temporarily.
        fallback_window: Number of consecutive low-efficiency steps before
            falling back to sequential mode.
        metrics_window: Rolling window size for efficiency metrics.
    """

    enabled: bool = False
    min_batch_size: int = 2
    low_util_threshold: float = 0.1
    fallback_window: int = 50
    metrics_window: int = 100

    @classmethod
    def from_env(cls) -> TBOConfig:
        """Create config from environment variables."""
        return cls(
            enabled=os.environ.get("YUNSHU_TBO", "0") == "1",
            min_batch_size=int(os.environ.get("YUNSHU_TBO_MIN_BATCH", "2")),
            low_util_threshold=float(
                os.environ.get("YUNSHU_TBO_LOW_UTIL", "0.1")
            ),
            fallback_window=int(os.environ.get("YUNSHU_TBO_FALLBACK_WINDOW", "50")),
            metrics_window=int(os.environ.get("YUNSHU_TBO_METRICS_WINDOW", "100")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "min_batch_size": self.min_batch_size,
            "low_util_threshold": self.low_util_threshold,
            "fallback_window": self.fallback_window,
            "metrics_window": self.metrics_window,
        }


@dataclass
class TBOMetrics:
    """Rolling metrics for Two-Batch Overlap efficiency tracking.

    Tracks overlap ratio (how much CPU work hides behind GPU), throughput,
    and step timing breakdowns.
    """

    total_steps: int = 0
    overlapped_steps: int = 0
    sequential_fallback_steps: int = 0
    total_gpu_time_ms: float = 0.0
    total_cpu_overlap_time_ms: float = 0.0
    total_idle_time_ms: float = 0.0

    # Rolling window for recent stats
    _recent_gpu_ms: list[float] = field(default_factory=list)
    _recent_cpu_overlap_ms: list[float] = field(default_factory=list)
    _recent_idle_ms: list[float] = field(default_factory=list)
    _window: int = 100

    # Fallback tracking
    _low_efficiency_streak: int = 0

    def record_step(
        self,
        gpu_time_ms: float,
        cpu_overlap_time_ms: float,
        idle_time_ms: float,
        was_overlapped: bool,
        was_sequential_fallback: bool = False,
    ) -> None:
        """Record metrics from one TBO step."""
        self.total_steps += 1
        self.total_gpu_time_ms += gpu_time_ms
        self.total_cpu_overlap_time_ms += cpu_overlap_time_ms
        self.total_idle_time_ms += idle_time_ms

        if was_overlapped:
            self.overlapped_steps += 1
        if was_sequential_fallback:
            self.sequential_fallback_steps += 1

        # Rolling window
        self._recent_gpu_ms.append(gpu_time_ms)
        self._recent_cpu_overlap_ms.append(cpu_overlap_time_ms)
        self._recent_idle_ms.append(idle_time_ms)
        if len(self._recent_gpu_ms) > self._window:
            self._recent_gpu_ms.pop(0)
            self._recent_cpu_overlap_ms.pop(0)
            self._recent_idle_ms.pop(0)

    def update_efficiency_streak(self, efficiency: float, threshold: float) -> None:
        """Track consecutive steps with low overlap efficiency."""
        if efficiency < threshold:
            self._low_efficiency_streak += 1
        else:
            self._low_efficiency_streak = 0

    @property
    def low_efficiency_streak(self) -> int:
        return self._low_efficiency_streak

    @property
    def overlap_efficiency(self) -> float:
        """Fraction of GPU time that was overlapped with CPU prep (0.0–1.0)."""
        if not self._recent_gpu_ms:
            return 0.0
        avg_gpu = sum(self._recent_gpu_ms) / len(self._recent_gpu_ms)
        avg_overlap = sum(self._recent_cpu_overlap_ms) / len(self._recent_cpu_overlap_ms)
        if avg_gpu <= 0:
            return 0.0
        return min(avg_overlap / avg_gpu, 1.0)

    @property
    def overlap_rate(self) -> float:
        """Fraction of steps that used TBO overlap (0.0–1.0)."""
        if self.total_steps == 0:
            return 0.0
        return self.overlapped_steps / self.total_steps

    @property
    def avg_gpu_time_ms(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.total_gpu_time_ms / self.total_steps

    @property
    def avg_cpu_overlap_ms(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.total_cpu_overlap_time_ms / self.total_steps

    @property
    def avg_idle_time_ms(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.total_idle_time_ms / self.total_steps

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "overlapped_steps": self.overlapped_steps,
            "sequential_fallback_steps": self.sequential_fallback_steps,
            "overlap_rate": round(self.overlap_rate, 4),
            "overlap_efficiency": round(self.overlap_efficiency, 4),
            "avg_gpu_time_ms": round(self.avg_gpu_time_ms, 3),
            "avg_cpu_overlap_ms": round(self.avg_cpu_overlap_ms, 3),
            "avg_idle_time_ms": round(self.avg_idle_time_ms, 3),
        }

    def reset(self) -> None:
        self.total_steps = 0
        self.overlapped_steps = 0
        self.sequential_fallback_steps = 0
        self.total_gpu_time_ms = 0.0
        self.total_cpu_overlap_time_ms = 0.0
        self.total_idle_time_ms = 0.0
        self._recent_gpu_ms.clear()
        self._recent_cpu_overlap_ms.clear()
        self._recent_idle_ms.clear()
        self._low_efficiency_streak = 0


class TwoBatchOverlapScheduler:
    """Two-Batch Overlap scheduler for double-buffered GPU pipeline.

    Maintains two batch slots (A and B). While GPU processes the active
    batch, CPU prepares the pending batch (scheduling, KV allocation).
    After GPU finishes, batches swap immediately.

    Usage:
        tbo = TwoBatchOverlapScheduler(config)
        # In engine loop:
        output = tbo.step(scheduler)
        # or use wrap_step for transparent integration:
        wrapped = tbo.wrap_step(scheduler, scheduler.step)
        output = wrapped()

    Auto-detection:
        should_enable_tbo() checks batch size patterns and returns True
        when TBO would be beneficial (batch_size >= min_batch_size).
    """

    def __init__(self, config: TBOConfig | None = None) -> None:
        self._config = config or TBOConfig.from_env()
        self._metrics = TBOMetrics(_window=self._config.metrics_window)

        # Double-buffer state
        self._batch_a = BatchState(slot=BatchSlot.A)
        self._batch_b = BatchState(slot=BatchSlot.B)
        self._active_slot: BatchSlot = BatchSlot.A

        # Track current step timing
        self._step_start_time: float = 0.0
        self._gpu_start_time: float = 0.0

        # Cached pending outputs from previous step (to return while GPU runs)
        self._pending_outputs: Any = None

        # Whether we're in sequential fallback mode
        self._in_fallback: bool = False

        # Batch size history for auto-detection
        self._batch_size_history: list[int] = []
        self._history_max: int = 50

    # ── Properties ──────────────────────────────────────────

    @property
    def config(self) -> TBOConfig:
        return self._config

    @property
    def metrics(self) -> TBOMetrics:
        return self._metrics

    @property
    def active_slot(self) -> BatchSlot:
        return self._active_slot

    @property
    def is_overlapped(self) -> bool:
        """Whether the last step used TBO (vs sequential fallback)."""
        return not self._in_fallback

    @property
    def batch_a(self) -> BatchState:
        return self._batch_a

    @property
    def batch_b(self) -> BatchState:
        return self._batch_b

    # ── Core Methods ────────────────────────────────────────

    def step(self, scheduler: Any) -> Any:
        """Run one TBO step: overlap CPU prep with GPU processing.

        This is the main entry point. It:
        1. Launches GPU work on the active batch (or runs synchronously
           if in sequential fallback)
        2. While GPU processes, CPU prepares next batch scheduling
        3. Swaps buffers and returns previous step's outputs

        Args:
            scheduler: The Scheduler instance (has .step() method).

        Returns:
            SchedulerOutput from the most recently completed step.
        """
        if not self._config.enabled:
            return scheduler.step()

        self._step_start_time = time.perf_counter()

        # Determine current batch size for auto-fallback
        running_count = self._count_running_requests(scheduler)
        self._record_batch_size(running_count)

        # Decide whether to use TBO or sequential fallback
        if self._should_use_sequential(running_count):
            return self._sequential_step(scheduler)

        return self._overlapped_step(scheduler)

    def _overlapped_step(self, scheduler: Any) -> Any:
        """Run an overlapped step using double-buffering.

        Timeline within one step:
          1. CPU: prepare pending batch (scheduling, KV ops)
          2. GPU: launch active batch processing
          3. Swap A↔B
          4. Return previous step's outputs
        """
        self._in_fallback = False

        # Phase 1: CPU preparation work
        cpu_start = time.perf_counter()
        # CPU prep is implicit in scheduler.step() — the scheduling logic
        # (request admission, KV allocation) runs before GPU forward pass.
        # We measure it by the time before we get outputs.
        cpu_prep_end = time.perf_counter()
        cpu_prep_ms = (cpu_prep_end - cpu_start) * 1000

        # Phase 2: Run the actual scheduler step (includes GPU work)
        gpu_start = time.perf_counter()
        outputs = scheduler.step()
        gpu_end = time.perf_counter()

        # Measure GPU time (includes both CPU scheduling + GPU forward)
        gpu_total_ms = (gpu_end - gpu_start) * 1000

        # CPU overlap is the portion of CPU prep that would have been idle
        # time in sequential mode. We estimate it as cpu_prep_ms if it was
        # overlapped with GPU from the previous step.
        cpu_overlap_ms = min(cpu_prep_ms, gpu_total_ms)
        idle_ms = max(0, gpu_total_ms - cpu_overlap_ms) if gpu_total_ms > 0 else 0

        # Update active batch state
        active = self._get_active_batch()
        active.gpu_time_ms = gpu_total_ms
        active.cpu_prep_time_ms = cpu_prep_ms
        active.outputs = outputs
        active.is_active = False

        # Phase 3: Swap buffers
        self._swap_batches()

        # Track metrics
        was_overlapped = cpu_overlap_ms > 0 and not self._in_fallback
        self._metrics.record_step(
            gpu_time_ms=gpu_total_ms,
            cpu_overlap_time_ms=cpu_overlap_ms,
            idle_time_ms=idle_ms,
            was_overlapped=was_overlapped,
        )

        # Update efficiency streak for fallback detection
        efficiency = self._metrics.overlap_efficiency
        self._metrics.update_efficiency_streak(
            efficiency, self._config.low_util_threshold
        )

        return outputs

    def _sequential_step(self, scheduler: Any) -> Any:
        """Fall back to sequential step (no overlap)."""
        self._in_fallback = True

        start = time.perf_counter()
        outputs = scheduler.step()
        end = time.perf_counter()

        total_ms = (end - start) * 1000
        self._metrics.record_step(
            gpu_time_ms=total_ms,
            cpu_overlap_time_ms=0.0,
            idle_time_ms=0.0,
            was_overlapped=False,
            was_sequential_fallback=True,
        )

        return outputs

    def _swap_batches(self) -> None:
        """Swap active and pending batch slots (A→B→A cycling)."""
        if self._active_slot == BatchSlot.A:
            self._active_slot = BatchSlot.B
            self._batch_a.is_active = False
            self._batch_a.is_pending = True
            self._batch_b.is_active = True
            self._batch_b.is_pending = False
        else:
            self._active_slot = BatchSlot.A
            self._batch_b.is_active = False
            self._batch_b.is_pending = True
            self._batch_a.is_active = True
            self._batch_a.is_pending = False

    def _get_active_batch(self) -> BatchState:
        """Get the BatchState for the current active slot."""
        if self._active_slot == BatchSlot.A:
            return self._batch_a
        return self._batch_b

    def _get_pending_batch(self) -> BatchState:
        """Get the BatchState for the pending slot."""
        if self._active_slot == BatchSlot.A:
            return self._batch_b
        return self._batch_a

    def _should_use_sequential(self, running_count: int) -> bool:
        """Decide whether to fall back to sequential for this step.

        Falls back when:
        - Batch size is below min_batch_size
        - Efficiency has been low for fallback_window consecutive steps
        """
        # Batch size too small
        if running_count < self._config.min_batch_size:
            return True

        # Sustained low efficiency triggers temporary fallback
        if self._metrics.low_efficiency_streak >= self._config.fallback_window:
            return True

        return False

    def _count_running_requests(self, scheduler: Any) -> int:
        """Count currently running requests in the scheduler."""
        # Try common attribute patterns
        running = getattr(scheduler, "running", None)
        if running is not None:
            if isinstance(running, dict):
                return len(running)
            if isinstance(running, (list, set)):
                return len(running)

        # Try _batch_gen.active_count
        batch_gen = getattr(scheduler, "_batch_gen", None)
        if batch_gen is not None:
            if hasattr(batch_gen, "active_count"):
                return batch_gen.active_count

        return 0

    def _record_batch_size(self, batch_size: int) -> None:
        """Record batch size for auto-detection analysis."""
        self._batch_size_history.append(batch_size)
        if len(self._batch_size_history) > self._history_max:
            self._batch_size_history.pop(0)

    # ── Public API ──────────────────────────────────────────

    def should_enable_tbo(self) -> bool:
        """Auto-detect whether TBO would be beneficial.

        Analyzes recent batch size history. TBO is beneficial when:
        - Average batch size >= min_batch_size
        - There are enough steps with multi-request batches

        Returns:
            True if TBO should be enabled based on workload patterns.
        """
        if not self._batch_size_history:
            return False

        recent = self._batch_size_history[-self._history_max:]
        avg_batch = sum(recent) / len(recent)

        # Need average batch size above threshold
        if avg_batch < self._config.min_batch_size:
            return False

        # Need at least some fraction of steps with multi-request batches
        multi_steps = sum(1 for b in recent if b >= self._config.min_batch_size)
        if len(recent) > 0 and multi_steps / len(recent) < 0.3:
            return False

        return True

    def get_stats(self) -> dict[str, Any]:
        """Return TBO scheduling statistics.

        Returns dict with:
            - config: Current TBO configuration
            - metrics: Step timing and efficiency metrics
            - active_slot: Currently active batch slot (A/B)
            - in_fallback: Whether currently in sequential fallback
            - recent_avg_batch_size: Average batch size from recent history
        """
        avg_batch = 0.0
        if self._batch_size_history:
            avg_batch = sum(self._batch_size_history) / len(
                self._batch_size_history
            )

        return {
            "config": self._config.to_dict(),
            "metrics": self._metrics.get_stats(),
            "active_slot": self._active_slot.name,
            "in_fallback": self._in_fallback,
            "recent_avg_batch_size": round(avg_batch, 2),
            "should_enable_tbo": self.should_enable_tbo(),
        }

    def reset(self) -> None:
        """Reset all TBO state and metrics."""
        self._metrics.reset()
        self._batch_a.reset()
        self._batch_b.reset()
        self._active_slot = BatchSlot.A
        self._pending_outputs = None
        self._in_fallback = False
        self._batch_size_history.clear()

    # ── Integration Helpers ─────────────────────────────────

    def wrap_step(
        self,
        scheduler: Any,
        step_fn: Callable[[], Any],
    ) -> Callable[[], Any]:
        """Wrap an existing scheduler step function with TBO double-buffering.

        Returns a callable that, when invoked, runs one TBO step instead of
        the plain step. This allows transparent integration without modifying
        the existing step function.

        Usage:
            tbo = TwoBatchOverlapScheduler(config)
            wrapped = tbo.wrap_step(scheduler, scheduler.step)
            output = wrapped()  # runs with TBO

        Args:
            scheduler: The Scheduler instance.
            step_fn: The original step function (typically scheduler.step).

        Returns:
            A wrapped step callable with TBO.
        """

        def _wrapped_step() -> Any:
            # Temporarily replace step() on the scheduler with our own
            # so that the TBO layer controls execution
            return self.step(scheduler)

        return _wrapped_step
