from __future__ import annotations

"""Modular scheduler mixins — composable scheduler components.

Each feature (metrics, profiling, spec-decode, memory pressure) is a composable
mixin that can be mixed in without modifying the core scheduler loop. (Single-node
only — the former distributed mixins, disaggregation / pipeline / data parallel,
were removed in the refocus.)

Each mixin wraps one concern and hooks into the scheduler lifecycle:
  - pre_step()   — called before scheduler.step()
  - post_step()  — called after scheduler.step() with output
  - on_add_request() — called when a request is added
  - on_finish()  — called when a request completes
  - get_stats()  — return mixin-specific metrics

Usage:
  scheduler = Scheduler(model, tokenizer, config)
  scheduler.add_mixin(MetricsMixin())
  scheduler.add_mixin(ProfilingMixin())
"""

import collections
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class SchedulerMixin(ABC):
    """Base class for scheduler mixins."""

    @abstractmethod
    def pre_step(self, scheduler: Any) -> None:
        """Called before each scheduler.step()."""

    @abstractmethod
    def post_step(self, scheduler: Any, output: Any) -> None:
        """Called after each scheduler.step() with the output."""

    def on_add_request(self, scheduler: Any, request: Any) -> None:
        """Called when a request is added."""

    def on_finish(self, scheduler: Any, request_id: str, output: Any) -> None:
        """Called when a request finishes."""

    def get_stats(self) -> dict:
        """Return mixin-specific stats."""
        return {}

    def shutdown(self) -> None:
        """Called on scheduler shutdown."""


class MetricsMixin(SchedulerMixin):
    """Collect per-step metrics: batch size, latency, throughput, memory."""

    def __init__(self, window_size: int = 100) -> None:
        self._window_size = window_size
        self._step_times: collections.deque[float] = collections.deque(
            maxlen=window_size
        )
        self._batch_sizes: deque[int] = deque(maxlen=self._window_size)
        self._throughput_window: list[tuple[float, int]] = []
        self._total_tokens = 0
        self._total_requests = 0
        self._start_time = time.monotonic()
        self._step_count = 0

    def pre_step(self, scheduler: Any) -> None:
        self._step_start = time.monotonic()

    def post_step(self, scheduler: Any, output: Any) -> None:
        now = time.monotonic()
        step_latency = now - self._step_start if hasattr(self, "_step_start") else 0.0
        self._step_times.append(step_latency)

        batch_size = 0
        tokens = 0
        finished = 0
        if hasattr(output, "outputs") and output.outputs:
            batch_size = len(output.outputs)
            for o in output.outputs:
                tokens += getattr(o, "completion_tokens", 1)
                if getattr(o, "finished", False):
                    finished += 1

        self._batch_sizes.append(batch_size)

        self._throughput_window.append((now, tokens))
        self._total_tokens += tokens
        self._total_requests += finished
        self._step_count += 1

        # Prune throughput window (keep last 60s)
        cutoff = now - 60.0
        self._throughput_window = [
            (t, n) for t, n in self._throughput_window if t > cutoff
        ]

    def on_finish(self, scheduler: Any, request_id: str, output: Any) -> None:
        pass

    def get_stats(self) -> dict:
        now = time.monotonic()
        uptime = now - self._start_time
        avg_step = (
            sum(self._step_times) / len(self._step_times) if self._step_times else 0.0
        )
        avg_batch = (
            sum(self._batch_sizes) / len(self._batch_sizes)
            if self._batch_sizes
            else 0.0
        )
        if len(self._throughput_window) >= 2:
            throughput_60s = sum(n for _, n in self._throughput_window)
            window_span = self._throughput_window[-1][0] - self._throughput_window[0][0]
            throughput_tps = throughput_60s / max(window_span, 0.001)
        else:
            throughput_tps = 0.0
        _sorted_steps = sorted(self._step_times) if self._step_times else []
        p50_step = _sorted_steps[len(_sorted_steps) // 2] if _sorted_steps else 0.0
        _p99_idx = (
            min(len(_sorted_steps) - 1, int(len(_sorted_steps) * 0.99))
            if _sorted_steps
            else 0
        )
        p99_step = _sorted_steps[_p99_idx] if _sorted_steps else 0.0
        return {
            "step_count": self._step_count,
            "uptime_seconds": round(uptime, 1),
            "avg_step_latency_ms": round(avg_step * 1000, 3),
            "p50_step_latency_ms": round(p50_step * 1000, 3),
            "p99_step_latency_ms": round(p99_step * 1000, 3),
            "avg_batch_size": round(avg_batch, 2),
            "throughput_tps_60s": round(throughput_tps, 1),
            "total_tokens": self._total_tokens,
            "total_requests": self._total_requests,
        }


@dataclass
class ProfilingSample:
    request_id: str
    step: int
    phase: str  # "prefill" | "decode" | "spec_verify"
    latency_ms: float
    tokens_generated: int
    batch_size: int
    memory_active_bytes: int = 0
    memory_peak_bytes: int = 0


class ProfilingMixin(SchedulerMixin):
    """Detailed step profiling with optional trace export.

    Captures per-step latency breakdowns (prefill vs decode vs spec verify),
    memory usage snapshots, and batch composition. Enable via
    YUNSHU_SCHEDULER_PROFILING=1.
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        max_samples: int = 10000,
    ) -> None:
        self._sample_rate = sample_rate
        self._max_samples = max_samples
        self._samples: deque[ProfilingSample] = deque(maxlen=self._max_samples)
        self._step_counter = 0
        self._enabled = True

    def pre_step(self, scheduler: Any) -> None:
        self._step_start = time.monotonic()

    def post_step(self, scheduler: Any, output: Any) -> None:
        self._step_counter += 1
        if not self._enabled:
            return
        import random

        if random.random() > self._sample_rate:
            return

        now = time.monotonic()
        step_latency = (
            (now - self._step_start) * 1000 if hasattr(self, "_step_start") else 0.0
        )

        batch_size = (
            len(output.outputs) if hasattr(output, "outputs") and output.outputs else 0
        )
        tokens = sum(getattr(o, "completion_tokens", 1) for o in (output.outputs or []))

        active_mem = 0
        peak_mem = 0
        try:
            import mlx.core as mx

            active_mem = mx.get_active_memory()
            peak_mem = mx.get_peak_memory()
        except Exception:
            logger.debug("operation failed", exc_info=True)

        phase = "decode"
        if output is not None and hasattr(output, "outputs"):
            for o in output.outputs or []:
                eop = getattr(o, "end_of_prompt", False)
                prog = getattr(o, "progress", None)
                if eop or (prog is not None and prog < 1.0):
                    phase = "prefill"
                    break
        # Fallback: if waiting queue is non-empty and output didn't
        # explicitly indicate decode, treat as prefill (the step likely
        # scheduled a prefill from the waiting queue).
        if phase == "decode" and hasattr(scheduler, "waiting") and scheduler.waiting:
            phase = "prefill"

        sample = ProfilingSample(
            request_id="",
            step=self._step_counter,
            phase=phase,
            latency_ms=round(step_latency, 3),
            tokens_generated=tokens,
            batch_size=batch_size,
            memory_active_bytes=active_mem,
            memory_peak_bytes=peak_mem,
        )
        self._samples.append(sample)

    def get_stats(self) -> dict:
        if not self._samples:
            return {"enabled": True, "samples": 0}

        prefill_samples = [s for s in self._samples if s.phase == "prefill"]
        decode_samples = [s for s in self._samples if s.phase == "decode"]

        def _avg_lat(samples: list[ProfilingSample]) -> float:
            return sum(s.latency_ms for s in samples) / len(samples) if samples else 0.0

        return {
            "enabled": True,
            "samples": len(self._samples),
            "steps_profiled": self._step_counter,
            "avg_prefill_latency_ms": round(_avg_lat(prefill_samples), 3),
            "avg_decode_latency_ms": round(_avg_lat(decode_samples), 3),
            "avg_batch_size": round(
                sum(s.batch_size for s in self._samples) / len(self._samples), 2
            ),
            "memory_active_mb": round(
                self._samples[-1].memory_active_bytes / 1024 / 1024, 1
            )
            if self._samples
            else 0,
            "memory_peak_mb": round(
                self._samples[-1].memory_peak_bytes / 1024 / 1024, 1
            )
            if self._samples
            else 0,
        }

    def export_traces(self) -> list[dict]:
        """Export profiling samples as JSON-serializable trace data."""
        return [
            {
                "step": s.step,
                "phase": s.phase,
                "latency_ms": s.latency_ms,
                "tokens": s.tokens_generated,
                "batch_size": s.batch_size,
                "memory_active_mb": round(s.memory_active_bytes / 1024 / 1024, 1),
                "memory_peak_mb": round(s.memory_peak_bytes / 1024 / 1024, 1),
            }
            for s in self._samples
        ]


class SpecDecodeMixin(SchedulerMixin):
    """Speculative decode integration with the scheduler.

    Tracks draft token production, verification results, and acceptance
    rates per proposer type. Provides adaptive draft length control.
    """

    def __init__(
        self,
        initial_draft_length: int = 5,
        max_draft_length: int = 10,
        min_draft_length: int = 2,
        acceptance_window: int = 50,
    ) -> None:
        self._draft_length = initial_draft_length
        self._max_draft_length = max_draft_length
        self._min_draft_length = min_draft_length
        self._acceptance_window = acceptance_window
        self._acceptances: deque[bool] = deque(maxlen=self._acceptance_window)
        self._total_drafts = 0
        self._total_accepted = 0
        self._total_rejected = 0
        self._proposer_stats: dict[str, dict] = defaultdict(
            lambda: {"drafts": 0, "accepted": 0, "rejected": 0}
        )

    def pre_step(self, scheduler: Any) -> None:
        pass

    def post_step(self, scheduler: Any, output: Any) -> None:
        if not hasattr(output, "outputs") or not output.outputs:
            return
        for o in output.outputs:
            spec_accepted = getattr(o, "spec_accepted", None)
            if spec_accepted is not None:
                self._acceptances.append(spec_accepted)
                if spec_accepted:
                    self._total_accepted += 1
                else:
                    self._total_rejected += 1
                self._total_drafts += 1

                proposer = getattr(o, "spec_proposer", "unknown")
                stats = self._proposer_stats[proposer]
                stats["drafts"] += 1
                if spec_accepted:
                    stats["accepted"] += 1
                else:
                    stats["rejected"] += 1

        # Adaptive draft length adjustment
        if len(self._acceptances) >= 10:
            acceptance_rate = sum(self._acceptances) / len(self._acceptances)
            if acceptance_rate > 0.85 and self._draft_length < self._max_draft_length:
                self._draft_length += 1
            elif acceptance_rate < 0.5 and self._draft_length > self._min_draft_length:
                self._draft_length -= 1

    @property
    def current_draft_length(self) -> int:
        return self._draft_length

    def get_stats(self) -> dict:
        rate = (
            sum(self._acceptances) / len(self._acceptances)
            if self._acceptances
            else 0.0
        )
        return {
            "draft_length": self._draft_length,
            "total_drafts": self._total_drafts,
            "total_accepted": self._total_accepted,
            "total_rejected": self._total_rejected,
            "acceptance_rate": round(rate, 4),
            "per_proposer": dict(self._proposer_stats),
        }


class MemoryPressureMixin(SchedulerMixin):
    """Memory pressure-aware scheduling with hysteresis.

    Monitors active memory and adjusts batch sizes / admission rates:
    - Normal: full batch size
    - Warning (>threshold): reduce batch size, pause new admissions
    - Critical (>95%): emergency eviction, reject new requests
    """

    def __init__(
        self,
        warning_threshold: float = 0.80,
        critical_threshold: float = 0.95,
        total_memory_bytes: int = 0,
        batch_size_normal: int = 32,
        batch_size_warning: int = 16,
        batch_size_critical: int = 4,
        hysteresis: float = 0.05,
    ) -> None:
        self._warning_threshold = warning_threshold
        self._critical_threshold = critical_threshold
        self._total_memory_bytes = total_memory_bytes
        self._batch_normal = batch_size_normal
        self._batch_warning = batch_size_warning
        self._batch_critical = batch_size_critical
        self._hysteresis = hysteresis
        self._current_state = "normal"  # normal | warning | critical
        self._transitions = 0
        self._last_memory_fraction = 0.0
        self._admission_paused = False

    @classmethod
    def from_env(cls) -> MemoryPressureMixin:
        return cls(
            warning_threshold=float(os.environ.get("YUNSHU_MEM_WARNING", "0.80")),
            critical_threshold=float(os.environ.get("YUNSHU_MEM_CRITICAL", "0.95")),
            batch_size_normal=int(os.environ.get("YUNSHU_BATCH_NORMAL", "32")),
            batch_size_warning=int(os.environ.get("YUNSHU_BATCH_WARNING", "16")),
            batch_size_critical=int(os.environ.get("YUNSHU_BATCH_CRITICAL", "4")),
        )

    def pre_step(self, scheduler: Any) -> None:
        try:
            import mlx.core as mx

            active = mx.get_active_memory()
            total = self._total_memory_bytes
            if total <= 0:
                return
            fraction = active / total
            self._last_memory_fraction = fraction

            prev_state = self._current_state
            if fraction >= self._critical_threshold:
                self._current_state = "critical"
                self._admission_paused = True
            elif fraction >= self._warning_threshold:
                if self._current_state == "critical":
                    if fraction < self._critical_threshold - self._hysteresis:
                        self._current_state = "warning"
                else:
                    self._current_state = "warning"
                self._admission_paused = False
            elif fraction < self._warning_threshold - self._hysteresis:
                self._current_state = "normal"
                self._admission_paused = False
            elif self._current_state != "critical":
                pass

            if prev_state != self._current_state:
                self._transitions += 1
                logger.info(
                    f"Memory pressure: {prev_state} → {self._current_state} "
                    f"({fraction:.1%})"
                )
        except Exception:
            logger.debug("operation failed", exc_info=True)

    def post_step(self, scheduler: Any, output: Any) -> None:
        pass

    @property
    def recommended_batch_size(self) -> int:
        if self._current_state == "critical":
            return self._batch_critical
        elif self._current_state == "warning":
            return self._batch_warning
        return self._batch_normal

    @property
    def is_admission_paused(self) -> bool:
        return self._admission_paused

    def get_stats(self) -> dict:
        return {
            "state": self._current_state,
            "memory_fraction": round(self._last_memory_fraction, 4),
            "recommended_batch_size": self.recommended_batch_size,
            "admission_paused": self._admission_paused,
            "state_transitions": self._transitions,
            "thresholds": {
                "warning": self._warning_threshold,
                "critical": self._critical_threshold,
            },
        }


class CompositionScheduler:
    """Composable scheduler that orchestrates mixins.

    Wraps a core Scheduler and applies mixins in order:
      pre_step hooks → core step → post_step hooks

    The lifecycle is:
      1. request added → on_add_request() hooks
      2. each step → pre_step() → scheduler.step() → post_step()
      3. request finished → on_finish() hooks
      4. shutdown → shutdown() hooks
    """

    def __init__(self, scheduler: Any) -> None:
        self._scheduler = scheduler
        self._mixins: list[SchedulerMixin] = []

    def add_mixin(self, mixin: SchedulerMixin) -> CompositionScheduler:
        self._mixins.append(mixin)
        return self

    @property
    def core(self) -> Any:
        return self._scheduler

    def add_request(self, request: Any) -> None:
        for m in self._mixins:
            m.on_add_request(self._scheduler, request)
        self._scheduler.add_request(request)

    def step(self) -> Any:
        for m in self._mixins:
            m.pre_step(self._scheduler)
        output = self._scheduler.step()
        for m in self._mixins:
            m.post_step(self._scheduler, output)
        # Check for finished requests
        if hasattr(output, "outputs") and output.outputs:
            for o in output.outputs:
                if getattr(o, "finished", False):
                    rid = getattr(o, "request_id", "")
                    for m in self._mixins:
                        m.on_finish(self._scheduler, rid, o)
        return output

    def has_requests(self) -> bool:
        return self._scheduler.has_requests()

    def abort_request(self, request_id: str) -> None:
        self._scheduler.abort_request(request_id)

    def remove_finished_request(self, request_id: str) -> None:
        self._scheduler.remove_finished_request(request_id)

    def fail_all_requests(self) -> list[str]:
        return self._scheduler.fail_all_requests()

    def shutdown(self) -> None:
        for m in self._mixins:
            m.shutdown()
        self._scheduler.shutdown()

    def get_stats(self) -> dict:
        stats = {}
        for _i, m in enumerate(self._mixins):
            name = type(m).__name__
            stats[name] = m.get_stats()
        return stats

    def get_mixin(self, mixin_type: type) -> SchedulerMixin | None:
        for m in self._mixins:
            if isinstance(m, mixin_type):
                return m
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._scheduler, name)
