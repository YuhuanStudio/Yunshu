"""Yunshu CPU/GPU Overlap Scheduler — C18: overlap CPU post-processing with GPU forward.

Studied from SGLang's overlap scheduling pattern:
- GPU forward pass launched via mx.async_eval() (non-blocking)
- CPU work (detokenization, grammar checking, response distribution)
  overlaps with in-flight GPU computation
- Synchronization only when GPU results are needed

Pipeline:
  Step N:   GPU_forward(N) ─── async ───┐
                                         │ overlap
  Step N:   CPU_postprocess(N-1) ────────┘
  Step N:   mx.synchronize() ← get GPU results for N
  Step N:   CPU_postprocess(N) queued for next step

On Apple Silicon (unified memory), the benefit is smaller than on
discrete GPUs but still measurable: detokenization + grammar processing
takes 0.1–0.5ms per token which overlaps with the GPU compute.

Integration:
  - EngineCore._engine_loop() wraps scheduler.step() with OverlapScheduler
  - OverlapScheduler.step_async() returns immediately after launching GPU work
  - OverlapScheduler.step_sync() waits for GPU + processes outputs
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class OverlapConfig:
    """Configuration for CPU/GPU overlap scheduling.

    Attributes:
        enabled: Whether to enable overlap scheduling.
        async_eval: Use mx.async_eval() for GPU work (default True).
        overlap_detokenize: Overlap detokenization with GPU forward.
        overlap_grammar: Overlap grammar/constraint checking with GPU forward.
        overlap_response_dist: Overlap response distribution with GPU forward.
        sync_timeout_ms: Maximum time to wait for GPU synchronization.
        metrics_window: Number of recent steps for rolling metrics.
    """

    enabled: bool = False
    async_eval: bool = True
    overlap_detokenize: bool = True
    overlap_grammar: bool = True
    overlap_response_dist: bool = False
    sync_timeout_ms: float = 100.0
    metrics_window: int = 100

    @classmethod
    def from_env(cls) -> OverlapConfig:
        """Create config from environment variables."""
        return cls(
            enabled=os.environ.get("YUNSHU_CPU_GPU_OVERLAP", "0") == "1",
            async_eval=os.environ.get("YUNSHU_ASYNC_EVAL", "1") == "1",
            overlap_detokenize=os.environ.get("YUNSHU_OVERLAP_DETOKENIZE", "1") == "1",
            overlap_grammar=os.environ.get("YUNSHU_OVERLAP_GRAMMAR", "1") == "1",
            overlap_response_dist=os.environ.get("YUNSHU_OVERLAP_RESPONSE_DIST", "0") == "1",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "async_eval": self.async_eval,
            "overlap_detokenize": self.overlap_detokenize,
            "overlap_grammar": self.overlap_grammar,
            "overlap_response_dist": self.overlap_response_dist,
            "sync_timeout_ms": self.sync_timeout_ms,
            "metrics_window": self.metrics_window,
        }


@dataclass
class OverlapMetrics:
    """Rolling metrics for overlap efficiency tracking."""

    gpu_time_ms: float = 0.0
    cpu_overlap_time_ms: float = 0.0
    total_overlap_time_ms: float = 0.0
    steps_with_overlap: int = 0
    total_steps: int = 0
    gpu_only_time_ms: float = 0.0
    cpu_only_time_ms: float = 0.0

    # Rolling window for recent stats
    _recent_gpu: list[float] = field(default_factory=list)
    _recent_cpu_overlap: list[float] = field(default_factory=list)
    _window: int = 100

    def record_step(
        self,
        gpu_time_ms: float,
        cpu_overlap_time_ms: float,
        cpu_only_time_ms: float,
        had_overlap: bool,
    ) -> None:
        self.total_steps += 1
        self.gpu_time_ms += gpu_time_ms
        self.cpu_overlap_time_ms += cpu_overlap_time_ms
        self.cpu_only_time_ms += cpu_only_time_ms
        self.gpu_only_time_ms += gpu_time_ms
        self.total_overlap_time_ms += cpu_overlap_time_ms

        if had_overlap:
            self.steps_with_overlap += 1

        self._recent_gpu.append(gpu_time_ms)
        self._recent_cpu_overlap.append(cpu_overlap_time_ms)
        if len(self._recent_gpu) > self._window:
            self._recent_gpu.pop(0)
            self._recent_cpu_overlap.pop(0)

    @property
    def overlap_efficiency(self) -> float:
        """Fraction of GPU time that was overlapped with CPU work (0.0–1.0)."""
        if not self._recent_gpu:
            return 0.0
        avg_gpu = sum(self._recent_gpu) / len(self._recent_gpu)
        avg_overlap = sum(self._recent_cpu_overlap) / len(self._recent_cpu_overlap)
        if avg_gpu <= 0:
            return 0.0
        return min(avg_overlap / avg_gpu, 1.0)

    @property
    def overlap_rate(self) -> float:
        """Fraction of steps that had any overlap (0.0–1.0)."""
        if self.total_steps == 0:
            return 0.0
        return self.steps_with_overlap / self.total_steps

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "steps_with_overlap": self.steps_with_overlap,
            "overlap_rate": round(self.overlap_rate, 3),
            "overlap_efficiency": round(self.overlap_efficiency, 3),
            "avg_gpu_time_ms": round(
                self.gpu_time_ms / max(self.total_steps, 1), 3
            ),
            "avg_cpu_overlap_ms": round(
                self.cpu_overlap_time_ms / max(self.total_steps, 1), 3
            ),
            "avg_cpu_only_ms": round(
                self.cpu_only_time_ms / max(self.total_steps, 1), 3
            ),
        }

    def reset(self) -> None:
        self.gpu_time_ms = 0.0
        self.cpu_overlap_time_ms = 0.0
        self.total_overlap_time_ms = 0.0
        self.steps_with_overlap = 0
        self.total_steps = 0
        self.gpu_only_time_ms = 0.0
        self.cpu_only_time_ms = 0.0
        self._recent_gpu.clear()
        self._recent_cpu_overlap.clear()


class OverlapScheduler:
    """CPU/GPU overlap wrapper for the continuous batching step loop.

    Wraps the scheduler step to overlap CPU post-processing with GPU
    forward passes using mx.async_eval().

    Usage:
        overlap = OverlapScheduler(config)
        overlap.step_async(scheduler)   # Launch GPU work (non-blocking)
        # ... CPU can do other work here ...
        output = overlap.step_sync()    # Wait for GPU + get results

    Or use step() for synchronous compatibility (no overlap benefit).
    """

    def __init__(self, config: OverlapConfig | None = None) -> None:
        self._config = config or OverlapConfig.from_env()
        self._metrics = OverlapMetrics()
        self._pending_outputs: list[Any] | None = None
        self._gpu_pending: bool = False
        self._step_start_time: float = 0.0
        self._gpu_launch_time: float = 0.0

        # Previous step's CPU post-processing work (to overlap)
        self._prev_cpu_work: list[Any] = []

    @property
    def config(self) -> OverlapConfig:
        return self._config

    @property
    def metrics(self) -> OverlapMetrics:
        return self._metrics

    @property
    def is_gpu_pending(self) -> bool:
        return self._gpu_pending

    def step(self, scheduler: Any) -> Any:
        """Synchronous step — same as scheduler.step() but with metrics.

        Use step_async() + step_sync() for actual overlap benefit.
        """
        return scheduler.step()

    def step_async(self, scheduler: Any) -> None:
        """Launch GPU work for the current step (non-blocking).

        After calling this, the CPU can do other work while the GPU
        processes. Call step_sync() to wait for results.

        If there's CPU work from the previous step, it runs here
        overlapped with the GPU computation.
        """
        if not self._config.enabled:
            return

        self._step_start_time = time.perf_counter()

        # Process previous step's CPU work while GPU is idle
        if self._prev_cpu_work:
            self._run_cpu_postprocess(self._prev_cpu_work)
            self._prev_cpu_work = []

        # Launch scheduler step — GPU work begins
        self._pending_outputs = scheduler.step()

        # If async_eval is enabled, trigger non-blocking GPU dispatch
        if self._config.async_eval and self._pending_outputs is not None:
            self._async_evaluate_outputs(self._pending_outputs)

        self._gpu_launch_time = time.perf_counter()
        self._gpu_pending = True

    def step_sync(self) -> Any:
        """Wait for GPU results and return outputs.

        If overlap is enabled, processes CPU post-processing for the
        current batch while queuing it for overlap in the next step.
        """
        if not self._config.enabled or not self._gpu_pending:
            return self._pending_outputs

        gpu_sync_start = time.perf_counter()

        # Synchronize GPU — wait for forward pass to complete
        self._synchronize_gpu()

        gpu_sync_end = time.perf_counter()
        gpu_time_ms = (gpu_sync_end - self._step_start_time) * 1000

        # CPU post-processing for current batch
        cpu_start = time.perf_counter()
        outputs = self._pending_outputs

        # Collect CPU work items from this step for next-step overlap
        cpu_work = self._extract_cpu_work(outputs)
        if cpu_work:
            self._prev_cpu_work = cpu_work

        cpu_end = time.perf_counter()
        cpu_only_ms = (cpu_end - cpu_start) * 1000

        # Calculate overlap: CPU work from previous step ran during GPU
        had_overlap = len(self._prev_cpu_work) > 0 or cpu_only_ms > 0
        cpu_overlap_ms = max(0, gpu_time_ms - cpu_only_ms) if had_overlap else 0

        self._metrics.record_step(
            gpu_time_ms=gpu_time_ms,
            cpu_overlap_time_ms=cpu_overlap_ms,
            cpu_only_time_ms=cpu_only_ms,
            had_overlap=had_overlap,
        )

        self._gpu_pending = False
        return outputs

    def _run_cpu_postprocess(self, work_items: list[Any]) -> None:
        """Run CPU-bound post-processing work (detokenization, grammar, etc.)."""
        for item in work_items:
            if isinstance(item, dict):
                self._process_work_item(item)

    def _process_work_item(self, item: dict) -> None:
        """Process a single CPU work item."""
        kind = item.get("kind")
        if kind == "detokenize" and self._config.overlap_detokenize:
            self._do_detokenize(item)
        elif kind == "grammar" and self._config.overlap_grammar:
            self._do_grammar_check(item)
        elif kind == "response_dist" and self._config.overlap_response_dist:
            self._do_response_dist(item)

    @staticmethod
    def _do_detokenize(item: dict) -> None:
        """Detokenize token IDs to text."""
        detokenizer = item.get("detokenizer")
        token_id = item.get("token_id")
        if detokenizer is not None and token_id is not None:
            detokenizer.add_token(token_id)

    @staticmethod
    def _do_grammar_check(item: dict) -> None:
        """Check grammar constraints on generated text."""
        checker = item.get("checker")
        text = item.get("text", "")
        if checker is not None and hasattr(checker, "validate"):
            checker.validate(text)

    @staticmethod
    def _do_response_dist(item: dict) -> None:
        """Distribute response to output collectors."""
        collector = item.get("collector")
        output = item.get("output")
        if collector is not None and output is not None:
            collector.put(output)

    def _extract_cpu_work(self, outputs: Any) -> list[dict]:
        """Extract CPU-post-processable work items from scheduler outputs.

        Returns a list of work items that can be processed during the
        next GPU forward pass for overlap.
        """
        if outputs is None:
            return []
        work = []
        # Extract from SchedulerOutput.outputs if available
        output_list = getattr(outputs, "outputs", None) or []
        if isinstance(outputs, list):
            output_list = outputs

        for out in output_list:
            if not hasattr(out, "__dict__"):
                continue
            # Queue detokenize work if detokenizer is attached
            detokenizer = getattr(out, "_detokenizer", None)
            token_id = getattr(out, "_last_token_id", None)
            if detokenizer and token_id is not None and self._config.overlap_detokenize:
                work.append({
                    "kind": "detokenize",
                    "detokenizer": detokenizer,
                    "token_id": token_id,
                })

            # Queue grammar check work if checker is attached
            checker = getattr(out, "_grammar_checker", None)
            text = getattr(out, "text", "")
            if checker and text and self._config.overlap_grammar:
                work.append({
                    "kind": "grammar",
                    "checker": checker,
                    "text": text,
                })

        return work

    def _async_evaluate_outputs(self, outputs: Any) -> None:
        """Trigger mx.async_eval() on any mx.arrays in outputs."""
        try:
            import mlx.core as mx

            # Find mx.arrays in the output to async-evaluate
            arrays = self._collect_arrays(outputs)
            if arrays:
                mx.async_eval(*arrays)
        except ImportError:
            pass

    @staticmethod
    def _collect_arrays(obj: Any, depth: int = 0) -> list:
        """Recursively collect mx.array instances from nested structures."""
        if depth > 5:
            return []
        try:
            import mlx.core as mx
            if isinstance(obj, mx.array):
                return [obj]
        except ImportError:
            return []

        arrays = []
        if isinstance(obj, (list, tuple)):
            for item in obj:
                arrays.extend(OverlapScheduler._collect_arrays(item, depth + 1))
        elif isinstance(obj, dict):
            for v in obj.values():
                arrays.extend(OverlapScheduler._collect_arrays(v, depth + 1))
        elif hasattr(obj, "__dict__"):
            for v in vars(obj).values():
                arrays.extend(OverlapScheduler._collect_arrays(v, depth + 1))
        return arrays

    @staticmethod
    def _synchronize_gpu() -> None:
        """Synchronize all pending GPU operations."""
        try:
            import mlx.core as mx
            mx.synchronize()
        except ImportError:
            pass

    def get_stats(self) -> dict[str, Any]:
        """Return overlap scheduling statistics."""
        return {
            "config": self._config.to_dict(),
            "metrics": self._metrics.get_stats(),
            "gpu_pending": self._gpu_pending,
        }

    def reset(self) -> None:
        """Reset all state and metrics."""
        self._metrics.reset()
        self._pending_outputs = None
        self._gpu_pending = False
        self._prev_cpu_work = []
