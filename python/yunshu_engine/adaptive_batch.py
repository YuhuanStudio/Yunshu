from __future__ import annotations

"""Yunshu Adaptive Batch Scheduler — load-aware batch sizing.

Dynamically adjusts batch size and prefill chunk size based on:
- GPU memory pressure (UMA utilisation)
- Generation latency (time-to-first-token + inter-token latency)
- Pending request queue depth

Uses exponential-weighted moving averages (EWMA) for smooth
transitions and hysteresis to avoid oscillation.

Integration points:
  - Engine calls compute_batch_size() each scheduling step
  - PrefillChunker calls compute_prefill_chunk_size() for long prompts
  - Metrics sink calls update_metrics() after each batch step
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveBatchConfig:
    """Configuration for adaptive batch scheduling.

    Attributes:
        min_batch: Minimum batch size (at least 1 request).
        max_batch: Maximum batch size (bounded by memory / policy).
        target_latency_ms: Target per-step latency in milliseconds.
        memory_threshold: Fraction of GPU memory beyond which we shrink batches.
        ewma_alpha: Smoothing factor for EWMA (0 < alpha <= 1). Higher = more responsive.
        scale_up_factor: Multiplier when scaling up batch size.
        scale_down_factor: Multiplier when scaling down batch size.
        prefill_memory_reserve: Fraction of free memory to reserve for prefill.
    """

    min_batch: int = 1
    max_batch: int = 32
    target_latency_ms: float = 100.0
    memory_threshold: float = 0.85
    ewma_alpha: float = 0.3
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.7
    prefill_memory_reserve: float = 0.7


@dataclass
class _RunningState:
    """Mutable state for EWMA tracking."""

    avg_latency_ms: float = 0.0
    avg_memory_usage: float = 0.0
    current_batch_size: int = 1
    total_updates: int = 0
    last_scale_time: float = 0.0


class AdaptiveBatchScheduler:
    """Adaptive batch size scheduler for continuous batching.

    Computes optimal batch size and prefill chunk size based on
    system load metrics. Uses EWMA to smooth out transient spikes
    and hysteresis (minimum interval between scale events) to
    prevent oscillation.
    """

    def __init__(self, config: AdaptiveBatchConfig | None = None) -> None:
        self._config = config or AdaptiveBatchConfig()
        self._state = _RunningState(
            current_batch_size=self._config.min_batch,
        )
        # Hysteresis: minimum seconds between consecutive scale events
        self._min_scale_interval_s: float = 1.0

    @property
    def config(self) -> AdaptiveBatchConfig:
        return self._config

    # ── Public API ──

    def compute_batch_size(
        self,
        current_memory_usage: float,
        avg_latency_ms: float,
        pending_count: int,
    ) -> int:
        """Compute optimal batch size for the next scheduling step.

        Decision logic:
        1. If memory usage exceeds threshold, scale down aggressively.
        2. If latency exceeds 2x target, scale down moderately.
        3. If latency is below target and memory is comfortable, scale up.
        4. Clamp result to [min_batch, min(max_batch, pending_count)].

        Args:
            current_memory_usage: GPU memory utilisation fraction [0.0, 1.0].
            avg_latency_ms: Average per-step latency in milliseconds.
            pending_count: Number of pending requests in the queue.

        Returns:
            Optimal batch size for the next step.
        """
        cfg = self._config
        batch = self._state.current_batch_size

        # --- Memory pressure ---
        if current_memory_usage > cfg.memory_threshold:
            # Over threshold: scale down aggressively
            batch = max(
                cfg.min_batch,
                int(batch * cfg.scale_down_factor),
            )
            logger.debug(
                "Memory pressure %.1f%% > threshold %.1f%%, batch -> %d",
                current_memory_usage * 100,
                cfg.memory_threshold * 100,
                batch,
            )

        # --- Latency pressure ---
        elif avg_latency_ms > cfg.target_latency_ms * 2.0:
            # Latency is 2x target: moderate scale-down
            batch = max(
                cfg.min_batch,
                int(batch * cfg.scale_down_factor),
            )
            logger.debug(
                "Latency pressure %.1fms > 2x target %.1fms, batch -> %d",
                avg_latency_ms,
                cfg.target_latency_ms,
                batch,
            )

        elif avg_latency_ms > cfg.target_latency_ms:
            # Latency is above target but within 2x: hold current
            pass

        else:
            # Latency below target, memory comfortable: scale up
            if current_memory_usage < cfg.memory_threshold * 0.75:
                batch = min(
                    cfg.max_batch,
                    int(batch * cfg.scale_up_factor),
                )
                logger.debug(
                    "Latency %.1fms < target %.1fms, memory %.1f%% comfortable, batch -> %d",
                    avg_latency_ms,
                    cfg.target_latency_ms,
                    current_memory_usage * 100,
                    batch,
                )

        # --- Clamp to pending count ---
        if pending_count == 0:
            return 0
        batch = max(cfg.min_batch, min(batch, pending_count))

        # --- Clamp to config bounds ---
        batch = max(cfg.min_batch, min(batch, cfg.max_batch))

        # --- Hysteresis: prevent rapid oscillation on scale-UP only ---
        # Scale-down under memory pressure must always be honoured to avoid OOM.
        if batch != self._state.current_batch_size:
            is_scale_down = batch < self._state.current_batch_size
            now = time.monotonic()
            if (
                not is_scale_down
                and now - self._state.last_scale_time < self._min_scale_interval_s
            ):
                # Too soon for scale-UP: suppress
                batch = self._state.current_batch_size
            else:
                self._state.last_scale_time = now

        self._state.current_batch_size = batch
        return batch

    def compute_prefill_chunk_size(
        self,
        seq_length: int,
        available_memory: float,
    ) -> int:
        """Compute safe prefill chunk size for a long prompt.

        Splits long prefills into chunks to avoid OOM. Uses a simple
        heuristic: allocate proportional to available memory, with
        a hard minimum of 128 tokens and hard maximum of 4096 tokens.

        Args:
            seq_length: Total prompt length in tokens.
            available_memory: Fraction of GPU memory available [0.0, 1.0].

        Returns:
            Maximum tokens to prefill in a single chunk.
        """
        MIN_CHUNK = 128
        MAX_CHUNK = 4096

        if seq_length <= MIN_CHUNK:
            return seq_length

        # Reserve fraction of available memory for prefill
        usable = available_memory * self._config.prefill_memory_reserve

        # Proportional scaling: more memory -> larger chunks
        # Baseline: at 0.5 available memory, allow full MAX_CHUNK
        if usable >= 0.5:
            chunk = MAX_CHUNK
        elif usable >= 0.25:
            chunk = int(MAX_CHUNK * (usable / 0.5))
        elif usable >= 0.1:
            chunk = int(MAX_CHUNK * 0.25 * (usable / 0.25))
        else:
            chunk = MIN_CHUNK

        # Don't exceed remaining sequence
        chunk = min(chunk, seq_length)

        # Clamp bounds
        chunk = max(MIN_CHUNK, min(chunk, MAX_CHUNK))

        logger.debug(
            "Prefill chunk: seq_length=%d, available_memory=%.1f%%, chunk=%d",
            seq_length,
            available_memory * 100,
            chunk,
        )

        return chunk

    def update_metrics(
        self,
        latency_ms: float,
        memory_usage: float,
        batch_size: int,
    ) -> None:
        """Update running EWMA averages with new observations.

        Args:
            latency_ms: Observed per-step latency in milliseconds.
            memory_usage: Observed GPU memory utilisation fraction.
            batch_size: The batch size used in this step.
        """
        alpha = self._config.ewma_alpha
        s = self._state

        if s.total_updates == 0:
            # First observation: initialise directly
            s.avg_latency_ms = latency_ms
            s.avg_memory_usage = memory_usage
        else:
            # EWMA update
            s.avg_latency_ms = alpha * latency_ms + (1 - alpha) * s.avg_latency_ms
            s.avg_memory_usage = alpha * memory_usage + (1 - alpha) * s.avg_memory_usage

        s.current_batch_size = batch_size
        s.total_updates += 1

    def get_stats(self) -> dict[str, Any]:
        """Return current scheduler statistics.

        Returns:
            Dict with current_batch_size, avg_latency_ms,
            avg_memory_usage, total_updates, and config.
        """
        s = self._state
        c = self._config
        return {
            "current_batch_size": s.current_batch_size,
            "avg_latency_ms": round(s.avg_latency_ms, 2),
            "avg_memory_usage": round(s.avg_memory_usage, 4),
            "total_updates": s.total_updates,
            "config": {
                "min_batch": c.min_batch,
                "max_batch": c.max_batch,
                "target_latency_ms": c.target_latency_ms,
                "memory_threshold": c.memory_threshold,
            },
        }

    def reset(self) -> None:
        """Reset scheduler state to defaults."""
        self._state = _RunningState(
            current_batch_size=self._config.min_batch,
        )
