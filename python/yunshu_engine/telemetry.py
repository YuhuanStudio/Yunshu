"""Lightweight telemetry — sampled metric collection with batch flushing.

Usage::

    from yunshu_engine.telemetry import TelemetryConfig, TelemetryCollector

    cfg = TelemetryConfig(enabled=True, endpoint="http://localhost:4318/v1/metrics")
    tc = TelemetryCollector(cfg)

    tc.collect("request_latency_ms", 42.5, tags={"model": "qwen-0.5b"})
    tc.collect("tokens_generated", 128, tags={"model": "qwen-0.5b"})

    assert tc.get_pending_count() > 0
    tc.flush()  # sends to endpoint (stub — clears batch)

When ``enabled=False``, all methods are no-ops.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TelemetryConfig:
    """Configuration for telemetry collection.

    Attributes:
        enabled: When False, all collect/flush calls are no-ops.
        endpoint: URL to send metrics to (used by flush).
        sample_rate: Fraction of metrics to actually collect (0.0-1.0).
            1.0 means collect everything, 0.0 means collect nothing.
        batch_size: Maximum number of metrics to buffer before auto-flush.
    """

    enabled: bool = False
    endpoint: str = "http://localhost:4318/v1/metrics"
    sample_rate: float = 1.0
    batch_size: int = 100

    def __post_init__(self) -> None:
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError(
                f"sample_rate must be between 0.0 and 1.0, got {self.sample_rate}"
            )
        if self.batch_size < 1:
            raise ValueError(
                f"batch_size must be >= 1, got {self.batch_size}"
            )


@dataclass
class _Metric:
    """Internal metric representation."""

    name: str
    value: float
    tags: dict[str, str]
    timestamp: float


class TelemetryCollector:
    """Thread-safe metric collector with sampling and batching.

    - ``collect()`` applies sampling based on ``sample_rate``.
    - ``flush()`` sends the batch to ``endpoint`` (stub implementation clears).
    - All methods are no-ops when config ``enabled=False``.
    """

    def __init__(self, config: TelemetryConfig | None = None) -> None:
        self._config = config or TelemetryConfig()
        self._lock = threading.Lock()
        self._batch: list[_Metric] = []

    @property
    def config(self) -> TelemetryConfig:
        return self._config

    def collect(
        self,
        metric_name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> bool:
        """Collect a metric. Returns True if the metric was accepted (sampled).

        When ``enabled=False`` or sampling rejects the metric, returns False.
        """
        if not self._config.enabled:
            return False

        # Sampling: keep only sample_rate fraction
        if self._config.sample_rate < 1.0:
            if random.random() >= self._config.sample_rate:
                return False

        metric = _Metric(
            name=metric_name,
            value=float(value),
            tags=dict(tags) if tags else {},
            timestamp=time.time(),
        )

        with self._lock:
            self._batch.append(metric)
            if len(self._batch) >= self._config.batch_size:
                self._do_flush()
            return True

    def flush(self) -> int:
        """Flush pending metrics to the configured endpoint.

        Returns the number of metrics flushed.
        When ``enabled=False``, returns 0 immediately.
        """
        if not self._config.enabled:
            return 0
        with self._lock:
            return self._do_flush()

    def get_pending_count(self) -> int:
        """Return the number of metrics waiting to be flushed.

        Returns 0 when ``enabled=False``.
        """
        if not self._config.enabled:
            return 0
        with self._lock:
            return len(self._batch)

    def _do_flush(self) -> int:
        """Internal flush — caller must hold ``self._lock``.

        Stub implementation: logs the batch and clears it.
        In production this would POST to ``self._config.endpoint``.
        """
        count = len(self._batch)
        if count == 0:
            return 0

        # Stub: in production, serialize and POST to endpoint
        logger.debug(
            "Flushing %d metrics to %s",
            count,
            self._config.endpoint,
        )

        self._batch.clear()
        return count
