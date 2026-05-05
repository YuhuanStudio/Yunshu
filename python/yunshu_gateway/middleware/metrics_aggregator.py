"""Yunshu Gateway — Time-windowed metrics aggregator.

Records individual request data points with timestamps and provides
rolling-window aggregation: sums, averages, percentiles (p50/p90/p95/p99).

The rolling window retains the last *N* minutes of data points
(configurable, default 10 minutes).  All operations are thread-safe.

This complements the cumulative _Metrics / PrometheusMetrics registries
by answering questions like "what was the p99 latency in the last 60 seconds?"
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Optional


@dataclass
class _DataPoint:
    """A single recorded request data point."""

    timestamp: float
    method: str
    path: str
    status: int
    duration_ms: float
    tokens_in: int
    tokens_out: int


class MetricsAggregator:
    """Thread-safe rolling-window metrics aggregator.

    Usage::

        agg = get_metrics_aggregator()
        agg.record_request("POST", "/v1/chat", 200, 150.5, 64, 128)
        summary = agg.get_summary(window_seconds=60)
        p99 = agg.get_percentiles("duration_ms")
    """

    def __init__(self, max_window_seconds: int = 600) -> None:
        self._lock = Lock()
        self._max_window = max_window_seconds
        self._points: list[_DataPoint] = []
        # Quick-access indices for numeric fields.
        self._field_index: dict[str, dict[str, list[float]]] = {
            "duration_ms": defaultdict(list),
            "tokens_in": defaultdict(list),
            "tokens_out": defaultdict(list),
        }

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_request(
        self,
        method: str,
        path: str,
        status: int,
        duration_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> None:
        """Record a completed request data point."""
        now = time.time()
        dp = _DataPoint(
            timestamp=now,
            method=method,
            path=path,
            status=status,
            duration_ms=duration_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        with self._lock:
            self._points.append(dp)
            # Also store in field indices keyed by path for faster querying.
            key = f"{method}:{path}"
            self._field_index["duration_ms"][key].append(duration_ms)
            self._field_index["tokens_in"][key].append(float(tokens_in))
            self._field_index["tokens_out"][key].append(float(tokens_out))
            # Prune expired entries.
            self._prune(now)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_summary(self, window_seconds: int = 60) -> dict[str, Any]:
        """Return aggregated metrics for the last *window_seconds*.

        Returns a dict with keys:
          total_requests, error_count, error_rate,
          avg_duration_ms, total_tokens_in, total_tokens_out,
          tokens_per_second, requests_per_second
        """
        cutoff = time.time() - window_seconds
        with self._lock:
            window = [p for p in self._points if p.timestamp >= cutoff]

        if not window:
            return {
                "total_requests": 0,
                "error_count": 0,
                "error_rate": 0.0,
                "avg_duration_ms": 0.0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "tokens_per_second": 0.0,
                "requests_per_second": 0.0,
                "window_seconds": window_seconds,
            }

        total = len(window)
        errors = sum(1 for p in window if p.status >= 400)
        durations = [p.duration_ms for p in window]
        total_tokens_in = sum(p.tokens_in for p in window)
        total_tokens_out = sum(p.tokens_out for p in window)

        # Compute actual time span covered by data points.
        earliest = window[0].timestamp
        latest = window[-1].timestamp
        span = max(latest - earliest, 1.0)

        return {
            "total_requests": total,
            "error_count": errors,
            "error_rate": round(errors / total, 4) if total else 0.0,
            "avg_duration_ms": round(sum(durations) / len(durations), 2),
            "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out,
            "tokens_per_second": round(total_tokens_out / span, 2),
            "requests_per_second": round(total / span, 2),
            "window_seconds": window_seconds,
        }

    def get_percentiles(self, metric: str = "duration_ms", window_seconds: int = 60) -> dict[str, float]:
        """Return p50/p90/p95/p99 for *metric* over the last *window_seconds*.

        Valid metric names: duration_ms, tokens_in, tokens_out.
        """
        cutoff = time.time() - window_seconds
        with self._lock:
            if metric not in self._field_index:
                return self._empty_percentiles()
            # Collect values from points within the window.
            values = sorted(
                p_duration
                for p in self._points
                if p.timestamp >= cutoff
                for p_duration in [getattr(p, metric, None)]
                if p_duration is not None
            )

        if not values:
            return self._empty_percentiles()

        return {
            "p50": round(self._percentile(values, 50), 4),
            "p90": round(self._percentile(values, 90), 4),
            "p95": round(self._percentile(values, 95), 4),
            "p99": round(self._percentile(values, 99), 4),
            "count": len(values),
            "min": round(values[0], 4),
            "max": round(values[-1], 4),
            "avg": round(sum(values) / len(values), 4),
        }

    def get_endpoint_breakdown(self, window_seconds: int = 60) -> list[dict[str, Any]]:
        """Return per-endpoint aggregated stats for the window."""
        cutoff = time.time() - window_seconds
        with self._lock:
            window = [p for p in self._points if p.timestamp >= cutoff]

        buckets: dict[str, list[_DataPoint]] = defaultdict(list)
        for p in window:
            buckets[f"{p.method} {p.path}"].append(p)

        result = []
        for key, points in sorted(buckets.items()):
            durations = [p.duration_ms for p in points]
            tokens_out = [p.tokens_out for p in points]
            result.append({
                "endpoint": key,
                "count": len(points),
                "avg_duration_ms": round(sum(durations) / len(durations), 2) if durations else 0.0,
                "total_tokens_out": sum(tokens_out),
            })
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prune(self, now: float) -> None:
        """Remove data points older than the max window. Caller holds lock."""
        cutoff = now - self._max_window
        # Find first index within window.
        idx = 0
        for i, p in enumerate(self._points):
            if p.timestamp >= cutoff:
                idx = i
                break
        else:
            idx = len(self._points)
        if idx > 0:
            self._points = self._points[idx:]
            # Rebuild field indices (simpler than incremental pruning).
            for field_name in self._field_index:
                self._field_index[field_name].clear()
            for p in self._points:
                key = f"{p.method}:{p.path}"
                self._field_index["duration_ms"][key].append(p.duration_ms)
                self._field_index["tokens_in"][key].append(float(p.tokens_in))
                self._field_index["tokens_out"][key].append(float(p.tokens_out))

    @staticmethod
    def _percentile(sorted_values: list[float], pct: int) -> float:
        """Compute percentile using linear interpolation (numpy-style)."""
        n = len(sorted_values)
        if n == 0:
            return 0.0
        if n == 1:
            return sorted_values[0]
        # Nearest-rank method.
        k = (pct / 100.0) * (n - 1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_values[int(k)]
        d0 = sorted_values[int(f)] * (c - k)
        d1 = sorted_values[int(c)] * (k - f)
        return d0 + d1

    @staticmethod
    def _empty_percentiles() -> dict[str, float]:
        return {
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "avg": 0.0,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: MetricsAggregator | None = None


def get_metrics_aggregator() -> MetricsAggregator:
    """Return the global MetricsAggregator singleton."""
    global _instance
    if _instance is None:
        _instance = MetricsAggregator()
    return _instance


def reset_metrics_aggregator() -> None:
    """Reset the singleton (for tests only)."""
    global _instance
    _instance = None
