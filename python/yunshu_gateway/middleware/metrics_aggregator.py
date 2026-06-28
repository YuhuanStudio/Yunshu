from __future__ import annotations

"""Yunshu Gateway — Time-windowed metrics aggregator.

Records individual request data points with timestamps and provides
rolling-window aggregation: sums, averages, percentiles (p50/p90/p95/p99).

The rolling window retains the last *N* minutes of data points
(configurable, default 10 minutes).  All operations are thread-safe.

This complements the cumulative _Metrics / PrometheusMetrics registries
by answering questions like "what was the p99 latency in the last 60 seconds?"
"""


import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock
from typing import Any


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
        # Track min/max timestamps for efficient pruning.
        self._oldest_ts: float = 0.0
        # Token throughput accumulator. The MetricsMiddleware records request
        # data points WITHOUT token counts (they aren't available at middleware scope, and
        # are recorded per-request by the routers' record_tokens for both streaming and
        # non-streaming), so per-request token fields were always 0 → the rolling-window
        # summary reported zero token throughput forever. Routers feed record_tokens() here.
        self._token_points: list[tuple[float, int, int]] = []  # (ts, tokens_in, tokens_out)

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
            if not self._points:
                self._oldest_ts = now
            self._points.append(dp)
            # Prune expired entries.
            self._prune(now)

    def record_tokens(self, tokens_in: int, tokens_out: int) -> None:
        """Record per-request token counts.

        Routers call this from their ``_record_metrics`` path (covering both
        streaming and non-streaming) because the MetricsMiddleware records the
        request data point at HTTP scope where token counts are not yet known.
        Without this, ``get_summary`` reported ``total_tokens_out`` /
        ``tokens_per_second`` as a permanent 0.
        """
        now = time.time()
        ti = int(tokens_in) if tokens_in else 0
        to = int(tokens_out) if tokens_out else 0
        if ti <= 0 and to <= 0:
            return
        with self._lock:
            self._token_points.append((now, ti, to))
            self._prune_tokens(now)

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
        # the buffer is physically pruned to _max_window (default 600s), but the
        # router accepts window up to 3600s — asking for a window LARGER than retention then
        # divided the (≤600s) data by the full requested span → rate/tps UNDERSTATED up to
        # ~6x. Clamp to what's actually retained so the denominator matches the data.
        window_seconds = min(window_seconds, self._max_window)
        cutoff = time.time() - window_seconds
        with self._lock:
            window = [p for p in self._points if p.timestamp >= cutoff]
            token_window = [tp for tp in self._token_points if tp[0] >= cutoff]

        # Token counts come from the dedicated record_tokens() stream (the
        # MetricsMiddleware request points carry no token data). Fall back to the
        # request points' own fields if some caller populated them directly.
        tokens_in_from_points = sum(p.tokens_in for p in window)
        tokens_out_from_points = sum(p.tokens_out for p in window)
        total_tokens_in = sum(tp[1] for tp in token_window) + tokens_in_from_points
        total_tokens_out = sum(tp[2] for tp in token_window) + tokens_out_from_points

        if not window and not token_window:
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

        # Compute actual time span covered by data points.
        # Use window_seconds for single-point cases (where earliest == latest)
        # to avoid inflating rate metrics from a single observation.
        # Derive the span from whichever stream has points so token throughput
        # is sane even when only token points exist for this window.
        # a windowed rate ("last N seconds") must divide by the WINDOW,
        # not the span between the first and last data point. The old data-span
        # denominator ignored idle gaps and clamped to 1.0, so two requests 1ms
        # apart in an otherwise-idle 60s window reported ~2 req/s instead of ~0.03
        # (up to ~30x overstatement). Idle time in the window IS part of the rate.
        span = max(float(window_seconds), 1.0)

        return {
            "total_requests": total,
            "error_count": errors,
            "error_rate": round(errors / total, 4) if total else 0.0,
            "avg_duration_ms": round(sum(durations) / len(durations), 2) if durations else 0.0,
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
        if metric not in ("duration_ms", "tokens_in", "tokens_out"):
            return self._empty_percentiles()
        window_seconds = min(window_seconds, self._max_window)  # see get_summary
        cutoff = time.time() - window_seconds
        with self._lock:
            if metric in ("tokens_in", "tokens_out"):
                # token counts live in the dedicated record_tokens()
                # stream, not on the request points (which carry 0).
                # Reading them off self._points alone gave a permanent all-zero
                # distribution for token_percentiles on the dashboard. Union the
                # token stream with any nonzero request-point token fields (some
                # callers populate record_request(..., tokens_out=N) directly).
                idx = 1 if metric == "tokens_in" else 2
                token_vals = [tp[idx] for tp in self._token_points if tp[0] >= cutoff]
                point_vals = [
                    getattr(p, metric) for p in self._points
                    if p.timestamp >= cutoff and getattr(p, metric, 0)
                ]
                values = sorted(token_vals + point_vals)
            else:
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
        """Return per-endpoint aggregated stats for the window.

        NOTE: per-endpoint ``total_tokens_out`` reflects only tokens
        carried on the request data points. In production the MetricsMiddleware
        records request points WITHOUT token counts (tokens arrive out-of-band
        via record_tokens(), which has no endpoint label), so this
        field is 0 unless a caller populates record_request(..., tokens_out=N)
        directly. Endpoint count + latency are always accurate; for true token
        throughput use get_summary()/get_percentiles("tokens_out").
        """
        window_seconds = min(window_seconds, self._max_window)  # see get_summary
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
        # Fast path: if oldest point is still within window, nothing to prune.
        if self._oldest_ts > 0 and self._oldest_ts >= now - self._max_window:
            return
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
        self._oldest_ts = self._points[0].timestamp if self._points else 0.0

    def _prune_tokens(self, now: float) -> None:
        """Remove token points older than the max window. Caller holds lock."""
        if not self._token_points:
            return
        if self._token_points[0][0] >= now - self._max_window:
            return
        cutoff = now - self._max_window
        idx = 0
        for i, tp in enumerate(self._token_points):
            if tp[0] >= cutoff:
                idx = i
                break
        else:
            idx = len(self._token_points)
        if idx > 0:
            self._token_points = self._token_points[idx:]

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
_instance_lock = threading.Lock()


def get_metrics_aggregator() -> MetricsAggregator:
    """Return the global MetricsAggregator singleton (thread-safe)."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = MetricsAggregator()
    return _instance


def reset_metrics_aggregator() -> None:
    """Reset the singleton (for tests only)."""
    global _instance
    with _instance_lock:
        _instance = None
