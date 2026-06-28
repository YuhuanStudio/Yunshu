"""metrics aggregator windowed rate understated when window > retention.

The buffer is physically pruned to _max_window (default 600s), but the /gw/monitoring/requests
router accepts window up to 3600s. Asking for a window LARGER than retention then divided the
(at most 600s of) retained data by the full requested span → requests_per_second / tokens_per_
second understated up to ~6x, and get_percentiles silently capped its sample set. Fix: clamp
window_seconds to _max_window in the read methods so the denominator matches the retained data.
"""
from __future__ import annotations

import time
import unittest.mock as m

from yunshu_gateway.middleware.metrics_aggregator import MetricsAggregator


def _fill(agg, n, span_s, now):
    for i in range(n):
        with m.patch("time.time", return_value=now - span_s + i * (span_s / n)):
            agg.record_request(method="GET", path="/x", status=200, duration_ms=10, tokens_out=5)


def test_rate_not_understated_when_window_exceeds_retention():
    agg = MetricsAggregator(max_window_seconds=600)
    now = time.time()
    _fill(agg, 600, 600, now)
    with m.patch("time.time", return_value=now):
        s600 = agg.get_summary(window_seconds=600)
        s3600 = agg.get_summary(window_seconds=3600)
    # the requested 3600s window is clamped to the 600s retention → same rate, not ~6x lower
    assert abs(s600["requests_per_second"] - s3600["requests_per_second"]) < 1e-6
    assert s3600["window_seconds"] == 600  # clamped to what's actually retained
    assert s3600["requests_per_second"] > 0.9  # ~1.0, not ~0.17


def test_percentiles_window_clamped():
    agg = MetricsAggregator(max_window_seconds=600)
    now = time.time()
    _fill(agg, 100, 600, now)
    with m.patch("time.time", return_value=now):
        p = agg.get_percentiles("duration_ms", window_seconds=3600)
    # still returns the retained samples (not empty), keyed correctly
    assert p["p50"] == 10.0


def test_small_window_unaffected():
    agg = MetricsAggregator(max_window_seconds=600)
    now = time.time()
    _fill(agg, 60, 60, now)
    with m.patch("time.time", return_value=now):
        s = agg.get_summary(window_seconds=60)
    assert s["window_seconds"] == 60  # a sub-retention window is untouched
    assert s["requests_per_second"] > 0.9
