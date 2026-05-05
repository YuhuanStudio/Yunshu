"""Unit tests for MetricsAggregator."""

import time
from unittest.mock import patch

import pytest

from yunshu_gateway.middleware.metrics_aggregator import (
    MetricsAggregator,
    get_metrics_aggregator,
    reset_metrics_aggregator,
)


@pytest.fixture(autouse=True)
def _fresh_instance():
    """Ensure each test gets a fresh singleton."""
    reset_metrics_aggregator()
    yield
    reset_metrics_aggregator()


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

class TestRecording:
    def test_record_single_request(self):
        agg = MetricsAggregator()
        agg.record_request("POST", "/v1/chat", 200, 150.5, 64, 128)
        summary = agg.get_summary(window_seconds=60)
        assert summary["total_requests"] == 1
        assert summary["total_tokens_in"] == 64
        assert summary["total_tokens_out"] == 128

    def test_record_multiple_requests(self):
        agg = MetricsAggregator()
        for _ in range(10):
            agg.record_request("POST", "/v1/chat", 200, 100.0, 10, 20)
        summary = agg.get_summary(window_seconds=60)
        assert summary["total_requests"] == 10
        assert summary["total_tokens_in"] == 100
        assert summary["total_tokens_out"] == 200

    def test_record_error_requests(self):
        agg = MetricsAggregator()
        agg.record_request("GET", "/v1/models", 200, 10.0)
        agg.record_request("POST", "/v1/chat", 500, 200.0)
        agg.record_request("POST", "/v1/chat", 429, 1.0)
        summary = agg.get_summary(window_seconds=60)
        assert summary["total_requests"] == 3
        assert summary["error_count"] == 2
        assert summary["error_rate"] == pytest.approx(2 / 3, abs=0.01)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

class TestSummary:
    def test_empty_summary(self):
        agg = MetricsAggregator()
        summary = agg.get_summary(window_seconds=60)
        assert summary["total_requests"] == 0
        assert summary["avg_duration_ms"] == 0.0
        assert summary["error_rate"] == 0.0

    def test_avg_duration(self):
        agg = MetricsAggregator()
        agg.record_request("GET", "/", 200, 100.0)
        agg.record_request("GET", "/", 200, 200.0)
        summary = agg.get_summary(window_seconds=60)
        assert summary["avg_duration_ms"] == pytest.approx(150.0, abs=0.1)

    def test_tokens_per_second(self):
        agg = MetricsAggregator()
        agg.record_request("POST", "/v1/chat", 200, 50.0, 10, 100)
        # Short sleep to get a measurable time span.
        time.sleep(0.05)
        agg.record_request("POST", "/v1/chat", 200, 50.0, 10, 100)
        summary = agg.get_summary(window_seconds=60)
        # Should have positive tokens_per_second.
        assert summary["tokens_per_second"] > 0

    def test_window_excludes_old_data(self):
        agg = MetricsAggregator()
        # Record with a fake old timestamp.
        old_time = time.time() - 120  # 2 minutes ago
        from yunshu_gateway.middleware.metrics_aggregator import _DataPoint
        agg._points.append(_DataPoint(
            timestamp=old_time, method="GET", path="/",
            status=200, duration_ms=50.0, tokens_in=0, tokens_out=0,
        ))
        # Record a recent one.
        agg.record_request("GET", "/health", 200, 10.0)
        summary = agg.get_summary(window_seconds=60)
        # Only the recent request should be in the window.
        assert summary["total_requests"] == 1


# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------

class TestPercentiles:
    def test_empty_percentiles(self):
        agg = MetricsAggregator()
        p = agg.get_percentiles("duration_ms", window_seconds=60)
        assert p["p50"] == 0.0
        assert p["p99"] == 0.0
        assert p["count"] == 0

    def test_percentiles_single_value(self):
        agg = MetricsAggregator()
        agg.record_request("GET", "/", 200, 100.0)
        p = agg.get_percentiles("duration_ms", window_seconds=60)
        assert p["p50"] == 100.0
        assert p["p99"] == 100.0
        assert p["count"] == 1

    def test_percentiles_multiple_values(self):
        agg = MetricsAggregator()
        # Values: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
        for i in range(1, 11):
            agg.record_request("GET", "/", 200, float(i * 10))
        p = agg.get_percentiles("duration_ms", window_seconds=60)
        assert p["p50"] == pytest.approx(55.0, abs=5.0)  # ~50
        assert p["p99"] == pytest.approx(100.0, abs=2.0)  # ~100
        assert p["min"] == 10.0
        assert p["max"] == 100.0
        assert p["avg"] == pytest.approx(55.0, abs=0.1)
        assert p["count"] == 10

    def test_percentiles_tokens_out(self):
        agg = MetricsAggregator()
        agg.record_request("POST", "/v1/chat", 200, 100.0, tokens_out=50)
        agg.record_request("POST", "/v1/chat", 200, 100.0, tokens_out=150)
        agg.record_request("POST", "/v1/chat", 200, 100.0, tokens_out=100)
        p = agg.get_percentiles("tokens_out", window_seconds=60)
        assert p["count"] == 3
        assert p["p50"] == pytest.approx(100.0, abs=1.0)

    def test_invalid_metric_returns_empty(self):
        agg = MetricsAggregator()
        agg.record_request("GET", "/", 200, 100.0)
        p = agg.get_percentiles("nonexistent_metric", window_seconds=60)
        assert p["count"] == 0


# ---------------------------------------------------------------------------
# Endpoint breakdown
# ---------------------------------------------------------------------------

class TestEndpointBreakdown:
    def test_breakdown_empty(self):
        agg = MetricsAggregator()
        assert agg.get_endpoint_breakdown() == []

    def test_breakdown_groups_by_endpoint(self):
        agg = MetricsAggregator()
        agg.record_request("GET", "/health", 200, 10.0)
        agg.record_request("GET", "/health", 200, 20.0)
        agg.record_request("POST", "/v1/chat", 200, 100.0)
        breakdown = agg.get_endpoint_breakdown(window_seconds=60)
        assert len(breakdown) == 2
        # Find the /health entry.
        health = next(b for b in breakdown if b["endpoint"] == "GET /health")
        assert health["count"] == 2
        assert health["avg_duration_ms"] == pytest.approx(15.0, abs=0.1)
        chat = next(b for b in breakdown if b["endpoint"] == "POST /v1/chat")
        assert chat["count"] == 1


# ---------------------------------------------------------------------------
# Window pruning
# ---------------------------------------------------------------------------

class TestPruning:
    def test_prune_removes_old_entries(self):
        agg = MetricsAggregator(max_window_seconds=2)
        from yunshu_gateway.middleware.metrics_aggregator import _DataPoint

        # Insert an old data point directly.
        old_time = time.time() - 10
        agg._points.append(_DataPoint(
            timestamp=old_time, method="GET", path="/old",
            status=200, duration_ms=10.0, tokens_in=0, tokens_out=0,
        ))
        # Recording a new request triggers pruning.
        agg.record_request("GET", "/new", 200, 5.0)

        # The old entry should be gone.
        assert len(agg._points) == 1
        assert agg._points[0].path == "/new"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_metrics_aggregator_returns_same(self):
        a = get_metrics_aggregator()
        b = get_metrics_aggregator()
        assert a is b

    def test_reset_creates_new_instance(self):
        a = get_metrics_aggregator()
        reset_metrics_aggregator()
        b = get_metrics_aggregator()
        assert a is not b
