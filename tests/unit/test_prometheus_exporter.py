"""Unit tests for PrometheusMetrics exporter."""

import pytest

from yunshu_gateway.middleware.prometheus_exporter import (
    PrometheusMetrics,
    get_prometheus_metrics,
    reset_prometheus_metrics,
)


@pytest.fixture(autouse=True)
def _fresh_instance():
    """Ensure each test gets a fresh singleton."""
    reset_prometheus_metrics()
    yield
    reset_prometheus_metrics()


# ---------------------------------------------------------------------------
# Counter tests
# ---------------------------------------------------------------------------


class TestCounterIncrement:
    def test_increment_default(self):
        pm = PrometheusMetrics()
        pm.inc_counter("request_total")
        assert pm.get_counter("request_total") == 1

    def test_increment_with_amount(self):
        pm = PrometheusMetrics()
        pm.inc_counter("request_total", amount=5)
        assert pm.get_counter("request_total") == 5

    def test_increment_with_labels(self):
        pm = PrometheusMetrics()
        pm.inc_counter("request_total", {"method": "POST", "status": "200"})
        pm.inc_counter("request_total", {"method": "POST", "status": "200"})
        pm.inc_counter("request_total", {"method": "GET", "status": "200"})
        assert pm.get_counter("request_total", {"method": "POST", "status": "200"}) == 2
        assert pm.get_counter("request_total", {"method": "GET", "status": "200"}) == 1

    def test_increment_different_labels_independent(self):
        pm = PrometheusMetrics()
        pm.inc_counter("request_total", {"status": "200"})
        pm.inc_counter("request_total", {"status": "500"})
        pm.inc_counter("request_total", {"status": "500"})
        assert pm.get_counter("request_total", {"status": "200"}) == 1
        assert pm.get_counter("request_total", {"status": "500"}) == 2

    def test_increment_unknown_counter_raises(self):
        pm = PrometheusMetrics()
        with pytest.raises(KeyError, match="nonexistent"):
            pm.inc_counter("nonexistent")

    def test_negative_increment_raises(self):
        pm = PrometheusMetrics()
        with pytest.raises(ValueError):
            pm.inc_counter("request_total", amount=-1)

    def test_tokens_generated_counter(self):
        pm = PrometheusMetrics()
        pm.inc_counter("tokens_generated_total", {"model": "test"}, amount=100)
        assert pm.get_counter("tokens_generated_total", {"model": "test"}) == 100


# ---------------------------------------------------------------------------
# Histogram tests
# ---------------------------------------------------------------------------


class TestHistogramObservation:
    def test_observe_and_generate(self):
        pm = PrometheusMetrics()
        pm.observe_histogram("request_duration_seconds", 0.1)
        pm.observe_histogram("request_duration_seconds", 0.5)
        text = pm.generate()
        assert "yunshu_request_duration_seconds" in text
        assert "# TYPE yunshu_request_duration_seconds histogram" in text
        assert "_sum" in text
        assert "_count" in text
        assert '_bucket{le="+Inf"}' in text

    def test_observe_with_labels(self):
        pm = PrometheusMetrics()
        pm.observe_histogram("request_duration_seconds", 0.1, {"endpoint": "/v1/chat"})
        pm.observe_histogram("inference_duration_seconds", 0.3, {"model": "qwen"})
        text = pm.generate()
        assert 'endpoint="/v1/chat"' in text
        assert 'model="qwen"' in text

    def test_observe_unknown_histogram_raises(self):
        pm = PrometheusMetrics()
        with pytest.raises(KeyError, match="nonexistent"):
            pm.observe_histogram("nonexistent", 1.0)

    def test_histogram_buckets_include_inf(self):
        pm = PrometheusMetrics()
        pm.observe_histogram("request_duration_seconds", 100.0)  # large value
        text = pm.generate()
        # The +Inf bucket should equal the count.
        assert '_bucket{le="+Inf"}' in text

    def test_histogram_empty_labels_format(self):
        pm = PrometheusMetrics()
        pm.observe_histogram("inference_duration_seconds", 0.05)
        text = pm.generate()
        assert "yunshu_inference_duration_seconds_sum 0.05" in text
        assert "yunshu_inference_duration_seconds_count 1" in text

    def test_over_range_value_only_counts_in_inf_bucket(self):
        """A value exceeding ALL finite buckets must NOT inflate any le= bucket.

        Regression: the over-range path previously incremented every finite
        bucket, dragging histogram_quantile() p50/p99 downward. Prometheus
        semantics: le="<upper>" counts observations <= upper. inference buckets
        max out at 10.0s; a 50s observation belongs ONLY to +Inf (== _count).
        """
        from yunshu_gateway.middleware.prometheus_exporter import _Histogram

        h = _Histogram("t", "t", buckets=(0.1, 0.5, 1.0))
        for v in (0.05, 0.1, 0.3, 0.5, 0.7, 2.0):
            h.observe(v)
        key = frozenset()
        # 2.0 must NOT appear in any finite bucket.
        assert h._bucket_counts[key] == [2, 4, 5]
        assert h._counts[key] == 6  # +Inf bucket == total count
        # Cumulative monotonic and every finite bucket <= count.
        assert h._bucket_counts[key][-1] <= h._counts[key]

        # All observations over-range → every finite bucket is 0, +Inf == count.
        h2 = _Histogram("t", "t", buckets=(1.0, 2.0))
        for v in (5.0, 6.0):
            h2.observe(v)
        assert h2._bucket_counts[frozenset()] == [0, 0]
        assert h2._counts[frozenset()] == 2

    def test_over_range_inference_histogram_pulls_percentile_correct(self):
        """End-to-end via the public API: a >10s TTFT must not pollute le=0.005."""
        pm = PrometheusMetrics()
        # 9 fast (1ms) + 1 very slow (50s, exceeds the 10s INFERENCE_BUCKETS max).
        for _ in range(9):
            pm.observe_histogram("ttft_seconds", 0.001)
        pm.observe_histogram("ttft_seconds", 50.0)
        text = pm.generate()
        # The slow one stays in +Inf only; the smallest finite buckets count the
        # 9 fast ones, NOT 10.
        assert 'yunshu_ttft_seconds_bucket{le="0.005"} 9' in text
        assert 'yunshu_ttft_seconds_bucket{le="+Inf"} 10' in text
        assert "yunshu_ttft_seconds_count 10" in text


# ---------------------------------------------------------------------------
# Gauge tests (via set/inc/dec)
# ---------------------------------------------------------------------------


class TestGaugeOperations:
    def test_set_and_get(self):
        pm = PrometheusMetrics()
        pm.set_gauge("gateway_active_requests", 5.0)
        assert pm._gauges["gateway_active_requests"].get() == 5.0

    def test_inc_and_dec(self):
        pm = PrometheusMetrics()
        pm.inc_gauge("gateway_active_requests")
        pm.inc_gauge("gateway_active_requests")
        pm.dec_gauge("gateway_active_requests")
        assert pm._gauges["gateway_active_requests"].get() == 1.0

    def test_kv_cache_gauges(self):
        pm = PrometheusMetrics()
        pm.set_gauge("kv_cache_blocks_used", 42.0, {"model": "qwen"})
        pm.set_gauge("kv_cache_blocks_total", 128.0, {"model": "qwen"})
        text = pm.generate()
        assert "yunshu_kv_cache_blocks_used" in text
        assert "yunshu_kv_cache_blocks_total" in text

    def test_unknown_gauge_raises(self):
        pm = PrometheusMetrics()
        with pytest.raises(KeyError, match="nonexistent"):
            pm.set_gauge("nonexistent", 1.0)


# ---------------------------------------------------------------------------
# Format generation tests
# ---------------------------------------------------------------------------


class TestFormatGeneration:
    def test_generate_produces_valid_text(self):
        pm = PrometheusMetrics()
        pm.inc_counter("request_total", {"method": "GET"})
        pm.observe_histogram("request_duration_seconds", 0.1, {"endpoint": "/"})
        pm.set_gauge("gateway_active_requests", 3)
        text = pm.generate()

        # Each metric family should have HELP and TYPE lines.
        assert "# HELP yunshu_request_total" in text
        assert "# TYPE yunshu_request_total counter" in text
        assert "# HELP yunshu_request_duration_seconds" in text
        assert "# TYPE yunshu_request_duration_seconds histogram" in text
        assert "# HELP yunshu_gateway_active_requests" in text
        assert "# TYPE yunshu_gateway_active_requests gauge" in text

    def test_generate_includes_uptime(self):
        pm = PrometheusMetrics()
        text = pm.generate()
        assert "yunshu_exporter_uptime_seconds" in text

    def test_generate_empty_registry(self):
        pm = PrometheusMetrics()
        text = pm.generate()
        # Should still produce valid text with counters/gauges/histograms.
        assert "# TYPE yunshu_request_total counter" in text
        assert "# TYPE yunshu_gateway_active_requests gauge" in text
        assert "# TYPE yunshu_request_duration_seconds histogram" in text


# ---------------------------------------------------------------------------
# Singleton tests
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_prometheus_metrics_returns_same(self):
        a = get_prometheus_metrics()
        b = get_prometheus_metrics()
        assert a is b

    def test_reset_creates_new_instance(self):
        a = get_prometheus_metrics()
        reset_prometheus_metrics()
        b = get_prometheus_metrics()
        assert a is not b


# ---------------------------------------------------------------------------
# Metric classification tests (counter vs gauge)
# ---------------------------------------------------------------------------


class TestMetricClassification:
    """Verify cumulative totals are Counters (not Gauges) for PromQL rate()."""

    def test_chunked_prefill_chunks_processed_is_counter(self):
        """chunked_prefill_total_chunks_processed is a cumulative counter, not a gauge."""
        pm = PrometheusMetrics()
        assert "chunked_prefill_total_chunks_processed" in pm._counters
        assert "chunked_prefill_total_chunks_processed" not in pm._gauges

    def test_chunked_prefill_active_chunks_is_gauge(self):
        """chunked_prefill_active_chunks is a gauge (current value, not cumulative)."""
        pm = PrometheusMetrics()
        assert "chunked_prefill_active_chunks" in pm._gauges

    def test_chunked_prefill_counter_exposed_as_counter_type(self):
        """The Prometheus exposition format should show TYPE counter."""
        pm = PrometheusMetrics()
        pm.set_counter("chunked_prefill_total_chunks_processed", 42)
        text = pm.generate()
        assert "yunshu_chunked_prefill_total_chunks_processed_total" in text
        assert (
            "# TYPE yunshu_chunked_prefill_total_chunks_processed_total counter" in text
        )


# ---------------------------------------------------------------------------
# Counter snapshot-based set() tests
# ---------------------------------------------------------------------------


class TestCounterSet:
    """Test set_counter() for snapshot-based reporting with monotonicity."""

    def test_set_counter_increases(self):
        pm = PrometheusMetrics()
        pm.set_counter("request_total", 10)
        pm.set_counter("request_total", 20)
        assert pm.get_counter("request_total") == 20

    def test_set_counter_handles_source_reset(self):
        """When source resets to lower value, exposed counter should not decrease."""
        pm = PrometheusMetrics()
        pm.set_counter("request_total", 100)
        # Source resets to 0 (engine restart)
        pm.set_counter("request_total", 0)
        # Exposed value should NOT go back to 0 (would break PromQL rate())
        assert pm.get_counter("request_total") == 100

    def test_set_counter_after_reset_then_growth(self):
        """After reset, growth from new base should still be monotonic."""
        pm = PrometheusMetrics()
        pm.set_counter("request_total", 100)
        pm.set_counter("request_total", 0)  # reset
        pm.set_counter("request_total", 50)  # new growth
        assert pm.get_counter("request_total") == 150  # 100 + 50
