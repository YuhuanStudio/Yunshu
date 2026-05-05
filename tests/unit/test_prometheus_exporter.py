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
        assert '# TYPE yunshu_request_duration_seconds histogram' in text
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


# ---------------------------------------------------------------------------
# Gauge tests (via set/inc/dec)
# ---------------------------------------------------------------------------

class TestGaugeOperations:
    def test_set_and_get(self):
        pm = PrometheusMetrics()
        pm.set_gauge("active_requests", 5.0)
        assert pm._gauges["active_requests"].get() == 5.0

    def test_inc_and_dec(self):
        pm = PrometheusMetrics()
        pm.inc_gauge("active_requests")
        pm.inc_gauge("active_requests")
        pm.dec_gauge("active_requests")
        assert pm._gauges["active_requests"].get() == 1.0

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
        pm.set_gauge("active_requests", 3)
        text = pm.generate()

        # Each metric family should have HELP and TYPE lines.
        assert "# HELP yunshu_request_total" in text
        assert "# TYPE yunshu_request_total counter" in text
        assert "# HELP yunshu_request_duration_seconds" in text
        assert "# TYPE yunshu_request_duration_seconds histogram" in text
        assert "# HELP yunshu_active_requests" in text
        assert "# TYPE yunshu_active_requests gauge" in text

    def test_generate_includes_uptime(self):
        pm = PrometheusMetrics()
        text = pm.generate()
        assert "yunshu_exporter_uptime_seconds" in text

    def test_generate_empty_registry(self):
        pm = PrometheusMetrics()
        text = pm.generate()
        # Should still produce valid text with counters/gauges/histograms.
        assert "# TYPE yunshu_request_total counter" in text
        assert "# TYPE yunshu_active_requests gauge" in text
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
