"""Tests for yunshu_engine.tracing — InferenceTracer, StructuredLogger,
MetricsAggregatorV2, HealthDashboard."""

import io
import json
import threading
import time
from unittest.mock import patch

import pytest

from yunshu_engine.tracing import (
    HealthDashboard,
    InferenceTracer,
    LogLevel,
    MetricsAggregatorV2,
    MetricType,
    Span,
    SpanKind,
    StructuredLogger,
    Trace,
    get_health_dashboard,
    get_inference_tracer,
    get_metrics_v2,
    get_structured_logger,
    reset_tracing,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_tracing()
    yield
    reset_tracing()


# ===========================================================================
# InferenceTracer tests
# ===========================================================================


class TestInferenceTracer:
    """Test InferenceTracer lifecycle: start, span, end, export, stats."""

    def test_start_trace_creates_trace(self):
        tracer = InferenceTracer()
        trace = tracer.start_trace("req-1", {"model": "test-model"})
        assert trace.trace_id == "req-1"
        assert trace.metadata == {"model": "test-model"}
        assert trace.start_time > 0
        assert trace.end_time == 0

    def test_start_trace_generates_id_if_empty(self):
        tracer = InferenceTracer()
        trace = tracer.start_trace("")
        assert len(trace.trace_id) > 0

    def test_span_adds_to_active_trace(self):
        tracer = InferenceTracer()
        tracer.start_trace("req-2")
        s = tracer.span("req-2", "prefill", {"tokens": 100})
        assert s is not None
        assert s.name == "prefill"
        assert s.attributes["tokens"] == 100
        assert s.start_time > 0

    def test_span_returns_none_for_unknown_trace(self):
        tracer = InferenceTracer()
        s = tracer.span("nonexistent", "test")
        assert s is None

    def test_end_span_closes_named_span(self):
        tracer = InferenceTracer()
        tracer.start_trace("req-3")
        tracer.span("req-3", "prefill")
        time.sleep(0.01)
        tracer.end_span("req-3", "prefill")
        trace = tracer.get_trace("req-3")
        assert trace is not None
        prefill_span = trace.spans[0]
        assert prefill_span.end_time >= prefill_span.start_time
        assert prefill_span.duration_ms > 0

    def test_end_trace_completes_trace(self):
        tracer = InferenceTracer()
        tracer.start_trace("req-4")
        tracer.span("req-4", "decode")
        trace = tracer.end_trace("req-4", {"tokens": 50})
        assert trace is not None
        assert trace.end_time > 0
        assert trace.result == {"tokens": 50}
        assert trace.span_count == 1

    def test_end_trace_ends_open_spans(self):
        tracer = InferenceTracer()
        tracer.start_trace("req-5")
        tracer.span("req-5", "prefill")
        tracer.span("req-5", "decode")
        trace = tracer.end_trace("req-5")
        for s in trace.spans:
            assert s.end_time > 0

    def test_end_trace_unknown_returns_none(self):
        tracer = InferenceTracer()
        result = tracer.end_trace("nonexistent")
        assert result is None

    def test_max_traces_eviction(self):
        tracer = InferenceTracer(max_traces=3)
        for i in range(5):
            tracer.start_trace(f"req-{i}")
            tracer.end_trace(f"req-{i}")
        stats = tracer.get_stats()
        assert stats["completed_traces"] == 3

    def test_get_trace_active_and_completed(self):
        tracer = InferenceTracer()
        tracer.start_trace("active-1")
        tracer.start_trace("done-1")
        tracer.end_trace("done-1")
        assert tracer.get_trace("active-1") is not None
        assert tracer.get_trace("done-1") is not None
        assert tracer.get_trace("nonexistent") is None

    def test_export_traces_json(self):
        tracer = InferenceTracer()
        tracer.start_trace("exp-1", {"model": "m"})
        tracer.span("exp-1", "prefill", {"tokens": 10})
        tracer.end_trace("exp-1", {"status": "ok"})
        output = tracer.export_traces("json")
        data = json.loads(output)
        assert "resourceSpans" in data
        assert "traces" in data
        assert len(data["traces"]) == 1
        assert data["traces"][0]["traceId"] == "exp-1"

    def test_export_traces_invalid_format_raises(self):
        tracer = InferenceTracer()
        with pytest.raises(ValueError, match="Unsupported"):
            tracer.export_traces("xml")

    def test_get_stats_empty(self):
        tracer = InferenceTracer()
        stats = tracer.get_stats()
        assert stats["active_traces"] == 0
        assert stats["completed_traces"] == 0

    def test_get_stats_with_data(self):
        tracer = InferenceTracer()
        tracer.start_trace("s-1")
        tracer.span("s-1", "prefill")
        tracer.span("s-1", "decode")
        tracer.end_trace("s-1")
        stats = tracer.get_stats()
        assert stats["completed_traces"] == 1
        assert stats["avg_spans_per_trace"] == 2.0
        assert stats["span_types"]["prefill"] == 1
        assert stats["span_types"]["decode"] == 1

    def test_clear(self):
        tracer = InferenceTracer()
        tracer.start_trace("c-1")
        tracer.end_trace("c-1")
        tracer.clear()
        assert tracer.get_stats()["active_traces"] == 0
        assert tracer.get_stats()["completed_traces"] == 0

    def test_concurrent_traces(self):
        tracer = InferenceTracer()
        errors = []

        def worker(i):
            try:
                rid = f"concurrent-{i}"
                tracer.start_trace(rid)
                tracer.span(rid, "work")
                tracer.end_trace(rid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        stats = tracer.get_stats()
        assert stats["completed_traces"] == 20


# ===========================================================================
# StructuredLogger tests
# ===========================================================================


class TestStructuredLogger:
    """Test StructuredLogger output format, levels, context binding."""

    def test_log_writes_json(self):
        buf = io.StringIO()
        slog = StructuredLogger(name="test", output=buf)
        slog.log("test_event", key="value")
        line = buf.getvalue().strip()
        data = json.loads(line)
        assert data["event"] == "test_event"
        assert data["key"] == "value"
        assert data["level"] == "INFO"
        assert "timestamp" in data

    def test_log_levels(self):
        buf = io.StringIO()
        slog = StructuredLogger(name="test", level=LogLevel.TRACE, output=buf)
        slog.trace("trace_ev")
        slog.debug("debug_ev")
        slog.info("info_ev")
        slog.warn("warn_ev")
        slog.error("error_ev")
        lines = [json.loads(l) for l in buf.getvalue().strip().split("\n")]
        assert len(lines) == 5
        assert lines[0]["level"] == "TRACE"
        assert lines[1]["level"] == "DEBUG"
        assert lines[2]["level"] == "INFO"
        assert lines[3]["level"] == "WARN"
        assert lines[4]["level"] == "ERROR"

    def test_log_level_filtering(self):
        buf = io.StringIO()
        slog = StructuredLogger(name="test", level=LogLevel.WARN, output=buf)
        slog.debug("should_not_appear")
        slog.info("also_not")
        slog.warn("should_appear")
        lines = buf.getvalue().strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["event"] == "should_appear"

    def test_bind_context(self):
        buf = io.StringIO()
        slog = StructuredLogger(name="test", output=buf)
        slog.bind_context(request_id="r-1", service="yunshu")
        slog.info("with_context")
        data = json.loads(buf.getvalue().strip())
        assert data["request_id"] == "r-1"
        assert data["service"] == "yunshu"

    def test_unbind_context(self):
        buf = io.StringIO()
        slog = StructuredLogger(name="test", output=buf)
        slog.bind_context(a="1", b="2")
        slog.unbind_context("a")
        slog.info("test")
        data = json.loads(buf.getvalue().strip())
        assert "a" not in data
        assert data["b"] == "2"

    def test_context_does_not_override_explicit_kwargs(self):
        buf = io.StringIO()
        slog = StructuredLogger(name="test", output=buf)
        slog.bind_context(model="base")
        slog.info("test", model="override")
        data = json.loads(buf.getvalue().strip())
        assert data["model"] == "override"

    def test_get_stats(self):
        slog = StructuredLogger(name="test")
        slog.info("ev1")
        slog.info("ev2")
        slog.error("ev3")
        stats = slog.get_stats()
        assert stats["total_entries"] == 3
        assert stats["log_counts_by_level"]["INFO"] == 2
        assert stats["log_counts_by_level"]["ERROR"] == 1
        assert stats["event_types"]["ev1"] == 1

    def test_get_stats_shows_bound_context_keys(self):
        slog = StructuredLogger(name="test")
        slog.bind_context(trace_id="t1", model="m1")
        stats = slog.get_stats()
        assert "trace_id" in stats["bound_context_keys"]
        assert "model" in stats["bound_context_keys"]

    def test_clear_stats(self):
        slog = StructuredLogger(name="test")
        slog.info("ev")
        slog.bind_context(k="v")
        slog.clear_stats()
        stats = slog.get_stats()
        assert stats["total_entries"] == 0
        assert stats["bound_context_keys"] == []

    def test_concurrent_logging(self):
        buf = io.StringIO()
        slog = StructuredLogger(name="test", output=buf)
        errors = []

        def worker(i):
            try:
                slog.info("concurrent_event", worker_id=i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert slog.get_stats()["total_entries"] == 20


# ===========================================================================
# MetricsAggregatorV2 tests
# ===========================================================================


class TestMetricsAggregatorV2:
    """Test MetricsAggregatorV2 counter, gauge, histogram, Prometheus output."""

    def test_counter_increments(self):
        m = MetricsAggregatorV2()
        m.counter("requests", {"method": "POST"})
        m.counter("requests", {"method": "POST"})
        m.counter("requests", {"method": "GET"})
        output = m.get_prometheus_output()
        assert 'requests{method="POST"} 2.0' in output
        assert 'requests{method="GET"} 1.0' in output

    def test_counter_with_value(self):
        m = MetricsAggregatorV2()
        m.counter("bytes", {"dir": "out"}, value=1024)
        m.counter("bytes", {"dir": "out"}, value=512)
        output = m.get_prometheus_output()
        assert "1536.0" in output

    def test_gauge_sets_value(self):
        m = MetricsAggregatorV2()
        m.gauge("temperature", {"model": "a"}, value=0.7)
        m.gauge("temperature", {"model": "a"}, value=0.9)
        output = m.get_prometheus_output()
        assert 'temperature{model="a"} 0.9' in output

    def test_gauge_no_labels(self):
        m = MetricsAggregatorV2()
        m.gauge("uptime", value=42.0)
        output = m.get_prometheus_output()
        assert "uptime 42.0" in output

    def test_histogram_records(self):
        m = MetricsAggregatorV2()
        m.register_metric("latency", MetricType.HISTOGRAM, "Latency in seconds")
        for v in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
            m.histogram("latency", {"endpoint": "/chat"}, value=v)
        output = m.get_prometheus_output()
        assert "# TYPE latency histogram" in output
        assert "# HELP latency Latency in seconds" in output
        assert "latency_count" in output
        assert "latency_sum" in output
        assert 'latency_bucket{le="+Inf"' in output

    def test_histogram_default_buckets(self):
        m = MetricsAggregatorV2()
        m.histogram("latency", value=0.1)
        output = m.get_prometheus_output()
        # Should contain bucket boundaries from DEFAULT_LATENCY_BUCKETS
        assert 'le="0.001"' in output
        assert 'le="60.0"' in output

    def test_histogram_custom_buckets(self):
        m = MetricsAggregatorV2()
        m.register_metric("custom_h", MetricType.HISTOGRAM, buckets=(0.1, 1.0, 10.0))
        m.histogram("custom_h", value=5.0)
        output = m.get_prometheus_output()
        assert 'le="0.1"' in output
        assert 'le="10.0"' in output
        assert 'le="+Inf"' in output

    def test_get_stats(self):
        m = MetricsAggregatorV2()
        m.counter("c1")
        m.gauge("g1", value=1)
        m.histogram("h1", value=0.1)
        stats = m.get_stats()
        assert stats["total_metrics"] == 3
        assert stats["counters"] == 1
        assert stats["gauges"] == 1
        assert stats["histograms"] == 1
        assert stats["scrape_count"] == 0

    def test_scrape_count_increments(self):
        m = MetricsAggregatorV2()
        m.counter("c")
        m.get_prometheus_output()
        m.get_prometheus_output()
        assert m.get_stats()["scrape_count"] == 2

    def test_prometheus_format_has_type_annotations(self):
        m = MetricsAggregatorV2()
        m.counter("reqs")
        m.gauge("temp")
        m.histogram("lat")
        output = m.get_prometheus_output()
        assert "# TYPE reqs counter" in output
        assert "# TYPE temp gauge" in output
        assert "# TYPE lat histogram" in output

    def test_concurrent_metrics(self):
        m = MetricsAggregatorV2()
        errors = []

        def worker(i):
            try:
                m.counter("reqs", {"worker": str(i)})
                m.gauge("temp", {"worker": str(i)}, value=float(i))
                m.histogram("lat", {"worker": str(i)}, value=0.01 * i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        stats = m.get_stats()
        assert stats["total_series"] >= 60  # 20 * 3

    def test_histogram_capping(self):
        m = MetricsAggregatorV2()
        for _ in range(100_001):
            m.histogram("big_h", value=0.01)
        # Should not raise or OOM
        output = m.get_prometheus_output()
        assert "big_h_count" in output


# ===========================================================================
# HealthDashboard tests
# ===========================================================================


class TestHealthDashboard:
    """Test HealthDashboard scoring and collection."""

    def test_compute_health_score_empty_report(self):
        hd = HealthDashboard()
        score = hd.compute_health_score({})
        assert 0 <= score <= 100

    def test_compute_health_score_healthy(self):
        hd = HealthDashboard()
        report = {
            "system": {"cpu_percent": 30, "memory_percent": 50, "gpu_utilization_pct": 40},
            "models": {"total": 1, "loaded": 1},
            "requests": {"error_rate": 0.0, "avg_latency_ms": 200},
            "memory_guard": {"active": True, "max_pressure": "normal"},
            "kv_cache": {"active": True, "avg_hit_rate": 0.8},
        }
        score = hd.compute_health_score(report)
        assert score >= 70  # Healthy system

    def test_compute_health_score_degraded(self):
        hd = HealthDashboard()
        report = {
            "system": {"cpu_percent": 95, "memory_percent": 97, "gpu_utilization_pct": 95},
            "models": {"total": 3, "loaded": 1},
            "requests": {"error_rate": 0.5, "avg_latency_ms": 8000},
            "memory_guard": {"active": True, "max_pressure": "critical"},
            "kv_cache": {"active": True, "avg_hit_rate": 0.1},
        }
        score = hd.compute_health_score(report)
        assert score < 50  # Degraded system

    def test_compute_health_score_no_report(self):
        hd = HealthDashboard()
        score = hd.compute_health_score(None)
        assert score == 50  # Unknown state

    def test_score_bounded_0_to_100(self):
        hd = HealthDashboard()
        # Extreme values
        report = {
            "system": {"cpu_percent": 0, "memory_percent": 0, "gpu_utilization_pct": 0},
            "models": {"total": 0, "loaded": 0},
            "requests": {"error_rate": 0, "avg_latency_ms": 0},
            "memory_guard": {"active": False},
            "kv_cache": {"active": False},
        }
        score = hd.compute_health_score(report)
        assert 0 <= score <= 100

    def test_collect_returns_report(self):
        hd = HealthDashboard()
        with patch.object(hd, "_collect_system", return_value={"cpu_percent": 50}):
            with patch.object(hd, "_collect_models", return_value={"total": 0, "loaded": 0}):
                with patch.object(hd, "_collect_requests", return_value={"active": 0}):
                    report = hd.collect()
        assert "timestamp" in report
        assert "health_score" in report
        assert 0 <= report["health_score"] <= 100

    def test_get_report_caches(self):
        hd = HealthDashboard()
        with patch.object(hd, "_collect_system", return_value={}):
            with patch.object(hd, "_collect_models", return_value={}):
                with patch.object(hd, "_collect_requests", return_value={}):
                    r1 = hd.get_report()
                    r2 = hd.get_report()
                    # Within 30s should return same report
                    assert r1["timestamp"] == r2["timestamp"]

    def test_memory_guard_pressure_levels(self):
        hd = HealthDashboard()
        scores = {}
        for level in ["normal", "warning", "critical"]:
            report = {
                "system": {"cpu_percent": 50, "memory_percent": 50, "gpu_utilization_pct": 50},
                "models": {"total": 1, "loaded": 1},
                "requests": {"error_rate": 0, "avg_latency_ms": 100},
                "memory_guard": {"active": True, "max_pressure": level},
                "kv_cache": {"active": False},
            }
            scores[level] = hd.compute_health_score(report)
        # Normal > Warning > Critical
        assert scores["normal"] > scores["warning"]
        assert scores["warning"] > scores["critical"]

    def test_model_loading_ratio(self):
        hd = HealthDashboard()
        # All loaded
        r1 = {
            "system": {"cpu_percent": 50, "memory_percent": 50, "gpu_utilization_pct": 50},
            "models": {"total": 3, "loaded": 3},
            "requests": {"error_rate": 0, "avg_latency_ms": 100},
            "memory_guard": {"active": False},
            "kv_cache": {"active": False},
        }
        # None loaded
        r2 = dict(r1)
        r2["models"] = {"total": 3, "loaded": 0}
        s1 = hd.compute_health_score(r1)
        s2 = hd.compute_health_score(r2)
        assert s1 > s2


# ===========================================================================
# Span / Trace dataclass tests
# ===========================================================================


class TestSpanTraceDataclasses:
    """Test Span and Trace property calculations."""

    def test_span_duration_ms(self):
        s = Span(span_id="s1", name="test", start_time=1.0, end_time=1.5)
        assert s.duration_ms == 500.0

    def test_span_duration_ms_zero_when_open(self):
        s = Span(span_id="s1", name="test", start_time=1.0)
        assert s.duration_ms == 0.0

    def test_span_to_dict(self):
        s = Span(span_id="s1", name="test", kind=SpanKind.SERVER,
                 start_time=1.0, end_time=2.0, attributes={"key": "val"})
        d = s.to_dict()
        assert d["spanId"] == "s1"
        assert d["name"] == "test"
        assert d["kind"] == "SERVER"
        assert d["attributes"]["key"] == "val"
        assert d["status"]["code"] == "UNSET"

    def test_trace_duration_ms(self):
        t = Trace(trace_id="t1", start_time=1.0, end_time=3.0)
        assert t.duration_ms == 2000.0

    def test_trace_span_count(self):
        t = Trace(trace_id="t1", start_time=1.0)
        t.spans.append(Span(span_id="s1", name="a"))
        t.spans.append(Span(span_id="s2", name="b"))
        assert t.span_count == 2

    def test_trace_to_dict(self):
        t = Trace(trace_id="t1", start_time=1.0, end_time=2.0,
                  metadata={"model": "m"}, result={"tokens": 10})
        t.spans.append(Span(span_id="s1", name="test"))
        d = t.to_dict()
        assert d["traceId"] == "t1"
        assert d["durationMs"] == 1000.0
        assert len(d["spans"]) == 1
        assert d["metadata"]["model"] == "m"


# ===========================================================================
# Singleton accessor tests
# ===========================================================================


class TestSingletons:
    """Test singleton getters and reset."""

    def test_get_inference_tracer(self):
        tracer = get_inference_tracer()
        assert isinstance(tracer, InferenceTracer)
        assert get_inference_tracer() is tracer  # Same instance

    def test_get_structured_logger(self):
        slog = get_structured_logger()
        assert isinstance(slog, StructuredLogger)
        assert get_structured_logger() is slog

    def test_get_metrics_v2(self):
        m = get_metrics_v2()
        assert isinstance(m, MetricsAggregatorV2)
        assert get_metrics_v2() is m

    def test_get_health_dashboard(self):
        hd = get_health_dashboard()
        assert isinstance(hd, HealthDashboard)
        assert get_health_dashboard() is hd

    def test_reset_clears_singletons(self):
        t1 = get_inference_tracer()
        reset_tracing()
        t2 = get_inference_tracer()
        assert t1 is not t2
