"""Tests for benchmark router and server metrics."""

import tempfile
import pytest
from pathlib import Path


class TestBenchRouter:
    """Test benchmark API endpoints."""

    def test_roofline_schema(self):
        from yunshu_gateway.routers.bench import RooflineRequest
        req = RooflineRequest()
        assert req.sizes == [64, 128, 256, 512, 1024, 2048, 4096, 8192]
        assert req.num_warmup == 5
        assert req.num_iters == 20

    def test_roofline_custom_sizes(self):
        from yunshu_gateway.routers.bench import RooflineRequest
        req = RooflineRequest(sizes=[64, 128], num_iters=5)
        assert req.sizes == [64, 128]
        assert req.num_iters == 5

    def test_latency_schema(self):
        from yunshu_gateway.routers.bench import LatencyRequest
        req = LatencyRequest()
        assert req.base_url == "http://localhost:8000"
        assert req.prompt_lengths == [32, 128, 512]
        assert req.num_requests == 3

    def test_throughput_schema(self):
        from yunshu_gateway.routers.bench import ThroughputRequest
        req = ThroughputRequest()
        assert req.concurrency_levels == [1, 2, 4]
        assert req.num_requests == 5

    def test_status_endpoint(self):
        from yunshu_gateway.routers.bench import _active_benchmark, _benchmark_results
        # These are module-level state, just verify they're accessible
        assert _active_benchmark is None or isinstance(_active_benchmark, str)
        assert isinstance(_benchmark_results, dict)


class TestServerMetrics:
    """Test oMLX-pattern ServerMetrics."""

    def test_create(self):
        from yunshu_engine.server_metrics import ServerMetrics
        m = ServerMetrics()
        snap = m.get_snapshot()
        assert snap["total_requests"] == 0
        assert snap["total_prompt_tokens"] == 0

    def test_record_request(self):
        from yunshu_engine.server_metrics import ServerMetrics
        m = ServerMetrics()
        m.record_request_complete(
            prompt_tokens=100,
            completion_tokens=50,
            cached_tokens=20,
            prefill_duration=0.5,
            generation_duration=1.0,
            model_id="test-model",
        )
        snap = m.get_snapshot()
        assert snap["total_requests"] == 1
        assert snap["total_prompt_tokens"] == 100
        assert snap["total_completion_tokens"] == 50
        assert snap["total_cached_tokens"] == 20
        assert snap["cache_efficiency_pct"] == 20.0
        assert snap["avg_generation_tps"] == 50.0

    def test_per_model(self):
        from yunshu_engine.server_metrics import ServerMetrics
        m = ServerMetrics()
        m.record_request_complete(100, 50, model_id="model-a")
        m.record_request_complete(200, 100, model_id="model-b")
        m.record_request_complete(50, 25, model_id="model-a")

        snap_a = m.get_snapshot(model_id="model-a")
        assert snap_a["total_prompt_tokens"] == 150
        assert snap_a["total_requests"] == 2

        snap_b = m.get_snapshot(model_id="model-b")
        assert snap_b["total_prompt_tokens"] == 200
        assert snap_b["total_requests"] == 1

        # Global
        snap = m.get_snapshot()
        assert snap["total_requests"] == 3
        assert snap["total_prompt_tokens"] == 350

    def test_alltime_scope(self):
        from yunshu_engine.server_metrics import ServerMetrics
        m = ServerMetrics()
        m.record_request_complete(100, 50, model_id="test")
        snap = m.get_snapshot(scope="alltime")
        assert snap["total_requests"] == 1
        assert snap["total_prompt_tokens"] == 100

    def test_persistence(self):
        from yunshu_engine.server_metrics import ServerMetrics
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.json"
            m1 = ServerMetrics(stats_path=path)
            m1.record_request_complete(100, 50, model_id="test")
            m1.save_alltime()

            # Reload
            m2 = ServerMetrics(stats_path=path)
            snap = m2.get_snapshot(scope="alltime")
            assert snap["total_requests"] == 1
            assert snap["total_prompt_tokens"] == 100

    def test_clear_session(self):
        from yunshu_engine.server_metrics import ServerMetrics
        m = ServerMetrics()
        m.record_request_complete(100, 50)
        m.clear_session()
        snap = m.get_snapshot()
        assert snap["total_requests"] == 0

    def test_singleton(self):
        from yunshu_engine.server_metrics import (
            get_server_metrics,
            reset_server_metrics,
        )
        reset_server_metrics()
        m1 = get_server_metrics()
        m2 = get_server_metrics()
        assert m1 is m2

    def test_cache_efficiency(self):
        from yunshu_engine.server_metrics import ServerMetrics
        m = ServerMetrics()
        m.record_request_complete(prompt_tokens=1000, completion_tokens=100, cached_tokens=800)
        snap = m.get_snapshot()
        assert snap["cache_efficiency_pct"] == 80.0

    def test_zero_division_safety(self):
        from yunshu_engine.server_metrics import ServerMetrics
        m = ServerMetrics()
        snap = m.get_snapshot()
        assert snap["avg_prefill_tps"] == 0.0
        assert snap["avg_generation_tps"] == 0.0
        assert snap["cache_efficiency_pct"] == 0.0
