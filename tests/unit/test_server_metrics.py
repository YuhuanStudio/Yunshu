"""Unit tests for ServerMetrics."""
import json
import tempfile
import time
from pathlib import Path

import pytest

from yunshu_engine.server_metrics import ServerMetrics, reset_server_metrics


@pytest.fixture(autouse=True)
def _reset():
    reset_server_metrics()
    yield
    reset_server_metrics()


class TestServerMetrics:
    def test_construction(self):
        m = ServerMetrics()
        assert m.total_requests == 0
        assert m.total_prompt_tokens == 0
        assert m.total_completion_tokens == 0

    def test_record_request_complete(self):
        m = ServerMetrics()
        m.record_request_complete(
            prompt_tokens=100,
            completion_tokens=50,
            cached_tokens=20,
            model_id="test-model",
        )
        assert m.total_requests == 1
        assert m.total_prompt_tokens == 100
        assert m.total_completion_tokens == 50
        assert m.total_cached_tokens == 20

    def test_per_model_tracking(self):
        m = ServerMetrics()
        m.record_request_complete(100, 50, model_id="model-a")
        m.record_request_complete(200, 100, model_id="model-b")
        m.record_request_complete(50, 25, model_id="model-a")

        snap_a = m.get_snapshot(model_id="model-a")
        assert snap_a["total_prompt_tokens"] == 150
        assert snap_a["total_completion_tokens"] == 75
        assert snap_a["total_requests"] == 2

        snap_b = m.get_snapshot(model_id="model-b")
        assert snap_b["total_prompt_tokens"] == 200
        assert snap_b["total_requests"] == 1

    def test_session_snapshot_metrics(self):
        m = ServerMetrics()
        m.record_request_complete(100, 50, prefill_duration=1.0, generation_duration=2.0)
        snap = m.get_snapshot()
        assert snap["total_prompt_tokens"] == 100
        assert snap["total_completion_tokens"] == 50
        assert snap["avg_prefill_tps"] == 100.0
        assert snap["avg_generation_tps"] == 25.0

    def test_alltime_snapshot(self):
        m = ServerMetrics()
        m.record_request_complete(100, 50)
        m.record_request_complete(200, 100)
        snap = m.get_snapshot(scope="alltime")
        assert snap["total_prompt_tokens"] == 300
        assert snap["total_completion_tokens"] == 150
        assert snap["total_requests"] == 2

    def test_clear_session(self):
        m = ServerMetrics()
        m.record_request_complete(100, 50, model_id="test")
        m.clear_session()
        assert m.total_requests == 0
        assert m.total_prompt_tokens == 0
        # Alltime should NOT be cleared
        snap = m.get_snapshot(scope="alltime")
        assert snap["total_requests"] == 1

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "stats.json"
            m = ServerMetrics(stats_path=path)
            m.record_request_complete(100, 50, model_id="test")
            m.save_alltime()

            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert data["total_prompt_tokens"] == 100
            assert data["total_requests"] == 1

            # Load in new instance
            m2 = ServerMetrics(stats_path=path)
            snap = m2.get_snapshot(scope="alltime")
            assert snap["total_prompt_tokens"] == 100
            assert snap["total_requests"] == 1

    def test_cache_efficiency(self):
        m = ServerMetrics()
        m.record_request_complete(100, 50, cached_tokens=80, prefill_duration=1.0)
        snap = m.get_snapshot()
        assert snap["cache_efficiency_pct"] == 80.0

    def test_zero_division_safe(self):
        m = ServerMetrics()
        snap = m.get_snapshot()
        assert snap["avg_prefill_tps"] == 0.0
        assert snap["avg_generation_tps"] == 0.0
        assert snap["cache_efficiency_pct"] == 0.0

    def test_unknown_model_returns_zeros(self):
        m = ServerMetrics()
        snap = m.get_snapshot(model_id="nonexistent")
        assert snap["total_prompt_tokens"] == 0
        assert snap["total_requests"] == 0

    def test_multiple_requests_accumulate(self):
        m = ServerMetrics()
        for i in range(10):
            m.record_request_complete(10, 5)
        assert m.total_requests == 10
        assert m.total_prompt_tokens == 100
        assert m.total_completion_tokens == 50

    def test_uptime_increases(self):
        m = ServerMetrics()
        time.sleep(0.05)
        snap = m.get_snapshot()
        assert snap["uptime_seconds"] >= 0.05

    def test_compute_utilization_empty(self):
        """No steps recorded → 0% utilization."""
        m = ServerMetrics()
        assert m.get_compute_utilization() == 0.0
        snap = m.get_snapshot()
        assert snap["compute_utilization_pct"] == 0.0

    def test_compute_utilization_100pct(self):
        """All compute steps, no idle → 100% utilization."""
        m = ServerMetrics()
        m.record_compute_step(10.0)
        m.record_compute_step(20.0)
        assert m.get_compute_utilization() == 100.0

    def test_compute_utilization_50pct(self):
        """Half compute, half idle → 50% utilization."""
        m = ServerMetrics()
        m.record_compute_step(10.0)
        m.record_compute_step(10.0, idle=True)
        assert m.get_compute_utilization() == 50.0

    def test_compute_utilization_clamped_at_100(self):
        """Utilization cannot exceed 100%."""
        m = ServerMetrics()
        m.record_compute_step(100.0)
        # Manually set wall time lower to test clamp
        m._total_wall_time_ms = 50.0
        util = m.get_compute_utilization()
        assert util == 100.0

    def test_compute_utilization_in_snapshot(self):
        m = ServerMetrics()
        m.record_request_complete(100, 50)
        m.record_compute_step(8.0)
        m.record_compute_step(2.0, idle=True)
        snap = m.get_snapshot()
        assert snap["compute_utilization_pct"] == 80.0
