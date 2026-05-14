"""Tests for rtt_routing.py — RTT-aware mesh request routing."""

import time
from unittest.mock import patch

import pytest

from yunshu_mesh.rtt_routing import (
    NodeRTT,
    RTTAwareRouter,
    RoutingScore,
)


class TestNodeRTT:
    def test_initial_values(self):
        node = NodeRTT(node_id="n1")
        assert node.rtt_ema == 0.0
        assert node.rtt_var == 0.0
        assert node.probe_count == 0

    def test_update_rtt_first(self):
        node = NodeRTT(node_id="n1")
        node.update_rtt(10.0)
        assert node.rtt_ema == 10.0
        assert node.probe_count == 1

    def test_update_rtt_ema(self):
        node = NodeRTT(node_id="n1")
        node.update_rtt(10.0)
        node.update_rtt(20.0)
        assert node.rtt_ema > 10.0
        assert node.rtt_ema < 20.0
        assert node.probe_count == 2

    def test_rtt_timeout(self):
        node = NodeRTT(node_id="n1")
        node.update_rtt(10.0)
        node.update_rtt(12.0)
        timeout = node.rtt_timeout_ms
        assert timeout > node.rtt_ema  # timeout > RTT

    def test_load_fraction(self):
        node = NodeRTT(node_id="n1", max_requests=10)
        node.active_requests = 3
        assert node.load_fraction == 0.3

    def test_load_fraction_zero_max(self):
        node = NodeRTT(node_id="n1", max_requests=0)
        assert node.load_fraction == 0.0

    def test_stability(self):
        """Many samples should converge."""
        node = NodeRTT(node_id="n1")
        for _ in range(100):
            node.update_rtt(10.0)
        assert abs(node.rtt_ema - 10.0) < 0.1


class TestRTTAwareRouter:
    def test_add_remove_node(self):
        router = RTTAwareRouter()
        router.add_node("n1")
        assert len(router._nodes) == 1
        router.remove_node("n1")
        assert len(router._nodes) == 0

    def test_route_empty(self):
        router = RTTAwareRouter()
        assert router.route() is None

    def test_route_single_node(self):
        router = RTTAwareRouter()
        router.add_node("n1")
        result = router.route()
        assert result is not None
        assert result.node_id == "n1"
        assert result.selected

    def test_route_prefers_lower_rtt(self):
        router = RTTAwareRouter(rtt_weight=1.0, load_weight=0.0)
        router.add_node("near")
        router.add_node("far")
        router.record_rtt("near", 5.0)
        router.record_rtt("far", 50.0)
        result = router.route()
        assert result.node_id == "near"

    def test_route_prefers_lower_load(self):
        router = RTTAwareRouter(rtt_weight=0.0, load_weight=1.0)
        router.add_node("n1", max_requests=10)
        router.add_node("n2", max_requests=10)
        router.record_rtt("n1", 10.0)
        router.record_rtt("n2", 10.0)
        router.record_request_start("n1")
        router.record_request_start("n1")
        router.record_request_start("n1")
        result = router.route()
        assert result.node_id == "n2"  # n1 has more load

    def test_route_combined(self):
        router = RTTAwareRouter(rtt_weight=0.5, load_weight=0.5)
        router.add_node("fast_loaded", max_requests=10)
        router.add_node("slow_empty", max_requests=10)
        router.record_rtt("fast_loaded", 5.0)
        router.record_rtt("slow_empty", 50.0)
        # fast_loaded has low RTT but some load
        router.record_request_start("fast_loaded")
        result = router.route()
        assert result is not None
        assert result.combined_score > 0

    def test_route_excludes(self):
        router = RTTAwareRouter()
        router.add_node("n1")
        router.add_node("n2")
        result = router.route(exclude={"n1"})
        assert result.node_id == "n2"

    def test_route_all_at_capacity(self):
        router = RTTAwareRouter()
        router.add_node("n1", max_requests=1)
        router.record_request_start("n1")
        result = router.route()
        assert result is None

    def test_record_request_lifecycle(self):
        router = RTTAwareRouter()
        router.add_node("n1", max_requests=10)
        router.record_request_start("n1")
        assert router._nodes["n1"].active_requests == 1
        router.record_request_end("n1")
        assert router._nodes["n1"].active_requests == 0

    def test_fallback_no_rtt(self):
        router = RTTAwareRouter()
        router.add_node("n1", max_requests=10)
        router.add_node("n2", max_requests=10)
        router.record_request_start("n2")
        result = router.route()
        assert result.node_id == "n1"  # least loaded fallback
        assert router._fallback_count == 1

    def test_route_all_scores(self):
        router = RTTAwareRouter()
        router.add_node("n1")
        router.add_node("n2")
        router.record_rtt("n1", 5.0)
        router.record_rtt("n2", 20.0)
        scores = router.route_all_scores()
        assert len(scores) == 2
        assert scores[0].selected
        assert scores[0].combined_score >= scores[1].combined_score

    def test_needs_probe(self):
        router = RTTAwareRouter(probe_interval=0.001)
        router.add_node("n1")
        assert router.needs_probe("n1")  # never probed
        router.record_rtt("n1", 10.0)
        assert not router.needs_probe("n1")  # just probed
        time.sleep(0.01)
        assert router.needs_probe("n1")  # interval elapsed

    def test_needs_probe_nonexistent(self):
        router = RTTAwareRouter()
        assert not router.needs_probe("nonexistent")

    def test_from_env(self):
        with patch.dict("os.environ", {
            "YUNSHU_RTT_WEIGHT": "0.8",
            "YUNSHU_LOAD_WEIGHT": "0.2",
            "YUNSHU_PROBE_INTERVAL": "10.0",
        }):
            router = RTTAwareRouter.from_env()
            assert router._rtt_weight == 0.8
            assert router._load_weight == 0.2

    def test_get_stats(self):
        router = RTTAwareRouter()
        router.add_node("n1")
        router.record_rtt("n1", 10.0)
        router.route()
        stats = router.get_stats()
        assert stats["num_nodes"] == 1
        assert stats["route_count"] == 1
        assert "n1" in stats["nodes"]
        assert stats["nodes"]["n1"]["rtt_ema_ms"] == 10.0

    def test_multiple_routes_track_load(self):
        router = RTTAwareRouter()
        router.add_node("n1", max_requests=5)
        router.add_node("n2", max_requests=5)
        router.record_rtt("n1", 10.0)
        router.record_rtt("n2", 10.0)

        # Route 3 requests
        for _ in range(3):
            result = router.route()
            router.record_request_start(result.node_id)

        stats = router.get_stats()
        total_load = sum(
            stats["nodes"][n]["active_requests"]
            for n in stats["nodes"]
        )
        assert total_load == 3
