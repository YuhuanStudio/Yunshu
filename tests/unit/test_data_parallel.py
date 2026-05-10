"""Tests for yunshu_mesh data parallel router and multi-tenant routing."""
from __future__ import annotations

import time

from yunshu_mesh.data_parallel import DataParallelRouter, NodeLoad


class TestNodeLoad:
    def test_record_request_start(self):
        load = NodeLoad(node_id="n0", rank=0)
        assert load.active_requests == 0
        load.record_request_start()
        assert load.active_requests == 1
        assert load.total_requests == 1

    def test_record_request_end(self):
        load = NodeLoad(node_id="n0", rank=0)
        load.record_request_start()
        load.record_request_end(100.0)
        assert load.active_requests == 0
        assert load.avg_latency_ms == 100.0

    def test_ema_latency(self):
        load = NodeLoad(node_id="n0", rank=0)
        load.record_request_start()
        load.record_request_end(100.0)
        load.record_request_start()
        load.record_request_end(200.0)
        # EMA: 0.3*200 + 0.7*100 = 130
        assert 125 < load.avg_latency_ms < 135

    def test_active_requests_never_negative(self):
        load = NodeLoad(node_id="n0", rank=0)
        load.record_request_end(50.0)
        assert load.active_requests == 0


class TestDataParallelRouterRoundRobin:
    def test_round_robin_cycling(self):
        router = DataParallelRouter(strategy="round_robin")
        router.add_node("n0", 0)
        router.add_node("n1", 1)
        router.add_node("n2", 2)

        selected = [router.select_node() for _ in range(6)]
        assert selected == ["n0", "n1", "n2", "n0", "n1", "n2"]

    def test_round_robin_single_node(self):
        router = DataParallelRouter(strategy="round_robin")
        router.add_node("n0", 0)
        assert router.select_node() == "n0"
        assert router.select_node() == "n0"

    def test_round_robin_no_nodes(self):
        router = DataParallelRouter(strategy="round_robin")
        assert router.select_node() is None


class TestDataParallelRouterLeastLoaded:
    def test_least_loaded_picks_fewest(self):
        router = DataParallelRouter(strategy="least_loaded")
        router.add_node("n0", 0)
        router.add_node("n1", 1)

        router.record_request_start("n0")
        router.record_request_start("n0")
        router.record_request_start("n1")

        assert router.select_node() == "n1"

    def test_least_loaded_tiebreak(self):
        router = DataParallelRouter(strategy="least_loaded")
        router.add_node("n0", 0)
        router.add_node("n1", 1)
        # Both have 0 requests — either could be selected
        selected = router.select_node()
        assert selected in ("n0", "n1")

    def test_least_loaded_respects_unavailable(self):
        router = DataParallelRouter(strategy="least_loaded")
        router.add_node("n0", 0)
        router.add_node("n1", 1)
        router.mark_unavailable("n0")
        assert router.select_node() == "n1"


class TestDataParallelRouterLatencyAware:
    def test_latency_aware_prefers_faster(self):
        router = DataParallelRouter(strategy="latency_aware")
        router.add_node("n0", 0)
        router.add_node("n1", 1)

        # n0 has higher latency
        router.record_request_start("n0")
        router.record_request_end("n0", 200.0)
        router.record_request_start("n1")
        router.record_request_end("n1", 50.0)

        assert router.select_node() == "n1"


class TestDataParallelRouterLifecycle:
    def test_remove_node(self):
        router = DataParallelRouter(strategy="round_robin")
        router.add_node("n0", 0)
        router.add_node("n1", 1)
        router.remove_node("n0")
        assert router.num_nodes == 1
        assert router.select_node() == "n1"

    def test_mark_available_again(self):
        router = DataParallelRouter(strategy="round_robin")
        router.add_node("n0", 0)
        router.mark_unavailable("n0")
        assert router.select_node() is None
        router.mark_available("n0")
        assert router.select_node() == "n0"

    def test_stats(self):
        router = DataParallelRouter(strategy="least_loaded")
        router.add_node("n0", 0)
        router.add_node("n1", 1)
        router.record_request_start("n0")
        stats = router.get_stats()
        assert stats["strategy"] == "least_loaded"
        assert stats["total_nodes"] == 2
        assert stats["available_nodes"] == 2
        assert stats["nodes"]["n0"]["active_requests"] == 1
        assert stats["nodes"]["n1"]["active_requests"] == 0


class TestMultiTenantRouting:
    """Test multi-tenant model/node assignment patterns."""

    def test_per_tenant_router(self):
        """Each tenant gets its own DataParallelRouter with its model's nodes."""
        # Tenant A: 2 nodes running model-A
        router_a = DataParallelRouter(strategy="least_loaded")
        router_a.add_node("node-1", 0)
        router_a.add_node("node-2", 1)

        # Tenant B: 1 node running model-B
        router_b = DataParallelRouter(strategy="least_loaded")
        router_b.add_node("node-3", 0)

        # Route requests
        assert router_a.select_node() in ("node-1", "node-2")
        assert router_b.select_node() == "node-3"

    def test_shared_node_pool(self):
        """Multiple tenants sharing the same node pool."""
        router = DataParallelRouter(strategy="least_loaded")
        router.add_node("node-1", 0)
        router.add_node("node-2", 1)

        # Both tenants compete for the same nodes
        router.record_request_start("node-1")
        assert router.select_node() == "node-2"

        router.record_request_start("node-2")
        # Now both have 1 request — either is valid
        assert router.select_node() in ("node-1", "node-2")
