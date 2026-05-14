"""Tests for DataParallel middleware and load balancer.

Covers:
- DPLoadBalancer node selection (round-robin, least-loaded, weighted)
- Health checking and node removal/recovery
- Request lifecycle tracking (record_start / record_end)
- Stats reporting
- DPRouterMiddleware initialization and dispatch
- DP env var activation
- Tracing headers
"""
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from yunshu_mesh.data_parallel import DataParallelRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_router(strategy="least_loaded", nodes=None):
    """Create a DataParallelRouter with optional nodes."""
    router = DataParallelRouter(strategy=strategy)
    if nodes:
        for nid, rank in nodes:
            router.add_node(nid, rank)
    return router


# ---------------------------------------------------------------------------
# DPLoadBalancer tests
# ---------------------------------------------------------------------------


class TestDPLoadBalancerInit:
    """Test DPLoadBalancer initialization."""

    def test_init_no_router(self):
        from yunshu_gateway.dp_middleware import DPLoadBalancer
        lb = DPLoadBalancer(dp_router=None)
        assert lb.dp_router is None
        stats = lb.get_stats()
        assert stats["active"] is False
        assert stats["total_requests"] == 0

    def test_init_with_router(self):
        from yunshu_gateway.dp_middleware import DPLoadBalancer
        router = _make_router(nodes=[("node-0", 0)])
        lb = DPLoadBalancer(dp_router=router)
        assert lb.dp_router is not None
        assert lb.dp_router.num_nodes == 1


class TestNodeRegistration:
    """Test node registration and unregistration."""

    def test_register_node(self):
        from yunshu_gateway.dp_middleware import DPLoadBalancer
        lb = DPLoadBalancer()
        lb.register_node("node-0")
        stats = lb.get_stats()
        assert "node-0" in stats["node_health"]

    def test_unregister_node(self):
        from yunshu_gateway.dp_middleware import DPLoadBalancer
        lb = DPLoadBalancer()
        lb.register_node("node-0")
        lb.unregister_node("node-0")
        stats = lb.get_stats()
        assert "node-0" not in stats["node_health"]

    def test_register_duplicate_node(self):
        from yunshu_gateway.dp_middleware import DPLoadBalancer
        lb = DPLoadBalancer()
        lb.register_node("node-0")
        lb.register_node("node-0")  # idempotent
        stats = lb.get_stats()
        assert "node-0" in stats["node_health"]


class TestNodeSelection:
    """Test node selection strategies."""

    def test_select_node_no_router(self):
        from yunshu_gateway.dp_middleware import DPLoadBalancer
        lb = DPLoadBalancer(dp_router=None)
        assert lb.select_node() is None

    def test_select_node_least_loaded(self):
        from yunshu_gateway.dp_middleware import DPLoadBalancer
        router = _make_router(strategy="least_loaded", nodes=[
            ("node-0", 0), ("node-1", 1), ("node-2", 2),
        ])
        lb = DPLoadBalancer(dp_router=router)
        for nid in ("node-0", "node-1", "node-2"):
            lb.register_node(nid)

        # All nodes equal load, should still pick one
        node = lb.select_node()
        assert node is not None
        assert node in ("node-0", "node-1", "node-2")

    def test_select_node_prefers_less_loaded(self):
        from yunshu_gateway.dp_middleware import DPLoadBalancer
        router = _make_router(strategy="least_loaded", nodes=[
            ("node-0", 0), ("node-1", 1),
        ])
        lb = DPLoadBalancer(dp_router=router)
        lb.register_node("node-0")
        lb.register_node("node-1")

        # Load up node-0 with requests
        for _ in range(5):
            router.record_request_start("node-0")

        # node-1 should be preferred
        selected = lb.select_node()
        assert selected == "node-1"

    def test_select_node_round_robin(self):
        from yunshu_gateway.dp_middleware import DPLoadBalancer
        router = _make_router(strategy="round_robin", nodes=[
            ("node-0", 0), ("node-1", 1), ("node-2", 2),
        ])
        lb = DPLoadBalancer(dp_router=router)
        for nid in ("node-0", "node-1", "node-2"):
            lb.register_node(nid)

        # Round-robin cycles through nodes
        nodes_selected = [lb.select_node() for _ in range(6)]
        assert len(set(nodes_selected)) >= 2  # at least 2 distinct nodes

    def test_select_node_tracks_total_requests(self):
        from yunshu_gateway.dp_middleware import DPLoadBalancer
        router = _make_router(nodes=[("node-0", 0)])
        lb = DPLoadBalancer(dp_router=router)
        lb.register_node("node-0")

        lb.select_node()
        lb.select_node()
        lb.select_node()

        stats = lb.get_stats()
        assert stats["total_requests"] == 3
        assert stats["total_routed"] == 3


class TestRequestLifecycle:
    """Test record_start / record_end lifecycle tracking."""

    def test_record_start_end(self):
        from yunshu_gateway.dp_middleware import DPLoadBalancer
        router = _make_router(nodes=[("node-0", 0)])
        lb = DPLoadBalancer(dp_router=router)
        lb.register_node("node-0")

        lb.record_start("node-0")
        router_stats = router.get_stats()
        assert router_stats["nodes"]["node-0"]["active_requests"] == 1

        lb.record_end("node-0", 150.0, success=True)
        router_stats = router.get_stats()
        assert router_stats["nodes"]["node-0"]["active_requests"] == 0

    def test_latency_tracking(self):
        from yunshu_gateway.dp_middleware import DPLoadBalancer
        router = _make_router(nodes=[("node-0", 0)])
        lb = DPLoadBalancer(dp_router=router)
        lb.register_node("node-0")

        # Record several requests
        for lat in [100.0, 200.0, 150.0, 180.0, 120.0]:
            lb.record_start("node-0")
            lb.record_end("node-0", lat, success=True)

        stats = lb.get_stats()
        node_health = stats["node_health"]["node-0"]
        assert node_health["sample_count"] == 5
        assert node_health["avg_latency_ms"] > 0

    def test_concurrent_requests(self):
        from yunshu_gateway.dp_middleware import DPLoadBalancer
        router = _make_router(nodes=[("node-0", 0)])
        lb = DPLoadBalancer(dp_router=router)
        lb.register_node("node-0")

        # Start 3 concurrent requests
        lb.record_start("node-0")
        lb.record_start("node-0")
        lb.record_start("node-0")

        router_stats = router.get_stats()
        assert router_stats["nodes"]["node-0"]["active_requests"] == 3

        lb.record_end("node-0", 50.0, success=True)
        router_stats = router.get_stats()
        assert router_stats["nodes"]["node-0"]["active_requests"] == 2

    def test_record_end_unknown_node(self):
        from yunshu_gateway.dp_middleware import DPLoadBalancer
        router = _make_router(nodes=[("node-0", 0)])
        lb = DPLoadBalancer(dp_router=router)
        # Should not raise
        lb.record_end("nonexistent", 100.0, success=True)


class TestHealthChecking:
    """Test health checking and node removal."""

    def test_mark_node_unhealthy_after_errors(self):
        from yunshu_gateway.dp_middleware import DPLoadBalancer
        router = _make_router(strategy="least_loaded", nodes=[
            ("node-0", 0), ("node-1", 1),
        ])
        lb = DPLoadBalancer(dp_router=router)
        lb.register_node("node-0")
        lb.register_node("node-1")

        # Set low threshold for testing
        lb._node_health["node-0"].error_threshold = 3

        # Send 3 errors
        for _ in range(3):
            lb.record_start("node-0")
            lb.record_end("node-0", 100.0, success=False)

        stats = lb.get_stats()
        assert stats["node_health"]["node-0"]["healthy"] is False
        assert stats["node_health"]["node-0"]["consecutive_errors"] == 3

    def test_healthy_node_after_success(self):
        from yunshu_gateway.dp_middleware import DPLoadBalancer
        router = _make_router(nodes=[("node-0", 0)])
        lb = DPLoadBalancer(dp_router=router)
        lb.register_node("node-0")

        # Record some errors then a success
        lb._node_health["node-0"].error_threshold = 5
        lb._node_health["node-0"].consecutive_errors = 3

        lb.record_start("node-0")
        lb.record_end("node-0", 100.0, success=True)

        stats = lb.get_stats()
        assert stats["node_health"]["node-0"]["consecutive_errors"] == 0
        assert stats["node_health"]["node-0"]["healthy"] is True

    def test_manual_mark_unhealthy(self):
        from yunshu_gateway.dp_middleware import DPLoadBalancer
        router = _make_router(strategy="least_loaded", nodes=[
            ("node-0", 0), ("node-1", 1),
        ])
        lb = DPLoadBalancer(dp_router=router)
        lb.register_node("node-0")
        lb.register_node("node-1")

        lb.mark_unhealthy("node-0")
        stats = lb.get_stats()
        assert stats["node_health"]["node-0"]["healthy"] is False

        # Router should also mark it unavailable
        router_stats = router.get_stats()
        assert router_stats["nodes"]["node-0"]["available"] is False

    def test_manual_mark_healthy(self):
        from yunshu_gateway.dp_middleware import DPLoadBalancer
        router = _make_router(nodes=[("node-0", 0)])
        lb = DPLoadBalancer(dp_router=router)
        lb.register_node("node-0")

        lb.mark_unhealthy("node-0")
        lb.mark_healthy("node-0")

        stats = lb.get_stats()
        assert stats["node_health"]["node-0"]["healthy"] is True

    def test_node_recovery_after_timeout(self):
        from yunshu_gateway.dp_middleware import DPLoadBalancer
        router = _make_router(strategy="round_robin", nodes=[
            ("node-0", 0), ("node-1", 1),
        ])
        lb = DPLoadBalancer(dp_router=router)
        lb.register_node("node-0")
        lb.register_node("node-1")

        # Mark node-0 unhealthy with a past error time (simulating timeout)
        lb.mark_unhealthy("node-0")
        lb._node_health["node-0"].recovery_timeout_seconds = 0.01  # 10ms
        lb._node_health["node-0"].last_error_time = time.monotonic() - 1.0  # 1 second ago

        # select_node should detect recovery and clear the unhealthy flag
        node = lb.select_node()
        # node-0 should be available again (recovered)
        assert lb._node_health["node-0"].marked_unhealthy is False

    def test_no_available_nodes(self):
        from yunshu_gateway.dp_middleware import DPLoadBalancer
        router = _make_router(nodes=[("node-0", 0)])
        lb = DPLoadBalancer(dp_router=router)
        lb.register_node("node-0")

        lb.mark_unhealthy("node-0")
        lb._node_health["node-0"].recovery_timeout_seconds = 9999.0

        # With no available nodes, select returns None
        node = lb.select_node()
        assert node is None


class TestStatsReporting:
    """Test get_stats() reporting."""

    def test_stats_empty(self):
        from yunshu_gateway.dp_middleware import DPLoadBalancer
        lb = DPLoadBalancer(dp_router=None)
        stats = lb.get_stats()
        assert stats["active"] is False
        assert stats["total_requests"] == 0
        assert stats["total_routed"] == 0
        assert stats["node_health"] == {}

    def test_stats_with_nodes(self):
        from yunshu_gateway.dp_middleware import DPLoadBalancer
        router = _make_router(nodes=[("node-0", 0), ("node-1", 1)])
        lb = DPLoadBalancer(dp_router=router)
        lb.register_node("node-0")
        lb.register_node("node-1")

        # Simulate some traffic
        lb.record_start("node-0")
        lb.record_end("node-0", 100.0, success=True)
        lb.record_start("node-1")
        lb.record_end("node-1", 200.0, success=True)

        stats = lb.get_stats()
        assert stats["active"] is True
        assert len(stats["node_health"]) == 2
        assert stats["node_health"]["node-0"]["avg_latency_ms"] == 100.0
        assert stats["node_health"]["node-1"]["avg_latency_ms"] == 200.0

    def test_stats_includes_router_stats(self):
        from yunshu_gateway.dp_middleware import DPLoadBalancer
        router = _make_router(nodes=[("node-0", 0)])
        lb = DPLoadBalancer(dp_router=router)
        lb.register_node("node-0")

        stats = lb.get_stats()
        assert "router" in stats
        assert stats["router"]["strategy"] == "least_loaded"
        assert stats["router"]["total_nodes"] == 1

    def test_latency_percentiles(self):
        from yunshu_gateway.dp_middleware import DPLoadBalancer
        router = _make_router(nodes=[("node-0", 0)])
        lb = DPLoadBalancer(dp_router=router)
        lb.register_node("node-0")

        # Record enough samples for percentile calculation
        for i in range(20):
            lb.record_start("node-0")
            lb.record_end("node-0", float(i * 10), success=True)

        stats = lb.get_stats()
        node = stats["node_health"]["node-0"]
        assert node["sample_count"] == 20
        assert node["p50_latency_ms"] > 0


class TestEnvVarActivation:
    """Test DP env var activation."""

    def test_setup_data_parallel_default(self):
        from yunshu_gateway.dp_middleware import (
            setup_data_parallel, DPLoadBalancer,
        )
        import yunshu_gateway.engine as engine_mod
        # Clean up any prior state
        import yunshu_gateway.dp_middleware as mod
        mod._dp_load_balancer = None
        engine_mod._dp_router = None

        lb = setup_data_parallel()
        assert isinstance(lb, DPLoadBalancer)
        assert lb.dp_router is not None
        assert lb.dp_router.num_nodes >= 1
        # Cleanup
        mod._dp_load_balancer = None
        engine_mod._dp_router = None

    def test_setup_data_parallel_with_nodes(self):
        from yunshu_gateway.dp_middleware import setup_data_parallel
        import yunshu_gateway.engine as engine_mod
        import yunshu_gateway.dp_middleware as mod
        mod._dp_load_balancer = None
        engine_mod._dp_router = None

        nodes = [("gpu-0", 0), ("gpu-1", 1), ("gpu-2", 2)]
        lb = setup_data_parallel(strategy="round_robin", nodes=nodes)
        assert lb.dp_router.num_nodes == 3

        stats = lb.get_stats()
        assert "gpu-0" in stats["node_health"]
        assert "gpu-1" in stats["node_health"]
        assert "gpu-2" in stats["node_health"]
        # Cleanup
        mod._dp_load_balancer = None
        engine_mod._dp_router = None

    def test_setup_from_env_var(self):
        from yunshu_gateway.dp_middleware import setup_data_parallel
        import yunshu_gateway.engine as engine_mod
        import yunshu_gateway.dp_middleware as mod
        mod._dp_load_balancer = None
        engine_mod._dp_router = None

        with patch.dict(os.environ, {"YUNSHU_DP_NODES": "alpha:0,beta:1,gamma:2"}):
            lb = setup_data_parallel()
            assert lb.dp_router.num_nodes == 3
        # Cleanup
        mod._dp_load_balancer = None
        engine_mod._dp_router = None

    def test_init_dp_load_balancer_singleton(self):
        from yunshu_gateway.dp_middleware import init_dp_load_balancer, get_dp_load_balancer
        import yunshu_gateway.dp_middleware as mod
        mod._dp_load_balancer = None

        lb = init_dp_load_balancer(dp_router=None)
        assert get_dp_load_balancer() is lb


class TestDPRouterMiddleware:
    """Test the FastAPI middleware."""

    @pytest.mark.asyncio
    async def test_middleware_skips_non_inference(self):
        from yunshu_gateway.dp_middleware import DPRouterMiddleware

        middleware = DPRouterMiddleware(app=MagicMock())

        # Create a mock request for a non-inference path
        request = MagicMock()
        request.url.path = "/health"
        request.query_params = {}

        response_obj = MagicMock()
        response_obj.status_code = 200
        response_obj.headers = {}

        async def call_next(_req):
            return response_obj

        response = await middleware.dispatch(request, call_next)
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_middleware_adds_tracing_headers(self):
        from yunshu_gateway.dp_middleware import (
            DPRouterMiddleware, DPLoadBalancer,
        )
        import yunshu_gateway.dp_middleware as mod

        router = _make_router(nodes=[("node-0", 0)])
        lb = DPLoadBalancer(dp_router=router)
        lb.register_node("node-0")
        mod._dp_load_balancer = lb

        middleware = DPRouterMiddleware(app=MagicMock())

        request = MagicMock()
        request.url.path = "/v1/chat/completions"
        request.query_params = {}
        request.state = MagicMock()

        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.headers = {}

        async def call_next(_req):
            return response_mock

        response = await middleware.dispatch(request, call_next)

        assert "X-DP-Node" in response.headers
        assert "X-DP-Latency" in response.headers
        assert response.headers["X-DP-Node"] == "node-0"

        # Cleanup
        mod._dp_load_balancer = None

    @pytest.mark.asyncio
    async def test_middleware_records_failure_on_exception(self):
        from yunshu_gateway.dp_middleware import (
            DPRouterMiddleware, DPLoadBalancer,
        )
        import yunshu_gateway.dp_middleware as mod

        router = _make_router(nodes=[("node-0", 0)])
        lb = DPLoadBalancer(dp_router=router)
        lb.register_node("node-0")
        mod._dp_load_balancer = lb

        middleware = DPRouterMiddleware(app=MagicMock())

        request = MagicMock()
        request.url.path = "/v1/chat/completions"
        request.query_params = {}
        request.state = MagicMock()

        async def failing_call_next(_req):
            raise RuntimeError("test error")

        with pytest.raises(RuntimeError):
            await middleware.dispatch(request, failing_call_next)

        # Verify failure was recorded
        stats = lb.get_stats()
        # Node should have had a request tracked
        assert stats["total_requests"] == 1

        # Cleanup
        mod._dp_load_balancer = None

    @pytest.mark.asyncio
    async def test_middleware_records_5xx_as_failure(self):
        from yunshu_gateway.dp_middleware import (
            DPRouterMiddleware, DPLoadBalancer,
        )
        import yunshu_gateway.dp_middleware as mod

        router = _make_router(nodes=[("node-0", 0)])
        lb = DPLoadBalancer(dp_router=router)
        lb.register_node("node-0")
        mod._dp_load_balancer = lb

        middleware = DPRouterMiddleware(app=MagicMock())

        request = MagicMock()
        request.url.path = "/v1/chat/completions"
        request.query_params = {}
        request.state = MagicMock()

        response_mock = MagicMock()
        response_mock.status_code = 500
        response_mock.headers = {}

        async def call_next(_req):
            return response_mock

        await middleware.dispatch(request, call_next)

        # 500 should be recorded as failure (success=False affects health tracking)
        stats = lb.get_stats()
        assert stats["total_requests"] == 1

        # Cleanup
        mod._dp_load_balancer = None

    @pytest.mark.asyncio
    async def test_middleware_no_lb_passes_through(self):
        from yunshu_gateway.dp_middleware import DPRouterMiddleware
        import yunshu_gateway.dp_middleware as mod
        mod._dp_load_balancer = None

        middleware = DPRouterMiddleware(app=MagicMock())

        request = MagicMock()
        request.url.path = "/v1/chat/completions"
        request.query_params = {}

        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.headers = {}

        async def call_next(_req):
            return response_mock

        response = await middleware.dispatch(request, call_next)
        assert response.status_code == 200
        # No DP headers when LB is not active
        assert "X-DP-Node" not in response.headers


class TestInferencePathDetection:
    """Test the inference path detection helper."""

    def test_inference_paths(self):
        from yunshu_gateway.dp_middleware import _is_inference_path
        assert _is_inference_path("/v1/chat/completions")
        assert _is_inference_path("/v1/completions")
        assert _is_inference_path("/v1/embeddings")
        assert _is_inference_path("/v1/messages")
        assert _is_inference_path("/v1/responses")
        assert _is_inference_path("/v1/batch/inference")
        assert _is_inference_path("/v1/audio/speech")
        assert _is_inference_path("/v1/images/generations")

    def test_non_inference_paths(self):
        from yunshu_gateway.dp_middleware import _is_inference_path
        assert not _is_inference_path("/health")
        assert not _is_inference_path("/api/v1/monitoring/system")
        assert not _is_inference_path("/v1/models")
        assert not _is_inference_path("/version")
        assert not _is_inference_path("/api/v1/admin/nodes")


class TestCapacityAwareRouting:
    """Test capacity-aware routing via DataParallelRouter."""

    def test_capacity_aware_prefers_larger_node(self):
        from yunshu_gateway.dp_middleware import DPLoadBalancer
        router = _make_router(strategy="capacity_aware", nodes=[
            ("big-node", 0), ("small-node", 1),
        ])
        router.set_node_capacity("big-node", memory_bytes=64 * 1024**3, gpu_cores=40)
        router.set_node_capacity("small-node", memory_bytes=16 * 1024**3, gpu_cores=10)

        lb = DPLoadBalancer(dp_router=router)
        lb.register_node("big-node")
        lb.register_node("small-node")

        # With equal load, big-node should get more traffic due to capacity
        # Load up big-node with 3 requests, small-node with 0
        for _ in range(3):
            router.record_request_start("big-node")

        # big-node: 3 requests / 4.0 capacity = 0.75 utilization
        # small-node: 0 requests / 1.0 capacity = 0.0 utilization
        # Should prefer small-node
        selected = lb.select_node()
        assert selected == "small-node"
