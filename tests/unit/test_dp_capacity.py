"""Tests for DP layer allocation — C17 memory-proportional + bandwidth-aware routing."""
import pytest

from yunshu_mesh.data_parallel import DataParallelRouter, NodeLoad


class TestNodeLoadCapacity:
    def test_default_weight(self):
        load = NodeLoad(node_id="n1", rank=0)
        assert load.capacity_weight == 1.0

    def test_custom_fields(self):
        load = NodeLoad(node_id="n1", rank=0, memory_bytes=16 * 1024**3, gpu_cores=10)
        assert load.memory_bytes == 16 * 1024**3
        assert load.gpu_cores == 10


class TestDataParallelRouterCapacityAware:
    def test_capacity_aware_strategy(self):
        router = DataParallelRouter(strategy="capacity_aware")
        router.add_node("small", 0)
        router.add_node("big", 1)
        router.set_node_capacity("small", 8 * 1024**3, gpu_cores=8)
        router.set_node_capacity("big", 32 * 1024**3, gpu_cores=24)
        # Big node should have higher capacity weight
        small = router._nodes["small"]
        big = router._nodes["big"]
        assert big.capacity_weight > small.capacity_weight

    def test_capacity_aware_distribution(self):
        router = DataParallelRouter(strategy="capacity_aware")
        router.add_node("small", 0)
        router.add_node("big", 1)
        router.set_node_capacity("small", 8 * 1024**3)
        router.set_node_capacity("big", 32 * 1024**3)
        # Route 10 requests — big should get more
        big_count = 0
        small_count = 0
        for _ in range(10):
            node = router.select_node()
            router.record_request_start(node)
            if node == "big":
                big_count += 1
            else:
                small_count += 1
        # Big node (4x memory) should get more requests
        assert big_count > small_count

    def test_set_node_capacity_unknown_node(self):
        router = DataParallelRouter()
        router.add_node("n1", 0)
        # Should not raise for unknown node
        router.set_node_capacity("unknown", 16 * 1024**3, gpu_cores=10)

    def test_recompute_weights(self):
        router = DataParallelRouter()
        router.add_node("n1", 0)
        router.add_node("n2", 1)
        router.add_node("n3", 2)
        router.set_node_capacity("n1", 8 * 1024**3)
        router.set_node_capacity("n2", 16 * 1024**3)
        router.set_node_capacity("n3", 32 * 1024**3)
        # n1 = 1.0 (baseline), n2 = 2.0, n3 = 4.0
        assert router._nodes["n1"].capacity_weight == 1.0
        assert router._nodes["n2"].capacity_weight == 2.0
        assert router._nodes["n3"].capacity_weight == 4.0

    def test_stats_includes_capacity(self):
        router = DataParallelRouter()
        router.add_node("n1", 0)
        router.set_node_capacity("n1", 16 * 1024**3, gpu_cores=10)
        stats = router.get_stats()
        node_stats = stats["nodes"]["n1"]
        assert "capacity_weight" in node_stats
        assert "memory_gb" in node_stats
        assert node_stats["memory_gb"] == 16.0
        assert node_stats["gpu_cores"] == 10

    def test_no_memory_defaults_to_weight_1(self):
        router = DataParallelRouter()
        router.add_node("n1", 0)
        assert router._nodes["n1"].capacity_weight == 1.0
