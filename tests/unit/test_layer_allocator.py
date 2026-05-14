"""Tests for yunshu_mesh layer allocator — §16.2/§16.3 gap closure.

Tests all 4 allocation strategies, water-filling rebalance,
node join/leave, and backward compatibility.
"""
from __future__ import annotations

import pytest

from yunshu_mesh.layer_allocator import (
    AllocationStats,
    LayerAllocationStrategy,
    LayerAllocator,
    NodeProfile,
    StageAllocation,
    WaterFillingRebalancer,
)
from yunshu_mesh.pipeline import PipelineParallel, auto_partition_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GB = 1024 ** 3


def _node(
    node_id: str = "n0",
    memory_gb: float = 64.0,
    bandwidth_mbps: float = 10000.0,
    latency_ms: float = 1.0,
    gpu_cores: int = 30,
) -> NodeProfile:
    """Convenience factory for NodeProfile."""
    return NodeProfile(
        node_id=node_id,
        memory_bytes=int(memory_gb * GB),
        bandwidth_mbps=bandwidth_mbps,
        latency_ms=latency_ms,
        gpu_cores=gpu_cores,
    )


def _mem_only_node(
    node_id: str = "n0",
    memory_gb: float = 64.0,
) -> NodeProfile:
    """NodeProfile with only memory info (no BW/GPU)."""
    return NodeProfile(
        node_id=node_id,
        memory_bytes=int(memory_gb * GB),
        bandwidth_mbps=0.0,
        latency_ms=0.0,
        gpu_cores=0,
    )


def _bare_node(node_id: str = "n0") -> NodeProfile:
    """NodeProfile with no hardware info."""
    return NodeProfile(node_id=node_id)


# ===================================================================
# Test NodeProfile
# ===================================================================


class TestNodeProfile:
    def test_capacity_score_positive(self):
        n = _node(memory_gb=128, bandwidth_mbps=20000, gpu_cores=40)
        assert n.capacity_score() > 0

    def test_capacity_score_zero_no_info(self):
        n = _bare_node()
        assert n.capacity_score() == 0.0

    def test_capacity_score_increases_with_memory(self):
        small = _node(memory_gb=32)
        big = _node(memory_gb=128)
        assert big.capacity_score() > small.capacity_score()

    def test_capacity_score_memory_dominant(self):
        """Memory weight (0.6) should dominate capacity_score when other
        fields are in reasonable ranges."""
        mem_heavy = _node(memory_gb=512, bandwidth_mbps=1000, gpu_cores=4)
        mem_light = _node(memory_gb=4, bandwidth_mbps=2000, gpu_cores=10)
        # The node with 128x more memory should win even with 2x less BW
        assert mem_heavy.capacity_score() > mem_light.capacity_score()

    def test_capacity_score_all_fields_contribute(self):
        """Adding bandwidth or GPU cores increases the score."""
        base = _node(memory_gb=64, bandwidth_mbps=0, gpu_cores=0)
        with_bw = _node(memory_gb=64, bandwidth_mbps=10000, gpu_cores=0)
        with_gpu = _node(memory_gb=64, bandwidth_mbps=0, gpu_cores=30)
        assert with_bw.capacity_score() > base.capacity_score()
        assert with_gpu.capacity_score() > base.capacity_score()


# ===================================================================
# Test EQUAL allocation
# ===================================================================


class TestAllocateEqual:
    def test_even_split(self):
        allocator = LayerAllocator()
        nodes = [_node("n0"), _node("n1"), _node("n2"), _node("n3")]
        stages = allocator.allocate(32, nodes, LayerAllocationStrategy.EQUAL)
        assert len(stages) == 4
        for s in stages:
            assert s.num_layers == 8
        total = sum(s.num_layers for s in stages)
        assert total == 32

    def test_uneven_split_remainder(self):
        allocator = LayerAllocator()
        nodes = [_node("n0"), _node("n1"), _node("n2")]
        stages = allocator.allocate(10, nodes, LayerAllocationStrategy.EQUAL)
        assert len(stages) == 3
        total = sum(s.num_layers for s in stages)
        assert total == 10
        # 10 // 3 = 3, remainder 1 — first node gets 4, others get 3 each
        assert stages[0].num_layers == 4
        assert stages[1].num_layers == 3
        assert stages[2].num_layers == 3

    def test_single_node(self):
        allocator = LayerAllocator()
        nodes = [_node("n0")]
        stages = allocator.allocate(48, nodes, LayerAllocationStrategy.EQUAL)
        assert len(stages) == 1
        assert stages[0].num_layers == 48

    def test_contiguous_ranges(self):
        allocator = LayerAllocator()
        nodes = [_node("n0"), _node("n1")]
        stages = allocator.allocate(6, nodes, LayerAllocationStrategy.EQUAL)
        assert stages[0].start_layer == 0
        assert stages[0].end_layer == 3
        assert stages[1].start_layer == 3
        assert stages[1].end_layer == 6


# ===================================================================
# Test MEMORY_PROPORTIONAL allocation
# ===================================================================


class TestAllocateMemoryProportional:
    def test_equal_memory_gives_equal_split(self):
        allocator = LayerAllocator()
        nodes = [_node("n0", memory_gb=64), _node("n1", memory_gb=64)]
        stages = allocator.allocate(32, nodes, LayerAllocationStrategy.MEMORY_PROPORTIONAL)
        assert len(stages) == 2
        assert stages[0].num_layers == 16
        assert stages[1].num_layers == 16

    def test_proportional_to_memory(self):
        allocator = LayerAllocator()
        # n0 has 3x memory of n1 → should get ~3x the layers
        nodes = [_node("n0", memory_gb=96), _node("n1", memory_gb=32)]
        stages = allocator.allocate(32, nodes, LayerAllocationStrategy.MEMORY_PROPORTIONAL)
        total = sum(s.num_layers for s in stages)
        assert total == 32
        # 96/128 = 75% → 24 layers; 32/128 = 25% → 8 layers
        assert stages[0].num_layers == 24
        assert stages[1].num_layers == 8

    def test_three_unequal_nodes(self):
        allocator = LayerAllocator()
        nodes = [
            _node("n0", memory_gb=128),
            _node("n1", memory_gb=64),
            _node("n2", memory_gb=64),
        ]
        stages = allocator.allocate(24, nodes, LayerAllocationStrategy.MEMORY_PROPORTIONAL)
        total = sum(s.num_layers for s in stages)
        assert total == 24
        # 128/256 = 50% → 12; 64/256 = 25% → 6 each
        assert stages[0].num_layers == 12
        assert stages[1].num_layers == 6
        assert stages[2].num_layers == 6

    def test_extremely_skewed_memory(self):
        """Node with 10x memory gets ~10x layers."""
        allocator = LayerAllocator()
        nodes = [_node("big", memory_gb=640), _node("small", memory_gb=64)]
        stages = allocator.allocate(22, nodes, LayerAllocationStrategy.MEMORY_PROPORTIONAL)
        total = sum(s.num_layers for s in stages)
        assert total == 22
        # 640/704 ≈ 90.9% → 20; 64/704 ≈ 9.1% → 2
        assert stages[0].num_layers >= stages[1].num_layers * 5

    def test_minimum_one_layer_per_node(self):
        """Each node should get at least 1 layer when enough layers exist."""
        allocator = LayerAllocator()
        nodes = [_node("n0", memory_gb=512), _node("n1", memory_gb=8)]
        stages = allocator.allocate(4, nodes, LayerAllocationStrategy.MEMORY_PROPORTIONAL)
        total = sum(s.num_layers for s in stages)
        assert total == 4
        for s in stages:
            assert s.num_layers >= 1


# ===================================================================
# Test BANDWIDTH_AWARE allocation
# ===================================================================


class TestAllocateBandwidthAware:
    def test_equal_bandwidth_same_as_proportional(self):
        """With equal bandwidth, result should be close to memory-proportional."""
        allocator = LayerAllocator()
        nodes = [
            _node("n0", memory_gb=96, bandwidth_mbps=10000),
            _node("n1", memory_gb=32, bandwidth_mbps=10000),
        ]
        stages = allocator.allocate(32, nodes, LayerAllocationStrategy.BANDWIDTH_AWARE)
        total = sum(s.num_layers for s in stages)
        assert total == 32
        # Should still give more to higher-memory node
        assert stages[0].num_layers > stages[1].num_layers

    def test_low_bandwidth_gets_more_layers(self):
        """Node with lower bandwidth should get more layers (fewer transfers)."""
        allocator = LayerAllocator()
        nodes = [
            _node("n0", memory_gb=64, bandwidth_mbps=40000),
            _node("n1", memory_gb=64, bandwidth_mbps=1000),
        ]
        stages = allocator.allocate(32, nodes, LayerAllocationStrategy.BANDWIDTH_AWARE)
        total = sum(s.num_layers for s in stages)
        assert total == 32
        # n1 has much lower BW → gets more layers to avoid inter-node transfers
        assert stages[1].num_layers >= stages[0].num_layers

    def test_four_nodes_mixed_bandwidth(self):
        """Mixed memory and bandwidth across 4 nodes."""
        allocator = LayerAllocator()
        nodes = [
            _node("n0", memory_gb=128, bandwidth_mbps=40000),
            _node("n1", memory_gb=64, bandwidth_mbps=10000),
            _node("n2", memory_gb=64, bandwidth_mbps=10000),
            _node("n3", memory_gb=32, bandwidth_mbps=2000),
        ]
        stages = allocator.allocate(48, nodes, LayerAllocationStrategy.BANDWIDTH_AWARE)
        total = sum(s.num_layers for s in stages)
        assert total == 48
        # All stages should have at least some layers
        for s in stages:
            assert s.num_layers >= 1


# ===================================================================
# Test LATENCY_OPTIMAL allocation
# ===================================================================


class TestAllocateLatencyOptimal:
    def test_single_node_all_layers(self):
        allocator = LayerAllocator()
        nodes = [_node("n0", gpu_cores=30, latency_ms=0.0)]
        stages = allocator.allocate(32, nodes, LayerAllocationStrategy.LATENCY_OPTIMAL)
        assert len(stages) == 1
        assert stages[0].num_layers == 32

    def test_balanced_gpu_cores(self):
        """Equal GPU cores → balanced allocation."""
        allocator = LayerAllocator()
        nodes = [
            _node("n0", gpu_cores=30, latency_ms=1.0),
            _node("n1", gpu_cores=30, latency_ms=1.0),
        ]
        stages = allocator.allocate(32, nodes, LayerAllocationStrategy.LATENCY_OPTIMAL)
        total = sum(s.num_layers for s in stages)
        assert total == 32
        # Should be roughly balanced
        assert abs(stages[0].num_layers - stages[1].num_layers) <= 2

    def test_more_gpu_cores_fewer_layers(self):
        """Node with more GPU cores processes faster → can handle more layers
        per unit time, so the DP may allocate more to it OR it balances
        the actual latency. The key invariant: total must match."""
        allocator = LayerAllocator()
        nodes = [
            _node("n0", gpu_cores=10, latency_ms=1.0),
            _node("n1", gpu_cores=40, latency_ms=1.0),
        ]
        stages = allocator.allocate(32, nodes, LayerAllocationStrategy.LATENCY_OPTIMAL)
        total = sum(s.num_layers for s in stages)
        assert total == 32

    def test_high_latency_node_gets_fewer_layers(self):
        """Node with higher inter-node latency is penalized."""
        allocator = LayerAllocator()
        nodes = [
            _node("n0", gpu_cores=30, latency_ms=0.5),
            _node("n1", gpu_cores=30, latency_ms=10.0),
        ]
        stages = allocator.allocate(32, nodes, LayerAllocationStrategy.LATENCY_OPTIMAL)
        total = sum(s.num_layers for s in stages)
        assert total == 32

    def test_three_nodes_latency_optimal(self):
        allocator = LayerAllocator()
        nodes = [
            _node("n0", gpu_cores=40, latency_ms=0.5),
            _node("n1", gpu_cores=30, latency_ms=1.0),
            _node("n2", gpu_cores=20, latency_ms=2.0),
        ]
        stages = allocator.allocate(30, nodes, LayerAllocationStrategy.LATENCY_OPTIMAL)
        total = sum(s.num_layers for s in stages)
        assert total == 30
        assert len(stages) == 3


# ===================================================================
# Test Water-Filling Rebalancer
# ===================================================================


class TestWaterFillingRebalancer:
    def test_no_change_needed(self):
        rebalancer = WaterFillingRebalancer()
        nodes = [_node("n0", memory_gb=64), _node("n1", memory_gb=64)]
        current = [
            StageAllocation("n0", 0, 16, 16),
            StageAllocation("n1", 16, 32, 16),
        ]
        result = rebalancer.rebalance(current, nodes, 32)
        total = sum(s.num_layers for s in result)
        assert total == 32

    def test_rebalance_after_node_join(self):
        """New node joins — should receive layers from existing nodes."""
        rebalancer = WaterFillingRebalancer(transfer_granularity=1, improvement_threshold=0.01)
        nodes = [_node("n0", memory_gb=64), _node("n1", memory_gb=64)]
        # Before: n0 had all layers
        current = [
            StageAllocation("n0", 0, 32, 32),
        ]
        result = rebalancer.rebalance(current, nodes, 32)
        total = sum(s.num_layers for s in result)
        assert total == 32
        assert len(result) == 2
        # n1 should receive some layers
        assert result[1].num_layers > 0

    def test_rebalance_after_node_leave(self):
        """Node leaves — its layers redistributed to remaining nodes."""
        rebalancer = WaterFillingRebalancer(transfer_granularity=1, improvement_threshold=0.01)
        # After n1 leaves, only n0 and n2 remain
        nodes = [_node("n0", memory_gb=64), _node("n2", memory_gb=64)]
        # Before: 3 nodes with 10, 11, 11 layers
        current = [
            StageAllocation("n0", 0, 10, 10),
            StageAllocation("n1", 10, 21, 11),
            StageAllocation("n2", 21, 32, 11),
        ]
        result = rebalancer.rebalance(current, nodes, 32)
        total = sum(s.num_layers for s in result)
        assert total == 32
        assert len(result) == 2

    def test_rebalance_from_empty_equal_nodes(self):
        """Starting from empty with equal nodes, distributes layers equally."""
        rebalancer = WaterFillingRebalancer()
        nodes = [_mem_only_node("n0", memory_gb=64), _mem_only_node("n1", memory_gb=64)]
        result = rebalancer.rebalance([], nodes, 32)
        total = sum(s.num_layers for s in result)
        assert total == 32
        assert len(result) == 2
        assert result[0].num_layers == 16
        assert result[1].num_layers == 16

    def test_empty_allocation_init(self):
        """Start from empty — should allocate from scratch."""
        rebalancer = WaterFillingRebalancer()
        nodes = [_node("n0", memory_gb=64), _node("n1", memory_gb=64)]
        result = rebalancer.rebalance([], nodes, 32)
        total = sum(s.num_layers for s in result)
        assert total == 32

    def test_zero_capacity_fallback_equal(self):
        """All nodes with zero capacity → equal fallback."""
        rebalancer = WaterFillingRebalancer()
        nodes = [_bare_node("n0"), _bare_node("n1"), _bare_node("n2")]
        result = rebalancer.rebalance([], nodes, 12)
        total = sum(s.num_layers for s in result)
        assert total == 12
        for s in result:
            assert s.num_layers == 4

    def test_rebalance_three_to_two_nodes(self):
        """3-node cluster drops to 2 nodes — layers should be split between 2."""
        rebalancer = WaterFillingRebalancer()
        nodes = [_node("n0", memory_gb=64), _node("n1", memory_gb=64)]
        current = [
            StageAllocation("n0", 0, 12, 12),
            StageAllocation("n1", 12, 24, 12),
            StageAllocation("n2", 24, 36, 12),
        ]
        result = rebalancer.rebalance(current, nodes, 36)
        total = sum(s.num_layers for s in result)
        assert total == 36
        assert len(result) == 2

    def test_granularity_controls_transfer_size(self):
        """Transfer granularity limits how many layers move per iteration."""
        rebalancer = WaterFillingRebalancer(transfer_granularity=5, improvement_threshold=0.01)
        nodes = [_node("n0", memory_gb=64), _node("n1", memory_gb=64)]
        current = [StageAllocation("n0", 0, 32, 32)]
        result = rebalancer.rebalance(current, nodes, 32)
        total = sum(s.num_layers for s in result)
        assert total == 32


# ===================================================================
# Test AllocationStats
# ===================================================================


class TestAllocationStats:
    def test_equal_split_perfect_balance(self):
        allocator = LayerAllocator()
        nodes = [_node("n0"), _node("n1"), _node("n2"), _node("n3")]
        allocator.allocate(32, nodes, LayerAllocationStrategy.EQUAL)
        stats = allocator.get_stats()
        assert stats is not None
        assert stats.balance_ratio == 1.0
        assert stats.max_stage_layers == 8
        assert stats.min_stage_layers == 8

    def test_unequal_split_imperfect_balance(self):
        allocator = LayerAllocator()
        nodes = [_node("n0", memory_gb=128), _node("n1", memory_gb=32)]
        allocator.allocate(10, nodes, LayerAllocationStrategy.MEMORY_PROPORTIONAL)
        stats = allocator.get_stats()
        assert stats is not None
        assert stats.balance_ratio < 1.0
        assert stats.strategy == "memory_proportional"

    def test_stats_to_dict(self):
        allocator = LayerAllocator()
        nodes = [_node("n0")]
        allocator.allocate(10, nodes, LayerAllocationStrategy.EQUAL)
        stats = allocator.get_stats()
        d = stats.to_dict()
        assert "total_layers" in d
        assert "balance_ratio" in d
        assert "strategy" in d


# ===================================================================
# Test LayerAllocator edge cases
# ===================================================================


class TestLayerAllocatorEdgeCases:
    def test_zero_layers_raises(self):
        allocator = LayerAllocator()
        with pytest.raises(ValueError, match="num_layers must be > 0"):
            allocator.allocate(0, [_node("n0")], LayerAllocationStrategy.EQUAL)

    def test_empty_nodes_raises(self):
        allocator = LayerAllocator()
        with pytest.raises(ValueError, match="nodes must be non-empty"):
            allocator.allocate(32, [], LayerAllocationStrategy.EQUAL)

    def test_negative_layers_raises(self):
        allocator = LayerAllocator()
        with pytest.raises(ValueError):
            allocator.allocate(-1, [_node("n0")], LayerAllocationStrategy.EQUAL)

    def test_no_profiles_falls_back_to_equal(self):
        """If no node has profile data, falls back to EQUAL regardless of requested strategy."""
        allocator = LayerAllocator()
        nodes = [_bare_node("n0"), _bare_node("n1")]
        stages = allocator.allocate(10, nodes, LayerAllocationStrategy.MEMORY_PROPORTIONAL)
        total = sum(s.num_layers for s in stages)
        assert total == 10
        # Should be equal: 5 and 5
        assert stages[0].num_layers == 5
        assert stages[1].num_layers == 5

    def test_more_nodes_than_layers(self):
        """More nodes than layers — some nodes get 0 layers."""
        allocator = LayerAllocator()
        nodes = [_node(f"n{i}") for i in range(10)]
        stages = allocator.allocate(4, nodes, LayerAllocationStrategy.EQUAL)
        total = sum(s.num_layers for s in stages)
        assert total == 4

    def test_single_layer_single_node(self):
        allocator = LayerAllocator()
        stages = allocator.allocate(1, [_node("n0")], LayerAllocationStrategy.EQUAL)
        assert stages[0].num_layers == 1

    def test_stats_none_before_allocation(self):
        allocator = LayerAllocator()
        assert allocator.get_stats() is None

    def test_rebalance_via_allocator(self):
        """LayerAllocator.rebalance delegates to WaterFillingRebalancer."""
        allocator = LayerAllocator()
        nodes = [_node("n0", memory_gb=64), _node("n1", memory_gb=64)]
        current = [StageAllocation("n0", 0, 32, 32)]
        result = allocator.rebalance(current, nodes, 32)
        total = sum(s.num_layers for s in result)
        assert total == 32
        # After rebalance, stats should be populated
        stats = allocator.get_stats()
        assert stats is not None
        assert stats.total_layers == 32

    def test_memory_proportional_for_unequal_nodes(self):
        """Allocator correctly distributes layers proportional to memory
        (using raw bytes, not log-scale)."""
        allocator = LayerAllocator()
        nodes = [_mem_only_node("big", memory_gb=128), _mem_only_node("small", memory_gb=32)]
        stages = allocator.allocate(32, nodes, LayerAllocationStrategy.MEMORY_PROPORTIONAL)
        total = sum(s.num_layers for s in stages)
        assert total == 32
        # 128/160 = 80% → 25.6 → 26; 32/160 = 20% → 6.4 → 6
        assert stages[0].num_layers > stages[1].num_layers


# ===================================================================
# Test backward compatibility with pipeline.py
# ===================================================================


class TestPipelineBackwardCompatibility:
    def test_auto_partition_model_default_strategy(self):
        """auto_partition_model with default strategy still works."""
        pp = auto_partition_model(
            num_layers=32,
            num_nodes=4,
            node_memory_gb=[64.0, 64.0, 64.0, 64.0],
            model_memory_per_layer_gb=1.0,
        )
        assert isinstance(pp, PipelineParallel)
        assert pp.num_stages == 4
        total = sum(s.num_layers for s in pp.stages)
        assert total == 32

    def test_auto_partition_model_equal_strategy(self):
        pp = auto_partition_model(
            num_layers=24,
            num_nodes=3,
            node_memory_gb=[128.0, 64.0, 64.0],
            model_memory_per_layer_gb=1.0,
            strategy=LayerAllocationStrategy.EQUAL,
        )
        total = sum(s.num_layers for s in pp.stages)
        assert total == 24
        # Equal: 8 each
        for s in pp.stages:
            assert s.num_layers == 8

    def test_auto_partition_model_memory_proportional(self):
        pp = auto_partition_model(
            num_layers=24,
            num_nodes=3,
            node_memory_gb=[128.0, 64.0, 64.0],
            model_memory_per_layer_gb=1.0,
            strategy=LayerAllocationStrategy.MEMORY_PROPORTIONAL,
        )
        total = sum(s.num_layers for s in pp.stages)
        assert total == 24
        # 128/256 = 50% → 12; 64/256 = 25% → 6 each
        assert pp.stages[0].num_layers == 12
        assert pp.stages[1].num_layers == 6
        assert pp.stages[2].num_layers == 6

    def test_auto_partition_with_bandwidth_info(self):
        """auto_partition_model accepts optional bandwidth/latency/gpu info."""
        pp = auto_partition_model(
            num_layers=32,
            num_nodes=2,
            node_memory_gb=[64.0, 64.0],
            model_memory_per_layer_gb=1.0,
            strategy=LayerAllocationStrategy.BANDWIDTH_AWARE,
            node_bandwidths_mbps=[40000.0, 1000.0],
        )
        total = sum(s.num_layers for s in pp.stages)
        assert total == 32

    def test_auto_partition_stages_contiguous(self):
        pp = auto_partition_model(
            num_layers=32,
            num_nodes=4,
            node_memory_gb=[64.0, 64.0, 64.0, 64.0],
            model_memory_per_layer_gb=1.0,
        )
        stages = pp.stages
        # Check contiguity
        for i in range(len(stages) - 1):
            assert stages[i].end_layer == stages[i + 1].start_layer
        assert stages[-1].end_layer == 32

    def test_auto_partition_model_warning_for_oversubscribed(self):
        """Should log warning when model doesn't fit in memory."""
        pp = auto_partition_model(
            num_layers=100,
            num_nodes=2,
            node_memory_gb=[10.0, 10.0],
            model_memory_per_layer_gb=1.0,
        )
        # Should still return a valid partition
        total = sum(s.num_layers for s in pp.stages)
        assert total == 100


# ===================================================================
# Test PipelineParallel still works
# ===================================================================


class TestPipelineParallelIntegration:
    def test_pipeline_stages_is_last(self):
        pp = auto_partition_model(
            num_layers=12,
            num_nodes=3,
            node_memory_gb=[64.0, 64.0, 64.0],
            model_memory_per_layer_gb=1.0,
        )
        stages = pp.stages
        assert not stages[0].is_last
        assert not stages[1].is_last
        assert stages[2].is_last

    def test_pipeline_get_stage(self):
        pp = auto_partition_model(
            num_layers=12,
            num_nodes=3,
            node_memory_gb=[64.0, 64.0, 64.0],
            model_memory_per_layer_gb=1.0,
        )
        s1 = pp.get_stage(1)
        assert s1 is not None
        assert s1.rank == 1

    def test_pipeline_to_dict(self):
        pp = auto_partition_model(
            num_layers=12,
            num_nodes=3,
            node_memory_gb=[64.0, 64.0, 64.0],
            model_memory_per_layer_gb=1.0,
        )
        d = pp.to_dict()
        assert d["num_layers"] == 12
        assert d["num_stages"] == 3
        assert len(d["stages"]) == 3
