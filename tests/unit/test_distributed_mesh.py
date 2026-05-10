"""Unit tests for yunshu_mesh distributed collective operations.

These tests validate the collective ops wrapper and mesh manager
in both single-node and distributed (ring backend) modes.

For distributed tests, run via launch_mesh.py:
  PYTHONPATH=. uv run python scripts/launch_mesh.py -n 2 -- pytest tests/unit/test_distributed_mesh.py -v
"""
from __future__ import annotations

import os

import mlx.core as mx
import pytest

from yunshu_mesh.collective import CollectiveOps
from yunshu_mesh.manager import MeshManager


class TestCollectiveOpsSingleNode:
    """Collective ops in single-node (no distributed) mode."""

    def test_single_node_no_init(self):
        ops = CollectiveOps()
        assert not ops.is_initialized
        assert ops.rank == 0
        assert ops.size == 1

    def test_single_node_all_reduce_passthrough(self):
        ops = CollectiveOps()
        x = mx.array([1.0, 2.0, 3.0])
        result = ops.all_reduce_sum(x)
        mx.eval(result)
        assert result.tolist() == [1.0, 2.0, 3.0]

    def test_single_node_all_gather_passthrough(self):
        ops = CollectiveOps()
        x = mx.array([5.0])
        result = ops.all_gather(x)
        mx.eval(result)
        assert result.tolist() == [5.0]

    def test_single_node_all_reduce_mean(self):
        ops = CollectiveOps()
        x = mx.array([4.0, 6.0])
        result = ops.all_reduce_mean(x)
        mx.eval(result)
        assert result.tolist() == [4.0, 6.0]


class TestCollectiveOpsDistributed:
    """Collective ops in distributed mode (requires ring backend).

    Run with: launch_mesh.py -n 2 -- pytest this_file.py -v
    """

    @pytest.fixture(autouse=True)
    def init_distributed(self):
        rank_env = os.environ.get("MLX_RANK")
        hostfile = os.environ.get("MLX_HOSTFILE")
        if not rank_env or not hostfile:
            pytest.skip("Distributed tests require MLX_RANK and MLX_HOSTFILE env vars")

        self.ops = CollectiveOps(backend="ring")
        assert self.ops.initialize(backend="ring")
        assert self.ops.is_initialized
        assert self.ops.size == 2

    def test_rank_and_size(self):
        assert self.ops.size == 2
        assert self.ops.rank in (0, 1)

    def test_all_sum_scalar(self):
        rank = self.ops.rank
        x = mx.array([float(rank + 1)])
        result = self.ops.all_reduce_sum(x)
        mx.eval(result)
        expected = 1.0 + 2.0  # rank0=1, rank1=2
        assert abs(result.item() - expected) < 0.01

    def test_all_sum_vector(self):
        rank = self.ops.rank
        v = mx.array([float(rank), float(rank * 2)])
        result = self.ops.all_reduce_sum(v)
        mx.eval(result)
        assert result.shape == (2,)

    def test_all_gather(self):
        rank = self.ops.rank
        y = mx.array([float(rank * 100)])
        gathered = self.ops.all_gather(y)
        mx.eval(gathered)
        assert gathered.tolist() == [0.0, 100.0]

    def test_all_gather_vector(self):
        rank = self.ops.rank
        z = mx.array([float(rank + 1), float((rank + 1) * 10)])
        gathered = self.ops.all_gather(z)
        mx.eval(gathered)
        assert gathered.shape[0] == 4  # 2 ranks x 2 elements

    def test_all_max(self):
        rank = self.ops.rank
        m = mx.array([float(rank * 7)])
        result = self.ops.all_max(m)
        mx.eval(result)
        assert result.item() == 7.0  # max(0, 7)

    def test_all_min(self):
        rank = self.ops.rank
        m = mx.array([float(rank * 7)])
        result = self.ops.all_min(m)
        mx.eval(result)
        assert result.item() == 0.0  # min(0, 7)


class TestMeshManager:
    """MeshManager integration tests."""

    def test_single_node_init(self):
        if os.environ.get("MLX_HOSTFILE"):
            pytest.skip("Single-node test in distributed environment")
        manager = MeshManager()
        manager.initialize()
        assert not manager.is_distributed
        assert manager.rank == 0
        assert manager.world_size == 1

    def test_stats_single_node(self):
        if os.environ.get("MLX_HOSTFILE"):
            pytest.skip("Single-node test in distributed environment")
        manager = MeshManager()
        manager.initialize(local_port=8000)
        stats = manager.get_stats()
        assert stats["distributed"] is False
        assert stats["rank"] == 0
        assert stats["world_size"] == 1
        assert stats["topology"] == "single"

    def test_distributed_init(self):
        if not os.environ.get("MLX_HOSTFILE"):
            pytest.skip("Distributed test requires MLX_HOSTFILE")
        manager = MeshManager()
        manager.initialize(backend="ring")
        assert manager.is_distributed
        assert manager.world_size == 2
        assert manager.rank in (0, 1)
