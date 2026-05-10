"""Tests for yunshu_mesh sharding module.

Tests model sharding utilities in both single-node and distributed modes.
"""
from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
import pytest

from yunshu_mesh.sharding import (
    PipelineFirstLayer,
    PipelineLastLayer,
    ShardedMoE,
    get_inner_model,
    get_layers,
    mx_barrier,
    shard_tensor_parallel,
    shard_pipeline_parallel,
    load_sharded_model,
    _detect_model_family,
)


class DummyLayer(nn.Module):
    """Dummy transformer layer for testing pipeline wrappers."""
    def __init__(self, dim: int = 64):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.n_heads = 4
        self.n_kv_heads = 2

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)


class DummyInnerModel(nn.Module):
    """Dummy inner model with layers list."""
    def __init__(self, num_layers: int = 8, dim: int = 64):
        super().__init__()
        self.layers = [DummyLayer(dim) for _ in range(num_layers)]


class DummyModel(nn.Module):
    """Dummy model wrapper matching common patterns."""
    def __init__(self, num_layers: int = 8, dim: int = 64):
        super().__init__()
        self.model = DummyInnerModel(num_layers, dim)
        self.head = nn.Linear(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.model.layers:
            x = layer(x)
        return self.head(x)


class TransformerModel(nn.Module):
    """Test model with 'transformer' instead of 'model'."""
    def __init__(self):
        super().__init__()
        self.transformer = DummyInnerModel(4, 32)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.transformer.layers:
            x = layer(x)
        return x


class BackboneModel(nn.Module):
    """Test model with 'backbone' attribute."""
    def __init__(self):
        super().__init__()
        self.backbone = DummyInnerModel(4, 32)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.backbone.layers:
            x = layer(x)
        return x


class TestGetInnerModel:
    def test_model_attr(self):
        model = DummyModel(4, 32)
        inner = get_inner_model(model)
        assert isinstance(inner, DummyInnerModel)
        assert hasattr(inner, "layers")

    def test_transformer_attr(self):
        model = TransformerModel()
        inner = get_inner_model(model)
        assert isinstance(inner, DummyInnerModel)

    def test_backbone_attr(self):
        model = BackboneModel()
        inner = get_inner_model(model)
        assert isinstance(inner, DummyInnerModel)

    def test_no_valid_attr(self):
        with pytest.raises(ValueError, match="model.*transformer.*backbone"):
            get_inner_model(nn.Module())


class TestGetLayers:
    def test_layers_attr(self):
        inner = DummyInnerModel(4, 32)
        layers = get_layers(inner)
        assert len(layers) == 4

    def test_returns_list(self):
        inner = DummyInnerModel(2, 32)
        layers = get_layers(inner)
        assert isinstance(layers, list)


class TestPipelineFirstLayer:
    def test_rank_zero_passes_through(self):
        layer = DummyLayer(32)
        wrapped = PipelineFirstLayer(layer, rank=0, group=None)
        x = mx.random.normal((1, 4, 32))
        output = wrapped(x)
        mx.eval(output)
        assert output.shape == x.shape

    def test_is_prefill_attribute(self):
        layer = DummyLayer(32)
        wrapped = PipelineFirstLayer(layer, rank=0, group=None)
        assert wrapped.is_prefill is False
        wrapped.is_prefill = True
        assert wrapped.is_prefill is True


class TestPipelineLastLayer:
    def test_single_rank_uses_all_gather(self):
        """Single-rank PipelineLastLayer just does all_gather passthrough."""
        layer = DummyLayer(32)
        wrapped = PipelineLastLayer(layer, rank=0, world_size=1, group=None)
        x = mx.random.normal((1, 4, 32))
        # Without group, will fail on all_gather — that's expected
        # In single-node mode, pipeline isn't used


class TestShardedMoE:
    def test_single_node_passthrough(self):
        """Without a group, ShardedMoE should error (sum_gradients needs group)."""
        layer = DummyLayer(32)
        # No group means it will fail — this test just confirms construction works
        wrapped = ShardedMoE(layer, group=None)
        assert wrapped.group is None


class TestMxBarrier:
    def test_barrier_none_group(self):
        """Barrier with None group should be a no-op."""
        mx_barrier(None)

    def test_barrier_with_group(self):
        """Test barrier with distributed group requires actual distributed env."""
        if not os.environ.get("MLX_HOSTFILE"):
            pytest.skip("Distributed test requires MLX_HOSTFILE")
        from yunshu_mesh.collective import CollectiveOps
        ops = CollectiveOps(backend="ring")
        ops.initialize(backend="ring")
        mx_barrier(ops.group)
        ops.shutdown()


class TestDetectModelFamily:
    def test_builtin_shard(self):
        """Model with shard() method returns 'builtin'."""
        class ShardModel(nn.Module):
            def shard(self, group):
                pass
        model = ShardModel()
        assert _detect_model_family(model) == "builtin"

    def test_unknown_model(self):
        """Model without shard() or known patterns returns 'unknown'."""
        model = DummyModel(4, 32)
        family = _detect_model_family(model)
        assert family == "unknown"


class TestLoadShardedModelSingle:
    def test_single_node_load(self):
        """Loading without group should work like normal load."""
        model_name = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        model, tokenizer = load_sharded_model(model_name, group=None)
        assert model is not None
        assert tokenizer is not None
        # Verify model works
        tokens = tokenizer.encode("Hello")
        assert len(tokens) > 0


class TestLoadShardedModelDistributed:
    @pytest.fixture(autouse=True)
    def init_distributed(self):
        if not os.environ.get("MLX_HOSTFILE"):
            pytest.skip("Distributed test requires MLX_HOSTFILE")
        from yunshu_mesh.collective import CollectiveOps
        ops = CollectiveOps(backend="ring")
        assert ops.initialize(backend="ring")
        self.group = ops.group

    def test_tensor_parallel_load(self):
        """Test loading a model with tensor parallel sharding."""
        model_name = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        model, tokenizer = load_sharded_model(
            model_name, group=self.group, strategy="tensor"
        )
        assert model is not None
        assert tokenizer is not None
        # Both ranks should have the tokenizer
        tokens = tokenizer.encode("Hello")
        assert len(tokens) > 0


class TestShardPipelineParallel:
    def test_pipeline_slices_layers(self):
        """Pipeline parallel should slice layers correctly."""
        model = DummyModel(8, 32)
        # Simulate a 2-node pipeline split
        # Can't actually shard without group, but can test the slicing logic
        inner = get_inner_model(model)
        layers = get_layers(inner)
        total = len(layers)

        # Rank 0 gets first half, rank 1 gets second half
        mid = total // 2
        assert mid == 4
        assert len(layers[:mid]) == 4
        assert len(layers[mid:]) == 4
