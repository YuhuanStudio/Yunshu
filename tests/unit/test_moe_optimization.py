"""Tests for MoE top-k optimization."""
import pytest


class FakeSwitchGLU:
    def __init__(self, top_k=4):
        self.top_k = top_k


class FakeLinear:
    pass


class FakeModel:
    """Mimics mlx.nn.Module.named_modules() flat iteration."""
    def __init__(self, children):
        self._children = children

    def named_modules(self):
        for name, mod in self._children.items():
            yield name, mod


class TestDetectMoEConfig:
    def test_detects_moe_model(self):
        from yunshu_engine.moe_optimization import detect_moe_config
        model = FakeModel({
            "layer.0.mlp": FakeSwitchGLU(top_k=8),
            "layer.1.mlp": FakeSwitchGLU(top_k=8),
        })
        config = detect_moe_config(model)
        assert config is not None
        assert config["moe_layers"] == 2
        assert config["top_k"] == 8

    def test_no_moe_model(self):
        from yunshu_engine.moe_optimization import detect_moe_config
        model = FakeModel({
            "layer.0": FakeLinear(),
            "layer.1": FakeLinear(),
        })
        config = detect_moe_config(model)
        assert config is None


class TestApplyMoETopK:
    def test_reduces_top_k(self):
        from yunshu_engine.moe_optimization import apply_moe_top_k
        layer1 = FakeSwitchGLU(top_k=8)
        layer2 = FakeSwitchGLU(top_k=8)
        model = FakeModel({"layer.0.mlp": layer1, "layer.1.mlp": layer2})

        result = apply_moe_top_k(model, target_top_k=4)
        assert result["patched_layers"] == 2
        assert result["original_top_k"] == 8
        assert result["new_top_k"] == 4
        assert layer1.top_k == 4
        assert layer2.top_k == 4

    def test_no_patch_if_already_at_target(self):
        from yunshu_engine.moe_optimization import apply_moe_top_k
        layer = FakeSwitchGLU(top_k=2)
        model = FakeModel({"layer.0.mlp": layer})

        result = apply_moe_top_k(model, target_top_k=4)
        assert result["patched_layers"] == 0
        assert layer.top_k == 2  # unchanged

    def test_no_moe_layers(self):
        from yunshu_engine.moe_optimization import apply_moe_top_k
        model = FakeModel({"layer.0": FakeLinear()})
        result = apply_moe_top_k(model, target_top_k=2)
        assert result["patched_layers"] == 0


class TestRestoreMoETopK:
    def test_restore(self):
        from yunshu_engine.moe_optimization import restore_moe_top_k
        layer1 = FakeSwitchGLU(top_k=4)
        layer2 = FakeSwitchGLU(top_k=4)
        model = FakeModel({"layer.0.mlp": layer1, "layer.1.mlp": layer2})

        restored = restore_moe_top_k(model, 8)
        assert restored == 2
        assert layer1.top_k == 8
        assert layer2.top_k == 8
