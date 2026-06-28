"""Tests for MoE top-k optimization.

the fakes now mirror the REAL mlx-lm structure — the routing `top_k` lives on a
gate/sparse-MoE block (with a `norm_topk_prob` renormalization flag), while the SwitchGLU
expert container has NO `top_k`. The old test's FakeSwitchGLU invented a `top_k` field the
real class lacks, which masked the bug where apply_moe_top_k matched only the (top_k-less)
container classes and patched nothing.
"""


class FakeSwitchGLU:
    """The expert-weight container — consumes pre-computed indices, has NO top_k."""
    pass


class FakeMoeGate:
    """A renormalizing MoE gate/block (e.g. Qwen3MoeSparseMoeBlock): top_k + norm_topk_prob."""
    def __init__(self, top_k=8, norm_topk_prob=True):
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.switch_mlp = FakeSwitchGLU()


class FakeNonRenormGate:
    """A gate that does NOT renormalize — reducing k would scale output down (unsafe)."""
    def __init__(self, top_k=8):
        self.top_k = top_k


class FakeLinear:
    pass


class FakeModel:
    """Mimics mlx.nn.Module.named_modules() flat iteration."""
    def __init__(self, children):
        self._children = children

    def named_modules(self):
        yield from self._children.items()


class TestDetectMoEConfig:
    def test_detects_moe_model(self):
        from yunshu_engine.moe_optimization import detect_moe_config
        model = FakeModel({
            "layer.0.mlp": FakeMoeGate(top_k=8),
            "layer.0.mlp.switch_mlp": FakeSwitchGLU(),
            "layer.1.mlp": FakeMoeGate(top_k=8),
            "layer.1.mlp.switch_mlp": FakeSwitchGLU(),
        })
        config = detect_moe_config(model)
        assert config is not None
        assert config["moe_layers"] == 2          # two gates
        assert config["top_k"] == 8               # read from the gate, not the container

    def test_no_moe_model(self):
        from yunshu_engine.moe_optimization import detect_moe_config
        model = FakeModel({"layer.0": FakeLinear(), "layer.1": FakeLinear()})
        assert detect_moe_config(model) is None


class TestApplyMoETopK:
    def test_reduces_top_k_on_renormalizing_gates(self):
        from yunshu_engine.moe_optimization import apply_moe_top_k
        g1, g2 = FakeMoeGate(top_k=8), FakeMoeGate(top_k=8)
        model = FakeModel({
            "layer.0.mlp": g1, "layer.0.mlp.switch_mlp": FakeSwitchGLU(),
            "layer.1.mlp": g2, "layer.1.mlp.switch_mlp": FakeSwitchGLU(),
        })
        result = apply_moe_top_k(model, target_top_k=4)
        assert result["patched_layers"] == 2
        assert result["original_top_k"] == 8 and result["new_top_k"] == 4
        assert g1.top_k == 4 and g2.top_k == 4

    def test_switchglu_container_is_not_matched(self):
        # the W1023 bug: the old code matched SwitchGLU (which has NO top_k) → patched 0.
        from yunshu_engine.moe_optimization import apply_moe_top_k
        model = FakeModel({"layer.0.mlp.switch_mlp": FakeSwitchGLU()})
        result = apply_moe_top_k(model, target_top_k=4)
        assert result["patched_layers"] == 0

    def test_non_renormalizing_gate_is_skipped_as_unsafe(self):
        from yunshu_engine.moe_optimization import apply_moe_top_k
        g = FakeNonRenormGate(top_k=8)
        model = FakeModel({"layer.0.mlp": g})
        result = apply_moe_top_k(model, target_top_k=4)
        assert result["patched_layers"] == 0
        assert result["skipped_unsafe"] == 1
        assert g.top_k == 8  # left at the trained value (no magnitude degradation)

    def test_no_patch_if_already_at_or_below_target(self):
        from yunshu_engine.moe_optimization import apply_moe_top_k
        g = FakeMoeGate(top_k=2)
        model = FakeModel({"layer.0.mlp": g})
        result = apply_moe_top_k(model, target_top_k=4)
        assert result["patched_layers"] == 0
        assert g.top_k == 2

    def test_no_moe_layers(self):
        from yunshu_engine.moe_optimization import apply_moe_top_k
        model = FakeModel({"layer.0": FakeLinear()})
        assert apply_moe_top_k(model, target_top_k=2)["patched_layers"] == 0


class TestRestoreMoETopK:
    def test_restore(self):
        from yunshu_engine.moe_optimization import restore_moe_top_k
        g1, g2 = FakeMoeGate(top_k=4), FakeMoeGate(top_k=4)
        model = FakeModel({"layer.0.mlp": g1, "layer.1.mlp": g2})
        restored = restore_moe_top_k(model, 8)
        assert restored == 2
        assert g1.top_k == 8 and g2.top_k == 8
