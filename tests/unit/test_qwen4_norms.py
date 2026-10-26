"""Legacy direct-gamma packs must not receive a second +1 (TF2bit garbage)."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from yunshu_engine.qwen4_norms import direct_gamma, infer_direct_gamma, normalize


def test_checkpoint_convention_uses_base_anchors_and_rejects_mixtures():
    assert direct_gamma([0.94] * 48)
    assert not direct_gamma([-0.06] * 48)
    assert not direct_gamma([1.0] * 2)
    with pytest.raises(ValueError, match="ambiguous"):
        direct_gamma([0.0] * 24 + [1.0] * 24)
    with pytest.raises(ValueError, match="nonfinite"):
        direct_gamma([float("nan")] * 48)
    with mx.stream(mx.cpu):
        weights = {
            f"language_model.model.layers.{i}.attn_hyper_connection.hc_norm.weight": mx.full(
                (4,), 0.94
            )
            for i in range(8)
        }
        # MTP has differently trained statistics, and is never used to infer centering.
        weights["mtp.layers.0.attn_hyper_connection.hc_norm.weight"] = mx.full((4,), -9)
        assert infer_direct_gamma(weights)


def test_recenter_only_qwen4_norms_and_preserve_direct_gamma_exactly():
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpRMSNorm

    with mx.stream(mx.cpu):
        hc = Qwen4ExpRMSNorm(2)
        ordinary = nn.RMSNorm(2)
        model = SimpleNamespace(named_modules=lambda: [("hc", hc), ("gdn", ordinary)])
        gamma = mx.array([0.001953125, 1.5], dtype=mx.bfloat16)
        weights = {"hc.weight": gamma, "gdn.weight": mx.ones((2,))}
        ordinary_weight = weights["gdn.weight"]
        assert normalize(model, weights, direct=False) == 0
        assert weights["hc.weight"] is gamma
        assert normalize(model, weights, direct=True) == 1
        assert mx.array_equal(
            1 + weights["hc.weight"].astype(mx.float32), gamma.astype(mx.float32)
        ).item()
        assert weights["hc.weight"].dtype == mx.float32
        assert weights["gdn.weight"] is ordinary_weight
