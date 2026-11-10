"""Qwen4 readiness D1 (2026-11-11): fused-kernel gates that fall back to upstream are counted, not silent."""

from types import SimpleNamespace

from yunshu_engine import fallbacks
from yunshu_engine.kernels import tf_hc_forward


def test_verify_batch_past_the_moe_row_cap_is_counted(monkeypatch):
    fallbacks.reset()
    monkeypatch.setitem(tf_hc_forward._STATE, "moe", True)
    monkeypatch.setattr(tf_hc_forward, "_moe_plan", lambda mlp: {"plan": 1})
    rows = tf_hc_forward.MAX_ROWS + 16
    assert tf_hc_forward._moe(object(), SimpleNamespace(shape=(rows, 512))) is None
    assert fallbacks.counts()["tf_moe_rows"]["count"] == 1
    # a prefill chunk takes the upstream MoE by design and is not a fallback
    assert tf_hc_forward._moe(object(), SimpleNamespace(shape=(2048, 512))) is None
    assert fallbacks.counts()["tf_moe_rows"]["count"] == 1
    fallbacks.reset()
