"""The fused Flash-Next GDN verify is reached from both qwen4_exp layer loops.

A pack whose HyperConnection weights are not affine (TensorFold 2-bit) keeps the upstream layer
loop; the GDN hook used to live only in the fused-HC loop, so that pack never reached the plan."""

import pytest

from yunshu_engine.kernels import qwen4_gdn, tf_hc_forward

language = pytest.importorskip("mlx_vlm.models.qwen4_exp.language")


def test_both_loops_reach_the_plan_and_the_fallback_is_the_parent(monkeypatch):
    cls = language.Qwen4ExpBatchInvariantForward
    parent = cls.__mro__[1]
    assert "_gated_delta" not in cls.__dict__
    calls = []
    monkeypatch.setattr(qwen4_gdn, "install_cache_patch", lambda: True)

    def fused(verifier, layer, inputs, mask, cache, original):
        calls.append(layer)
        return original(verifier, layer, inputs, mask, cache)

    monkeypatch.setattr(qwen4_gdn, "gated_delta", fused)
    # Resolved per call: a later patch of the parent (the oMLX prework) is what a fallback runs.
    monkeypatch.setattr(parent, "_gated_delta", lambda self, *a: ("parent", a))
    monkeypatch.setitem(tf_hc_forward._STATE, "gdn_original", None)
    try:
        assert tf_hc_forward.install_qwen4_gdn()
        assert tf_hc_forward.install_qwen4_gdn()  # idempotent
        verifier = object.__new__(cls)
        assert verifier._gated_delta("L", "x", None, "c") == (
            "parent",
            ("L", "x", None, "c"),
        )
        assert calls == ["L"]
    finally:
        monkeypatch.setitem(tf_hc_forward._STATE, "installed", True)
        monkeypatch.setitem(tf_hc_forward._STATE, "original", cls._model)
        tf_hc_forward.uninstall()
    assert "_gated_delta" not in cls.__dict__
