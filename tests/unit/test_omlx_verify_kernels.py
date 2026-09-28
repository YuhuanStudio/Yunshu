"""Vendored oMLX verify kernels install once and arm only inside MTP verify."""

import pytest

pytest.importorskip("mlx_vlm.speculative.mtp")


def test_apply_installs_and_is_idempotent(monkeypatch):
    import mlx_vlm.speculative.mtp as mtp

    from yunshu_engine.kernels import omlx
    from yunshu_engine.kernels.omlx import qwen35_verify_qmm

    applied = omlx.apply()
    assert applied["gdn_prework"] and applied["sdpa_split"] and applied["verify_qmm"]
    wrapped = mtp._mtp_verify_target
    assert getattr(wrapped, "_yunshu_armed", False)
    assert omlx.apply(row_exact=True) is applied
    assert mtp._mtp_verify_target is wrapped

    seen = []
    inner = wrapped.__closure__
    # Only row-exact mode is armed during verify, and it is cleared afterwards.
    original = next(c.cell_contents for c in inner if callable(c.cell_contents))
    monkeypatch.setattr(
        mtp, "_mtp_verify_target", wrapped
    )  # keep the wrapper installed
    import types

    probe = types.SimpleNamespace(
        run=lambda *a, **k: seen.append(
            (qwen35_verify_qmm._is_armed(), qwen35_verify_qmm.is_row_exact_armed())
        )
    )
    for cell in inner:
        if cell.cell_contents is original:
            cell.cell_contents = probe.run
    try:
        omlx._STATE["row_exact"] = True
        wrapped()
        omlx._STATE["row_exact"] = False
        wrapped()
    finally:
        for cell in inner:
            if cell.cell_contents is probe.run:
                cell.cell_contents = original
    assert seen == [(False, True), (False, False)]
    assert qwen35_verify_qmm._is_armed() is False
    assert qwen35_verify_qmm.is_row_exact_armed() is False


def test_invariant_linear_inactive_uses_fallback():
    import mlx.core as mx
    import mlx.nn as nn

    from yunshu_engine.kernels import batch_invariant as bi

    lin = nn.QuantizedLinear(64, 64, bias=False, group_size=64, bits=4)
    lin._yunshu_invariant = True
    x = mx.zeros((1, 1, 64))
    calls = []
    try:
        bi.set_active(False)
        bi.invariant_linear(lin, x, lambda layer, v: calls.append(1) or v)
    finally:
        bi.set_active(True)
    assert calls == [1]
