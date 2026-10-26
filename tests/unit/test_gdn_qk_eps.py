"""GDN q/k norm epsilon follows the installed mlx-vlm (b8bf04c) and ScaledQuantizedLinear rows (63fc979).

CPU-only. The fused Metal kernels take their epsilon text from ``gdn_qk_eps``; Metal parity of the
fused norm against stock after the 0.7.10 bump is on the Mac verification list.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from yunshu_engine.kernels import batch_invariant, gdn_prefill, gdn_qk_eps
from yunshu_engine.kernels.omlx import qwen35_gdn_prework as prework


def _stock(q, k):
    from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet

    return Qwen3_5GatedDeltaNet._normalize_qk(None, q, k)


def test_eps_matches_installed_upstream_function_exactly():
    kind = gdn_qk_eps.qk_norm_kind()
    assert kind is not None
    eps = gdn_qk_eps.qk_norm_eps(128)
    x = mx.random.normal((1, 5, 3, 128)) * 1e-3
    q, k = _stock(x, x * 2)
    inv = 128**-0.5
    with mx.stream(mx.cpu):
        assert mx.array_equal(q, (inv**2) * mx.fast.rms_norm(x, None, eps)).item()
        assert mx.array_equal(k, inv * mx.fast.rms_norm(x * 2, None, eps)).item()
    # the pinned mlx-vlm (0.7.8) is the fixed form; 0.7.10 flips to per_head_dim with no code change
    import importlib.metadata as md
    from packaging.version import Version

    expected = "per_head_dim" if Version(md.version("mlx-vlm")) >= Version("0.7.10") else "fixed"
    assert kind == expected


@pytest.mark.parametrize(
    "eps_fn, kind",
    [
        (lambda d: 1e-6, "fixed"),
        (lambda d: 1e-6 / d, "per_head_dim"),
    ],
)
def test_measure_recognises_both_upstream_forms(eps_fn, kind):
    def normalize(q, k):
        s = k.shape[-1] ** -0.5
        e = eps_fn(k.shape[-1])
        return (s**2) * mx.fast.rms_norm(q, None, e), s * mx.fast.rms_norm(k, None, e)

    assert gdn_qk_eps.measure(normalize) == kind


def test_measure_rejects_unknown_eps():
    def normalize(q, k):
        s = k.shape[-1] ** -0.5
        return (s**2) * mx.fast.rms_norm(q, None, 1e-5), s * mx.fast.rms_norm(k, None, 1e-5)

    assert gdn_qk_eps.measure(normalize) is None


def test_expression_text_is_float32_exact():
    fixed = gdn_qk_eps.expression(128, "fixed")
    per = gdn_qk_eps.expression(128, "per_head_dim")
    assert np.float32(float(fixed[:-1])) == np.float32(1e-6)
    assert np.float32(float(per[:-1])) == np.float32(1e-6) / np.float32(128)
    assert gdn_qk_eps.expression("float(DK)", "per_head_dim") == "(1e-06f / float(DK))"
    assert gdn_qk_eps.expression("float(DK)", "fixed") == fixed


def test_kernel_sources_carry_the_measured_epsilon(monkeypatch):
    assert gdn_qk_eps.MARKER in gdn_prefill._NORM_SCALE_SOURCE
    assert gdn_qk_eps.MARKER in prework._SOURCE
    for kind in ("fixed", "per_head_dim"):
        monkeypatch.setattr(gdn_qk_eps, "qk_norm_kind", lambda kind=kind: kind)
        fused = gdn_qk_eps.render(gdn_prefill._NORM_SCALE_SOURCE)
        arm = gdn_qk_eps.render(prework._SOURCE, "float(DK)")
        assert gdn_qk_eps.MARKER not in fused + arm
        assert f"ss / 128.0f + {gdn_qk_eps.expression(128, kind)}" in fused
        assert f"sumsq / float(DK) + {gdn_qk_eps.expression('float(DK)', kind)}" in arm
    # L2 arms keep mlx-lm's sum + 1e-6
    assert "T(1e-6f)" in prework._SOURCE


def test_scaled_quantized_linear_rows_are_singleton_calls():
    types = batch_invariant.scaled_linear_types()
    if not types:
        pytest.skip("installed mlx-vlm has no ScaledQuantizedLinear")
    lin = nn.Linear(64, 32, bias=False)
    ql = nn.QuantizedLinear.from_linear(lin, group_size=16, bits=4, mode="nvfp4")
    scaled = types[0](ql, mx.array(0.5), "weight_global_scale")
    x = mx.random.normal((2, 3, 64)).astype(mx.bfloat16)
    out = batch_invariant.scaled_rows(scaled, x)
    assert out.shape == (2, 3, 32)
    for b in range(2):
        for t in range(3):
            assert mx.array_equal(out[b : b + 1, t : t + 1], scaled(x[b : b + 1, t : t + 1])).item()
    # not a scaled layer, or a single row: left to the caller
    assert batch_invariant.scaled_rows(ql, x) is None
    assert batch_invariant.scaled_rows(scaled, x[:1, :1]) is None
