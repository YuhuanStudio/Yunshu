"""The fused Qwen4 RMSNorm kernel agrees with the compiled reference and is row-invariant."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
language = pytest.importorskip("mlx_vlm.models.qwen4_exp.language")
from yunshu_engine.kernels import fused_rmsnorm as fr  # noqa: E402


@pytest.fixture(autouse=True)
def _restore():
    yield
    fr.uninstall()


def _ref(x, weight, group_size, eps):
    return language._qwen4_rms_norm(group_size, eps)(x, weight)


@pytest.mark.parametrize(
    ("width", "group"),
    [(2560, None), (10240, 2560), (128, None), (256, None), (96, None)],
)
def test_matches_reference(width, group):
    mx.random.seed(3)
    x = (mx.random.normal((7, width)) * 3.0).astype(mx.bfloat16)
    w = mx.random.normal((width,)).astype(mx.float32) * 0.2
    got = fr.rms_norm(x, w, group, 1e-6)
    want = _ref(x, w, group, 1e-6)
    assert got.dtype == x.dtype and got.shape == x.shape
    g, r = np.array(got.astype(mx.float32)), np.array(want.astype(mx.float32))
    # the statistic can differ in the last fp32 bit; at most one bf16 ulp (2^-7 relative) in a few elements
    assert np.abs(g - r).max() <= 0.02 * np.abs(r).max() / 8 + 1e-3
    assert (g == r).mean() > 0.97


def test_row_invariant_across_batch_shapes():
    mx.random.seed(5)
    x = mx.random.normal((9, 2560)).astype(mx.bfloat16)
    w = mx.random.normal((2560,)).astype(mx.float32) * 0.1
    full = fr.rms_norm(x, w, None, 1e-6)
    for i in (0, 4, 8):
        one = fr.rms_norm(x[i : i + 1], w, None, 1e-6)
        assert mx.array_equal(one[0], full[i]).item()
    three = fr.rms_norm(x.reshape(3, 3, 2560), w, None, 1e-6)
    assert mx.array_equal(three.reshape(9, 2560), full).item()


def test_install_routes_module_and_uninstall_restores():
    norm = language.Qwen4ExpRMSNorm(2560, eps=1e-6)
    norm.weight = mx.random.normal((2560,)).astype(mx.float32) * 0.1
    x = mx.random.normal((2, 2560)).astype(mx.bfloat16)
    before = norm(x)
    assert fr.install() and fr.install()
    after = norm(x)
    assert after.shape == before.shape
    assert (
        np.abs(
            np.array(after.astype(mx.float32)) - np.array(before.astype(mx.float32))
        ).max()
        < 0.05
    )
    fr.uninstall()
    assert mx.array_equal(norm(x), before).item()
    f32 = mx.random.normal((2, 2560)).astype(mx.float32)
    assert fr.install()
    assert norm(f32).dtype == mx.float32  # fp32 input keeps the reference path
