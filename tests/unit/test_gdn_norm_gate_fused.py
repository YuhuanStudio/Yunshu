"""The fused verifier norm must use the served RMS / FP32 SiLU arithmetic.

Upstream omlx e37e07f5: released and nightly MLX builds differ in their float32
exponential, so the kernel probes which expression matches ``_precise_swiglu``
and declines fusion when none does.
"""

import mlx.core as mx
import pytest

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")

language = pytest.importorskip("mlx_vlm.models.qwen3_5.language")
fused = pytest.importorskip("yunshu_engine.kernels.omlx.qwen35_gdn_verify_fused")


def _fused(norm, x, gate):
    rows = x.size // 128
    return fused._norm_gate_kernel(norm.eps)(
        inputs=[x, gate, norm.weight],
        template=[("InT", x.dtype)],
        grid=(32, rows, 1),
        threadgroup=(32, 8, 1),
        output_shapes=[x.shape, (rows * 2,)],
        output_dtypes=[x.dtype, mx.float32],
    )[0]


def test_probe_selects_an_expression():
    assert fused._sigmoid_exp() in ("metal::exp", "metal::precise::exp")


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_fused_norm_gate_matches_served_arithmetic(dtype, eps):
    norm = language.Qwen3_5RMSNormGated(128, eps=eps)
    for seed in range(20):
        mx.random.seed(seed)
        norm.weight = (1 + 0.2 * mx.random.normal((128,))).astype(dtype)
        x = (mx.random.normal((1, 1, 32, 128)) * 10 ** (seed % 8 - 4)).astype(dtype)
        gate = (mx.random.normal(x.shape) * (seed / 2 + 0.125)).astype(dtype)
        expected, actual = norm(x, gate), _fused(norm, x, gate)
        mx.eval(expected, actual)
        assert mx.array_equal(expected.view(mx.uint16), actual.view(mx.uint16)).item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_fused_norm_gate_matches_every_gate_encoding(dtype):
    mx.random.seed(42)
    norm = language.Qwen3_5RMSNormGated(128)
    norm.weight = (1 + 0.2 * mx.random.normal((128,))).astype(dtype)
    x = mx.ones((1, 1, 512, 128), dtype)
    gate = (
        mx.arange(65536, dtype=mx.uint32).astype(mx.uint16).view(dtype).reshape(x.shape)
    )
    expected, actual = norm(x, gate), _fused(norm, x, gate)
    mx.eval(expected, actual)
    assert mx.array_equal(expected.view(mx.uint16), actual.view(mx.uint16)).item()


def test_probe_declines_unknown_served_arithmetic(monkeypatch):
    monkeypatch.setattr(
        language, "_precise_swiglu", lambda h, gate, x: mx.ones_like(gate)
    )
    fused._sigmoid_exp.cache_clear()
    try:
        assert fused._sigmoid_exp() is None
    finally:
        monkeypatch.undo()
        fused._sigmoid_exp.cache_clear()
    assert fused._sigmoid_exp() is not None
