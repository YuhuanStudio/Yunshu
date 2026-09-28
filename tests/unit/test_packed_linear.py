"""4-bit NAX packed projections vs mx.quantized_matmul."""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from yunshu_engine.kernels.omlx import is_nax_available  # noqa: E402

pytestmark = pytest.mark.skipif(
    not is_nax_available(), reason="needs an M5-class GPU with NAX kernels"
)


def _linear(K, N, bits, seed=0):
    mx.random.seed(seed)
    lin = nn.QuantizedLinear(K, N, bias=False, group_size=64, bits=bits)
    w = mx.random.normal((N, K)) * 0.05
    q, s, b = mx.quantize(w, group_size=64, bits=bits)
    lin.weight, lin.scales, lin.biases = q, s.astype(mx.bfloat16), b.astype(mx.bfloat16)
    mx.eval(lin.parameters())
    return lin


def _packed(lin):
    from yunshu_engine.kernels.omlx import qwen35_packed_linear as pl

    assert pl.eligible(lin)
    (packed,) = pl._pack([lin])
    return packed


@pytest.mark.parametrize("K,N", [(2048, 1024), (4096, 2048)])
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16, 64])
def test_packed_matches_quantized_matmul(K, N, rows):
    lin = _linear(K, N, 4)
    packed = _packed(lin)
    mx.random.seed(1)
    x = (mx.random.normal((rows, K)) * 0.5).astype(mx.bfloat16)
    ref = mx.quantized_matmul(
        x.astype(mx.float32),
        lin.weight,
        lin.scales.astype(mx.float32),
        lin.biases.astype(mx.float32),
        transpose=True,
        group_size=64,
        bits=4,
    )
    err = mx.abs(packed(x).astype(mx.float32) - ref)
    # bf16 output rounding plus fp32 accumulation order.
    assert float(mx.max(err / mx.maximum(mx.abs(ref), 1.0))) < 2e-2


def test_packed_rows_2_to_8_are_row_invariant():
    packed = _packed(_linear(2048, 1024, 4, seed=3))
    mx.random.seed(4)
    x = (mx.random.normal((8, 2048)) * 0.5).astype(mx.bfloat16)
    full = packed(x)
    for rows in range(2, 9):
        assert mx.array_equal(packed(x[:rows]), full[:rows]).item(), rows


def test_only_4bit_is_packed():
    from yunshu_engine.kernels.omlx import qwen35_packed_linear as pl

    assert pl.eligible(_linear(2048, 1024, 4))
    assert not pl.eligible(_linear(2048, 1024, 5))
