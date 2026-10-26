"""Row-exact qmv admission guard (mlx2 1436775): K = 64 / 128 decline.

CPU-only: the guard is pure layout logic, so it runs without Metal. The Metal
parity check at these K values stays on the Mac verification list.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from yunshu_engine.kernels.omlx import row_exact_qmv as rq


def _linear(k, bits=4, group_size=32):
    lin = nn.Linear(k, 16, bias=False)
    lin.weight = (mx.random.normal((16, k)) * 0.02).astype(mx.bfloat16)
    q = nn.QuantizedLinear.from_linear(lin, group_size=group_size, bits=bits)
    q.set_dtype(mx.bfloat16)
    return q


@pytest.mark.parametrize("k", [64, 128])
@pytest.mark.parametrize("bits", [4, 8])
def test_quad_k_declines(k, bits):
    q = _linear(k, bits)
    x = mx.zeros((3, k), mx.bfloat16)
    assert rq._kernel_supported(q, x) is False
    assert rq._plan(q, x, 3) is None


@pytest.mark.parametrize("k", [32, 96, 160, 192, 256, 1024])
def test_other_k_still_admitted(k):
    q = _linear(k)
    assert rq._kernel_supported(q, mx.zeros((3, k), mx.bfloat16)) is True


def test_declined_layer_takes_exact_per_row_form():
    q = _linear(128)
    x = (mx.random.normal((1, 4, 128)) * 0.5).astype(mx.bfloat16)
    out = rq.quantized_linear(q, x)
    for r in range(4):
        assert mx.array_equal(out[:, r : r + 1], q(x[:, r : r + 1])).item()


def test_group_and_one_row_paths_decline():
    q = _linear(128)
    assert rq._group_plan([q, q], mx.zeros((3, 128), mx.bfloat16), 3) is None
    assert (
        rq.one_row_qmv(q.weight, q.scales, q.biases, 4, 32, "affine", mx.bfloat16, 4)
        is None
    )
    assert frozenset({64, 128}) == rq.STOCK_QUAD_K
