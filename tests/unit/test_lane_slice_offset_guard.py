"""Tensor slice() offsets >= 2 GiB route to stock (mlx2 b9db287, llama.cpp #28748).

CPU-only: arrays are lazy (never evaluated), so a >2 GiB weight is only a shape.
Metal behaviour at the boundary stays on the Mac verification list.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from yunshu_engine.kernels import int_code_linear, lane_linear
from yunshu_engine.kernels.tensorfold import lane_qmm


def test_offset_math():
    # untiled: last tile's first row + one row; 4-bit K=5120 -> 2560 B per row
    assert lane_qmm.max_tensor_slice_offset(64, 5120, 4) == 33 * 2560
    assert lane_qmm.max_tensor_slice_offset(32, 64, 8) == 64
    assert lane_qmm.max_tensor_slice_offset(33, 64, 8) == 33 * 64
    # tiled weights only slice inside a row
    assert lane_qmm.max_tensor_slice_offset(10**9, 5120, 8, tiled=True) == 5120


def test_boundary():
    k, bits = 8192, 8  # 8192 B per row
    rows_at_limit = lane_qmm.SLICE_OFFSET_LIMIT // k  # 262144 rows
    n_ok = rows_at_limit - 64
    n_bad = rows_at_limit + 64
    assert lane_qmm.slice_offset_ok(n_ok, k, bits)
    assert not lane_qmm.slice_offset_ok(n_bad, k, bits)
    # the same weight tiled is fine: tiles come from 64-bit pointers
    assert lane_qmm.slice_offset_ok(n_bad, k, bits, tiled=True)


def _lazy_linear(n, k, bits):
    q = nn.QuantizedLinear(64, 4, bias=False, group_size=64, bits=bits)
    q.weight = mx.zeros((n, k * bits // 32), mx.uint32)
    q.scales = mx.zeros((n, k // 64), mx.bfloat16)
    q.biases = mx.zeros((n, k // 64), mx.bfloat16)
    return q


@pytest.fixture
def ready(monkeypatch):
    monkeypatch.setattr(lane_qmm, "ready", lambda: True)


def test_supports_declines_deep_untiled(ready):
    k, bits = 8192, 8
    n = lane_qmm.SLICE_OFFSET_LIMIT // k + 64
    q = _lazy_linear(n, k, bits)
    x = mx.zeros((4, k), mx.bfloat16)
    assert not lane_qmm.supports(q.weight, q.scales, x, bits, 64, "affine")
    assert lane_qmm.supports(q.weight, q.scales, x, bits, 64, "affine", tiled=True)
    small = _lazy_linear(256, k, bits)
    assert lane_qmm.supports(small.weight, small.scales, x, bits, 64, "affine")


def test_eligible_gates(ready):
    k, bits = 8192, 8
    deep_untiled = lane_qmm.SLICE_OFFSET_LIMIT // k + 66  # n % 32 != 0 -> untiled
    assert deep_untiled % lane_qmm.NT
    assert not lane_linear.eligible(_lazy_linear(deep_untiled, k, bits))
    deep_tiled = (deep_untiled // lane_qmm.NT + 1) * lane_qmm.NT
    assert lane_linear.eligible(_lazy_linear(deep_tiled, k, bits))
    assert lane_linear.eligible(_lazy_linear(100, k, bits))
    # IntCodeLinear never tiles
    assert not int_code_linear.eligible(_lazy_linear(deep_tiled, k, bits))
    assert int_code_linear.eligible(_lazy_linear(100, k, bits))


def test_lane_matmul_refuses_before_launch():
    k, bits = 8192, 8
    n = lane_qmm.SLICE_OFFSET_LIMIT // k + 64
    w = mx.zeros((n, k * bits // 32), mx.uint32)
    sbt = mx.zeros((k // 64, n, 2), mx.bfloat16)
    with pytest.raises(ValueError, match="2 GiB"):
        lane_qmm.lane_matmul(mx.zeros((2, k), mx.bfloat16), w, sbt, tiled=False)
