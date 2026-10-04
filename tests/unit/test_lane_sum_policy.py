"""Measured context gates retain original input identity only where enabled."""

import math
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
from yunshu_engine.kernels import lane_linear as ll  # noqa: E402


class Array:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.size = math.prod(shape)

    def reshape(self, *shape):
        shape = list(shape)
        if -1 in shape:
            shape[shape.index(-1)] = self.size // math.prod(n for n in shape if n != -1)
        return Array(shape, self.dtype)

    def astype(self, dtype):
        return Array(self.shape, dtype)


class Projection(SimpleNamespace):
    def __contains__(self, key):
        return False

    _rows = ll.LaneLinear._rows


@pytest.mark.parametrize(
    "rows,dtype,enabled,original",
    [
        (16, mx.bfloat16, True, True),
        (16, mx.bfloat16, False, False),
        (16, mx.float32, True, False),
        (512, mx.bfloat16, True, False),
    ],
)
def test_original_bf_identity_reaches_the_matmul_boundary(
    monkeypatch, rows, dtype, enabled, original
):
    calls = []

    def matmul(x, *_, **__):
        calls.append(x)
        return Array((*x.shape[:-1], 32), mx.bfloat16)

    monkeypatch.setattr(ll.lane_qmm, "lane_matmul", matmul)
    monkeypatch.setattr(ll, "STOCK_ROWS", 0)
    monkeypatch.setattr(ll, "_SUM_REUSE", enabled)
    projection = Projection(
        input_dims=64,
        output_dims=32,
        group_size=64,
        bits=4,
        tiled=True,
        weight=None,
        sbt=None,
    )
    x = Array((1, rows, 64), dtype)
    y = ll.LaneLinear.__call__(projection, x)
    assert (calls[0] is x) == original
    assert calls[0].dtype == mx.bfloat16
    assert y.shape == (1, rows, 32) and y.dtype == dtype


def test_unqualified_contexts_and_cleared_requests_disable_reuse(monkeypatch):
    monkeypatch.setattr(ll, "_SUM_POLICY", False)
    monkeypatch.setattr(ll, "_SUM_REUSE", False)
    ll.set_sum_context(range(32768))
    assert not ll.sum_reuse_enabled()
    ll.configure_sum_policy(True)
    for size in (1034, 8203, 32767):
        ll.set_sum_context(range(size))
        assert not ll.sum_reuse_enabled()
    ll.set_sum_context(range(32768))
    assert ll.sum_reuse_enabled()
    ll.set_sum_context(None)
    assert not ll.sum_reuse_enabled()
    ll.set_sum_reuse(True)
    ll.configure_sum_policy(False)
    assert not ll.sum_reuse_enabled()
