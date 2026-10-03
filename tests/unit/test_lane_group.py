"""Column grouping keeps each original reduction and owns no loaded model."""

import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from yunshu_engine.kernels import lane_group as lg  # noqa: E402


class Projection:
    def __init__(self, n=32, k=64):
        self.bits, self.group_size = 4, 64
        self.input_dims, self.output_dims = k, n
        self.tiled = True
        self.weight = np.ones((n, k), dtype=np.float32)
        self.sbt = np.zeros((1, n, 2), dtype=np.float32)

    def __contains__(self, key):
        return key in self.__dict__

    def parameters(self):
        return {"weight": self.weight, "sbt": self.sbt}


@pytest.fixture
def calls(monkeypatch):
    recorded = []
    monkeypatch.setattr(lg.lane_linear, "LaneLinear", Projection)
    monkeypatch.setattr(lg.lane_linear, "STOCK_ROWS", 0)
    monkeypatch.setattr(
        lg,
        "mx",
        SimpleNamespace(
            concatenate=np.concatenate, eval=lambda *_: None, bfloat16=np.float16
        ),
    )
    monkeypatch.setattr(lg.lane_qmm, "split_k", lambda n, k: 2 if n == 32 else 4)

    def matmul(x, weight, sbt, **kw):
        recorded.append((x, weight, sbt, kw))
        return x.astype(np.float32) @ weight.T

    monkeypatch.setattr(lg.lane_qmm, "lane_matmul", matmul)
    return recorded


def test_preserves_member_split_and_dtype_without_duplicate_buffers(calls):
    a, b = Projection(), Projection()
    x = np.ones((1, 20, 64), dtype=np.float32)
    out = lg.grouped_linears((a, b), x)
    assert len(out) == 2
    assert all(y.shape == (1, 20, 32) and y.dtype == x.dtype for y in out)
    assert calls[0][3]["sk"] == 2  # combined width would choose 4
    assert calls[0][3]["row_block"] == 16
    assert calls[0][0].dtype == np.float16
    assert np.shares_memory(a.weight, calls[0][1])
    assert np.shares_memory(b.weight, calls[0][1])
    lg.grouped_linears((a, b), x)
    assert calls[0][1] is calls[1][1]


@pytest.mark.parametrize("reason", ["split", "bias", "alignment", "rows", "dtype"])
def test_incompatible_projection_falls_back_before_repacking(calls, reason):
    a, b = Projection(), Projection()
    x = np.ones((1, 2, 64), dtype=np.float32)
    if reason == "split":
        b.output_dims = 64
    elif reason == "bias":
        b.bias = np.zeros(32)
    elif reason == "alignment":
        b.output_dims = 48
    elif reason == "rows":
        x = np.ones((1, 129, 64), dtype=np.float32)
    else:
        b.bits = 8
    original = a.weight
    assert lg.grouped_linears((a, b), x) is None
    assert a.weight is original and not calls


def test_reordered_overlapping_or_replaced_members_fall_back(calls):
    a, b, c = Projection(), Projection(), Projection()
    x = np.ones((1, 2, 64), dtype=np.float32)
    lg.grouped_linears((a, b), x)
    assert lg.grouped_linears((b, a), x) is None
    assert lg.grouped_linears((c, b), x) is None
    b.weight = b.weight.copy()
    assert lg.grouped_linears((a, b), x) is None
    assert len(calls) == 1


def test_group_does_not_retain_modules_after_unload(calls):
    a, b = Projection(), Projection()
    refs = weakref.ref(a), weakref.ref(b)
    lg.grouped_linears((a, b), np.ones((1, 2, 64), dtype=np.float32))
    group = a._yunshu_column_group
    del a, b
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert all(ref() is None for ref in group.members)
