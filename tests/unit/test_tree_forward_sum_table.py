import mlx.core as mx
import pytest

from yunshu_engine import tree_verify as tv
from yunshu_engine.kernels.omlx import qwen35_verify_qmm as vq


def _table():
    return getattr(vq._ROUTE_ARMED, "sums", None)


@pytest.mark.parametrize("fail", [False, True])
def test_tree_forward_leaves_no_group_sums_behind(monkeypatch, fail):
    def fake(*args, **kwargs):
        x = mx.zeros((1, 64))
        vq.register_group_sums(x, mx.zeros((1, 1)))
        assert _table()
        if fail:
            raise RuntimeError("forward failed")
        return "result"

    monkeypatch.setattr(tv, "_tree_forward", fake)
    vq.clear_group_sums()
    if fail:
        with pytest.raises(RuntimeError):
            tv.tree_forward()
    else:
        assert tv.tree_forward() == "result"
    assert not _table()
