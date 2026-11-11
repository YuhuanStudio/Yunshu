"""Build replay outputs from a device path without mutating caches; apply only a whole path."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from yunshu_engine import tree_verify as tv


@pytest.fixture(autouse=True)
def cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


class Cell(list):
    pass


def result():
    layer = SimpleNamespace(head_v_dim=4, num_v_heads=1)
    state = mx.array([17.0])
    prev = mx.arange(6).reshape(1, 3, 2).astype(mx.float32)
    mixed = mx.arange(16).reshape(1, 8, 2).astype(mx.float32) + 100

    def fused(layer, state, path, count, prev, mixed, idx):
        mask = mx.arange(path.size) < count[0]
        value = state + mx.sum(mx.where(mask, path + 1, 0))
        tail = mx.take(mx.concatenate([prev, mixed], axis=1)[0], idx, axis=0)[None]
        return value, tail

    rows = SimpleNamespace(replay_tail=fused)
    res = tv.TreeResult(tv.TreeShape([-1, 0, 1, 1, 2, 3, 5, 6]), 0, None)
    res.records = {0: ("gdn", layer, state, prev, mixed, rows)}
    return res


@pytest.mark.parametrize("path", [[0], [0, 1], [0, 1, 3], [0, 1, 3, 5, 6]])
def test_prepared_replay_equals_host_path_and_does_not_mutate_cache(path):
    res = result()
    cache = [Cell([res.records[0][3], res.records[0][2]])]
    before = list(cache[0])
    padded = mx.array(path + [0] * (res.shape.width - len(path)), dtype=mx.int32)
    prepared = tv.prepare_tree_commit(
        res, padded, mx.array([len(path)], dtype=mx.int32)
    )
    assert all(a is b for a, b in zip(cache[0], before, strict=True))
    other = [Cell(before)]
    tv.tree_commit(None, other, res, path)
    tv.tree_commit(None, cache, res, path, prepared=prepared)
    assert all(mx.array_equal(a, b) for a, b in zip(cache[0], other[0], strict=True))


def test_commit_consumes_prepared_outputs_without_building_replay_again():
    res = result()
    prepared = {0: (mx.array([123.0]), mx.full((1, 3, 2), 456.0))}

    def forbidden(*args):
        pytest.fail("replay was rebuilt after readback")

    res.records[0][-1].replay_tail = forbidden
    cache = [Cell([res.records[0][3], res.records[0][2]])]
    tv.tree_commit(None, cache, res, [0, 1], prepared=prepared)
    assert cache[0][1].tolist() == [123.0]
    assert cache[0][0].tolist() == [[[456.0, 456.0]] * 3]
