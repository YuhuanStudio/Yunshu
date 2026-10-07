from types import SimpleNamespace

import mlx.core as mx

from yunshu_engine import tree_verify as tv


def _cache(seed, cap=40):
    k = mx.random.normal((1, 2, cap, 4), key=mx.random.key(seed))
    return SimpleNamespace(keys=k, values=k * 2)


def test_compact_kv_moves_accepted_rows_to_consecutive_slots():
    n0, path = 10, [0, 3, 5, 6]
    caches = [_cache(1), _cache(2)]
    before = [(mx.array(c.keys), mx.array(c.values)) for c in caches]
    src = mx.array([n0 + r for r in path[1:]], dtype=mx.int32)
    tv.compact_kv(caches, src, n0, len(path))
    for c, (k, v) in zip(caches, before, strict=True):
        for dst, row in enumerate(path[1:], start=1):
            assert mx.array_equal(c.keys[..., n0 + dst, :], k[..., n0 + row, :])
            assert mx.array_equal(c.values[..., n0 + dst, :], v[..., n0 + row, :])
        assert mx.array_equal(c.keys[..., : n0 + 1, :], k[..., : n0 + 1, :])
        assert mx.array_equal(
            c.keys[..., n0 + len(path) :, :], k[..., n0 + len(path) :, :]
        )
