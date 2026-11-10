"""APC row extraction preserves int8 codes and both scale tensors."""

import pytest

mx = pytest.importorskip("mlx.core")


def test_extract_preserves_packed_state_and_detaches_writes():
    from yunshu_engine.kernels.int8_kv import Int8KVCache

    cache = Int8KVCache()
    cache.state = (
        mx.arange(2 * 64 * 32).reshape(2, 1, 64, 32).astype(mx.int8),
        mx.ones((2, 1, 64, 32), mx.int8),
        mx.full((2, 1, 64, 1), 0.125, mx.float16),
        mx.full((2, 1, 64, 1), 0.25, mx.float16),
    )
    cache.offset = 17
    for index in (0, 1, -1):
        row = cache.extract(index)
        assert isinstance(row, Int8KVCache)
        assert row.offset == 17
        for actual, source in zip(row.state, cache.state, strict=True):
            expected = source[index % 2 : index % 2 + 1]
            assert actual.dtype == source.dtype
            assert bool(mx.array_equal(actual, expected))
        row.keys[..., 0, :] = 0
        assert bool(
            mx.array_equal(
                cache.keys, mx.arange(2 * 64 * 32).reshape(2, 1, 64, 32).astype(mx.int8)
            )
        )
    for index in (-3, 2):
        with pytest.raises(IndexError):
            cache.extract(index)


def test_extract_empty_int8_cache():
    from yunshu_engine.kernels.int8_kv import Int8KVCache

    cache = Int8KVCache()
    for index in (0, -1):
        row = cache.extract(index)
        assert isinstance(row, Int8KVCache)
        assert row.state == (None, None, None, None)
        assert row.offset == 0
    with pytest.raises(IndexError):
        cache.extract(1)
