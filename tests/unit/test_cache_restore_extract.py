import mlx.core as mx
from mlx_vlm.models.cache import BatchKVCache

from yunshu_engine.kernels.cache_restore import extract_single_row


def _cache(pad):
    c = BatchKVCache([pad])
    k = mx.random.normal((1, 2, 6, 4))
    v = mx.random.normal((1, 2, 6, 4))
    c.update_and_fetch(k, v)
    return c


def test_matches_upstream_extract_including_left_padding():
    for pad in (0, 2):
        c = _cache(pad)
        want = c.extract(0)
        got = extract_single_row(c)
        assert got.offset == want.offset
        assert mx.array_equal(got.keys, want.keys).item()
        assert mx.array_equal(got.values, want.values).item()


def test_declines_multi_row_and_empty():
    assert extract_single_row(BatchKVCache([0, 0])) is None
    assert extract_single_row(BatchKVCache([0])) is None
