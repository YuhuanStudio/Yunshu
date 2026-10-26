"""A text prefix-cache lookup counts as a hit only when the reuse is accepted (omlx 5628bb7e): an exact
hit whose refeed trim fails is reported to the caller as a miss and must not stay in the hit counter."""

from unittest.mock import MagicMock

import mlx.core as mx

from yunshu_engine.kv_prefix_cache import KVPrefixCache


def _cache(offset, fail_trim=False):
    c = MagicMock()
    c.offset = offset
    c.keys = mx.zeros((1, 4, 8))
    c.values = mx.zeros((1, 4, 8))
    c.trim = MagicMock(side_effect=RuntimeError("trim failed") if fail_trim else None)
    return c


def test_accepted_exact_hit_counts_once():
    cache = KVPrefixCache(min_prefix_length=4)
    tokens = mx.array([1, 2, 3, 4, 5, 6, 7, 8])
    cache.add(tokens, [_cache(8)])
    result, _, matched = cache.get(tokens)
    assert result is not None and matched == 7
    assert cache._total_hits == 1 and cache._total_lookups == 1


def test_exact_hit_whose_trim_fails_is_a_miss_not_a_hit():
    cache = KVPrefixCache(min_prefix_length=4)
    tokens = mx.array([1, 2, 3, 4, 5, 6, 7, 8])
    cache.add(tokens, [_cache(8, fail_trim=True)])
    result, _, matched = cache.get(tokens)
    assert result is None and matched == 0
    assert cache._total_hits == 0 and cache._total_lookups == 1
