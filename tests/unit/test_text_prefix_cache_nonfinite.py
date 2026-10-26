"""A cache with a non-finite recurrent state is never published to the text path's prefix cache."""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

from yunshu_engine.kv_prefix_cache import KVPrefixCache, recurrent_state_finite

N = 64


def _cache(value=0.5, dtype=mx.float16):
    rec = ArraysCache(size=2)
    rec.cache = [mx.ones((1, 4, 8), dtype=dtype), mx.full((1, 2, 3), value, dtype)]
    kv = KVCache()
    kv.update_and_fetch(mx.ones((1, 2, N, 4)), mx.ones((1, 2, N, 4)))
    mx.eval(rec.cache, kv.keys)
    return [rec, kv]


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_recurrent_state_is_not_stored(bad):
    pc = KVPrefixCache(max_entries=4, min_prefix_length=8)
    pc.add(mx.arange(N), _cache(bad))
    assert pc.size == 0
    assert recurrent_state_finite(_cache(bad)) is False


def test_finite_state_is_still_stored():
    pc = KVPrefixCache(max_entries=4, min_prefix_length=8)
    pc.add(mx.arange(N), _cache())
    assert pc.size == 1
    assert recurrent_state_finite(_cache()) is True


def test_dense_kv_only_cache_is_not_inspected():
    kv = KVCache()
    kv.update_and_fetch(mx.ones((1, 2, N, 4)), mx.ones((1, 2, N, 4)))
    assert recurrent_state_finite([kv]) is True


def test_prompt_cache_store_refuses_a_non_finite_state():
    from yunshu_engine.prompt_cache import PromptCacheManager

    pm = PromptCacheManager()
    assert pm.store("bad", _cache(float("nan")), token_count=N) is False
    assert pm.lookup("bad") is None
    assert pm.store("ok", _cache(), token_count=N) is True
