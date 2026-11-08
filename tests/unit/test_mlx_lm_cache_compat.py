"""Cache-state extraction, reasoning tracking and stop matchers against the locked mlx-lm."""

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache

from yunshu_kv.mlx_cache import extract_cache_state


def test_extract_cache_state_trims_to_offset():
    c = KVCache()
    k = mx.ones((1, 2, 3, 4))
    c.update_and_fetch(k, k)
    state = extract_cache_state(c)
    assert state["offset"] == 3 and state["keys"].shape[2] == 3


def test_extract_cache_state_rotating():
    c = RotatingKVCache(max_size=8)
    k = mx.ones((1, 2, 3, 4))
    c.update_and_fetch(k, k)
    state = extract_cache_state(c)
    assert state["keys"].shape[2] == 3 and state["max_size"] == 8


class _Tok:
    has_thinking = True
    think_start_tokens = (10,)
    think_end_tokens = (11, 12)
