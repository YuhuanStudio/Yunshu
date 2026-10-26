"""Hoisting the query np.asarray out of the per-entry loops must not change lookups."""

import random

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache

from yunshu_engine.kv_prefix_cache import KVPrefixCache, get_prefix_length


def _kv(n):
    c = KVCache()
    c.update_and_fetch(mx.ones((1, 1, n, 4)), mx.ones((1, 1, n, 4)))
    return [c]


def test_get_prefix_length_numpy_query_same_as_mx_query():
    rng = random.Random(3)
    for _ in range(200):
        a = [rng.randrange(3) for _ in range(rng.randrange(0, 30))]
        b = [rng.randrange(3) for _ in range(rng.randrange(0, 30))]
        assert get_prefix_length(
            np.asarray(a, dtype=np.int32), mx.array(b)
        ) == get_prefix_length(mx.array(a), mx.array(b))


def test_no_trim_picks_longest_full_prefix_entry():
    rng = random.Random(11)
    base = [rng.randrange(1000) for _ in range(80)]
    cache = KVPrefixCache(min_prefix_length=4, max_entries=64)
    cache._no_trim_mode = True
    entries = []
    for n in (8, 16, 24, 40, 56):
        entries.append(base[:n])
        cache.add(mx.array(base[:n]), _kv(n))
    # a diverging entry that is never a full prefix
    div = base[:20] + [5000, 5001, 5002, 5003]
    cache.add(mx.array(div), _kv(len(div)))
    for qlen in (3, 8, 9, 20, 41, 56, 57, 80):
        q = base[:qlen]
        hit, remaining, matched = cache.get(mx.array(q))
        # oracle: longest stored entry that is a full prefix of q and shorter than q
        cands = [
            len(e)
            for e in entries
            if len(e) <= qlen and q[: len(e)] == e and len(e) >= 4
        ]
        expect = max(cands) if cands and max(cands) < qlen else 0
        assert matched == expect, (qlen, matched, expect)
        assert (hit is not None) == (expect > 0)
        assert remaining == qlen - expect
