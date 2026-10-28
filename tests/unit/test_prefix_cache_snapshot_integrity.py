"""ollama b859a945 / 6137793a / a5b659e6 re-check on the text prefix cache.

ollama's radix trie could merge two snapshot edges as if adjacent and report a hit over a buffer
holding other positions. KVPrefixCache keeps flat entries (no merging); these tests pin the guards
that make the same failure impossible here: a hit never reports more tokens than the snapshot
holds, a hybrid (no-trim) hit needs the snapshot to be exactly its key, and a snapshot handed out
is detached (mutating it cannot change the stored entry).
"""

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache

from yunshu_engine.kv_prefix_cache import KVPrefixCache, cache_length


def _kv(n):
    c = KVCache()
    c.update_and_fetch(mx.ones((1, 1, n, 4)), mx.ones((1, 1, n, 4)))
    return c


def _tokens(n, base=0):
    return mx.array(list(range(base, base + n)))


def test_hit_never_reports_more_tokens_than_the_snapshot_holds():
    cache = KVPrefixCache(min_prefix_length=4)
    # key says 60 tokens, the snapshot holds 40 (an entry whose buffer is shorter than its key)
    cache.add(_tokens(60), [_kv(40)])
    got, remaining, matched = cache.get(mx.array(list(range(60)) + [999, 998]))
    assert got is not None and matched == 40 and cache_length(got) == 40
    assert remaining == 62 - matched


def test_exact_key_match_over_a_short_snapshot_is_declined():
    cache = KVPrefixCache(min_prefix_length=4)
    cache.add(_tokens(60), [_kv(40)])
    got, remaining, matched = cache.get(_tokens(60))
    assert got is None and matched == 0 and remaining == 60


def test_no_trim_hit_requires_the_snapshot_to_be_exactly_its_key():
    cache = KVPrefixCache(min_prefix_length=4)
    cache._no_trim_mode = True
    cache.add(_tokens(60), [_kv(40)])
    got, remaining, matched = cache.get(mx.array(list(range(60)) + [999, 998]))
    assert got is None and matched == 0 and remaining == 62


def test_no_trim_hit_on_a_consistent_snapshot_is_exact_and_detached():
    cache = KVPrefixCache(min_prefix_length=4)
    cache._no_trim_mode = True
    rec = ArraysCache(1)
    rec.cache = [mx.ones((1, 2, 3))]
    kv = _kv(48)
    cache.add(_tokens(48), [rec, kv])
    query = mx.array(list(range(48)) + [900, 901])
    got, remaining, matched = cache.get(query)
    assert got is not None and matched == 48 and remaining == 2
    assert cache_length([got[1]]) == 48
    got[1].keys[...] = 7
    got[0].cache[0] = got[0].cache[0] * 0
    again, _, m2 = cache.get(query)
    assert m2 == 48
    assert again[1].keys[..., :4, :].tolist() == [[[[1.0] * 4] * 4]]
    assert again[0].cache[0].tolist() == [[[1.0] * 3] * 2]


def test_divergent_query_never_matches_past_the_divergence():
    cache = KVPrefixCache(min_prefix_length=4)
    cache.add(_tokens(100), [_kv(100)])
    q = list(range(100))
    q[70] = 12345  # diverges at 70
    got, remaining, matched = cache.get(mx.array(q + [1, 2, 3]))
    assert got is not None and matched <= 70
    assert cache_length(got) == matched
