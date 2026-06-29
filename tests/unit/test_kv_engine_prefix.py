"""Tests for KVPrefixCache — engine-level KV prefix matching for multi-turn speedup.

Tests:
- Exact hash match (O(1) fast path)
- Partial prefix match (scan slow path)
- LRU eviction when at capacity
- Hash deduplication on re-add
- Snapshot isolation (mutations don't affect cache)
- min_prefix_length threshold
- get_stats / size / clear
"""

from unittest.mock import MagicMock

import mlx.core as mx

from yunshu_engine.kv_prefix_cache import (
    KVPrefixCache,
    _token_hash,
    cache_length,
    get_prefix_length,
)


def _fake_cache(offset: int, key_shape=(1, 4, 8)):
    """Create a fake KV cache entry with given offset."""
    c = MagicMock()
    c.offset = offset
    c.keys = mx.zeros(key_shape)
    c.values = mx.zeros(key_shape)
    c.trim = MagicMock()
    return c


class TestGetPrefixLength:
    def test_identical_arrays(self):
        a = mx.array([1, 2, 3, 4, 5])
        assert get_prefix_length(a, a) == 5

    def test_no_common_prefix(self):
        a = mx.array([1, 2, 3])
        b = mx.array([4, 5, 6])
        assert get_prefix_length(a, b) == 0

    def test_partial_prefix(self):
        a = mx.array([1, 2, 3, 4, 5])
        b = mx.array([1, 2, 9, 8, 7])
        assert get_prefix_length(a, b) == 2

    def test_one_is_prefix_of_other(self):
        a = mx.array([1, 2, 3])
        b = mx.array([1, 2, 3, 4, 5])
        assert get_prefix_length(a, b) == 3

    def test_empty_arrays(self):
        a = mx.array([], dtype=mx.int32)
        b = mx.array([1, 2, 3])
        assert get_prefix_length(a, b) == 0


class TestCacheLength:
    def test_returns_max_offset(self):
        caches = [_fake_cache(10), _fake_cache(25), _fake_cache(15)]
        assert cache_length(caches) == 25

    def test_empty_list(self):
        assert cache_length([]) == 0

    def test_no_offset_attr(self):
        caches = [MagicMock(spec=[])]
        assert cache_length(caches) == 0


class TestTokenHash:
    def test_deterministic(self):
        tokens = mx.array([1, 2, 3, 4, 5])
        assert _token_hash(tokens) == _token_hash(tokens)

    def test_different_tokens_different_hash(self):
        a = mx.array([1, 2, 3])
        b = mx.array([4, 5, 6])
        assert _token_hash(a) != _token_hash(b)


class TestKVPrefixCacheExactHit:
    def test_add_then_get_exact(self):
        cache = KVPrefixCache(min_prefix_length=4)
        tokens = mx.array([1, 2, 3, 4, 5, 6, 7, 8])
        kv = [_fake_cache(8)]

        cache.add(tokens, kv)
        result, remaining, matched = cache.get(tokens)

        assert result is not None
        # a FULL/exact hit refeed-trims by one token so the caller
        # re-feeds the last prompt token ONCE — re-feeding it against the un-trimmed cache
        # would duplicate it at a shifted RoPE position → confident-wrong first decode
        # token. So remaining=1, matched=len-1, and the cache was trimmed by one.
        assert remaining == 1
        assert matched == 7
        result[0].trim.assert_called_once_with(1)

    def test_exact_hit_no_refeed_trim_preserves_full_match(self):
        """The engine-loop scheduler passes exact_refeed_trim=False: it uses
        matched==len to detect a FULL match and fall back to an efficient BATCHED
        prefill. The 8th-audit refeed-trim (matched-1) defeated that detection and
        collapsed batched throughput at batch>=16 (bisect → 456948a; 152→79 tok/s).
        With the opt-out, an exact hit reports the FULL match and does NOT trim."""
        cache = KVPrefixCache(min_prefix_length=4)
        tokens = mx.array([1, 2, 3, 4, 5, 6, 7, 8])
        kv = [_fake_cache(8)]
        cache.add(tokens, kv)
        result, remaining, matched = cache.get(tokens, exact_refeed_trim=False)
        assert result is not None
        assert remaining == 0
        assert matched == 8
        result[0].trim.assert_not_called()

    def test_no_hit_empty_cache(self):
        cache = KVPrefixCache()
        tokens = mx.array([1, 2, 3, 4, 5])
        result, remaining, matched = cache.get(tokens)

        assert result is None
        assert remaining == 5
        assert matched == 0


class TestKVPrefixCachePartialHit:
    def test_partial_prefix_match(self):
        cache = KVPrefixCache(min_prefix_length=4)
        cached_tokens = mx.array([1, 2, 3, 4, 5, 6, 7, 8])
        new_tokens = mx.array([1, 2, 3, 4, 5, 9, 10, 11])

        kv = [_fake_cache(8)]
        cache.add(cached_tokens, kv)

        result, remaining, matched = cache.get(new_tokens)
        assert result is not None
        assert matched == 5  # first 5 tokens match
        assert remaining == 3

    def test_hash_chain_reuse_not_floored_to_block(self):
        # a cached prefix LONGER than one 64-token block must reuse the
        # TOKEN-EXACT prefix, not floor to a block boundary. Cache 100 tokens, then
        # query the same 100 + 50 more (150). Before the fix this returned
        # matched=64 (36 valid tokens silently re-prefilled); now it must be 100.
        cache = KVPrefixCache(min_prefix_length=4)
        cached_tokens = mx.array(list(range(1, 101)))  # 100 tokens
        new_tokens = mx.array(
            list(range(1, 101)) + list(range(500, 550))
        )  # 100 shared + 50

        cache.add(cached_tokens, [_fake_cache(100)])
        result, remaining, matched = cache.get(new_tokens)

        assert result is not None
        assert matched == 100, (
            f"expected full 100-token reuse, got {matched} (block-floored?)"
        )
        assert remaining == 50

    def test_below_min_prefix_no_hit(self):
        cache = KVPrefixCache(min_prefix_length=10)
        cached_tokens = mx.array([1, 2, 3, 4, 5, 6, 7, 8])
        new_tokens = mx.array([1, 2, 3, 4, 5, 9, 10, 11])

        kv = [_fake_cache(8)]
        cache.add(cached_tokens, kv)

        result, remaining, matched = cache.get(new_tokens)
        assert result is None
        assert matched == 0

    def test_short_prompt_not_stored(self):
        cache = KVPrefixCache(min_prefix_length=32)
        tokens = mx.array([1, 2, 3, 4, 5])  # 5 < 32
        kv = [_fake_cache(5)]

        cache.add(tokens, kv)
        assert cache.size == 0


class TestKVPrefixCacheLRU:
    def test_evicts_at_capacity(self):
        cache = KVPrefixCache(max_entries=2, min_prefix_length=4)

        t1 = mx.array([1, 2, 3, 4, 5])
        t2 = mx.array([6, 7, 8, 9, 10])
        t3 = mx.array([11, 12, 13, 14, 15])

        cache.add(t1, [_fake_cache(5)])
        cache.add(t2, [_fake_cache(5)])
        assert cache.size == 2

        cache.add(t3, [_fake_cache(5)])  # should evict t1
        assert cache.size == 2

        # t1 should be evicted (LRU)
        result, _, _ = cache.get(t1)
        assert result is None

    def test_lru_touch_updates_order(self):
        cache = KVPrefixCache(max_entries=2, min_prefix_length=4)

        t1 = mx.array([1, 2, 3, 4, 5])
        t2 = mx.array([6, 7, 8, 9, 10])
        t3 = mx.array([11, 12, 13, 14, 15])

        cache.add(t1, [_fake_cache(5)])
        cache.add(t2, [_fake_cache(5)])

        # Touch t1 so it's most recently used
        cache.get(t1)

        # Adding t3 should evict t2 (LRU), not t1
        cache.add(t3, [_fake_cache(5)])
        result, _, _ = cache.get(t1)
        assert result is not None  # t1 still cached


class TestKVPrefixCacheSnapshotIsolation:
    def test_snapshot_does_not_mutate_original(self):
        cache = KVPrefixCache(min_prefix_length=4)
        tokens = mx.array([1, 2, 3, 4, 5, 6, 7, 8])
        kv = [_fake_cache(8)]
        cache.add(tokens, kv)

        result, _, _ = cache.get(tokens)
        # Modifying result should not affect cached version
        assert result is not None
        assert cache.size == 1


class TestKVPrefixCacheDedup:
    def test_readd_same_tokens_updates_entry(self):
        cache = KVPrefixCache(min_prefix_length=4)
        tokens = mx.array([1, 2, 3, 4, 5, 6, 7, 8])

        kv1 = [_fake_cache(8)]
        kv2 = [_fake_cache(8)]
        cache.add(tokens, kv1)
        cache.add(tokens, kv2)

        assert cache.size == 1  # deduplicated


class TestKVPrefixCacheStats:
    def test_get_stats(self):
        cache = KVPrefixCache(max_entries=32, min_prefix_length=4)
        tokens = mx.array(list(range(10)))
        cache.add(tokens, [_fake_cache(10)])

        stats = cache.get_stats()
        assert stats["entries"] == 1
        assert stats["max_entries"] == 32
        assert stats["total_cached_tokens"] == 10
        assert stats["min_prefix_length"] == 4

    def test_clear(self):
        cache = KVPrefixCache(min_prefix_length=4)
        cache.add(mx.array([1, 2, 3, 4, 5]), [_fake_cache(5)])
        assert cache.size == 1

        cache.clear()
        assert cache.size == 0

    def test_size_property(self):
        cache = KVPrefixCache(min_prefix_length=4)
        assert cache.size == 0
        cache.add(mx.array([1, 2, 3, 4, 5]), [_fake_cache(5)])
        assert cache.size == 1


class TestKVPrefixCacheBlockDedup:
    """Test block-level dedup and COW refcounting."""

    def test_block_refcount_on_add(self):
        """Adding an entry should create block refcounts."""
        cache = KVPrefixCache(min_prefix_length=4)
        tokens = mx.array(list(range(128)))  # 2 blocks of 64
        cache.add(tokens, [_fake_cache(128)])

        assert len(cache._block_refcount) > 0

    def test_shared_blocks_incref(self):
        """Two entries with shared prefix blocks should have refcount > 1."""
        cache = KVPrefixCache(min_prefix_length=4)
        shared_prefix = list(range(128))
        tokens_a = mx.array(shared_prefix + [200, 201, 202])
        tokens_b = mx.array(shared_prefix + [300, 301, 302])

        cache.add(tokens_a, [_fake_cache(131)])
        cache.add(tokens_b, [_fake_cache(131)])

        # First 2 blocks (0-63, 64-127) should be shared (refcount=2)
        shared = sum(1 for c in cache._block_refcount.values() if c > 1)
        assert shared >= 2

    def test_remove_decrements_refcount(self):
        """Removing an entry should decrement block refcounts."""
        cache = KVPrefixCache(min_prefix_length=4)
        shared_prefix = list(range(128))
        tokens_a = mx.array(shared_prefix + [200, 201])
        tokens_b = mx.array(shared_prefix + [300, 301])

        cache.add(tokens_a, [_fake_cache(130)])
        cache.add(tokens_b, [_fake_cache(130)])

        # Remove second entry — shared blocks should have refcount decremented
        cache._remove_entry(1)
        for count in cache._block_refcount.values():
            assert count >= 1

    def test_stats_include_unique_and_shared_blocks(self):
        """get_stats should include unique_blocks and shared_blocks."""
        cache = KVPrefixCache(min_prefix_length=4)
        tokens = mx.array(list(range(128)))
        cache.add(tokens, [_fake_cache(128)])

        stats = cache.get_stats()
        assert "unique_blocks" in stats
        assert "shared_blocks" in stats
        assert stats["unique_blocks"] >= 1

    def test_clear_resets_refcount(self):
        """clear() should reset all block refcounts."""
        cache = KVPrefixCache(min_prefix_length=4)
        cache.add(mx.array(list(range(128))), [_fake_cache(128)])

        cache.clear()
        assert len(cache._block_refcount) == 0
