"""Tests for PromptCacheManager — exact-match KV state cache for prompt reuse."""

import time
import threading

import pytest

from yunshu_engine.prompt_cache import (
    CacheEntry,
    PromptCacheManager,
    PromptCacheStats,
    compute_messages_hash,
)


# ── compute_messages_hash ──


class TestComputeMessagesHash:
    def test_deterministic(self):
        msgs = [{"role": "user", "content": "Hello"}]
        h1 = compute_messages_hash(msgs)
        h2 = compute_messages_hash(msgs)
        assert h1 == h2

    def test_different_messages_different_hash(self):
        msgs_a = [{"role": "user", "content": "Hello"}]
        msgs_b = [{"role": "user", "content": "World"}]
        h_a = compute_messages_hash(msgs_a)
        h_b = compute_messages_hash(msgs_b)
        assert h_a != h_b

    def test_includes_params(self):
        msgs = [{"role": "user", "content": "Hello"}]
        h1 = compute_messages_hash(msgs, model="m1")
        h2 = compute_messages_hash(msgs, model="m2")
        assert h1 != h2

    def test_params_order_invariant(self):
        msgs = [{"role": "user", "content": "Hello"}]
        h1 = compute_messages_hash(msgs, a=1, b=2)
        h2 = compute_messages_hash(msgs, b=2, a=1)
        assert h1 == h2

    def test_empty_messages(self):
        h = compute_messages_hash([])
        assert isinstance(h, str)
        assert len(h) > 0

    def test_hash_is_hex_string(self):
        h = compute_messages_hash([{"role": "user", "content": "test"}])
        assert all(c in "0123456789abcdef" for c in h)

    def test_no_params_same_as_empty_params(self):
        msgs = [{"role": "user", "content": "test"}]
        h1 = compute_messages_hash(msgs)
        h2 = compute_messages_hash(msgs)
        assert h1 == h2


# ── CacheEntry ──


class TestCacheEntry:
    def test_fields(self):
        entry = CacheEntry(
            key="abc",
            kv_state=[1, 2, 3],
            messages_hash="abc",
            size_bytes=1024,
            token_count=100,
        )
        assert entry.key == "abc"
        assert entry.kv_state == [1, 2, 3]
        assert entry.size_bytes == 1024
        assert entry.token_count == 100
        assert entry.access_count == 0

    def test_default_timestamps(self):
        before = time.monotonic()
        entry = CacheEntry(key="k", kv_state=None, messages_hash="k")
        after = time.monotonic()
        assert before <= entry.created_at <= after
        assert entry.last_accessed >= entry.created_at


# ── PromptCacheManager — store / lookup ──


class TestStoreAndLookup:
    def test_store_and_lookup(self):
        cache = PromptCacheManager(max_entries=10, max_memory_mb=1.0)
        h = "hash-1"
        assert cache.store(h, kv_state=[1, 2, 3], token_count=10, size_bytes=100) is True
        entry = cache.lookup(h)
        assert entry is not None
        assert entry.kv_state == [1, 2, 3]
        assert entry.token_count == 10

    def test_lookup_miss(self):
        cache = PromptCacheManager()
        result = cache.lookup("nonexistent")
        assert result is None

    def test_store_overwrites_existing(self):
        cache = PromptCacheManager(max_entries=10, max_memory_mb=1.0)
        h = "hash-1"
        cache.store(h, kv_state="old", size_bytes=100)
        cache.store(h, kv_state="new", size_bytes=200)
        entry = cache.lookup(h)
        assert entry is not None
        assert entry.kv_state == "new"
        assert cache.get_memory_usage_bytes() == 200

    def test_store_rejects_oversized(self):
        cache = PromptCacheManager(max_memory_mb=0.001)  # 1KB
        result = cache.store("h", kv_state="x", size_bytes=10_000_000)
        assert result is False

    def test_has(self):
        cache = PromptCacheManager(max_entries=10, max_memory_mb=1.0)
        cache.store("h1", kv_state=None, size_bytes=100)
        assert cache.has("h1") is True
        assert cache.has("h2") is False


# ── LRU eviction ──


class TestLRUEviction:
    def test_evicts_on_capacity(self):
        cache = PromptCacheManager(max_entries=3, max_memory_mb=100.0)
        cache.store("h1", kv_state="a", size_bytes=100)
        cache.store("h2", kv_state="b", size_bytes=100)
        cache.store("h3", kv_state="c", size_bytes=100)
        # Adding 4th should evict the LRU (h1)
        cache.store("h4", kv_state="d", size_bytes=100)
        assert cache.has("h1") is False
        assert cache.has("h4") is True
        assert cache.get_entry_count() == 3

    def test_lru_order_updated_on_lookup(self):
        cache = PromptCacheManager(max_entries=3, max_memory_mb=100.0)
        cache.store("h1", kv_state="a", size_bytes=100)
        cache.store("h2", kv_state="b", size_bytes=100)
        cache.store("h3", kv_state="c", size_bytes=100)
        # Access h1 to make it recently used
        cache.lookup("h1")
        # h2 is now LRU
        cache.store("h4", kv_state="d", size_bytes=100)
        assert cache.has("h1") is True  # accessed, not evicted
        assert cache.has("h2") is False  # LRU, evicted

    def test_evicts_on_memory_pressure(self):
        cache = PromptCacheManager(max_entries=100, max_memory_mb=0.001)  # 1KB
        cache.store("h1", kv_state="a", size_bytes=500)
        cache.store("h2", kv_state="b", size_bytes=500)
        # Now memory is 1000/1000. Storing h3 (500 bytes) should trigger eviction
        cache.store("h3", kv_state="c", size_bytes=500)
        # h1 should be evicted (LRU)
        assert cache.has("h1") is False
        assert cache.has("h2") is True
        assert cache.has("h3") is True


# ── TTL expiration ──


class TestTTLExpiration:
    def test_expired_entry_returns_none(self):
        cache = PromptCacheManager(max_entries=10, max_memory_mb=1.0, ttl_seconds=0.01)
        cache.store("h1", kv_state="data", size_bytes=100)
        time.sleep(0.02)
        entry = cache.lookup("h1")
        assert entry is None

    def test_non_expired_entry_works(self):
        cache = PromptCacheManager(max_entries=10, max_memory_mb=1.0, ttl_seconds=60.0)
        cache.store("h1", kv_state="data", size_bytes=100)
        entry = cache.lookup("h1")
        assert entry is not None

    def test_prune_expired(self):
        cache = PromptCacheManager(max_entries=10, max_memory_mb=1.0, ttl_seconds=0.01)
        cache.store("h1", kv_state="a", size_bytes=100)
        cache.store("h2", kv_state="b", size_bytes=100)
        time.sleep(0.02)
        pruned = cache.prune_expired()
        assert pruned == 2
        assert cache.get_entry_count() == 0

    def test_prune_mixed(self):
        cache = PromptCacheManager(max_entries=10, max_memory_mb=1.0, ttl_seconds=0.05)
        cache.store("h1", kv_state="a", size_bytes=100)
        time.sleep(0.06)
        cache.store("h2", kv_state="b", size_bytes=100)
        pruned = cache.prune_expired()
        assert pruned == 1
        assert cache.has("h1") is False
        assert cache.has("h2") is True


# ── invalidate ──


class TestInvalidate:
    def test_invalidate_existing(self):
        cache = PromptCacheManager(max_entries=10, max_memory_mb=1.0)
        cache.store("h1", kv_state="a", size_bytes=100)
        result = cache.invalidate("h1")
        assert result is True
        assert cache.has("h1") is False
        assert cache.get_memory_usage_bytes() == 0

    def test_invalidate_nonexistent(self):
        cache = PromptCacheManager()
        result = cache.invalidate("no-such-key")
        assert result is False

    def test_invalidate_all(self):
        cache = PromptCacheManager(max_entries=10, max_memory_mb=1.0)
        cache.store("h1", kv_state="a", size_bytes=100)
        cache.store("h2", kv_state="b", size_bytes=100)
        count = cache.invalidate_all()
        assert count == 2
        assert cache.get_entry_count() == 0
        assert cache.get_memory_usage_bytes() == 0

    def test_invalidate_updates_stats(self):
        cache = PromptCacheManager(max_entries=10, max_memory_mb=1.0)
        cache.store("h1", kv_state="a", size_bytes=100)
        cache.invalidate("h1")
        stats = cache.get_stats()
        assert stats["invalidations"] == 1


# ── access metadata ──


class TestAccessMetadata:
    def test_access_count_increments(self):
        cache = PromptCacheManager(max_entries=10, max_memory_mb=1.0)
        cache.store("h1", kv_state="a", size_bytes=100)
        cache.lookup("h1")
        cache.lookup("h1")
        entry = cache.lookup("h1")
        assert entry.access_count == 3  # 3 lookups

    def test_last_accessed_updates(self):
        cache = PromptCacheManager(max_entries=10, max_memory_mb=1.0)
        cache.store("h1", kv_state="a", size_bytes=100)
        t1 = cache.lookup("h1").last_accessed
        time.sleep(0.01)
        t2 = cache.lookup("h1").last_accessed
        assert t2 >= t1


# ── thread safety ──


class TestThreadSafety:
    def test_concurrent_store_lookup(self):
        cache = PromptCacheManager(max_entries=100, max_memory_mb=100.0)
        errors = []

        def worker(thread_id):
            try:
                for i in range(50):
                    h = f"hash-{thread_id}-{i}"
                    cache.store(h, kv_state=f"data-{i}", size_bytes=100)
                    entry = cache.lookup(h)
                    assert entry is not None
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert cache.get_entry_count() > 0

    def test_concurrent_invalidate(self):
        cache = PromptCacheManager(max_entries=100, max_memory_mb=100.0)
        for i in range(50):
            cache.store(f"h-{i}", kv_state=f"v-{i}", size_bytes=100)

        errors = []

        def invalidator():
            try:
                for i in range(50):
                    cache.invalidate(f"h-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=invalidator) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ── stats ──


class TestGetStats:
    def test_initial_stats(self):
        cache = PromptCacheManager()
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["stores"] == 0
        assert stats["evictions"] == 0
        assert stats["entries"] == 0

    def test_hit_rate_after_operations(self):
        cache = PromptCacheManager(max_entries=10, max_memory_mb=1.0)
        cache.store("h1", kv_state="a", size_bytes=100)
        cache.lookup("h1")  # hit
        cache.lookup("h1")  # hit
        cache.lookup("h2")  # miss
        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_memory_utilization(self):
        cache = PromptCacheManager(max_entries=10, max_memory_mb=0.001)  # ~1KB
        cache.store("h1", kv_state="a", size_bytes=500)
        stats = cache.get_stats()
        assert stats["memory_usage_bytes"] == 500
        expected_max = int(0.001 * 1024 * 1024)
        assert stats["max_memory_bytes"] == expected_max
        assert stats["memory_utilization"] == pytest.approx(500.0 / expected_max, abs=0.01)

    def test_stats_after_eviction(self):
        cache = PromptCacheManager(max_entries=2, max_memory_mb=100.0)
        cache.store("h1", kv_state="a", size_bytes=100)
        cache.store("h2", kv_state="b", size_bytes=100)
        cache.store("h3", kv_state="c", size_bytes=100)  # evicts h1
        stats = cache.get_stats()
        assert stats["evictions"] == 1
        assert stats["stores"] == 3


# ── size estimation ──


class TestSizeEstimation:
    def test_estimate_none(self):
        assert PromptCacheManager._estimate_size(None) == 0

    def test_estimate_list(self):
        size = PromptCacheManager._estimate_size([1, 2, 3])
        assert size == 3 * 1024  # default per-item

    def test_estimate_fallback(self):
        size = PromptCacheManager._estimate_size("some string")
        assert size == 1024  # fallback
