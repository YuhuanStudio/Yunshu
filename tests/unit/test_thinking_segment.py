"""Unit tests for ThinkingSegmentSubstore — reasoning token KV cache reuse."""
import time

import pytest

from yunshu_kv.thinking_segment import (
    ThinkingSegment,
    ThinkingSegmentConfig,
    ThinkingSegmentSubstore,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tokens(n: int, start: int = 0) -> list[int]:
    """Return a list of n sequential token IDs starting from *start*."""
    return list(range(start, start + n))


def _dummy_kv(seed: int = 0) -> dict:
    """Deterministic dummy KV data so tests can inspect stored values."""
    return {"tensors": [seed, seed + 1, seed + 2]}


# ---------------------------------------------------------------------------
# TestThinkingSegmentConfig
# ---------------------------------------------------------------------------


class TestThinkingSegmentConfig:
    """Tests for the ThinkingSegmentConfig dataclass defaults and overrides."""

    def test_defaults(self):
        cfg = ThinkingSegmentConfig()
        assert cfg.max_segments_per_conversation == 10
        assert cfg.max_total_segments == 1000
        assert cfg.min_tokens_to_cache == 32
        assert cfg.ttl_seconds == 3600.0
        assert cfg.enable_ssd is False
        assert cfg.ssd_cache_dir == ""

    def test_custom_values(self):
        cfg = ThinkingSegmentConfig(
            max_segments_per_conversation=3,
            max_total_segments=50,
            min_tokens_to_cache=8,
            ttl_seconds=120.0,
            enable_ssd=True,
            ssd_cache_dir="/tmp/ts_cache",
        )
        assert cfg.max_segments_per_conversation == 3
        assert cfg.max_total_segments == 50
        assert cfg.min_tokens_to_cache == 8
        assert cfg.ttl_seconds == 120.0
        assert cfg.enable_ssd is True
        assert cfg.ssd_cache_dir == "/tmp/ts_cache"


# ---------------------------------------------------------------------------
# TestThinkingSegmentSubstore
# ---------------------------------------------------------------------------


class TestThinkingSegmentSubstore:
    """Tests for ThinkingSegmentSubstore core operations."""

    # -- store() basic operation ------------------------------------------------

    def test_store_returns_step_hash(self):
        store = ThinkingSegmentSubstore()
        thinking = _make_tokens(64)
        context = _make_tokens(20)
        kv = _dummy_kv()

        result = store.store("conv-1", thinking, context, kv)

        assert result is not None
        assert isinstance(result, str)
        assert len(result) == 16  # hexdigest()[:16]

    def test_store_persists_segment(self):
        store = ThinkingSegmentSubstore()
        thinking = _make_tokens(64)
        context = _make_tokens(20)
        kv = _dummy_kv(42)

        step_hash = store.store("conv-1", thinking, context, kv)

        segs = store.get_conversation_segments("conv-1")
        assert len(segs) == 1
        assert segs[0].step_hash == step_hash
        assert segs[0].kv_data == kv
        assert segs[0].num_tokens == 64

    def test_store_updates_stats(self):
        store = ThinkingSegmentSubstore()
        store.store("conv-1", _make_tokens(64), _make_tokens(5), _dummy_kv())

        stats = store.get_stats()
        assert stats["stored"] == 1
        assert stats["total_segments"] == 1

    # -- store() skips short thinking ------------------------------------------

    def test_store_skips_short_thinking(self):
        cfg = ThinkingSegmentConfig(min_tokens_to_cache=32)
        store = ThinkingSegmentSubstore(cfg)

        thinking = _make_tokens(16)  # below min_tokens_to_cache
        result = store.store("conv-1", thinking, _make_tokens(5), _dummy_kv())

        assert result is None
        assert store.get_stats()["stored"] == 0
        assert store.get_stats()["total_segments"] == 0

    def test_store_at_exact_min_tokens_is_accepted(self):
        cfg = ThinkingSegmentConfig(min_tokens_to_cache=32)
        store = ThinkingSegmentSubstore(cfg)

        thinking = _make_tokens(32)  # exactly at the boundary
        result = store.store("conv-1", thinking, _make_tokens(5), _dummy_kv())

        assert result is not None
        assert store.get_stats()["stored"] == 1

    # -- store() deduplication --------------------------------------------------

    def test_store_dedup_returns_existing_hash(self):
        store = ThinkingSegmentSubstore()
        thinking = _make_tokens(64)
        context = _make_tokens(10)
        kv_a = _dummy_kv(1)
        kv_b = _dummy_kv(2)

        hash_a = store.store("conv-1", thinking, context, kv_a)
        hash_b = store.store("conv-1", thinking, context, kv_b)

        # Same inputs => same hash returned
        assert hash_a == hash_b
        # No second segment created
        assert store.get_stats()["stored"] == 1
        assert store.get_stats()["total_segments"] == 1

    def test_store_dedup_increments_access_count(self):
        store = ThinkingSegmentSubstore()
        thinking = _make_tokens(64)
        context = _make_tokens(10)

        store.store("conv-1", thinking, context, _dummy_kv())
        store.store("conv-1", thinking, context, _dummy_kv())

        seg = store.get_conversation_segments("conv-1")[0]
        assert seg.access_count >= 1  # dedup touch increments access_count

    # -- lookup() hit and miss -------------------------------------------------

    def test_lookup_hit(self):
        store = ThinkingSegmentSubstore()
        thinking = _make_tokens(64)
        context = _make_tokens(10)
        kv = _dummy_kv(99)

        step_hash = store.store("conv-1", thinking, context, kv)
        result = store.lookup("conv-1", step_hash)

        assert result is not None
        assert result.step_hash == step_hash
        assert result.kv_data == kv

    def test_lookup_miss_unknown_hash(self):
        store = ThinkingSegmentSubstore()
        result = store.lookup("conv-1", "nonexistent_hash")

        assert result is None

    def test_lookup_updates_hit_stats(self):
        store = ThinkingSegmentSubstore()
        thinking = _make_tokens(64)
        context = _make_tokens(10)

        step_hash = store.store("conv-1", thinking, context, _dummy_kv())
        store.lookup("conv-1", step_hash)

        stats = store.get_stats()
        assert stats["hits"] == 1
        assert stats["tokens_saved"] == 64

    def test_lookup_miss_updates_miss_stats(self):
        store = ThinkingSegmentSubstore()
        store.lookup("conv-1", "nope")

        assert store.get_stats()["misses"] == 1

    def test_lookup_increments_access_count(self):
        store = ThinkingSegmentSubstore()
        thinking = _make_tokens(64)
        context = _make_tokens(10)

        step_hash = store.store("conv-1", thinking, context, _dummy_kv())

        seg1 = store.lookup("conv-1", step_hash)
        assert seg1 is not None
        assert seg1.access_count == 1

        seg2 = store.lookup("conv-1", step_hash)
        assert seg2 is not None
        assert seg2.access_count == 2

    # -- lookup() TTL expiration -----------------------------------------------

    def test_lookup_ttl_expired(self):
        cfg = ThinkingSegmentConfig(ttl_seconds=0.01)  # 10 ms TTL
        store = ThinkingSegmentSubstore(cfg)

        thinking = _make_tokens(64)
        context = _make_tokens(10)
        step_hash = store.store("conv-1", thinking, context, _dummy_kv())

        time.sleep(0.05)  # wait for TTL to expire

        result = store.lookup("conv-1", step_hash)
        assert result is None
        assert store.get_stats()["misses"] == 1

    def test_lookup_ttl_not_expired(self):
        cfg = ThinkingSegmentConfig(ttl_seconds=60.0)
        store = ThinkingSegmentSubstore(cfg)

        thinking = _make_tokens(64)
        context = _make_tokens(10)
        step_hash = store.store("conv-1", thinking, context, _dummy_kv())

        # Immediate lookup should succeed
        result = store.lookup("conv-1", step_hash)
        assert result is not None
        assert store.get_stats()["hits"] == 1

    # -- lookup_by_context() exact prefix match --------------------------------

    def test_lookup_by_context_exact_match(self):
        store = ThinkingSegmentSubstore()
        thinking = _make_tokens(64)
        context = _make_tokens(10)

        store.store("conv-1", thinking, context, _dummy_kv())

        result = store.lookup_by_context("conv-1", context, thinking)
        assert result is not None
        assert result.num_tokens == 64

    def test_lookup_by_context_no_match(self):
        store = ThinkingSegmentSubstore()
        thinking = _make_tokens(64)
        context = _make_tokens(10)

        store.store("conv-1", thinking, context, _dummy_kv())

        # Different thinking tokens — won't match
        other_thinking = _make_tokens(64, start=999)
        result = store.lookup_by_context("conv-1", context, other_thinking)
        assert result is None

    def test_lookup_by_context_wrong_conversation(self):
        store = ThinkingSegmentSubstore()
        thinking = _make_tokens(64)
        context = _make_tokens(10)

        store.store("conv-1", thinking, context, _dummy_kv())

        result = store.lookup_by_context("conv-2", context, thinking)
        assert result is None

    def test_lookup_by_context_empty_conversation(self):
        store = ThinkingSegmentSubstore()
        result = store.lookup_by_context("conv-1", _make_tokens(10), _make_tokens(64))
        assert result is None

    # -- get_conversation_segments() -------------------------------------------

    def test_get_conversation_segments_empty(self):
        store = ThinkingSegmentSubstore()
        assert store.get_conversation_segments("conv-1") == []

    def test_get_conversation_segments_returns_all(self):
        store = ThinkingSegmentSubstore()
        ctx_base = _make_tokens(10)

        # Three different thinking segments (different token content)
        store.store("conv-1", _make_tokens(40, start=0), ctx_base, _dummy_kv(1))
        store.store("conv-1", _make_tokens(40, start=100), ctx_base, _dummy_kv(2))
        store.store("conv-1", _make_tokens(40, start=200), ctx_base, _dummy_kv(3))

        segs = store.get_conversation_segments("conv-1")
        assert len(segs) == 3

    def test_get_conversation_segments_isolation(self):
        """Segments from one conversation don't leak into another."""
        store = ThinkingSegmentSubstore()

        store.store("conv-1", _make_tokens(40, start=0), _make_tokens(5), _dummy_kv())
        store.store("conv-2", _make_tokens(40, start=100), _make_tokens(5), _dummy_kv())

        assert len(store.get_conversation_segments("conv-1")) == 1
        assert len(store.get_conversation_segments("conv-2")) == 1

    def test_get_conversation_segments_returns_copy(self):
        """Returned list is a copy — mutations don't affect internal state."""
        store = ThinkingSegmentSubstore()
        store.store("conv-1", _make_tokens(40), _make_tokens(5), _dummy_kv())

        segs = store.get_conversation_segments("conv-1")
        segs.clear()

        assert len(store.get_conversation_segments("conv-1")) == 1

    # -- clear_conversation() --------------------------------------------------

    def test_clear_conversation_returns_count(self):
        store = ThinkingSegmentSubstore()
        ctx_base = _make_tokens(10)

        store.store("conv-1", _make_tokens(40, start=0), ctx_base, _dummy_kv())
        store.store("conv-1", _make_tokens(40, start=100), ctx_base, _dummy_kv())

        removed = store.clear_conversation("conv-1")
        assert removed == 2

    def test_clear_conversation_removes_segments(self):
        store = ThinkingSegmentSubstore()
        ctx_base = _make_tokens(10)

        h = store.store("conv-1", _make_tokens(40), ctx_base, _dummy_kv())
        store.clear_conversation("conv-1")

        assert store.get_conversation_segments("conv-1") == []
        assert store.lookup("conv-1", h) is None

    def test_clear_conversation_updates_total(self):
        store = ThinkingSegmentSubstore()
        ctx_base = _make_tokens(10)

        store.store("conv-1", _make_tokens(40, start=0), ctx_base, _dummy_kv())
        store.store("conv-2", _make_tokens(40, start=100), ctx_base, _dummy_kv())

        store.clear_conversation("conv-1")

        assert store.get_stats()["total_segments"] == 1

    def test_clear_nonexistent_conversation(self):
        store = ThinkingSegmentSubstore()
        removed = store.clear_conversation("ghost")
        assert removed == 0

    # -- get_stats() -----------------------------------------------------------

    def test_get_stats_initial(self):
        store = ThinkingSegmentSubstore()
        stats = store.get_stats()

        assert stats["total_segments"] == 0
        assert stats["conversations_tracked"] == 0
        assert stats["stored"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["evictions"] == 0
        assert stats["tokens_saved"] == 0
        assert stats["hit_rate"] == 0.0

    def test_get_stats_after_operations(self):
        store = ThinkingSegmentSubstore()
        thinking = _make_tokens(64)
        context = _make_tokens(10)

        step_hash = store.store("conv-1", thinking, context, _dummy_kv())
        store.lookup("conv-1", step_hash)   # hit
        store.lookup("conv-1", "nope")       # miss

        stats = store.get_stats()
        assert stats["stored"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_get_stats_hit_rate_zero_denominator(self):
        store = ThinkingSegmentSubstore()
        assert store.get_stats()["hit_rate"] == 0.0

    # -- Per-conversation eviction ----------------------------------------------

    def test_eviction_per_conversation(self):
        cfg = ThinkingSegmentConfig(
            max_segments_per_conversation=2,
            max_total_segments=100,
            min_tokens_to_cache=1,
        )
        store = ThinkingSegmentSubstore(cfg)
        ctx_base = _make_tokens(5)

        # Store 3 segments in the same conversation; limit is 2
        h1 = store.store("conv-1", _make_tokens(10, start=0), ctx_base, _dummy_kv(1))
        time.sleep(0.01)
        h2 = store.store("conv-1", _make_tokens(10, start=100), ctx_base, _dummy_kv(2))
        time.sleep(0.01)
        h3 = store.store("conv-1", _make_tokens(10, start=200), ctx_base, _dummy_kv(3))

        # Only 2 segments remain
        assert store.get_stats()["total_segments"] == 2
        assert store.get_stats()["evictions"] >= 1

        # The oldest (h1) should have been evicted
        assert store.lookup("conv-1", h1) is None
        assert store.lookup("conv-1", h2) is not None
        assert store.lookup("conv-1", h3) is not None

    # -- Global eviction -------------------------------------------------------

    def test_eviction_global_limit(self):
        cfg = ThinkingSegmentConfig(
            max_segments_per_conversation=100,
            max_total_segments=3,
            min_tokens_to_cache=1,
        )
        store = ThinkingSegmentSubstore(cfg)

        # Fill from different conversations
        store.store("conv-1", _make_tokens(10, start=0), _make_tokens(5), _dummy_kv())
        time.sleep(0.01)
        store.store("conv-2", _make_tokens(10, start=100), _make_tokens(5), _dummy_kv())
        time.sleep(0.01)
        store.store("conv-3", _make_tokens(10, start=200), _make_tokens(5), _dummy_kv())

        assert store.get_stats()["total_segments"] == 3

        # Fourth store should trigger global eviction of the oldest
        store.store("conv-4", _make_tokens(10, start=300), _make_tokens(5), _dummy_kv())

        assert store.get_stats()["total_segments"] <= 3
        assert store.get_stats()["evictions"] >= 1

    def test_eviction_global_oldest_removed_first(self):
        cfg = ThinkingSegmentConfig(
            max_segments_per_conversation=100,
            max_total_segments=2,
            min_tokens_to_cache=1,
        )
        store = ThinkingSegmentSubstore(cfg)

        h1 = store.store("conv-1", _make_tokens(10, start=0), _make_tokens(5), _dummy_kv())
        time.sleep(0.01)
        h2 = store.store("conv-2", _make_tokens(10, start=100), _make_tokens(5), _dummy_kv())
        time.sleep(0.01)
        # Third store evicts the globally oldest (h1)
        h3 = store.store("conv-3", _make_tokens(10, start=200), _make_tokens(5), _dummy_kv())

        assert store.lookup("conv-1", h1) is None   # evicted
        assert store.lookup("conv-2", h2) is not None  # survived
        assert store.lookup("conv-3", h3) is not None  # just stored

    # -- compute_step_hash -----------------------------------------------------

    def test_compute_step_hash_deterministic(self):
        store = ThinkingSegmentSubstore()
        thinking = _make_tokens(32)
        context = _make_tokens(16)

        h1 = store.compute_step_hash(thinking, context)
        h2 = store.compute_step_hash(thinking, context)
        assert h1 == h2

    def test_compute_step_hash_different_thinking(self):
        store = ThinkingSegmentSubstore()
        context = _make_tokens(10)

        h1 = store.compute_step_hash(_make_tokens(32, start=0), context)
        h2 = store.compute_step_hash(_make_tokens(32, start=50), context)
        assert h1 != h2

    def test_compute_step_hash_different_context(self):
        store = ThinkingSegmentSubstore()
        thinking = _make_tokens(32)

        h1 = store.compute_step_hash(thinking, _make_tokens(10, start=0))
        h2 = store.compute_step_hash(thinking, _make_tokens(10, start=50))
        assert h1 != h2

    # -- TTL cleanup during eviction -------------------------------------------

    def test_ttl_cleanup_during_store(self):
        cfg = ThinkingSegmentConfig(
            ttl_seconds=0.01,
            max_segments_per_conversation=100,
            max_total_segments=100,
            min_tokens_to_cache=1,
        )
        store = ThinkingSegmentSubstore(cfg)

        h = store.store("conv-1", _make_tokens(10), _make_tokens(5), _dummy_kv())
        assert store.lookup("conv-1", h) is not None

        time.sleep(0.05)  # TTL expired

        # Storing a new segment triggers _maybe_evict which does TTL cleanup
        store.store("conv-1", _make_tokens(10, start=999), _make_tokens(5), _dummy_kv())

        # The expired segment should have been cleaned up
        assert store.lookup("conv-1", h) is None

    # -- Edge cases ------------------------------------------------------------

    def test_store_multiple_conversations(self):
        store = ThinkingSegmentSubstore()
        ctx_base = _make_tokens(10)

        store.store("conv-1", _make_tokens(40, start=0), ctx_base, _dummy_kv())
        store.store("conv-2", _make_tokens(40, start=100), ctx_base, _dummy_kv())
        store.store("conv-3", _make_tokens(40, start=200), ctx_base, _dummy_kv())

        stats = store.get_stats()
        assert stats["total_segments"] == 3
        assert stats["conversations_tracked"] == 3

    def test_clear_one_conversation_preserves_others(self):
        store = ThinkingSegmentSubstore()
        ctx_base = _make_tokens(10)

        store.store("conv-1", _make_tokens(40, start=0), ctx_base, _dummy_kv())
        h2 = store.store("conv-2", _make_tokens(40, start=100), ctx_base, _dummy_kv())

        store.clear_conversation("conv-1")

        assert store.lookup("conv-2", h2) is not None
        assert store.get_stats()["total_segments"] == 1

    def test_thinking_segment_dataclass_defaults(self):
        seg = ThinkingSegment(
            conversation_id="conv-1",
            step_hash="abc123",
            kv_data=_dummy_kv(),
            num_tokens=64,
            thinking_text_hash="def456",
        )
        assert seg.access_count == 0
        assert seg.created_at > 0
        assert seg.last_accessed > 0
        assert seg.created_at == pytest.approx(seg.last_accessed, abs=0.01)
