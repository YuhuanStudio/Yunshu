"""Tests for the Thinking Segment KV Substore."""


class TestThinkingSegmentSubstore:
    """Test the ThinkingSegmentSubstore directly."""

    def test_store_and_lookup(self):
        from yunshu_kv.thinking_segment import (
            ThinkingSegmentConfig,
            ThinkingSegmentSubstore,
        )

        store = ThinkingSegmentSubstore(ThinkingSegmentConfig(min_tokens_to_cache=4))
        # Store a segment with 10 thinking tokens
        thinking = list(range(10))
        context = list(range(20, 30))
        step_hash = store.store("conv1", thinking, context, kv_data="mock_kv")
        assert step_hash is not None
        assert len(step_hash) == 16

        # Lookup by hash
        result = store.lookup("conv1", step_hash)
        assert result is not None
        assert result.num_tokens == 10
        assert result.conversation_id == "conv1"

    def test_store_skips_short_thinking(self):
        from yunshu_kv.thinking_segment import (
            ThinkingSegmentConfig,
            ThinkingSegmentSubstore,
        )

        store = ThinkingSegmentSubstore(ThinkingSegmentConfig(min_tokens_to_cache=32))
        # Only 5 tokens — below threshold
        step_hash = store.store("conv1", list(range(5)), list(range(10)), kv_data="kv")
        assert step_hash is None

    def test_lookup_by_context(self):
        from yunshu_kv.thinking_segment import (
            ThinkingSegmentConfig,
            ThinkingSegmentSubstore,
        )

        store = ThinkingSegmentSubstore(ThinkingSegmentConfig(min_tokens_to_cache=4))
        thinking = list(range(10))
        context = list(range(20, 30))
        store.store("conv1", thinking, context, kv_data="mock_kv")

        # Lookup by context prefix
        result = store.lookup_by_context("conv1", context, thinking)
        assert result is not None

    def test_get_conversation_segments(self):
        from yunshu_kv.thinking_segment import (
            ThinkingSegmentConfig,
            ThinkingSegmentSubstore,
        )

        store = ThinkingSegmentSubstore(ThinkingSegmentConfig(min_tokens_to_cache=4))
        store.store("conv1", list(range(10)), list(range(5)), kv_data="kv1")
        store.store("conv1", list(range(20, 30)), list(range(5)), kv_data="kv2")

        segments = store.get_conversation_segments("conv1")
        assert len(segments) == 2

    def test_clear_conversation(self):
        from yunshu_kv.thinking_segment import (
            ThinkingSegmentConfig,
            ThinkingSegmentSubstore,
        )

        store = ThinkingSegmentSubstore(ThinkingSegmentConfig(min_tokens_to_cache=4))
        store.store("conv1", list(range(10)), list(range(5)), kv_data="kv")
        count = store.clear_conversation("conv1")
        assert count == 1
        assert store.get_conversation_segments("conv1") == []

    def test_stats_tracking(self):
        from yunshu_kv.thinking_segment import (
            ThinkingSegmentConfig,
            ThinkingSegmentSubstore,
        )

        store = ThinkingSegmentSubstore(ThinkingSegmentConfig(min_tokens_to_cache=4))
        store.store("conv1", list(range(10)), list(range(5)), kv_data="kv")
        store.lookup_by_context("conv1", list(range(5)), list(range(10)))

        stats = store.get_stats()
        assert stats["stored"] == 1
        assert stats["hits"] == 1
        assert stats["total_segments"] == 1

    def test_ttl_expiry(self):
        import time

        from yunshu_kv.thinking_segment import (
            ThinkingSegmentConfig,
            ThinkingSegmentSubstore,
        )

        store = ThinkingSegmentSubstore(
            ThinkingSegmentConfig(
                min_tokens_to_cache=4,
                ttl_seconds=0.01,  # 10ms TTL
            )
        )
        step_hash = store.store("conv1", list(range(10)), list(range(5)), kv_data="kv")
        assert step_hash is not None

        time.sleep(0.02)  # Wait for expiry
        result = store.lookup("conv1", step_hash)
        assert result is None

    def test_eviction_per_conversation(self):
        from yunshu_kv.thinking_segment import (
            ThinkingSegmentConfig,
            ThinkingSegmentSubstore,
        )

        store = ThinkingSegmentSubstore(
            ThinkingSegmentConfig(
                min_tokens_to_cache=4,
                max_segments_per_conversation=2,
            )
        )
        store.store("conv1", list(range(10)), list(range(5)), kv_data="kv1")
        store.store("conv1", list(range(20, 30)), list(range(5)), kv_data="kv2")
        # Third store should trigger eviction of oldest
        store.store("conv1", list(range(30, 40)), list(range(5)), kv_data="kv3")

        segments = store.get_conversation_segments("conv1")
        assert len(segments) <= 2
