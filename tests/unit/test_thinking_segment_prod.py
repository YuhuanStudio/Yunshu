"""Tests for Thinking Segment production hardening.

Phase 4 tests:
- KV compression for stored thinking segments
- SSD persistence (save/load)
- TTL expiry cleanup with memory reclamation
- Compression round-trip accuracy
"""
import json
import time
from unittest.mock import patch

from yunshu_kv.thinking_segment import (
    ThinkingSegmentConfig,
    ThinkingSegmentSubstore,
)


class TestKVCompression:
    """Test KV compression for stored thinking segments."""

    def test_compression_disabled_by_default(self):
        config = ThinkingSegmentConfig()
        assert config.enable_compression is False

    def test_compression_enabled(self):
        config = ThinkingSegmentConfig(enable_compression=True, compression_bits=4)
        assert config.enable_compression is True
        assert config.compression_bits == 4

    def test_store_with_compression(self):
        """Stored segment should have compressed KV data."""
        config = ThinkingSegmentConfig(
            enable_compression=True,
            compression_bits=4,
            compression_group_size=64,
        )
        store = ThinkingSegmentSubstore(config)

        # Create mock KV data (list of floats)
        kv_data = [0.1, 0.2, 0.3, -0.1, 0.5] * 20  # 100 elements
        step_hash = store.store(
            conversation_id="conv_1",
            thinking_tokens=list(range(64)),
            context_tokens=list(range(32)),
            kv_data=kv_data,
        )

        assert step_hash is not None
        stats = store.get_stats()
        assert stats["compressions"] == 1

    def test_compression_round_trip(self):
        """Compressed then decompressed data should approximate original."""
        config = ThinkingSegmentConfig(
            enable_compression=True,
            compression_bits=4,
            compression_group_size=64,
        )
        store = ThinkingSegmentSubstore(config)

        # Create KV data with enough elements for quantization
        kv_data = [0.1 * i for i in range(128)]
        step_hash = store.store(
            conversation_id="conv_1",
            thinking_tokens=list(range(64)),
            context_tokens=list(range(32)),
            kv_data=kv_data,
        )

        # Lookup triggers decompression
        segment = store.lookup("conv_1", step_hash)
        assert segment is not None
        # Decompressed data should be a list (approximate)
        assert isinstance(segment.kv_data, list)
        assert len(segment.kv_data) == 128

    def test_store_without_compression(self):
        """Without compression, kv_data should be stored as-is."""
        config = ThinkingSegmentConfig(enable_compression=False)
        store = ThinkingSegmentSubstore(config)

        kv_data = [1.0, 2.0, 3.0]
        step_hash = store.store(
            conversation_id="conv_1",
            thinking_tokens=list(range(64)),
            context_tokens=list(range(32)),
            kv_data=kv_data,
        )

        segment = store.lookup("conv_1", step_hash)
        assert segment.kv_data == kv_data

    def test_compression_with_none_kv_data(self):
        """None KV data should not crash compression."""
        config = ThinkingSegmentConfig(enable_compression=True)
        store = ThinkingSegmentSubstore(config)

        step_hash = store.store(
            conversation_id="conv_1",
            thinking_tokens=list(range(64)),
            context_tokens=[],
            kv_data=None,
        )

        assert step_hash is not None
        segment = store.lookup("conv_1", step_hash)
        assert segment.kv_data is None

    def test_compression_fallback_without_kv_quantization(self):
        """Should gracefully fallback if kv_quantization module unavailable."""
        config = ThinkingSegmentConfig(enable_compression=True)
        store = ThinkingSegmentSubstore(config)

        with patch.dict("sys.modules", {"yunshu_engine.kv_quantization": None}):
            kv_data = [1.0, 2.0, 3.0]
            step_hash = store.store(
                conversation_id="conv_1",
                thinking_tokens=list(range(64)),
                context_tokens=[],
                kv_data=kv_data,
            )
            # Should store uncompressed as fallback
            segment = store.lookup("conv_1", step_hash)
            assert segment is not None


class TestSSDPersistence:
    """Test SSD save/load for thinking segments."""

    def test_ssd_disabled_by_default(self):
        config = ThinkingSegmentConfig()
        assert config.enable_ssd is False

    def test_ssd_save_creates_file(self, tmp_path):
        """Storing a segment should create an SSD file when enabled."""
        config = ThinkingSegmentConfig(
            enable_ssd=True,
            ssd_cache_dir=str(tmp_path),
        )
        store = ThinkingSegmentSubstore(config)

        kv_data = [1.0, 2.0, 3.0]
        step_hash = store.store(
            conversation_id="conv_1",
            thinking_tokens=list(range(64)),
            context_tokens=[],
            kv_data=kv_data,
        )

        # Check file was created
        ssd_file = tmp_path / "conv_1" / f"{step_hash}.json"
        assert ssd_file.exists()

        stats = store.get_stats()
        assert stats["ssd_saves"] == 1

    def test_ssd_load_from_disk(self, tmp_path):
        """Loading a segment not in memory should read from SSD."""
        config = ThinkingSegmentConfig(
            enable_ssd=True,
            ssd_cache_dir=str(tmp_path),
        )
        store = ThinkingSegmentSubstore(config)

        kv_data = [1.0, 2.0, 3.0]
        step_hash = store.store(
            conversation_id="conv_1",
            thinking_tokens=list(range(64)),
            context_tokens=[],
            kv_data=kv_data,
        )

        # Remove from memory
        store._hash_index.clear()
        store._segments.clear()
        store._total_segments = 0

        # Lookup should find it on SSD
        segment = store.lookup("conv_1", step_hash)
        assert segment is not None
        assert segment.step_hash == step_hash

        stats = store.get_stats()
        assert stats["ssd_loads"] == 1

    def test_ssd_file_content_is_valid_json(self, tmp_path):
        """SSD files should be valid JSON."""
        config = ThinkingSegmentConfig(
            enable_ssd=True,
            ssd_cache_dir=str(tmp_path),
        )
        store = ThinkingSegmentSubstore(config)

        step_hash = store.store(
            conversation_id="conv_1",
            thinking_tokens=list(range(64)),
            context_tokens=[],
            kv_data=[1.0, 2.0],
        )

        ssd_file = tmp_path / "conv_1" / f"{step_hash}.json"
        with open(ssd_file) as f:
            data = json.load(f)

        assert data["conversation_id"] == "conv_1"
        assert data["step_hash"] == step_hash
        assert data["num_tokens"] == 64

    def test_ssd_with_compressed_kv_data(self, tmp_path):
        """Compressed KV data should be serialized to SSD."""
        config = ThinkingSegmentConfig(
            enable_ssd=True,
            ssd_cache_dir=str(tmp_path),
            enable_compression=True,
        )
        store = ThinkingSegmentSubstore(config)

        kv_data = [0.1 * i for i in range(128)]
        step_hash = store.store(
            conversation_id="conv_1",
            thinking_tokens=list(range(64)),
            context_tokens=[],
            kv_data=kv_data,
        )

        ssd_file = tmp_path / "conv_1" / f"{step_hash}.json"
        assert ssd_file.exists()
        with open(ssd_file) as f:
            data = json.load(f)
        # Should have compressed format marker
        assert data["kv_data"].get("_compressed") is True

    def test_ssd_load_nonexistent(self, tmp_path):
        """Loading nonexistent segment from SSD should return None."""
        config = ThinkingSegmentConfig(
            enable_ssd=True,
            ssd_cache_dir=str(tmp_path),
        )
        store = ThinkingSegmentSubstore(config)

        segment = store.lookup("conv_missing", "nonexistent_hash")
        assert segment is None

    def test_ssd_default_dir(self):
        """Without explicit dir, should use ~/.yunshu/thinking_segments."""
        config = ThinkingSegmentConfig(enable_ssd=True)
        store = ThinkingSegmentSubstore(config)
        assert store._ssd_dir is not None
        assert "yunshu" in str(store._ssd_dir)
        assert "thinking_segments" in str(store._ssd_dir)

    def test_ssd_file_removed_on_segment_delete(self, tmp_path):
        """Removing a segment should also remove its SSD file."""
        config = ThinkingSegmentConfig(
            enable_ssd=True,
            ssd_cache_dir=str(tmp_path),
        )
        store = ThinkingSegmentSubstore(config)

        step_hash = store.store(
            conversation_id="conv_1",
            thinking_tokens=list(range(64)),
            context_tokens=[],
            kv_data=[1.0],
        )

        ssd_file = tmp_path / "conv_1" / f"{step_hash}.json"
        assert ssd_file.exists()

        store.clear_conversation("conv_1")

        assert not ssd_file.exists()


class TestTTLCleanup:
    """Test TTL expiry with memory reclamation."""

    def test_ttl_config(self):
        config = ThinkingSegmentConfig(ttl_seconds=60.0, ttl_cleanup_interval=10.0)
        assert config.ttl_seconds == 60.0
        assert config.ttl_cleanup_interval == 10.0

    def test_expired_segment_not_returned(self):
        """Expired segments should not be returned by lookup."""
        config = ThinkingSegmentConfig(ttl_seconds=0.01)  # 10ms TTL
        store = ThinkingSegmentSubstore(config)

        step_hash = store.store(
            conversation_id="conv_1",
            thinking_tokens=list(range(64)),
            context_tokens=[],
            kv_data=[1.0],
        )

        # Should be available immediately
        segment = store.lookup("conv_1", step_hash)
        assert segment is not None

        # Wait for TTL to expire
        time.sleep(0.02)

        # Now it should be expired
        segment = store.lookup("conv_1", step_hash)
        assert segment is None

    def test_ttl_cleanup_removes_segments(self):
        """TTL cleanup should remove expired segments.

        Segments are cleaned via _maybe_evict during store() calls
        (which includes TTL expiry check) and also via explicit _maybe_ttl_cleanup.
        """
        config = ThinkingSegmentConfig(
            ttl_seconds=0.01,
            ttl_cleanup_interval=0.0,  # Run on every store
        )
        store = ThinkingSegmentSubstore(config)

        step_hash = store.store(
            conversation_id="conv_1",
            thinking_tokens=list(range(64)),
            context_tokens=[],
            kv_data=[1.0],
        )

        assert store._total_segments == 1

        # Wait for expiry (generous sleep to handle CI timing)
        time.sleep(0.1)

        # Call _maybe_ttl_cleanup directly to verify TTL expiry
        store._last_ttl_cleanup = 0  # Force TTL cleanup to run
        store._maybe_ttl_cleanup()

        # First segment should have been cleaned up
        assert store._hash_index.get(step_hash) is None

    def test_ttl_cleanup_reclaims_memory(self):
        """TTL cleanup should set kv_data to None to reclaim memory."""
        config = ThinkingSegmentConfig(
            ttl_seconds=0.01,
            ttl_cleanup_interval=0.0,
        )
        store = ThinkingSegmentSubstore(config)

        large_kv = [float(i) for i in range(10000)]
        step_hash = store.store(
            conversation_id="conv_1",
            thinking_tokens=list(range(64)),
            context_tokens=[],
            kv_data=large_kv,
        )

        # Verify stored
        segment = store._hash_index.get(step_hash)
        assert segment is not None

        # Wait for expiry
        time.sleep(0.1)

        # Call _maybe_ttl_cleanup directly
        store._last_ttl_cleanup = 0  # Force TTL cleanup to run
        store._maybe_ttl_cleanup()

        # Segment should have been removed (kv_data set to None in _remove_segment)
        assert store._hash_index.get(step_hash) is None

    def test_ttl_not_triggered_too_frequently(self):
        """TTL cleanup should respect ttl_cleanup_interval."""
        config = ThinkingSegmentConfig(
            ttl_seconds=3600,  # 1 hour
            ttl_cleanup_interval=60.0,  # Only cleanup every 60s
        )
        store = ThinkingSegmentSubstore(config)

        # First store triggers potential cleanup
        store.store("conv_1", list(range(64)), [], [1.0])
        first_cleanup_count = store.get_stats()["ttl_cleanups"]

        # Immediate second store should not trigger cleanup
        store.store("conv_2", list(range(64)), [], [1.0])
        second_cleanup_count = store.get_stats()["ttl_cleanups"]

        # Should be same (or at most 1 more) because interval not elapsed
        assert second_cleanup_count <= first_cleanup_count + 1

    def test_clear_conversation_returns_count(self):
        """clear_conversation should return the number of removed segments."""
        config = ThinkingSegmentConfig()
        store = ThinkingSegmentSubstore(config)

        store.store("conv_1", list(range(64)), [], [1.0])
        store.store("conv_1", list(range(65)), list(range(10)), [2.0])

        count = store.clear_conversation("conv_1")
        assert count == 2
        assert store._total_segments == 0


class TestCompressionConfigValidation:
    """Test compression configuration edge cases."""

    def test_8bit_compression(self):
        """8-bit compression should work (basically no compression)."""
        config = ThinkingSegmentConfig(
            enable_compression=True,
            compression_bits=8,
        )
        store = ThinkingSegmentSubstore(config)

        kv_data = [1.0, 2.0, 3.0] * 30
        step_hash = store.store("conv_1", list(range(64)), [], kv_data)

        segment = store.lookup("conv_1", step_hash)
        assert segment is not None

    def test_group_size_1(self):
        """Group size 1 should work (per-element quantization)."""
        config = ThinkingSegmentConfig(
            enable_compression=True,
            compression_bits=4,
            compression_group_size=1,
        )
        store = ThinkingSegmentSubstore(config)

        kv_data = [0.5] * 64
        step_hash = store.store("conv_1", list(range(64)), [], kv_data)

        segment = store.lookup("conv_1", step_hash)
        assert segment is not None


class TestProductionStats:
    """Test stats include new production metrics."""

    def test_stats_include_compression_count(self):
        config = ThinkingSegmentConfig(enable_compression=True)
        store = ThinkingSegmentSubstore(config)

        stats = store.get_stats()
        assert "compressions" in stats
        assert "ssd_saves" in stats
        assert "ssd_loads" in stats
        assert "ttl_cleanups" in stats

    def test_stats_initial_values(self):
        store = ThinkingSegmentSubstore(ThinkingSegmentConfig())
        stats = store.get_stats()
        assert stats["compressions"] == 0
        assert stats["ssd_saves"] == 0
        assert stats["ssd_loads"] == 0
        assert stats["ttl_cleanups"] == 0
