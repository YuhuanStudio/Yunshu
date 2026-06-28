"""Tests for BoundarySnapshotSSDStore — serialize/deserialize round-trip."""

import numpy as np
import pytest

from yunshu_kv.boundary_snapshot import BoundarySnapshotSSDStore


class TestBoundarySnapshotSerializeDeserialize:
    """Test binary serialization round-trip."""

    def test_ndarray_roundtrip(self, tmp_path):
        store = BoundarySnapshotSSDStore(tmp_path)
        state = {
            "weights": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "bias": np.array([0.5], dtype=np.float16),
        }
        data = store._serialize(state)
        result = store._deserialize(data)
        np.testing.assert_array_equal(result["weights"], state["weights"])
        np.testing.assert_array_equal(result["bias"], state["bias"])

    def test_string_roundtrip(self, tmp_path):
        store = BoundarySnapshotSSDStore(tmp_path)
        state = {"name": "KVCACHE", "extra": "hello world"}
        data = store._serialize(state)
        result = store._deserialize(data)
        assert result["name"] == "KVCACHE"
        assert result["extra"] == "hello world"

    def test_int_roundtrip(self, tmp_path):
        store = BoundarySnapshotSSDStore(tmp_path)
        state = {"offset": 42, "length": 1024}
        data = store._serialize(state)
        result = store._deserialize(data)
        assert result["offset"] == 42
        assert result["length"] == 1024

    def test_float_roundtrip(self, tmp_path):
        store = BoundarySnapshotSSDStore(tmp_path)
        state = {"temperature": 0.7, "scale": 1.5}
        data = store._serialize(state)
        result = store._deserialize(data)
        assert abs(result["temperature"] - 0.7) < 1e-10
        assert abs(result["scale"] - 1.5) < 1e-10

    def test_bool_roundtrip(self, tmp_path):
        store = BoundarySnapshotSSDStore(tmp_path)
        state = {"trained": True, "frozen": False}
        data = store._serialize(state)
        result = store._deserialize(data)
        assert result["trained"] is True
        assert result["frozen"] is False

    def test_mixed_types_roundtrip(self, tmp_path):
        store = BoundarySnapshotSSDStore(tmp_path)
        state = {
            "cache_type": "ROTATING_KVCACHE",
            "offset": 128,
            "rate": 0.99,
            "active": True,
            "keys": np.random.randn(2, 8, 128, 64).astype(np.float32),
        }
        data = store._serialize(state)
        result = store._deserialize(data)
        assert result["cache_type"] == "ROTATING_KVCACHE"
        assert result["offset"] == 128
        assert abs(result["rate"] - 0.99) < 1e-10
        assert result["active"] is True
        np.testing.assert_array_almost_equal(result["keys"], state["keys"])

    def test_unsupported_list_value_raises_not_drops(self, tmp_path):
        """a list value (ArraysCache/CacheList recurrent state) must
        fail loudly, not be silently dropped — the old serializer skipped it and
        produced a snapshot missing the key, with no error on reload."""
        store = BoundarySnapshotSSDStore(tmp_path)
        state = {"cache_type": "ARRAYS_CACHE", "arrays": [np.zeros(4), np.ones(4)]}
        with pytest.raises(TypeError, match="unsupported value type"):
            store._serialize(state)

    def test_unsupported_none_value_raises(self, tmp_path):
        store = BoundarySnapshotSSDStore(tmp_path)
        with pytest.raises(TypeError, match="unsupported value type"):
            store._serialize({"cache_type": "KVCACHE", "meta_state": None})

    def test_invalid_magic_fails(self, tmp_path):
        store = BoundarySnapshotSSDStore(tmp_path)
        with pytest.raises(ValueError, match="Invalid boundary snapshot magic"):
            store._deserialize(b"BADMAGIC" + b"\x00" * 20)


class TestBoundarySnapshotSaveLoad:
    """Test save/load with pending writes buffer."""

    def test_save_returns_path(self, tmp_path):
        store = BoundarySnapshotSSDStore(tmp_path)
        state = {"data": np.array([1.0], dtype=np.float32)}
        path = store.save("req-1", 0, state)
        # Path is returned even if async write hasn't completed
        assert str(path).endswith("req-1_L0.bin")

    def test_load_from_pending(self, tmp_path):
        store = BoundarySnapshotSSDStore(tmp_path)
        state = {"value": np.array([42.0], dtype=np.float32)}
        store.save("req-1", 0, state)
        # Should find it in pending writes
        result = store.load("req-1", 0)
        assert result is not None
        np.testing.assert_array_equal(result["value"], state["value"])

    def test_load_missing_returns_none(self, tmp_path):
        store = BoundarySnapshotSSDStore(tmp_path)
        assert store.load("nonexistent", 0) is None

    def test_cleanup_removes_entries(self, tmp_path):
        store = BoundarySnapshotSSDStore(tmp_path)
        state = {"x": np.array([1.0], dtype=np.float32)}
        store.save("req-1", 0, state)
        store.save("req-1", 1, state)
        store.cleanup("req-1")
        assert store.load("req-1", 0) is None
        assert store.load("req-1", 1) is None


class TestBoundarySnapshotLifecycle:
    """Test start/stop lifecycle."""

    def test_start_stop(self, tmp_path):
        store = BoundarySnapshotSSDStore(tmp_path)
        store.start()
        assert store._writer_thread is not None
        assert store._writer_thread.is_alive()
        store.stop()
        assert store._writer_thread is None
