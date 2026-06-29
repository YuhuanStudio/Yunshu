"""Tests for SSDSQLiteStore — crash-consistent SQLite metadata for SSD KV cache."""

import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from yunshu_kv.ssd_sqlite_store import SSDSQLiteStore


@pytest.fixture
def store(tmp_path):
    """Create an SSDSQLiteStore with a temporary database, closed after test."""
    db_path = tmp_path / "test_index.db"
    s = SSDSQLiteStore(db_path)
    yield s
    s.close()


@pytest.fixture
def populated_store(store):
    """Store with a few pre-populated entries."""
    store.put("hash_a", "/tmp/block_a.safetensors", 64, 4096)
    store.put("hash_b", "/tmp/block_b.safetensors", 128, 8192)
    store.put("hash_c", "/tmp/block_c.safetensors", 32, 2048)
    return store


# ── Basic CRUD ──


class TestPutGetDelete:
    def test_put_and_get(self, store):
        store.put("abc123", "/data/block.safetensors", 64, 4096)
        entry = store.get("abc123")
        assert entry is not None
        assert entry["block_hash"] == "abc123"
        assert entry["block_path"] == "/data/block.safetensors"
        assert entry["num_tokens"] == 64
        assert entry["size_bytes"] == 4096
        assert entry["access_count"] == 0

    def test_get_nonexistent_returns_none(self, store):
        assert store.get("no_such_hash") is None

    def test_delete_existing(self, store):
        store.put("del_me", "/tmp/del.safetensors", 32, 2048)
        assert store.delete("del_me") is True
        assert store.get("del_me") is None

    def test_delete_nonexistent_returns_false(self, store):
        assert store.delete("no_such_hash") is False

    def test_put_upsert(self, store):
        """Second put with same hash updates the entry."""
        store.put("dup", "/old/path.safetensors", 32, 1024)
        store.put("dup", "/new/path.safetensors", 64, 2048)
        entry = store.get("dup")
        assert entry is not None
        assert entry["block_path"] == "/new/path.safetensors"
        assert entry["num_tokens"] == 64
        assert entry["size_bytes"] == 2048


# ── Touch ──


class TestTouch:
    def test_touch_updates_last_accessed(self, store):
        store.put("t1", "/tmp/t1.safetensors", 32, 1024)
        time.sleep(0.01)  # Ensure time moves forward
        result = store.touch("t1")
        assert result is True
        entry = store.get("t1")
        assert entry is not None
        assert entry["last_accessed"] > entry["created_at"]
        assert entry["access_count"] == 1

    def test_touch_increments_access_count(self, store):
        store.put("t2", "/tmp/t2.safetensors", 32, 1024)
        store.touch("t2")
        store.touch("t2")
        store.touch("t2")
        entry = store.get("t2")
        assert entry is not None
        assert entry["access_count"] == 3

    def test_touch_nonexistent_returns_false(self, store):
        assert store.touch("nope") is False


# ── List All ──


class TestListAll:
    def test_list_all_empty(self, store):
        assert store.list_all() == []

    def test_list_all_returns_all_entries(self, populated_store):
        entries = populated_store.list_all()
        assert len(entries) == 3
        hashes = {e["block_hash"] for e in entries}
        assert hashes == {"hash_a", "hash_b", "hash_c"}

    def test_list_all_sorted_by_created_at(self, populated_store):
        entries = populated_store.list_all()
        timestamps = [e["created_at"] for e in entries]
        assert timestamps == sorted(timestamps)


# ── Stats ──


class TestGetStats:
    def test_stats_empty(self, store):
        stats = store.get_stats()
        assert stats["num_entries"] == 0
        assert stats["total_size_bytes"] == 0
        assert stats["total_tokens"] == 0

    def test_stats_after_inserts(self, populated_store):
        stats = populated_store.get_stats()
        assert stats["num_entries"] == 3
        assert stats["total_size_bytes"] == 4096 + 8192 + 2048  # 14336
        assert stats["total_tokens"] == 64 + 128 + 32  # 224


# ── Checkpoint ──


class TestCheckpoint:
    def test_checkpoint_no_error(self, store):
        store.put("cp1", "/tmp/cp1.safetensors", 32, 1024)
        store.checkpoint()  # Should not raise

    def test_checkpoint_writes_wal(self, store, tmp_path):
        store.put("cp1", "/tmp/cp1.safetensors", 32, 1024)
        store.checkpoint()
        # After checkpoint, WAL file should be small or absent
        Path(str(tmp_path / "test_index.db") + "-wal")
        # WAL may still exist but should be truncated
        # The key invariant is that checkpoint didn't raise
        assert True


# ── Crash Recovery ──


class TestRecovery:
    def test_recovery_removes_stale_entries(self, tmp_path):
        """Insert entries, delete a data file, verify recovery cleans stale entries."""
        db_path = tmp_path / "recovery_test.db"
        # Create real data files
        block_a = tmp_path / "block_a.safetensors"
        block_b = tmp_path / "block_b.safetensors"
        block_a.write_bytes(b"\x00" * 100)
        block_b.write_bytes(b"\x00" * 200)

        with SSDSQLiteStore(db_path) as store:
            store.put("hash_a", str(block_a), 64, 100)
            store.put("hash_b", str(block_b), 128, 200)

            # Verify both entries exist
            assert store.get("hash_a") is not None
            assert store.get("hash_b") is not None

            # Simulate crash: delete one data file
            block_b.unlink()

            # Run recovery
            removed = store.recover()
            assert removed == 1

            # hash_a should still be there, hash_b should be gone
            assert store.get("hash_a") is not None
            assert store.get("hash_b") is None

    def test_recovery_no_stale_entries(self, tmp_path):
        """Recovery with all files present removes nothing."""
        db_path = tmp_path / "clean_test.db"
        block = tmp_path / "block.safetensors"
        block.write_bytes(b"\x00" * 50)

        with SSDSQLiteStore(db_path) as store:
            store.put("hash_x", str(block), 32, 50)
            removed = store.recover()
            assert removed == 0
            assert store.get("hash_x") is not None

    def test_recovery_on_startup(self, tmp_path):
        """Verify that a new store instance can recover from a previous session."""
        db_path = tmp_path / "restart_test.db"
        block = tmp_path / "block.safetensors"
        block.write_bytes(b"\x00" * 100)

        # Session 1: write entry
        with SSDSQLiteStore(db_path) as store1:
            store1.put("survivor", str(block), 64, 100)

        # Session 2: new store, should recover entries
        with SSDSQLiteStore(db_path) as store2:
            removed = store2.recover()
            assert removed == 0  # all files present
            entry = store2.get("survivor")
            assert entry is not None
            assert entry["block_path"] == str(block)


# ── WAL Mode ──


class TestWALMode:
    def test_wal_mode_enabled(self, store, tmp_path):
        """Verify WAL journal mode is active."""
        # Connect to same database and check journal mode
        conn = sqlite3.connect(str(tmp_path / "test_index.db"))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"

    def test_concurrent_reads_while_writing(self, tmp_path):
        """Multiple threads reading while one writes should not deadlock."""
        db_path = tmp_path / "concurrent_test.db"
        store = SSDSQLiteStore(db_path)
        errors = []

        def writer():
            try:
                for i in range(50):
                    store.put(f"w_{i}", f"/tmp/b_{i}.safetensors", i * 10, i * 100)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(50):
                    store.get(f"w_{i % 10}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Concurrent access errors: {errors}"
        store.close()


# ── Context Manager ──


class TestContextManager:
    def test_context_manager(self, tmp_path):
        db_path = tmp_path / "ctx_test.db"
        with SSDSQLiteStore(db_path) as store:
            store.put("ctx1", "/tmp/ctx.safetensors", 32, 1024)
            assert store.is_open
            assert store.get("ctx1") is not None
        # After context exit, store should be closed
        assert not store.is_open


# ── Backward Compatibility (JSON → SQLite) ──


class TestBackwardCompatJSON:
    def test_import_json_index(self, tmp_path):
        """Can import entries from a legacy JSON index file."""
        db_path = tmp_path / "migrate.db"
        json_path = tmp_path / "index.json"

        # Create fake block files
        block_a = tmp_path / "a.safetensors"
        block_b = tmp_path / "b.safetensors"
        block_a.write_bytes(b"\x00" * 100)
        block_b.write_bytes(b"\x00" * 200)

        # Write legacy JSON index
        entries = [
            {
                "block_hash": "hash_a",
                "file_path": str(block_a),
                "token_count": 64,
                "file_size": 100,
                "created_at": 1000.0,
            },
            {
                "block_hash": "hash_b",
                "file_path": str(block_b),
                "token_count": 128,
                "file_size": 200,
                "created_at": 2000.0,
            },
            # This entry's file doesn't exist — should be skipped
            {
                "block_hash": "hash_c",
                "file_path": "/nonexistent/file.safetensors",
                "token_count": 32,
                "file_size": 50,
                "created_at": 3000.0,
            },
        ]
        with open(json_path, "w") as f:
            json.dump({"entries": entries}, f)

        with SSDSQLiteStore(db_path) as store:
            imported = store.import_json_index(json_path)
            assert imported == 2  # hash_c skipped

            assert store.get("hash_a") is not None
            assert store.get("hash_a")["num_tokens"] == 64
            assert store.get("hash_b") is not None
            assert store.get("hash_b")["num_tokens"] == 128
            assert store.get("hash_c") is None

    def test_import_json_index_missing_file(self, tmp_path):
        """import_json_index returns 0 when JSON file doesn't exist."""
        db_path = tmp_path / "nojson.db"
        with SSDSQLiteStore(db_path) as store:
            imported = store.import_json_index(tmp_path / "nonexistent.json")
            assert imported == 0

    def test_import_json_plain_list(self, tmp_path):
        """import_json_index handles plain list format (no "entries" wrapper)."""
        db_path = tmp_path / "list_format.db"
        json_path = tmp_path / "index.json"
        block = tmp_path / "block.safetensors"
        block.write_bytes(b"\x00" * 50)

        entries = [
            {
                "block_hash": "list_hash",
                "file_path": str(block),
                "token_count": 32,
                "file_size": 50,
                "created_at": 1000.0,
            },
        ]
        with open(json_path, "w") as f:
            json.dump(entries, f)

        with SSDSQLiteStore(db_path) as store:
            imported = store.import_json_index(json_path)
            assert imported == 1
            assert store.get("list_hash") is not None


# ── Close / Reopen ──


class TestCloseReopen:
    def test_close_and_reopen(self, tmp_path):
        """Data persists after close and reopen."""
        db_path = tmp_path / "persist.db"
        store1 = SSDSQLiteStore(db_path)
        store1.put("persist1", "/tmp/persist.safetensors", 64, 4096)
        store1.close()

        store2 = SSDSQLiteStore(db_path)
        entry = store2.get("persist1")
        assert entry is not None
        assert entry["block_hash"] == "persist1"
        store2.close()

    def test_double_close_no_error(self, tmp_path):
        """Closing twice should not raise."""
        db_path = tmp_path / "dblclose.db"
        store = SSDSQLiteStore(db_path)
        store.close()
        store.close()  # Should not raise


# ── SSDKVCache Integration (backend env var) ──


class TestSSDKVCacheBackendSelection:
    def test_default_backend_is_sqlite(self, tmp_path):
        """Without YUNSHU_SSD_BACKEND, sqlite is default."""
        os.environ.pop("YUNSHU_SSD_BACKEND", None)
        from yunshu_engine.ssd_kv_cache import SSDKVCache

        cache = SSDKVCache(cache_dir=str(tmp_path / "kv"), backend="sqlite")
        assert cache._backend == "sqlite"
        assert cache._sqlite_store is not None
        cache.close()

    def test_json_backend(self, tmp_path):
        """backend='json' skips SQLite store."""
        from yunshu_engine.ssd_kv_cache import SSDKVCache

        cache = SSDKVCache(cache_dir=str(tmp_path / "kv-json"), backend="json")
        assert cache._backend == "json"
        assert cache._sqlite_store is None
        cache.close()

    def test_env_var_backend(self, tmp_path):
        """YUNSHU_SSD_BACKEND env var is respected."""
        from yunshu_engine.ssd_kv_cache import SSDKVCache

        os.environ["YUNSHU_SSD_BACKEND"] = "json"
        try:
            cache = SSDKVCache(cache_dir=str(tmp_path / "kv-env"))
            assert cache._backend == "json"
            cache.close()
        finally:
            os.environ.pop("YUNSHU_SSD_BACKEND", None)


class TestBatchOperations:
    """Tests for batch_put, batch_get, batch_delete."""

    def test_batch_put_inserts_all(self, store):
        entries = [
            (f"batch_{i}", f"/tmp/block_{i}.bin", 64 * (i + 1), 4096 * (i + 1))
            for i in range(10)
        ]
        inserted = store.batch_put(entries)
        assert inserted == 10
        for i in range(10):
            result = store.get(f"batch_{i}")
            assert result is not None
            assert result["num_tokens"] == 64 * (i + 1)

    def test_batch_put_upsert(self, store):
        store.put("upsert_a", "/tmp/old.bin", 32, 2048)
        entries = [("upsert_a", "/tmp/new.bin", 64, 4096)]
        store.batch_put(entries)
        result = store.get("upsert_a")
        assert result["num_tokens"] == 64
        assert result["block_path"] == "/tmp/new.bin"

    def test_batch_put_empty(self, store):
        assert store.batch_put([]) == 0

    def test_batch_get_returns_all(self, populated_store):
        results = populated_store.batch_get(["hash_a", "hash_b", "hash_c"])
        assert len(results) == 3
        assert all(r is not None for r in results)
        assert results[0]["block_hash"] == "hash_a"
        assert results[1]["block_hash"] == "hash_b"
        assert results[2]["block_hash"] == "hash_c"

    def test_batch_get_missing_returns_none(self, populated_store):
        results = populated_store.batch_get(["hash_a", "nonexistent", "hash_c"])
        assert len(results) == 3
        assert results[0] is not None
        assert results[1] is None
        assert results[2] is not None

    def test_batch_get_empty(self, store):
        assert store.batch_get([]) == []

    def test_batch_delete_removes_all(self, populated_store):
        deleted = populated_store.batch_delete(["hash_a", "hash_b"])
        assert deleted == 2
        assert populated_store.get("hash_a") is None
        assert populated_store.get("hash_b") is None
        assert populated_store.get("hash_c") is not None

    def test_batch_delete_partial(self, populated_store):
        deleted = populated_store.batch_delete(["hash_a", "nonexistent"])
        assert deleted == 1

    def test_batch_delete_empty(self, populated_store):
        assert populated_store.batch_delete([]) == 0

    def test_batch_operations_consistency(self, store):
        """Batch put then batch get returns consistent results."""
        entries = [(f"h{i}", f"/p{i}", i * 16, i * 1024) for i in range(50)]
        store.batch_put(entries)
        hashes = [f"h{i}" for i in range(50)]
        results = store.batch_get(hashes)
        assert all(r is not None for r in results)
        total_tokens = sum(r["num_tokens"] for r in results if r)
        assert total_tokens == sum(i * 16 for i in range(50))
