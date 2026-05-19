from __future__ import annotations
"""Yunshu SSD KV Cache — SQLite-backed metadata store for crash consistency (C13).

Replaces the JSON-based index with SQLite in WAL mode for:
- Crash consistency: WAL journal survives process crashes mid-write
- Concurrent access: readers never block writers
- Atomic transactions: multi-row updates are all-or-nothing
- Fast lookups: indexed PRIMARY KEY + last_accessed

Thread safety:
- SQLite connection used with check_same_thread=False + explicit Lock
- WAL mode allows concurrent reads while a write is in progress
- All public methods are thread-safe
"""

import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS kv_entries (
    block_hash TEXT PRIMARY KEY,
    block_path TEXT NOT NULL,
    num_tokens INTEGER NOT NULL,
    created_at REAL NOT NULL,
    last_accessed REAL NOT NULL DEFAULT 0,
    access_count INTEGER NOT NULL DEFAULT 0,
    size_bytes INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_kv_last_accessed
    ON kv_entries(last_accessed);
CREATE INDEX IF NOT EXISTS idx_kv_num_tokens
    ON kv_entries(num_tokens);
"""


class SSDSQLiteStore:
    """Crash-consistent SQLite metadata store for SSD KV cache blocks.

    Usage:
        store = SSDSQLiteStore("~/.cache/yunshu/kv-ssd/index.db")
        store.put("abc123", "/path/to/block.safetensors", 64, 4096)
        entry = store.get("abc123")   # {"block_hash": ..., "block_path": ...}
        store.touch("abc123")         # update last_accessed
        store.delete("abc123")
        store.close()

    Or as context manager:
        with SSDSQLiteStore(path) as store:
            ...
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(os.path.expanduser(str(db_path)))
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._open()

    # ── Context Manager ──

    def __enter__(self) -> SSDSQLiteStore:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ── Lifecycle ──

    def _open(self) -> None:
        """Open database connection, enable WAL, create schema."""
        try:
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                timeout=10.0,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA temp_store=MEMORY")
            self._conn.execute("PRAGMA mmap_size=67108864")  # 64 MB
            self._conn.executescript(_SCHEMA_V1)
            self._conn.commit()
            logger.debug("SSDSQLiteStore: opened %s (WAL mode)", self._db_path)
        except Exception:
            logger.error("SSDSQLiteStore: failed to open %s", self._db_path, exc_info=True)
            self._conn = None

    def close(self) -> None:
        """Close the database connection and checkpoint WAL."""
        if self._conn is not None:
            try:
                self.checkpoint()
            except Exception:
                logger.debug("checkpoint during close failed", exc_info=True)
            try:
                self._conn.close()
            except Exception:
                logger.debug("connection close failed", exc_info=True)
            self._conn = None

    # ── Core CRUD ──

    def put(
        self,
        block_hash: str,
        block_path: str,
        num_tokens: int,
        size_bytes: int,
    ) -> None:
        """Insert or update a block entry.

        Args:
            block_hash: Hex-encoded block hash (PRIMARY KEY).
            block_path: Absolute path to the block data file.
            num_tokens: Number of tokens in this block.
            size_bytes: Size of the block data file in bytes.
        """
        if self._conn is None:
            return
        now = time.monotonic()
        with self._lock:
            try:
                self._conn.execute(
                    """
                    INSERT INTO kv_entries
                        (block_hash, block_path, num_tokens, created_at,
                         last_accessed, access_count, size_bytes)
                    VALUES (?, ?, ?, ?, ?, 0, ?)
                    ON CONFLICT(block_hash) DO UPDATE SET
                        block_path = excluded.block_path,
                        num_tokens = excluded.num_tokens,
                        size_bytes = excluded.size_bytes,
                        last_accessed = excluded.last_accessed
                    """,
                    (block_hash, block_path, num_tokens, now, now, size_bytes),
                )
                self._conn.commit()
            except Exception:
                logger.debug("SSDSQLiteStore.put failed for %s", block_hash[:16], exc_info=True)

    def get(self, block_hash: str) -> Optional[dict]:
        """Look up a block by hash.

        Returns:
            Dict with keys: block_hash, block_path, num_tokens, created_at,
            last_accessed, access_count, size_bytes.
            None if not found.
        """
        if self._conn is None:
            return None
        with self._lock:
            try:
                row = self._conn.execute(
                    """
                    SELECT block_hash, block_path, num_tokens, created_at,
                           last_accessed, access_count, size_bytes
                    FROM kv_entries WHERE block_hash = ?
                    """,
                    (block_hash,),
                ).fetchone()
            except Exception:
                logger.debug("SSDSQLiteStore.get failed for %s", block_hash[:16], exc_info=True)
                return None
        if row is None:
            return None
        return {
            "block_hash": row[0],
            "block_path": row[1],
            "num_tokens": row[2],
            "created_at": row[3],
            "last_accessed": row[4],
            "access_count": row[5],
            "size_bytes": row[6],
        }

    def delete(self, block_hash: str) -> bool:
        """Delete a block entry.

        Returns:
            True if a row was deleted, False otherwise.
        """
        if self._conn is None:
            return False
        with self._lock:
            try:
                cursor = self._conn.execute(
                    "DELETE FROM kv_entries WHERE block_hash = ?",
                    (block_hash,),
                )
                self._conn.commit()
                return cursor.rowcount > 0
            except Exception:
                logger.debug("SSDSQLiteStore.delete failed for %s", block_hash[:16], exc_info=True)
                return False

    def touch(self, block_hash: str) -> bool:
        """Update last_accessed and increment access_count for a block.

        Returns:
            True if the row was updated, False if not found.
        """
        if self._conn is None:
            return False
        now = time.monotonic()
        with self._lock:
            try:
                cursor = self._conn.execute(
                    """
                    UPDATE kv_entries
                    SET last_accessed = ?, access_count = access_count + 1
                    WHERE block_hash = ?
                    """,
                    (now, block_hash),
                )
                self._conn.commit()
                return cursor.rowcount > 0
            except Exception:
                logger.debug("SSDSQLiteStore.touch failed for %s", block_hash[:16], exc_info=True)
                return False

    def batch_put(
        self,
        entries: list[tuple[str, str, int, int]],
    ) -> int:
        """Insert or update multiple block entries in a single transaction.

        Args:
            entries: List of (block_hash, block_path, num_tokens, size_bytes) tuples.

        Returns:
            Number of entries successfully inserted/updated.
        """
        if self._conn is None or not entries:
            return 0
        now = time.monotonic()
        inserted = 0
        with self._lock:
            try:
                self._conn.executemany(
                    """
                    INSERT INTO kv_entries
                        (block_hash, block_path, num_tokens, created_at,
                         last_accessed, access_count, size_bytes)
                    VALUES (?, ?, ?, ?, ?, 0, ?)
                    ON CONFLICT(block_hash) DO UPDATE SET
                        block_path = excluded.block_path,
                        num_tokens = excluded.num_tokens,
                        size_bytes = excluded.size_bytes,
                        last_accessed = excluded.last_accessed
                    """,
                    [(h, p, n, now, now, s) for h, p, n, s in entries],
                )
                self._conn.commit()
                inserted = len(entries)
            except Exception:
                logger.debug("SSDSQLiteStore.batch_put failed (%d entries)", len(entries), exc_info=True)
        return inserted

    def batch_get(self, block_hashes: list[str]) -> list[Optional[dict]]:
        """Look up multiple blocks by hash in a single query.

        Args:
            block_hashes: List of hex-encoded block hashes.

        Returns:
            List of dicts (same order as input). Missing entries are None.
        """
        if self._conn is None or not block_hashes:
            return [None] * len(block_hashes)
        with self._lock:
            try:
                placeholders = ",".join("?" * len(block_hashes))
                rows = self._conn.execute(
                    f"""
                    SELECT block_hash, block_path, num_tokens, created_at,
                           last_accessed, access_count, size_bytes
                    FROM kv_entries WHERE block_hash IN ({placeholders})
                    """,
                    block_hashes,
                ).fetchall()
            except Exception:
                logger.debug("SSDSQLiteStore.batch_get failed (%d hashes)", len(block_hashes), exc_info=True)
                return [None] * len(block_hashes)
        # Build lookup by hash
        lookup: dict[str, dict] = {}
        for r in rows:
            lookup[r[0]] = {
                "block_hash": r[0],
                "block_path": r[1],
                "num_tokens": r[2],
                "created_at": r[3],
                "last_accessed": r[4],
                "access_count": r[5],
                "size_bytes": r[6],
            }
        return [lookup.get(h) for h in block_hashes]

    def batch_delete(self, block_hashes: list[str]) -> int:
        """Delete multiple block entries in a single transaction.

        Returns:
            Number of entries deleted.
        """
        if self._conn is None or not block_hashes:
            return 0
        with self._lock:
            try:
                placeholders = ",".join("?" * len(block_hashes))
                cursor = self._conn.execute(
                    f"DELETE FROM kv_entries WHERE block_hash IN ({placeholders})",
                    block_hashes,
                )
                self._conn.commit()
                return cursor.rowcount
            except Exception:
                logger.debug("SSDSQLiteStore.batch_delete failed (%d hashes)", len(block_hashes), exc_info=True)
                return 0

    def list_all(self) -> list[dict]:
        """Return all block entries as a list of dicts (oldest first by created_at)."""
        if self._conn is None:
            return []
        with self._lock:
            try:
                rows = self._conn.execute(
                    """
                    SELECT block_hash, block_path, num_tokens, created_at,
                           last_accessed, access_count, size_bytes
                    FROM kv_entries ORDER BY created_at
                    """
                ).fetchall()
            except Exception:
                logger.debug("SSDSQLiteStore.list_all failed", exc_info=True)
                return []
        return [
            {
                "block_hash": r[0],
                "block_path": r[1],
                "num_tokens": r[2],
                "created_at": r[3],
                "last_accessed": r[4],
                "access_count": r[5],
                "size_bytes": r[6],
            }
            for r in rows
        ]

    def get_stats(self) -> dict:
        """Return aggregate statistics."""
        if self._conn is None:
            return {
                "num_entries": 0,
                "total_size_bytes": 0,
                "total_tokens": 0,
            }
        with self._lock:
            try:
                row = self._conn.execute(
                    """
                    SELECT
                        COUNT(*),
                        COALESCE(SUM(size_bytes), 0),
                        COALESCE(SUM(num_tokens), 0)
                    FROM kv_entries
                    """
                ).fetchone()
            except Exception:
                logger.debug("SSDSQLiteStore.get_stats failed", exc_info=True)
                return {
                    "num_entries": 0,
                    "total_size_bytes": 0,
                    "total_tokens": 0,
                }
        return {
            "num_entries": row[0],
            "total_size_bytes": row[1],
            "total_tokens": row[2],
        }

    # ── Maintenance ──

    def checkpoint(self) -> None:
        """Force a WAL checkpoint (flush WAL to main database file).

        Call this periodically or on graceful shutdown for durability.
        """
        if self._conn is None:
            return
        with self._lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                logger.debug("SSDSQLiteStore.checkpoint failed", exc_info=True)

    def recover(self) -> int:
        """Verify all block_path files exist, remove stale entries.

        Call on startup to clean up orphaned index entries after a crash.

        Returns:
            Number of stale entries removed.
        """
        if self._conn is None:
            return 0
        removed = 0
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT block_hash, block_path FROM kv_entries"
                ).fetchall()
                for bh, fpath in rows:
                    if not Path(fpath).exists():
                        self._conn.execute(
                            "DELETE FROM kv_entries WHERE block_hash = ?",
                            (bh,),
                        )
                        removed += 1
                if removed > 0:
                    self._conn.commit()
                    logger.info("SSDSQLiteStore.recover: removed %d stale entries", removed)
            except Exception:
                logger.debug("SSDSQLiteStore.recover failed", exc_info=True)
        return removed

    # ── Backward Compatibility ──

    def import_json_index(self, json_path: str | Path) -> int:
        """Import entries from a legacy JSON index file.

        Skips entries whose block_path files no longer exist.

        Returns:
            Number of entries imported.
        """
        import json

        json_path = Path(json_path)
        if not json_path.exists():
            return 0

        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception:
            logger.warning("Failed to read JSON index: %s", json_path, exc_info=True)
            return 0

        entries = data if isinstance(data, list) else data.get("entries", [])
        imported = 0
        for entry in entries:
            fpath = entry.get("file_path", entry.get("block_path", ""))
            if not fpath or not Path(fpath).exists():
                continue
            bh = entry.get("block_hash", "")
            # Handle bytes format
            if isinstance(bh, bytes):
                bh = bh.hex()
            num_tokens = int(entry.get("token_count", entry.get("num_tokens", 0)))
            size_bytes = int(entry.get("file_size", entry.get("size_bytes", 0)))
            created_at = float(entry.get("created_at", 0))
            if not bh:
                continue
            self.put(bh, fpath, num_tokens, size_bytes)
            imported += 1

        if imported > 0:
            logger.info("SSDSQLiteStore: imported %d entries from JSON index", imported)
        return imported

    @property
    def is_open(self) -> bool:
        """Check whether the database connection is active."""
        return self._conn is not None
