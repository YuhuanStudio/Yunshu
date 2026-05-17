from __future__ import annotations
"""Yunshu SSD-Tier KV Cache — persistent KV block storage for prefix reuse.

Studied from oMLX's PagedSSDCacheManager, adapted for Yunshu:
- Background writer thread for non-blocking saves (oMLX pattern)
- safetensors-based serialization (thread-safe, no Metal API in writer)
- Hot RAM cache + SSD index for fast lookup
- LRU eviction from hot → SSD, SSD → delete
- Block-level hashing integration with KVPrefixCache
- Dynamic disk budget awareness
- SQLite-backed metadata for crash consistency (C13 audit item)

Backend selection via YUNSHU_SSD_BACKEND env var:
  "sqlite" (default) — crash-consistent WAL-mode SQLite (yunshu_kv.SSDSQLiteStore)
  "json" — legacy JSON index (backward compat, no crash safety)

Architecture:
  KVPrefixCache
    → SSDKVCache (this module)
      → hot_cache: OrderedDict[bytes, list] (recent blocks in RAM)
      → _index: dict[bytes, _BlockMeta] (all known blocks, on disk or hot)
      → _sqlite_store: SSDSQLiteStore (crash-consistent metadata, when backend=sqlite)
      → _write_queue: Queue (background writer thread)

Thread safety:
  - Tensor bytes extracted on inference thread (Metal-safe)
  - Background writer does only file I/O (no Metal API calls)
  - Index mutations protected by threading.Lock
"""

import json
import logging
import os
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_FORMAT_VERSION = "1"
_READABLE_VERSIONS = frozenset({"1"})

# MLX dtype to safetensors dtype string mapping
_DTYPE_MAP = {
    "float16": "F16",
    "mlx.core.float16": "F16",
    "float32": "F32",
    "mlx.core.float32": "F32",
    "bfloat16": "BF16",
    "mlx.core.bfloat16": "BF16",
    "int8": "I8",
    "mlx.core.int8": "I8",
    "int32": "I32",
    "mlx.core.int32": "I32",
    "uint8": "U8",
    "mlx.core.uint8": "U8",
    "uint16": "U16",
    "mlx.core.uint16": "U16",
    "uint32": "U32",
    "mlx.core.uint32": "U32",
    "bool_": "BOOL",
    "mlx.core.bool_": "BOOL",
}


@dataclass
class _BlockMeta:
    """Metadata for a cached block (in-memory index entry)."""
    block_hash: bytes
    file_path: str
    token_count: int
    model_name: str
    created_at: float
    file_size: int = 0
    last_accessed: float = 0.0


@dataclass
class SSDCacheStats:
    """Statistics for the SSD-tier KV cache."""
    hot_cache_entries: int = 0
    hot_cache_bytes: int = 0
    disk_entries: int = 0
    disk_bytes: int = 0
    total_entries: int = 0
    max_size_bytes: int = 0
    writes_completed: int = 0
    reads_completed: int = 0
    evictions: int = 0


def _extract_tensor_bytes(arr) -> tuple[bytes, str, list[int]]:
    """Extract raw bytes from an mx.array for safetensors serialization.

    Runs on the inference thread (Metal-safe for eval'd arrays).
    """
    import mlx.core as mx

    dtype_name = str(arr.dtype)
    st_dtype = _DTYPE_MAP.get(dtype_name, "F16")
    shape = list(arr.shape)

    # Ensure array is evaluated before extracting bytes
    mx.eval(arr)

    if dtype_name == "bfloat16":
        raw = bytes(memoryview(arr.astype(mx.uint16)))
    else:
        try:
            raw = bytes(memoryview(arr))
        except (TypeError, BufferError):
            raw = bytes(arr)

    return raw, st_dtype, shape


def _write_safetensors(path: str, tensors: dict[str, tuple[bytes, str, list[int]]],
                       metadata: dict[str, str] | None = None) -> int:
    """Write safetensors file without using mlx API (thread-safe).

    Format: [8-byte header_size LE uint64][header JSON padded to 8 bytes][tensor data]
    """
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = metadata

    offset = 0
    for name, (raw_bytes, dtype, shape) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(raw_bytes)],
        }
        offset += len(raw_bytes)

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad header to 8-byte alignment
    padding = (8 - (len(header_json) % 8)) % 8
    header_json += b" " * padding

    header_size = len(header_json)
    total_size = 8 + header_size + offset

    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "wb") as f:
            f.write(struct.pack("<Q", header_size))
            f.write(header_json)
            for name in tensors:
                f.write(tensors[name][0])
        os.rename(tmp_path, path)
    except Exception:
        logger.debug("safetensors write failed", exc_info=True)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return total_size


def _read_safetensors_metadata(path: str) -> dict[str, Any]:
    """Read safetensors header without loading tensor data."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    return header


class SSDKVCache:
    """Persistent SSD-tier KV cache for block-level prefix cache persistence.

    Integrates with KVPrefixCache to provide:
    - Background non-blocking save of KV blocks to disk
    - Fast hot cache for recently accessed blocks
    - LRU eviction from hot → SSD → delete
    - Dynamic disk budget management

    Thread safety:
    - All public methods are thread-safe (Lock-protected)
    - Background writer thread handles all file I/O
    """

    def __init__(
        self,
        cache_dir: str = "~/.cache/yunshu/kv-ssd",
        max_size_bytes: int = 10 * 1024 ** 3,  # 10 GB default
        hot_cache_size: int = 100,
        writer_queue_size: int = 64,
        backend: str | None = None,
    ):
        self._cache_dir = Path(os.path.expanduser(cache_dir))
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Create hex-bucket subdirectories (oMLX pattern)
        for h in "0123456789abcdef":
            (self._cache_dir / h).mkdir(exist_ok=True)

        self._max_size_bytes = max_size_bytes
        self._hot_cache_size = hot_cache_size
        self._lock = threading.Lock()

        # Hot cache: block_hash_hex → (cache_data, token_count)
        self._hot_cache: OrderedDict[str, tuple[list, int]] = OrderedDict()

        # On-disk index: block_hash_hex → _BlockMeta
        self._index: dict[str, _BlockMeta] = {}

        # Backend selection: "sqlite" (default, crash-consistent) or "json" (legacy)
        self._backend = (
            backend
            or os.environ.get("YUNSHU_SSD_BACKEND", "sqlite")
        ).lower()
        if self._backend not in ("sqlite", "json"):
            logger.warning("Unknown YUNSHU_SSD_BACKEND=%r, falling back to sqlite", self._backend)
            self._backend = "sqlite"

        # SQLite metadata store for crash consistency (C13)
        self._sqlite_store: SSDSQLiteStore | None = None
        self._db = None  # kept for backward compat alias
        if self._backend == "sqlite":
            from yunshu_kv.ssd_sqlite_store import SSDSQLiteStore
            db_path = self._cache_dir / "index.db"
            self._sqlite_store = SSDSQLiteStore(db_path)
            # Attempt JSON → SQLite migration for backward compat
            json_index_path = self._cache_dir / "index.json"
            if json_index_path.exists() and db_path.exists():
                migrated = self._sqlite_store.import_json_index(json_index_path)
                if migrated > 0:
                    logger.info("SSD KV cache: migrated %d entries from JSON to SQLite", migrated)

        # Background writer
        self._write_queue: list[tuple] = []
        self._writer_thread: threading.Thread | None = None
        self._writer_stop = threading.Event()
        self._writer_lock = threading.Lock()
        self._writer_queue_size = writer_queue_size

        # Stats
        self._writes_completed = 0
        self._reads_completed = 0
        self._evictions = 0

        # Recover existing index from disk
        self._recover_index()

    def _start_writer(self) -> None:
        """Start the background writer thread if not running."""
        with self._writer_lock:
            if self._writer_thread is not None and self._writer_thread.is_alive():
                return
            self._writer_stop.clear()
            self._writer_thread = threading.Thread(
                target=self._writer_loop, daemon=True, name="ssd-kv-writer"
            )
            self._writer_thread.start()

    def _writer_loop(self) -> None:
        """Background writer loop — processes save/delete operations."""
        while not self._writer_stop.is_set():
            item = None
            with self._writer_lock:
                if not self._write_queue:
                    break
                item = self._write_queue.pop(0)

            if item is None:
                break

            try:
                op = item[0]
                if op == "save":
                    _, block_hash_hex, tensors_raw, meta_dict, file_path = item
                    file_size = _write_safetensors(file_path, tensors_raw, meta_dict)
                    with self._lock:
                        if block_hash_hex in self._index:
                            self._index[block_hash_hex].file_size = file_size
                            self._sqlite_upsert(block_hash_hex, self._index[block_hash_hex])
                        self._writes_completed += 1
                elif op == "delete":
                    _, file_path = item
                    try:
                        os.unlink(file_path)
                    except OSError:
                        pass
            except Exception as e:
                logger.debug("SSD writer error", exc_info=True)

    def _flush_writer(self) -> None:
        """Flush all pending writes."""
        # Signal writer to stop and wait for it
        self._writer_stop.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=5.0)
        self._writer_thread = None
        self._writer_stop.clear()
        # Process remaining items synchronously (no locks needed — writer is stopped)
        for item in self._write_queue:
            try:
                op = item[0]
                if op == "save":
                    _, block_hash_hex, tensors_raw, meta_dict, file_path = item
                    _write_safetensors(file_path, tensors_raw, meta_dict)
                    self._writes_completed += 1
                elif op == "delete":
                    try:
                        os.unlink(item[1])
                    except OSError:
                        pass
            except Exception:
                logger.debug("SSD writer flush failed", exc_info=True)
        self._write_queue.clear()

    def _recover_index(self) -> None:
        """Recover block index from SQLite store, or scan cache directory."""
        # Try SQLite store first (fast, crash-consistent)
        if self._sqlite_store is not None:
            try:
                # Run recovery to clean up orphaned entries
                self._sqlite_store.recover()
                count = self._recover_from_sqlite_store()
                if count > 0:
                    logger.info(f"SSD KV cache: recovered {count} blocks from SQLite store")
                    return
            except Exception:
                logger.debug("SQLite store recovery failed, falling back to scan", exc_info=True)

        # Fallback: scan safetensors headers
        count = 0
        for bucket in self._cache_dir.iterdir():
            if not bucket.is_dir() or len(bucket.name) != 1:
                continue
            for f in bucket.iterdir():
                if not f.name.endswith(".safetensors"):
                    continue
                block_hash_hex = f.stem
                try:
                    header = _read_safetensors_metadata(str(f))
                    meta = header.get("__metadata__", {})
                    version = meta.get("yunshu_cache_version", "unknown")
                    if version not in _READABLE_VERSIONS:
                        continue
                    self._index[block_hash_hex] = _BlockMeta(
                        block_hash=bytes.fromhex(block_hash_hex),
                        file_path=str(f),
                        token_count=int(meta.get("token_count", "0")),
                        model_name=meta.get("model_name", ""),
                        created_at=float(meta.get("created_at", "0")),
                        file_size=f.stat().st_size,
                    )
                    # Populate SQLite store if available (migration from file scan)
                    if self._sqlite_store is not None:
                        self._sqlite_store.put(
                            block_hash_hex, str(f),
                            int(meta.get("token_count", "0")),
                            f.stat().st_size,
                        )
                    count += 1
                except Exception:
                    logger.debug("SSD block index recovery failed", exc_info=True)
        if count > 0:
            logger.info(f"SSD KV cache: recovered {count} blocks from disk scan")

    def _recover_from_sqlite_store(self) -> int:
        """Recover block metadata from SSDSQLiteStore."""
        entries = self._sqlite_store.list_all()
        count = 0
        for entry in entries:
            bh_hex = entry["block_hash"]
            fpath = entry["block_path"]
            self._index[bh_hex] = _BlockMeta(
                block_hash=bytes.fromhex(bh_hex),
                file_path=fpath,
                token_count=entry["num_tokens"],
                model_name="",  # model_name not in SSDSQLiteStore schema
                created_at=entry["created_at"],
                file_size=entry["size_bytes"],
                last_accessed=entry["last_accessed"],
            )
            count += 1
        return count

    def _sqlite_upsert(self, hex_hash: str, meta: _BlockMeta) -> None:
        """Insert or update a block entry in the metadata store."""
        if self._sqlite_store is not None:
            self._sqlite_store.put(
                hex_hash, meta.file_path, meta.token_count, meta.file_size,
            )

    def _sqlite_delete(self, hex_hash: str) -> None:
        """Delete a block entry from the metadata store."""
        if self._sqlite_store is not None:
            self._sqlite_store.delete(hex_hash)

    def _block_path(self, block_hash: bytes) -> str:
        """Get file path for a block hash."""
        hex_hash = block_hash.hex()
        return str(self._cache_dir / hex_hash[0] / f"{hex_hash}.safetensors")

    def save_block(
        self,
        block_hash: bytes,
        cache_data: list,
        token_count: int,
        model_name: str = "",
    ) -> None:
        """Save a KV block to the SSD cache (non-blocking).

        Extracts tensor bytes on the calling thread (Metal-safe),
        then enqueues for background writing.

        Args:
            block_hash: Block content hash (from KVPrefixCache block hashing)
            cache_data: Per-layer KV cache data list
            token_count: Number of tokens in this block
            model_name: Model name for isolation
        """
        # Extract tensor bytes on the inference thread
        tensors_raw: dict[str, tuple[bytes, str, list[int]]] = {}
        for i, layer_data in enumerate(cache_data):
            if isinstance(layer_data, (list, tuple)):
                for k, tensor in enumerate(layer_data):
                    if tensor is not None and hasattr(tensor, 'shape'):
                        key = f"layer_{i}_state_{k}"
                        tensors_raw[key] = _extract_tensor_bytes(tensor)
            elif hasattr(layer_data, 'keys') and hasattr(layer_data, 'values'):
                tensors_raw[f"layer_{i}_keys"] = _extract_tensor_bytes(layer_data.keys)
                tensors_raw[f"layer_{i}_values"] = _extract_tensor_bytes(layer_data.values)
            elif hasattr(layer_data, 'shape'):
                # Raw mx.array layer (no keys/values wrapper)
                tensors_raw[f"layer_{i}_state_0"] = _extract_tensor_bytes(layer_data)

        if not tensors_raw:
            return

        meta = {
            "yunshu_cache_version": _CACHE_FORMAT_VERSION,
            "token_count": str(token_count),
            "num_layers": str(len(cache_data)),
            "model_name": model_name,
            "created_at": str(time.time()),
        }

        hex_hash = block_hash.hex()
        file_path = self._block_path(block_hash)

        # Add to hot cache and index
        with self._lock:
            self._hot_cache[hex_hash] = (cache_data, token_count)
            self._hot_cache.move_to_end(hex_hash)
            self._evict_hot_if_full()
            self._index[hex_hash] = _BlockMeta(
                block_hash=block_hash,
                file_path=file_path,
                token_count=token_count,
                model_name=model_name,
                created_at=time.time(),
            )
            self._sqlite_upsert(hex_hash, self._index[hex_hash])

        # Enqueue for background writing (no lock held here).
        # NOTE: We never silently drop queued saves — dropping would leave
        # the index/SQLite claiming the block exists on disk when it was
        # never written (data loss).  Instead we block the caller until
        # the queue drains below capacity.
        with self._writer_lock:
            self._write_queue.append(("save", hex_hash, tensors_raw, meta, file_path))

        # For small caches, write synchronously to avoid thread management overhead
        # Background writer is started only when queue exceeds threshold
        if len(self._write_queue) <= 2:
            self._process_pending_writes()
        else:
            self._start_writer()

    def _process_pending_writes(self) -> None:
        """Process pending writes synchronously."""
        while True:
            with self._writer_lock:
                if not self._write_queue:
                    break
                item = self._write_queue.pop(0)
            try:
                op = item[0]
                if op == "save":
                    _, block_hash_hex, tensors_raw, meta_dict, file_path = item
                    file_size = _write_safetensors(file_path, tensors_raw, meta_dict)
                    with self._lock:
                        if block_hash_hex in self._index:
                            self._index[block_hash_hex].file_size = file_size
                            self._sqlite_upsert(block_hash_hex, self._index[block_hash_hex])
                        self._writes_completed += 1
                elif op == "delete":
                    try:
                        os.unlink(item[1])
                    except OSError:
                        pass
            except Exception:
                logger.debug("SSD write error", exc_info=True)

    def load_block(self, block_hash: bytes) -> list | None:
        """Load a KV block from hot cache or SSD.

        Returns:
            Cache data list, or None if not found.
        """
        hex_hash = block_hash.hex()

        with self._lock:
            # Check hot cache first
            if hex_hash in self._hot_cache:
                self._hot_cache.move_to_end(hex_hash)
                self._reads_completed += 1
                return self._hot_cache[hex_hash][0]

            # Check disk index — capture meta while holding the lock
            # to prevent a concurrent delete_block from racing with us.
            meta = self._index.get(hex_hash)
            if meta is None:
                return None

        # Load from disk (blocking, runs on inference thread)
        if meta is None:
            return None

        try:
            import mlx.core as mx
            data, header = mx.load(meta.file_path, return_metadata=True)

            # Reconstruct cache data from safetensors keys
            num_layers = int(header.get("num_layers", "0"))
            cache_data = [None] * num_layers

            for i in range(num_layers):
                keys_key = f"layer_{i}_keys"
                vals_key = f"layer_{i}_values"
                if keys_key in data and vals_key in data:
                    cache_data[i] = (data[keys_key], data[vals_key])
                else:
                    # Collect state_N entries
                    states = []
                    k = 0
                    while f"layer_{i}_state_{k}" in data:
                        states.append(data[f"layer_{i}_state_{k}"])
                        k += 1
                    if states:
                        cache_data[i] = states

            # Promote to hot cache
            token_count = int(header.get("token_count", "0"))
            with self._lock:
                self._hot_cache[hex_hash] = (cache_data, token_count)
                self._hot_cache.move_to_end(hex_hash)
                self._evict_hot_if_full()
                self._reads_completed += 1

            return cache_data

        except Exception:
            logger.debug(f"SSD KV load failed for {hex_hash[:16]}", exc_info=True)
            return None

    def has_block(self, block_hash: bytes) -> bool:
        """Check if a block exists in hot cache or on disk."""
        hex_hash = block_hash.hex()
        with self._lock:
            return hex_hash in self._hot_cache or hex_hash in self._index

    def delete_block(self, block_hash: bytes) -> None:
        """Delete a block from both hot cache and disk."""
        hex_hash = block_hash.hex()
        with self._lock:
            self._hot_cache.pop(hex_hash, None)
            meta = self._index.pop(hex_hash, None)
            self._sqlite_delete(hex_hash)

        if meta is not None and meta.file_path:
            with self._writer_lock:
                self._write_queue.append(("delete", meta.file_path))
            self._process_pending_writes()

    def clear(self) -> int:
        """Clear all cached blocks from memory and disk. Returns count cleared."""
        with self._lock:
            count = len(self._index)
            self._hot_cache.clear()
            paths = [m.file_path for m in self._index.values()]
            self._index.clear()
            # Clear SQLite store
            if self._sqlite_store is not None:
                # Delete all entries individually for correctness
                try:
                    for entry in self._sqlite_store.list_all():
                        self._sqlite_store.delete(entry["block_hash"])
                except Exception:
                    logger.debug("SQLite store clear failed", exc_info=True)

        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass

        return count

    def enforce_size_limit(self) -> int:
        """Evict LRU disk blocks until under budget. Returns count evicted."""
        evicted = 0
        with self._lock:
            total_size = sum(m.file_size for m in self._index.values())
            # Sort by last accessed (oldest first)
            sorted_entries = sorted(
                self._index.items(), key=lambda x: x[1].last_accessed
            )
            for hex_hash, meta in sorted_entries:
                if total_size <= self._get_effective_max():
                    break
                total_size -= meta.file_size
                self._index.pop(hex_hash, None)
                self._hot_cache.pop(hex_hash, None)
                self._sqlite_delete(hex_hash)
                try:
                    os.unlink(meta.file_path)
                except OSError:
                    pass
                evicted += 1

        self._evictions += evicted
        return evicted

    def _evict_hot_if_full(self) -> None:
        """Evict oldest hot cache entries when over capacity."""
        while len(self._hot_cache) > self._hot_cache_size:
            hex_hash, _ = self._hot_cache.popitem(last=False)
            # Data stays on disk — just evict from RAM

    def _get_effective_max(self) -> int:
        """Get effective max size considering disk free space."""
        try:
            stat = os.statvfs(self._cache_dir)
            free = stat.f_frsize * stat.f_bavail
            return min(self._max_size_bytes, int(free * 0.95))
        except Exception:
            logger.debug("statvfs for cache dir failed", exc_info=True)
            return self._max_size_bytes

    def get_stats(self) -> SSDCacheStats:
        """Return cache statistics."""
        with self._lock:
            hot_bytes = 0
            disk_bytes = 0
            for hex_hash in self._hot_cache:
                if hex_hash in self._index:
                    hot_bytes += self._index[hex_hash].file_size
            for meta in self._index.values():
                disk_bytes += meta.file_size

            return SSDCacheStats(
                hot_cache_entries=len(self._hot_cache),
                hot_cache_bytes=hot_bytes,
                disk_entries=len(self._index) - len(self._hot_cache),
                disk_bytes=disk_bytes,
                total_entries=len(self._index),
                max_size_bytes=self._max_size_bytes,
                writes_completed=self._writes_completed,
                reads_completed=self._reads_completed,
                evictions=self._evictions,
            )

    def close(self) -> None:
        """Flush and stop the background writer."""
        self._flush_writer()
        # Close SQLite store
        if self._sqlite_store is not None:
            self._sqlite_store.close()
        # Ensure thread reference is cleared
        self._writer_thread = None
