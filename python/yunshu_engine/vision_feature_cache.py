from __future__ import annotations

"""Yunshu Vision Feature Cache — two-tier LRU + SSD cache for vision encoder outputs.

Studied from oMLX's cache/vision_feature_cache.py, adapted for Yunshu:
- In-memory LRU (OrderedDict): fast lookup for recently seen images
- SSD persistence (safetensors): survives engine restarts
- Keyed by (model_name, image_hash) for per-model isolation
- Background writer thread for non-blocking SSD writes
- Atomic rename for crash safety

Caches the output of vision_tower + projector (image features projected
into language model space). Avoids re-running the vision encoder when
the same image appears with different text contexts across multi-turn
conversations.
"""

import contextlib
import hashlib
import json
import logging
import os
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class _SSDEntry:
    """Metadata for a cached vision feature stored on SSD."""
    key_hash: str
    model_name: str
    image_hash: str
    file_path: Path
    file_size: int
    created_at: float
    last_access: float
    num_tensors: int = 1


def compute_image_hash(image_data: bytes) -> str:
    """Compute SHA256 hash of raw image bytes for cache keying."""
    return hashlib.sha256(image_data).hexdigest()


def _composite_key(model_name: str, image_hash: str) -> str:
    return f"{model_name}:{image_hash}"


def _composite_hash(model_name: str, image_hash: str) -> str:
    return hashlib.sha256(f"{model_name}:{image_hash}".encode()).hexdigest()


class VisionFeatureCache:
    """Two-tier vision feature cache: in-memory LRU + SSD persistence.

    Args:
        cache_dir: SSD storage directory. None for memory-only mode.
        max_size_bytes: Maximum SSD cache size in bytes (default 10GB).
        max_memory_entries: Maximum in-memory LRU entries (default 20).
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        max_size_bytes: int = 10 * 1024**3,
        max_memory_entries: int = 20,
    ):
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._max_size_bytes = max_size_bytes
        self._max_memory_entries = max_memory_entries

        # In-memory LRU
        self._memory_cache: OrderedDict[str, Any] = OrderedDict()
        self._memory_lock = threading.Lock()

        # SSD index
        self._ssd_index: OrderedDict[str, _SSDEntry] = OrderedDict()
        self._ssd_lock = threading.RLock()
        self._ssd_total_size: int = 0

        # Background writer
        self._write_queue: queue.Queue = queue.Queue(maxsize=32)
        self._writer_shutdown = threading.Event()
        self._pending_write_keys: set = set()
        self._pending_lock = threading.Lock()

        # Stats
        self._stats: dict[str, int] = {
            "hits": 0, "misses": 0, "saves": 0,
            "ssd_loads": 0, "errors": 0,
        }

        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._scan_existing_files()

        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="vision-cache-writer",
        )
        self._writer_thread.start()

    def get(self, image_hash: str, model_name: str) -> Any | None:
        """Look up cached vision features.

        Checks memory LRU first, then SSD. Returns None on miss.
        """
        key = _composite_key(model_name, image_hash)

        with self._memory_lock:
            if key in self._memory_cache:
                self._memory_cache.move_to_end(key)
                self._stats["hits"] += 1
                return self._memory_cache[key]

        if self._cache_dir is not None:
            features = self._load_from_ssd(key)
            if features is not None:
                with self._memory_lock:
                    self._memory_put(key, features)
                self._stats["hits"] += 1
                self._stats["ssd_loads"] += 1
                return features

        self._stats["misses"] += 1
        return None

    def put(self, image_hash: str, model_name: str, features: Any) -> None:
        """Store vision features. Must be called after mx.eval(features)."""
        key = _composite_key(model_name, image_hash)

        with self._memory_lock:
            self._memory_put(key, features)

        if self._cache_dir is not None:
            self._enqueue_ssd_write(key, image_hash, model_name, features)

        self._stats["saves"] += 1

    def close(self) -> None:
        """Shut down the background writer and flush pending writes."""
        self._writer_shutdown.set()
        with contextlib.suppress(queue.Full):
            self._write_queue.put_nowait(None)
        self._writer_thread.join(timeout=10.0)

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def _memory_put(self, key: str, features: Any) -> None:
        if key in self._memory_cache:
            self._memory_cache.move_to_end(key)
            self._memory_cache[key] = features
            return
        while self._memory_cache and len(self._memory_cache) >= self._max_memory_entries:
            self._memory_cache.popitem(last=False)
        self._memory_cache[key] = features

    def _file_path_for_key(self, key: str) -> Path:
        h = hashlib.sha256(key.encode()).hexdigest()
        subdir = self._cache_dir / h[0]
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{h}.safetensors"

    def _enqueue_ssd_write(
        self, key: str, image_hash: str, model_name: str, features: Any,
    ) -> None:
        with self._pending_lock:
            if key in self._pending_write_keys:
                return
            self._pending_write_keys.add(key)

        with self._ssd_lock:
            if key in self._ssd_index:
                self._ssd_index[key].last_access = time.time()
                self._ssd_index.move_to_end(key)
                with self._pending_lock:
                    self._pending_write_keys.discard(key)
                return

        try:
            import numpy as np
            tensors_raw: dict[str, tuple[bytes, str, list[int]]] = {}
            num_tensors = 1

            if isinstance(features, list):
                num_tensors = len(features)
                for i, feat in enumerate(features):
                    arr = np.array(feat).astype(np.float16)
                    tensors_raw[f"feature_{i}"] = (
                        arr.tobytes(), "F16", list(arr.shape),
                    )
            else:
                arr = np.array(features).astype(np.float16)
                tensors_raw["feature"] = (
                    arr.tobytes(), "F16", list(arr.shape),
                )

            metadata = {
                "image_hash": image_hash,
                "model_name": model_name,
                "num_tensors": str(num_tensors),
                "created_at": str(time.time()),
            }

            file_path = self._file_path_for_key(key)
            estimated_size = sum(len(raw) for raw, _, _ in tensors_raw.values())

            now = time.time()
            entry = _SSDEntry(
                key_hash=_composite_hash(model_name, image_hash),
                model_name=model_name,
                image_hash=image_hash,
                file_path=file_path,
                file_size=estimated_size,
                created_at=now,
                last_access=now,
                num_tensors=num_tensors,
            )
            with self._ssd_lock:
                self._ssd_index[key] = entry
                self._ssd_total_size += estimated_size
                self._evict_ssd_if_needed()

            try:
                self._write_queue.put_nowait(
                    (key, tensors_raw, metadata, file_path),
                )
            except queue.Full:
                with self._ssd_lock:
                    if key in self._ssd_index:
                        self._ssd_total_size -= self._ssd_index[key].file_size
                        del self._ssd_index[key]
                with self._pending_lock:
                    self._pending_write_keys.discard(key)

        except Exception as e:
            logger.debug(f"Failed to prepare vision feature for SSD write: {e}")
            with self._pending_lock:
                self._pending_write_keys.discard(key)

    def _evict_ssd_if_needed(self) -> None:
        while self._ssd_total_size > self._max_size_bytes and self._ssd_index:
            _, oldest = self._ssd_index.popitem(last=False)
            self._ssd_total_size -= oldest.file_size
            try:
                if oldest.file_path.exists():
                    oldest.file_path.unlink()
            except Exception:
                logger.debug("failed", exc_info=True)

    def _load_from_ssd(self, key: str) -> Any | None:
        with self._ssd_lock:
            entry = self._ssd_index.get(key)
            if entry is None:
                return None
            file_path = entry.file_path
            num_tensors = entry.num_tensors

        try:
            if not file_path.exists():
                with self._ssd_lock:
                    if key in self._ssd_index:
                        self._ssd_total_size -= self._ssd_index[key].file_size
                        del self._ssd_index[key]
                return None

            arrays = mx.load(str(file_path))

            if num_tensors == 1 and "feature" in arrays:
                features = arrays["feature"]
            else:
                features = []
                for i in range(num_tensors):
                    tensor_key = f"feature_{i}"
                    if tensor_key in arrays:
                        features.append(arrays[tensor_key])
                    else:
                        return None

            with self._ssd_lock:
                if key in self._ssd_index:
                    self._ssd_index[key].last_access = time.time()
                    self._ssd_index.move_to_end(key)

            return features

        except Exception as e:
            logger.warning(f"Failed to load vision features from {file_path}: {e}")
            self._stats["errors"] += 1
            with self._ssd_lock:
                if key in self._ssd_index:
                    self._ssd_total_size -= self._ssd_index[key].file_size
                    del self._ssd_index[key]
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception:
                logger.debug("failed", exc_info=True)
            return None

    def _scan_existing_files(self) -> None:
        if self._cache_dir is None:
            return

        scanned = indexed = errors = 0
        for subdir in self._cache_dir.iterdir():
            if not subdir.is_dir():
                continue
            for file_path in subdir.glob("*.safetensors"):
                scanned += 1
                try:
                    _, metadata = mx.load(str(file_path), return_metadata=True)
                    image_hash = metadata.get("image_hash", "")
                    model_name = metadata.get("model_name", "")
                    num_tensors = int(metadata.get("num_tensors", "1"))

                    if not image_hash or not model_name:
                        errors += 1
                        continue

                    key = _composite_key(model_name, image_hash)
                    stat = file_path.stat()

                    self._ssd_index[key] = _SSDEntry(
                        key_hash=_composite_hash(model_name, image_hash),
                        model_name=model_name,
                        image_hash=image_hash,
                        file_path=file_path,
                        file_size=stat.st_size,
                        created_at=stat.st_ctime,
                        last_access=stat.st_mtime,
                        num_tensors=num_tensors,
                    )
                    self._ssd_total_size += stat.st_size
                    indexed += 1
                except Exception:
                    logger.debug("vision cache SSD file scan failed for %s", file_path, exc_info=True)
                    errors += 1

        if scanned > 0:
            logger.info(
                f"Vision cache scan: scanned={scanned}, indexed={indexed}, "
                f"errors={errors}, total={self._ssd_total_size / (1024**2):.1f}MB",
            )

    def _writer_loop(self) -> None:
        """Background writer thread — writes safetensors files via pure Python I/O."""
        while True:
            try:
                item = self._write_queue.get(timeout=1.0)
            except queue.Empty:
                if self._writer_shutdown.is_set():
                    break
                continue

            if item is None:
                break

            key, tensors_raw, metadata, file_path = item
            temp_path = None

            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = file_path.with_name(file_path.stem + "_tmp.safetensors")

                # Write safetensors file manually (avoids mx.save on non-MLX thread)
                actual_size = self._write_safetensors(
                    str(temp_path), tensors_raw, metadata,
                )
                os.rename(str(temp_path), str(file_path))

                with self._ssd_lock:
                    if key in self._ssd_index:
                        old_size = self._ssd_index[key].file_size
                        self._ssd_index[key].file_size = actual_size
                        self._ssd_total_size += actual_size - old_size

            except Exception as e:
                logger.warning(f"Vision cache write failed: {e}")
                self._stats["errors"] += 1
                with self._ssd_lock:
                    if key in self._ssd_index:
                        self._ssd_total_size -= self._ssd_index[key].file_size
                        del self._ssd_index[key]
                for p in (temp_path, file_path):
                    try:
                        if p is not None and p.exists():
                            p.unlink()
                    except Exception:
                        logger.debug("failed", exc_info=True)
            finally:
                with self._pending_lock:
                    self._pending_write_keys.discard(key)

    @staticmethod
    def _write_safetensors(
        path: str,
        tensors: dict[str, tuple[bytes, str, list[int]]],
        metadata: dict[str, str],
    ) -> int:
        """Write a safetensors file with the given tensors and metadata.

        Returns the actual file size written.
        Uses the safetensors binary format header (JSON header + binary data).
        """
        # Build header
        header: dict[str, Any] = {}
        if metadata:
            header["__metadata__"] = metadata

        # Compute offsets
        offset = 0
        tensor_entries = {}
        for name, (data, dtype, shape) in tensors.items():
            tensor_entries[name] = {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + len(data)],
            }
            offset += len(data)

        header.update(tensor_entries)

        # Encode header as JSON with padding to 8-byte alignment
        header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
        padding = (8 - (len(header_json) + 8) % 8) % 8
        header_json += b" " * padding

        # Write file: [8-byte header_size][header_json][tensor_data...]
        with open(path, "wb") as f:
            f.write(len(header_json).to_bytes(8, "little"))
            f.write(header_json)
            for name in tensor_entries:
                f.write(tensors[name][0])

        return 8 + len(header_json) + sum(len(data) for data, _, _ in tensors.values())
