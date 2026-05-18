from __future__ import annotations
"""Yunshu PromptCacheManager — exact-match KV state cache for prompt reuse.

Caches complete prompt KV states keyed by content hash of messages + params.
When a new request matches a cached prompt exactly, reuses the cached KV
state directly — zero prefill compute, instant time-to-first-token.

Design:
  - Content hash of (messages + sampling_params) as cache key
  - LRU eviction with configurable max entries and memory budget
  - Stats tracking: hits, misses, memory usage, entry count
  - Thread-safe for concurrent access from engine loop

Integration:
  EngineCore.add_request()
    → PromptCacheManager.lookup(messages_hash)
      → if hit: inject cached KV, skip prefill
    → PromptCacheManager.store(messages_hash, kv_state)
      → after prefill, cache for future reuse
"""

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached KV state with metadata."""

    key: str
    kv_state: Any
    messages_hash: str
    created_at: float = field(default_factory=time.monotonic)
    last_accessed: float = field(default_factory=time.monotonic)
    access_count: int = 0
    size_bytes: int = 0
    token_count: int = 0


@dataclass
class PromptCacheStats:
    """Accumulated statistics for PromptCacheManager."""

    hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0
    invalidations: int = 0
    total_size_bytes: int = 0


def compute_messages_hash(messages: list[dict], **params) -> str:
    """Compute a content hash for messages and optional parameters.

    Args:
        messages: Chat messages to hash.
        **params: Additional parameters to include in hash
                  (e.g., model, temperature, enable_thinking).

    Returns:
        Hex digest of the blake2b hash.
    """
    payload = {"messages": messages}
    if params:
        # Sort keys for deterministic hashing
        payload["params"] = dict(sorted(params.items()))

    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=24).hexdigest()


class PromptCacheManager:
    """Exact-match KV state cache with LRU eviction.

    Usage:
        cache = PromptCacheManager(max_entries=256, max_memory_mb=512)
        h = compute_messages_hash(messages)
        cache.store(h, kv_state, token_count=1024)
        ...
        entry = cache.lookup(h)
        if entry:
            reuse_kv(entry.kv_state)
    """

    def __init__(
        self,
        max_entries: int = 256,
        max_memory_mb: float = 512.0,
        ttl_seconds: float = 3600.0,
        eviction_on_store: bool = True,
    ) -> None:
        self._max_entries = max_entries
        self._max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self._ttl_seconds = ttl_seconds
        self._eviction_on_store = eviction_on_store

        # OrderedDict for LRU ordering (oldest at front)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._stats = PromptCacheStats()

    # ── Public API ──

    def lookup(self, messages_hash: str) -> Optional[CacheEntry]:
        """Look up a cached KV state by messages hash.

        Args:
            messages_hash: The hash computed from messages + params.

        Returns:
            CacheEntry if found (and not expired), None otherwise.
        """
        with self._lock:
            entry = self._cache.get(messages_hash)
            if entry is None:
                self._stats.misses += 1
                return None

            # Check TTL
            if self._is_expired(entry):
                self._remove_entry(messages_hash)
                self._stats.misses += 1
                self._stats.evictions += 1
                return None

            # Update access metadata (move to end = most recently used)
            entry.last_accessed = time.monotonic()
            entry.access_count += 1
            self._cache.move_to_end(messages_hash)

            self._stats.hits += 1
            return entry

    def store(
        self,
        messages_hash: str,
        kv_state: Any,
        token_count: int = 0,
        size_bytes: int = 0,
    ) -> bool:
        """Store a KV state in the cache.

        Args:
            messages_hash: Hash key for the messages.
            kv_state: The KV cache state to store.
            token_count: Number of tokens in the prefilled state.
            size_bytes: Estimated memory size in bytes.

        Returns:
            True if stored, False if rejected (e.g., too large).
        """
        # Estimate size if not provided
        if size_bytes == 0:
            size_bytes = self._estimate_size(kv_state)

        # Reject if single entry exceeds budget
        if size_bytes > self._max_memory_bytes:
            return False

        with self._lock:
            # If key already exists, update in place
            if messages_hash in self._cache:
                old_entry = self._cache[messages_hash]
                self._stats.total_size_bytes -= old_entry.size_bytes

                entry = CacheEntry(
                    key=messages_hash,
                    kv_state=kv_state,
                    messages_hash=messages_hash,
                    created_at=old_entry.created_at,
                    access_count=old_entry.access_count,
                    size_bytes=size_bytes,
                    token_count=token_count,
                )
                self._cache[messages_hash] = entry
                self._cache.move_to_end(messages_hash)
                self._stats.total_size_bytes += size_bytes
                self._stats.stores += 1
                return True

            # Evict if necessary
            if self._eviction_on_store:
                self._evict_if_needed(size_bytes)

            # Check capacity
            if len(self._cache) >= self._max_entries:
                if not self._evict_one():
                    return False

            entry = CacheEntry(
                key=messages_hash,
                kv_state=kv_state,
                messages_hash=messages_hash,
                size_bytes=size_bytes,
                token_count=token_count,
            )
            self._cache[messages_hash] = entry
            self._stats.total_size_bytes += size_bytes
            self._stats.stores += 1
            return True

    def invalidate(self, messages_hash: str) -> bool:
        """Remove a specific entry from the cache.

        Args:
            messages_hash: Hash key to remove.

        Returns:
            True if the entry was found and removed.
        """
        with self._lock:
            if messages_hash in self._cache:
                self._remove_entry(messages_hash)
                self._stats.invalidations += 1
                return True
            return False

    def invalidate_all(self) -> int:
        """Remove all entries from the cache.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats.total_size_bytes = 0
            self._stats.invalidations += count
            return count

    def has(self, messages_hash: str) -> bool:
        """Check if a key exists in the cache (without updating LRU)."""
        with self._lock:
            return messages_hash in self._cache

    def get_entry_count(self) -> int:
        """Return the number of entries currently in the cache."""
        with self._lock:
            return len(self._cache)

    def get_memory_usage_bytes(self) -> int:
        """Return estimated total memory usage in bytes."""
        with self._lock:
            return self._stats.total_size_bytes

    def prune_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries pruned.
        """
        pruned = 0
        with self._lock:
            expired_keys = [
                k for k, v in self._cache.items() if self._is_expired(v)
            ]
            for key in expired_keys:
                self._remove_entry(key)
                pruned += 1
            self._stats.evictions += pruned
        return pruned

    def get_stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            total = self._stats.hits + self._stats.misses
            return {
                "hits": self._stats.hits,
                "misses": self._stats.misses,
                "hit_rate": round(self._stats.hits / total, 4) if total > 0 else 0.0,
                "stores": self._stats.stores,
                "evictions": self._stats.evictions,
                "invalidations": self._stats.invalidations,
                "entries": len(self._cache),
                "max_entries": self._max_entries,
                "memory_usage_bytes": self._stats.total_size_bytes,
                "max_memory_bytes": self._max_memory_bytes,
                "memory_utilization": round(
                    self._stats.total_size_bytes / max(1, self._max_memory_bytes), 4
                ),
            }

    # ── Internal ──

    def _remove_entry(self, key: str) -> None:
        """Remove an entry and update total size."""
        entry = self._cache.pop(key, None)
        if entry is not None:
            self._stats.total_size_bytes -= entry.size_bytes

    def _evict_if_needed(self, incoming_bytes: int) -> None:
        """Evict LRU entries until there's room for incoming_bytes."""
        target = self._max_memory_bytes - incoming_bytes
        while self._stats.total_size_bytes > target and self._cache:
            if not self._evict_one():
                break

    def _evict_one(self) -> bool:
        """Evict the least recently used entry.

        Returns:
            True if an entry was evicted.
        """
        if not self._cache:
            return False
        # popitem(last=False) evicts the oldest (LRU)
        key, entry = self._cache.popitem(last=False)
        self._stats.total_size_bytes -= entry.size_bytes
        self._stats.evictions += 1
        logger.debug(f"PromptCache: evicted {key} ({entry.size_bytes} bytes)")
        return True

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if an entry has exceeded its TTL."""
        age = time.monotonic() - entry.created_at
        return age > self._ttl_seconds

    @staticmethod
    def _estimate_size(kv_state: Any) -> int:
        """Estimate memory size of a KV state.

        Handles:
        - Lists of mx.array (typical KV cache)
        - Nested lists
        - Raw byte estimates
        """
        if kv_state is None:
            return 0

        total = 0

        if isinstance(kv_state, list):
            for item in kv_state:
                total += PromptCacheManager._estimate_size(item)
        elif hasattr(kv_state, "nbytes"):
            # mx.array or numpy array
            total = kv_state.nbytes
        elif hasattr(kv_state, "shape") and hasattr(kv_state, "dtype"):
            # Estimate from shape + dtype
            try:
                itemsize = {
                    "float32": 4, "float16": 2, "bfloat16": 2,
                    "int32": 4, "int64": 8, "bool": 1,
                }.get(str(kv_state.dtype), 4)
                total = 1
                for dim in kv_state.shape:
                    total *= dim
                total *= itemsize
            except Exception:
                logger.debug("tensor size estimation failed", exc_info=True)
                total = 1024  # safe default per item
        else:
            total = 1024  # safe default per item

        return total
