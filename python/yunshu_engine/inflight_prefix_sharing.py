from __future__ import annotations
"""Inflight prefix sharing — new requests share KV with in-flight prefills.

SGLang pattern: when a request is being prefilled, its partially-built KV
cache is immediately available for other requests that share the same prefix.
This eliminates redundant prefill for system prompts and shared context.

How it works:
1. When _generate_fast() starts prefilling, it registers the token sequence
   and partial KV cache blocks with InflightPrefixTracker.
2. When a new request arrives, _generate_fast() checks the tracker for
   matching in-flight prefixes before falling back to completed prefix cache.
3. When the prefill completes, the full prefix is committed to KVPrefixCache
   and removed from the tracker.

Thread safety: all operations run inside the MLX executor (single-threaded),
so no locking is needed for the tracker itself. The asyncio API uses
thread-safe delegation to the executor.
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class InflightEntry:
    request_id: str
    token_ids: list[int]
    kv_cache_ref: object  # Reference to the MLX prompt cache
    last_update_len: int = 0
    created_at: float = field(default_factory=time.monotonic)
    last_updated_at: float = field(default_factory=time.monotonic)
    model_name: str = ""
    # Track all prefix tuples indexed for this request so _deindex_prefix
    # can clean up every entry, including intermediate prefixes added by update().
    _indexed_prefixes: set[tuple[int, ...]] = field(default_factory=set)


class InflightPrefixTracker:
    """Tracks in-flight prefills and enables prefix sharing between them.

    This is the SGLang cache_unfinished_req pattern adapted for Yunshu's
    single-request fast path. Instead of sharing at the scheduler level
    (which requires continuous batching), we share at the engine level
    by allowing concurrent generate() calls to find each other's partial KV.

    Usage in _generate_fast():
        tracker = get_inflight_tracker()
        # Register our prefill
        tracker.register(req_id, ids, cache)
        # Before prefill, check for shared prefix
        shared = tracker.find_prefix(ids, model_name)
        # After generation, unregister
        tracker.unregister(req_id)
    """

    def __init__(self, max_entries: int = 64, ttl_seconds: float = 300.0):
        self._entries: dict[str, InflightEntry] = {}
        # Token-indexed lookup: first N tokens -> set of request IDs
        self._prefix_index: dict[tuple[int, ...], set[str]] = defaultdict(set)
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._stats = {
            "registrations": 0,
            "prefix_hits": 0,
            "prefix_misses": 0,
            "evictions": 0,
        }

    def register(self, request_id: str, token_ids: list[int],
                 kv_cache: object, model_name: str = "") -> None:
        """Register a new in-flight prefill."""
        with self._lock:
            self._evict_expired()
            if len(self._entries) >= self._max_entries:
                self._evict_oldest()

            entry = InflightEntry(
                request_id=request_id,
                token_ids=list(token_ids),
                kv_cache_ref=kv_cache,
                last_update_len=len(token_ids),
                model_name=model_name,
            )
            self._entries[request_id] = entry
            self._index_prefix(request_id, token_ids)
            self._stats["registrations"] += 1
            logger.debug(
                "inflight prefix registered: req=%s, tokens=%d, model=%s",
                request_id[:12], len(token_ids), model_name,
            )

    def update(self, request_id: str, new_token_ids: list[int]) -> None:
        """Update an in-flight prefill with more tokens (chunked prefill)."""
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is None:
                return
            old_len = len(entry.token_ids)
            entry.token_ids = list(new_token_ids)
            entry.last_update_len = len(new_token_ids)
            entry.last_updated_at = time.monotonic()
            # Update prefix index with new tokens at checkpoint boundaries only.
            # This avoids the O(n) per-update blowup that caused memory leaks
            # when update() indexed every intermediate position but _deindex_prefix
            # only cleaned up at checkpoints.
            self._index_checkpoints(entry, start_offset=old_len)

    # Checkpoint lengths used for prefix indexing — power-of-2 for dense coverage
    _CHECKPOINTS = sorted(set(2 ** i for i in range(0, 14)))

    def find_prefix(self, token_ids: list[int], model_name: str = "") -> InflightEntry | None:
        """Find the longest matching in-flight prefix.

        Searches the tracker for the in-flight request with the longest
        shared prefix with the given token_ids. Returns the entry if found,
        allowing the caller to reuse its KV cache.
        """
        with self._lock:
            if not self._entries or not token_ids:
                self._stats["prefix_misses"] += 1
                return None

            best_entry = None
            best_len = 0

            # Only check at checkpoint boundaries and the full token length,
            # since those are the only lengths indexed by _index_checkpoints.
            max_len = min(len(token_ids), 8192)
            check_lengths = [cl for cl in self._CHECKPOINTS if cl <= max_len]
            if max_len not in check_lengths:
                check_lengths.append(max_len)
            # Search from longest to shortest for early exit
            check_lengths.sort(reverse=True)

            for check_len in check_lengths:
                prefix = tuple(token_ids[:check_len])
                candidates = self._prefix_index.get(prefix)
                if candidates:
                    for req_id in candidates:
                        entry = self._entries.get(req_id)
                        if entry is None:
                            continue
                        # Model isolation: only share within same model
                        if model_name and entry.model_name and model_name != entry.model_name:
                            continue
                        # Verify full prefix match
                        entry_prefix = entry.token_ids[:check_len]
                        if entry_prefix == token_ids[:check_len] and check_len > best_len:
                            best_entry = entry
                            best_len = check_len
                            break  # Found at this length, no need to check others
                    if best_entry is not None:
                        break

            if best_entry is not None:
                self._stats["prefix_hits"] += 1
                logger.debug(
                    "inflight prefix hit: shared=%d tokens from req=%s",
                    best_len, best_entry.request_id[:12],
                )
            else:
                self._stats["prefix_misses"] += 1
            return best_entry

    def unregister(self, request_id: str) -> None:
        """Remove an in-flight prefill (request completed or cancelled)."""
        with self._lock:
            entry = self._entries.pop(request_id, None)
            if entry is None:
                return
            # Clean up ALL prefix index entries using the tracked set
            self._deindex_all(request_id, entry)

    def get_stats(self) -> dict:
        with self._lock:
            return {
                **self._stats,
                "active_entries": len(self._entries),
                "prefix_index_size": sum(len(v) for v in self._prefix_index.values()),
            }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._prefix_index.clear()

    def _index_prefix(self, request_id: str, token_ids: list[int]) -> None:
        """Index prefix at checkpoint boundaries for efficient lookup."""
        entry = self._entries.get(request_id)
        self._index_checkpoints(entry, token_ids=token_ids, start_offset=0)

    def _index_checkpoints(
        self,
        entry: InflightEntry | None,
        token_ids: list[int] | None = None,
        start_offset: int = 0,
    ) -> None:
        """Index prefix entries at checkpoint boundaries.

        Uses the entry's _indexed_prefixes set to track all indexed tuples
        so they can be fully cleaned up during deindex.

        Args:
            entry: The InflightEntry to index. If None, does nothing.
            token_ids: Token IDs to index. Uses entry.token_ids if not given.
            start_offset: Only index checkpoints at or after this offset.
        """
        if entry is None:
            return
        tids = token_ids if token_ids is not None else entry.token_ids
        rid = entry.request_id
        checkpoints = set(2 ** i for i in range(0, 14))
        for cp in sorted(checkpoints):
            if cp <= len(tids) and cp > start_offset:
                prefix = tuple(tids[:cp])
                self._prefix_index[prefix].add(rid)
                entry._indexed_prefixes.add(prefix)
        # Always index the full prefix
        full = tuple(tids)
        if full and len(full) > start_offset:
            self._prefix_index[full].add(rid)
            entry._indexed_prefixes.add(full)

    def _deindex_all(self, request_id: str, entry: InflightEntry) -> None:
        """Remove ALL prefix index entries for a request using tracked prefixes."""
        for prefix in entry._indexed_prefixes:
            bucket = self._prefix_index.get(prefix)
            if bucket is not None:
                bucket.discard(request_id)
                if not bucket:
                    del self._prefix_index[prefix]
        entry._indexed_prefixes.clear()

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [
            rid for rid, entry in self._entries.items()
            if now - entry.last_updated_at > self._ttl_seconds
        ]
        for rid in expired:
            self.unregister(rid)
            self._stats["evictions"] += 1

    def _evict_oldest(self) -> None:
        if not self._entries:
            return
        # Evict least-recently-used (by last_updated_at, not created_at)
        oldest_id = min(self._entries, key=lambda k: self._entries[k].last_updated_at)
        self.unregister(oldest_id)
        self._stats["evictions"] += 1


_tracker_instance: InflightPrefixTracker | None = None
_tracker_lock = threading.Lock()


def get_inflight_tracker() -> InflightPrefixTracker:
    global _tracker_instance
    if _tracker_instance is None:
        with _tracker_lock:
            if _tracker_instance is None:
                _tracker_instance = InflightPrefixTracker()
    return _tracker_instance
