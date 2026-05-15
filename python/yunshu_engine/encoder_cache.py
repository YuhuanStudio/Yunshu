from __future__ import annotations
"""Yunshu EncoderCacheManager — encoder hidden-state cache for encoder-decoder models.

Audit gap §12.2: Yunshu lacked encoder-decoder support while vLLM has
EncoderCacheManager. This module provides:

- ``EncoderCacheEntry``: stores encoder outputs (hidden states) with a TTL
  deadline for automatic eviction.
- ``EncoderCacheManager``: bounded LRU-like cache that maps request_id to
  encoder hidden states. Supports put/get/evict with configurable max
  entries, TTL, and stats tracking (hits, misses, evictions, memory_bytes).

Design notes (vLLM EncoderCacheManager pattern):
- Encoder outputs are typically large (batch_size × seq_len × hidden_dim)
  so a bounded cache with TTL eviction is essential for memory hygiene.
- TTL defaults to 300 s (5 min) — encoder outputs for a request are only
  needed while the decoder is still generating, which rarely exceeds this.
- ``evict_all_expired()`` is called periodically from the scheduler step
  loop to reclaim memory from stale entries.
- Memory tracking uses ``sys.getsizeof`` as a lower bound and falls back
  to element-count estimation for numpy/MLX arrays.
"""

import logging
import sys
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EncoderCacheEntry:
    """A single cached encoder output with TTL deadline.

    Attributes:
        request_id: The request this encoder output belongs to.
        encoder_outputs: The encoder hidden states (typically an mx.array
            or a list/tuple of layer-wise arrays).
        created_at: Monotonic timestamp when the entry was created.
        deadline: Monotonic timestamp after which the entry is expired.
        memory_bytes: Estimated memory footprint in bytes.
    """

    request_id: str
    encoder_outputs: Any
    created_at: float
    deadline: float
    memory_bytes: int = 0


def _estimate_memory(obj: Any) -> int:
    """Estimate memory footprint of an object (bytes).

    For numpy arrays and MLX arrays, uses ``nbytes`` / ``size`` when
    available. Falls back to ``sys.getsizeof`` for other objects.
    """
    if obj is None:
        return 0

    # Single array with nbytes attribute (numpy, mlx, torch)
    nbytes = getattr(obj, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)

    # Tuple/list of arrays
    if isinstance(obj, (list, tuple)):
        return sum(_estimate_memory(item) for item in obj)

    # Generic object
    return sys.getsizeof(obj)


@dataclass
class EncoderCacheStats:
    """Running statistics for an EncoderCacheManager."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_bytes: int = 0
    num_entries: int = 0
    total_puts: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "memory_bytes": self.memory_bytes,
            "num_entries": self.num_entries,
            "total_puts": self.total_puts,
            "hit_rate": round(self.hits / max(self.hits + self.misses, 1), 4),
        }


class EncoderCacheManager:
    """Bounded LRU cache for encoder hidden states (vLLM EncoderCacheManager pattern).

    Manages encoder outputs keyed by ``request_id`` with:
    - Configurable max entries (default 64) — oldest entry evicted when full.
    - Configurable TTL (default 300 s) — entries expire after deadline.
    - Stats tracking (hits, misses, evictions, memory_bytes).

    Usage::

        cache = EncoderCacheManager(max_entries=128, ttl_seconds=600)
        cache.put("req-abc", encoder_hidden_states)
        states = cache.get("req-abc")
        cache.evict("req-abc")

    Thread-safety: This class is designed for use within the scheduler's
    single-threaded step loop. If concurrent access is needed, callers
    should add their own locking.
    """

    def __init__(
        self,
        max_entries: int = 64,
        ttl_seconds: float = 300.0,
    ) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: dict[str, EncoderCacheEntry] = {}
        self._stats = EncoderCacheStats()
        # Insertion-order tracking for LRU eviction (oldest first)
        self._insertion_order: list[str] = []

    # ── Core API ──

    def put(self, request_id: str, encoder_outputs: Any) -> None:
        """Store encoder outputs for a request.

        If the cache is full, the oldest (first-inserted) entry is evicted.
        If an entry for ``request_id`` already exists, it is replaced
        (memory is updated accordingly).

        Args:
            request_id: Unique identifier for the request.
            encoder_outputs: Encoder hidden states to cache.
        """
        now = time.monotonic()

        # If overwriting an existing entry, remove old one first
        if request_id in self._entries:
            self._remove_entry(request_id)

        # Evict oldest entries if at capacity
        while len(self._entries) >= self.max_entries:
            self._evict_oldest()

        mem_bytes = _estimate_memory(encoder_outputs)
        entry = EncoderCacheEntry(
            request_id=request_id,
            encoder_outputs=encoder_outputs,
            created_at=now,
            deadline=now + self.ttl_seconds,
            memory_bytes=mem_bytes,
        )
        self._entries[request_id] = entry
        self._insertion_order.append(request_id)

        self._stats.total_puts += 1
        self._stats.num_entries = len(self._entries)
        self._stats.memory_bytes += mem_bytes

    def get(self, request_id: str) -> Any | None:
        """Retrieve encoder outputs for a request.

        Returns ``None`` if the entry does not exist or has expired
        (expired entries are evicted on access).

        Args:
            request_id: The request ID to look up.

        Returns:
            Encoder hidden states, or ``None`` if not found / expired.
        """
        entry = self._entries.get(request_id)
        if entry is None:
            self._stats.misses += 1
            return None

        # Check TTL expiration
        if time.monotonic() > entry.deadline:
            self.evict(request_id)
            self._stats.misses += 1
            return None

        self._stats.hits += 1
        return entry.encoder_outputs

    def evict(self, request_id: str) -> bool:
        """Explicitly evict a specific entry.

        Args:
            request_id: The entry to evict.

        Returns:
            ``True`` if the entry was found and evicted, ``False`` otherwise.
        """
        if request_id not in self._entries:
            return False

        self._remove_entry(request_id)
        self._stats.evictions += 1
        return True

    def evict_all_expired(self) -> int:
        """Evict all entries whose TTL has elapsed.

        Should be called periodically (e.g., every N scheduler steps)
        to reclaim memory from stale entries.

        Returns:
            Number of entries evicted.
        """
        now = time.monotonic()
        expired_ids = [
            rid
            for rid, entry in self._entries.items()
            if now > entry.deadline
        ]
        for rid in expired_ids:
            self._remove_entry(rid)
        self._stats.evictions += len(expired_ids)
        if expired_ids:
            logger.debug(
                "EncoderCacheManager: evicted %d expired entries",
                len(expired_ids),
            )
        return len(expired_ids)

    def clear(self) -> None:
        """Remove all entries and reset stats counters."""
        count = len(self._entries)
        self._entries.clear()
        self._insertion_order.clear()
        self._stats.memory_bytes = 0
        self._stats.num_entries = 0
        self._stats.evictions += count
        if count:
            logger.debug("EncoderCacheManager: cleared %d entries", count)

    # ── Stats ──

    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics as a dictionary."""
        self._stats.num_entries = len(self._entries)
        self._stats.memory_bytes = sum(
            e.memory_bytes for e in self._entries.values()
        )
        stats = self._stats.as_dict()
        stats["max_entries"] = self.max_entries
        stats["ttl_seconds"] = self.ttl_seconds
        return stats

    @property
    def num_entries(self) -> int:
        return len(self._entries)

    # ── Internal ──

    def _remove_entry(self, request_id: str) -> None:
        """Remove an entry and update memory tracking."""
        entry = self._entries.pop(request_id, None)
        if entry is not None:
            self._stats.memory_bytes -= entry.memory_bytes
        try:
            self._insertion_order.remove(request_id)
        except ValueError:
            pass

    def _evict_oldest(self) -> None:
        """Evict the oldest (first-inserted) entry to make room."""
        if not self._insertion_order:
            return
        oldest_id = self._insertion_order[0]
        self._remove_entry(oldest_id)
        self._stats.evictions += 1
        logger.debug("EncoderCacheManager: evicted oldest entry %s", oldest_id)
