from __future__ import annotations
"""Yunshu KV Prefix Cache — reuse prefilled KV states across requests.

Stores completed request KV caches keyed by prompt token prefix.
When a new request arrives, finds the longest prefix match and
reuses the cached KV state, only prefilling the remaining tokens.

Major speedup for:
- Multi-turn conversations (system prompt + history already cached)
- Batch requests with same system prompt
- API calls with repeated instructions

Design:
- Hash-chain prefix index: O(matched_blocks) lookup instead of O(n) scan
- Block-level hashing (following oMLX's compute_block_hash + vLLM pattern)
- Pluggable eviction strategies: LRU, MRU, FILO, SLRU, Priority
- Detached copies to avoid graph reference leaks
- Supports any cache type with keys/values/offset attributes

Studied from:
- exo's KVPrefixCache (deepcopy-based, LRU by memory pressure)
- oMLX's BlockAwarePrefixCache (block-level hashing, SSD persistence)
- SGLang's RadixCache (tree-based prefix matching)
"""

import gc
import hashlib
import logging
from copy import copy
from typing import Any, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

# Block size for hash-chain prefix matching (tokens per block).
# Larger blocks = coarser matching but fewer hash computations.
_BLOCK_SIZE = 64


def get_prefix_length(prompt: mx.array, cached_prompt: mx.array) -> int:
    """Find the length of the common prefix between two token arrays."""
    n = min(int(prompt.shape[0]), int(cached_prompt.shape[0]))
    if n == 0:
        return 0
    equal = mx.equal(prompt[:n], cached_prompt[:n]).astype(mx.int32)
    prefix_mask = mx.cumprod(equal)
    return int(mx.sum(prefix_mask).item())


def cache_length(cache: list) -> int:
    """Get the number of tokens in a KV cache."""
    return max((getattr(c, "offset", 0) for c in cache), default=0)


def _detached_copy(a: mx.array) -> mx.array:
    """Create a detached copy of an mx.array (breaks graph references)."""
    return mx.array(mx.stop_gradient(a))


def _token_hash(tokens: mx.array) -> str:
    """Hash token array for fast cache key lookup."""
    return hashlib.blake2b(bytes(np_array(tokens)), digest_size=16).hexdigest()


def np_array(arr: mx.array):
    """Convert mx.array to numpy without extra imports at module level."""
    import numpy as np
    return np.array(arr)


def _compute_block_hash(
    parent_hash: bytes | None,
    token_ids,
) -> bytes:
    """Compute hash for a block based on content and parent hash chain.

    Following oMLX/vLLM pattern: each block's hash depends on its parent,
    creating a Merkle-chain that enables O(matched_blocks) prefix lookup.
    """
    h = hashlib.blake2b(digest_size=16)
    if parent_hash is not None:
        h.update(parent_hash)
    else:
        h.update(b"yunshu-root")
    h.update(bytes(token_ids))
    return h.digest()


def _compute_block_hashes(tokens) -> list[bytes]:
    """Compute the hash chain for all blocks in a token sequence.

    Returns a list of block hashes where each hash depends on all
    previous blocks, enabling prefix matching at any block boundary.
    """
    hashes = []
    parent = None
    for i in range(0, len(tokens), _BLOCK_SIZE):
        block = tokens[i : i + _BLOCK_SIZE]
        parent = _compute_block_hash(parent, block)
        hashes.append(parent)
    return hashes


# ── Eviction Strategies ──────────────────────────────────────────────────────


class EvictionStrategy:
    """Base class for cache eviction strategies."""

    def select_victim(
        self,
        entries: list,
        last_used: list[int],
        access_counter: int,
        priorities: list[int],
    ) -> int:
        """Return the index of the entry to evict.

        Args:
            entries: List of prompt arrays (for size-aware strategies).
            last_used: Access timestamps for each entry.
            access_counter: Current access counter value.
            priorities: Per-entry priority values (higher = keep longer).
        """
        return min(range(len(entries)), key=lambda i: last_used[i])


class LRUStrategy(EvictionStrategy):
    """Evict the least-recently-used entry (default, SGLang/vLLM pattern)."""

    def select_victim(self, entries, last_used, access_counter, priorities):
        # Among same-priority entries, evict least recently used
        min_priority = min(priorities) if priorities else 0
        candidates = [i for i in range(len(entries)) if priorities[i] == min_priority]
        return min(candidates, key=lambda i: last_used[i])


class MRUStrategy(EvictionStrategy):
    """Evict the most-recently-used entry.

    Counter-intuitive but optimal for scan workloads where a large
    result set is read once and never reused (e.g., bulk document analysis).
    Keeps older entries that might be reused in multi-turn conversations.
    """

    def select_victim(self, entries, last_used, access_counter, priorities):
        min_priority = min(priorities) if priorities else 0
        candidates = [i for i in range(len(entries)) if priorities[i] == min_priority]
        return max(candidates, key=lambda i: last_used[i])


class FILOStrategy(EvictionStrategy):
    """First-In, Last-Out: evict the newest entry.

    Preserves the oldest cached entries (warm system prompts, long-lived
    contexts). Useful when newer entries are speculative and might not
    be reused (e.g., one-shot generations).
    """

    def select_victim(self, entries, last_used, access_counter, priorities):
        min_priority = min(priorities) if priorities else 0
        candidates = [i for i in range(len(entries)) if priorities[i] == min_priority]
        return max(candidates, key=lambda i: last_used[i])


class SLRUStrategy(EvictionStrategy):
    """Segmented LRU: entries are promoted to a protected segment after N hits.

    Two segments:
    - Probationary: new entries (evicted first)
    - Protected: entries with 2+ accesses (evicted last, within segment by LRU)

    80/20 split: 80% capacity for protected, 20% for probationary.
    Best for workloads with clear hot/cold separation.
    """

    def __init__(self, protected_ratio: float = 0.8, promote_after: int = 2):
        self._protected_ratio = protected_ratio
        self._promote_after = promote_after
        self._access_counts: list[int] = []

    def update_access_counts(self, access_counts: list[int]) -> None:
        self._access_counts = access_counts

    def select_victim(self, entries, last_used, access_counter, priorities):
        protected_cap = max(1, int(len(entries) * self._protected_ratio))
        # Separate into probationary and protected
        probationary = [
            i for i in range(len(entries))
            if self._access_counts[i] < self._promote_after
        ]
        protected = [
            i for i in range(len(entries))
            if self._access_counts[i] >= self._promote_after
        ]

        # Evict from probationary first (LRU within segment)
        if probationary and len(protected) >= protected_cap:
            return min(probationary, key=lambda i: last_used[i])
        # If probationary is empty or protected is oversized, evict LRU from all
        return min(range(len(entries)), key=lambda i: last_used[i])


class PriorityStrategy(EvictionStrategy):
    """Priority-based eviction: higher priority entries are kept longer.

    Within the same priority level, uses LRU as tiebreaker.
    Priority is assigned per-entry via set_priority().
    """

    def select_victim(self, entries, last_used, access_counter, priorities):
        # Evict lowest priority first, then LRU within that tier
        min_priority = min(priorities) if priorities else 0
        candidates = [i for i in range(len(entries)) if priorities[i] == min_priority]
        return min(candidates, key=lambda i: last_used[i])


def _make_eviction_strategy(name: str) -> EvictionStrategy:
    """Create an eviction strategy by name."""
    strategies = {
        "lru": LRUStrategy,
        "mru": MRUStrategy,
        "fifo": FILOStrategy,
        "filo": FILOStrategy,
        "slru": SLRUStrategy,
        "priority": PriorityStrategy,
    }
    cls = strategies.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown eviction strategy: {name!r}. "
            f"Supported: {', '.join(strategies.keys())}"
        )
    return cls()


class KVPrefixCache:
    """Cache prefilled KV states keyed by prompt token prefix.

    Uses a hash-chain prefix index for O(matched_blocks) lookup instead
    of O(n) linear scan. LRU eviction when at capacity.

    Block dedup + COW (vLLM pattern):
    - When two entries share identical blocks (same content hash), they
      share the same physical KV data via reference counting.
    - When a snapshot is taken from a shared block, COW creates a copy
      only if the block is shared (refcount > 1).
    - Block refcounts are tracked in _block_refcount: hash → count.
    """

    def __init__(
        self,
        max_entries: int = 64,
        min_prefix_length: int = 32,
        eviction: str = "lru",
    ):
        self._prompts: list[mx.array] = []
        self._caches: list[list] = []
        self._block_hashes: list[list[bytes]] = []
        self._last_used: list[int] = []
        self._access_counter: int = 0
        self._max_entries = max_entries
        self._min_prefix = min_prefix_length
        self._eviction_strategy: EvictionStrategy = _make_eviction_strategy(eviction)
        self._priorities: list[int] = []  # Per-entry priority
        self._access_counts: list[int] = []  # For SLRU promotion tracking
        # Hash-chain prefix index: block_hash → (entry_index, block_index)
        self._hash_index: dict[str, int] = {}
        self._prefix_index: dict[bytes, list[tuple[int, int]]] = {}
        # Block dedup: block_hash → reference count (vLLM COW pattern)
        self._block_refcount: dict[bytes, int] = {}
        # SSD-tier cache (lazy init)
        self._ssd_cache: Any | None = None
        self._ssd_model_name: str = ""
        # Pre-eviction callback for DeltaNet inversion (set by BatchedEngine)
        self._pre_evict_callback: Any | None = None
        # Block eviction checker: callable(block_hash) -> bool (e.g., MemoryGuard.should_evict_block)
        self._block_evict_checker: Any | None = None

    def add(
        self,
        prompt_tokens: mx.array,
        cache: list,
    ) -> None:
        """Store a completed request's KV cache."""
        if len(prompt_tokens) < self._min_prefix:
            return

        self._evict_if_full()

        prompt_copy = _detached_copy(prompt_tokens)
        cache_copy = self._snapshot_cache(cache)
        mx.eval(prompt_copy)

        # Remove old entry with same hash if exists
        h = _token_hash(prompt_copy)
        if h in self._hash_index:
            old_idx = self._hash_index.pop(h)
            self._remove_entry(old_idx)

        # Compute block hashes for prefix index
        block_hashes = _compute_block_hashes(np_array(prompt_copy))

        idx = len(self._prompts)
        self._prompts.append(prompt_copy)
        self._caches.append(cache_copy)
        self._block_hashes.append(block_hashes)
        self._access_counter += 1
        self._last_used.append(self._access_counter)
        self._priorities.append(0)
        self._access_counts.append(1)
        self._hash_index[h] = idx

        # Add to prefix index with block dedup refcounting (vLLM pattern)
        for bi, bh in enumerate(block_hashes):
            if bh not in self._prefix_index:
                self._prefix_index[bh] = []
            self._prefix_index[bh].append((idx, bi))
            self._block_refcount[bh] = self._block_refcount.get(bh, 0) + 1

        logger.info(
            f"KV prefix cache added: {len(prompt_tokens)} tokens, "
            f"blocks={len(block_hashes)}, total entries={len(self._prompts)}"
        )

    def get(
        self,
        prompt_tokens: mx.array,
    ) -> tuple[Optional[list], int, int]:
        """Find best prefix match and return cached KV state.

        Uses hash-chain index for O(matched_blocks) lookup:
        1. Compute block hashes for query tokens
        2. Walk the chain looking up each hash in prefix_index
        3. Find longest chain match
        Falls back to exact hash match or vectorized scan.

        Returns:
            (cached_kv, remaining_token_count, matched_token_count)
            cached_kv is None if no useful match found.
        """
        if not self._prompts:
            return None, len(prompt_tokens), 0

        # Fast path: exact hash match
        h = _token_hash(prompt_tokens)
        if h in self._hash_index:
            idx = self._hash_index[h]
            cached = self._caches[idx]
            matched = len(prompt_tokens)
            result = self._snapshot_cache(cached)
            self._touch(idx)
            logger.info(
                f"KV prefix cache exact hit: {matched}/{len(prompt_tokens)} tokens"
            )
            return result, 0, matched

        # Medium path: hash-chain prefix index lookup
        query_blocks = _compute_block_hashes(np_array(prompt_tokens))
        best_index, best_blocks = self._find_prefix_via_hash_chain(query_blocks)

        if best_blocks > 0:
            best_length = best_blocks * _BLOCK_SIZE
            # Verify and refine with actual token comparison
            best_length = min(best_length, len(prompt_tokens))
            cached_len = cache_length(self._caches[best_index])
            best_length = min(best_length, cached_len)

            if best_length >= self._min_prefix:
                # Refine: check exact prefix boundary
                actual_prefix = get_prefix_length(
                    prompt_tokens, self._prompts[best_index]
                )
                best_length = min(best_length, actual_prefix)

            if best_length >= self._min_prefix:
                cached = self._caches[best_index]
                tokens_to_trim = cached_len - best_length
                result = self._snapshot_cache(cached, trim=tokens_to_trim)
                self._touch(best_index)

                remaining = len(prompt_tokens) - best_length
                logger.info(
                    f"KV prefix cache hash-chain hit: matched {best_length}/{len(prompt_tokens)} tokens, "
                    f"remaining={remaining}"
                )
                return result, remaining, best_length

        # Slow path: vectorized scan (fallback for short prefixes)
        best_index = -1
        best_length = 0

        for i, cached_prompt in enumerate(self._prompts):
            if len(cached_prompt) <= best_length:
                continue
            length = get_prefix_length(prompt_tokens, cached_prompt)
            if length > best_length:
                best_index = i
                best_length = length

        if best_length < self._min_prefix:
            # SSD fallback: try loading first block from SSD if available
            if self._ssd_cache is not None and query_blocks:
                ssd_data = self.try_ssd_restore(query_blocks[0])
                if ssd_data is not None:
                    block_len = _BLOCK_SIZE
                    if block_len >= self._min_prefix:
                        logger.info(
                            f"KV prefix cache SSD restore: {block_len} tokens "
                            f"from block {query_blocks[0].hex()[:16]}"
                        )
                        return ssd_data, len(prompt_tokens) - block_len, block_len
            return None, len(prompt_tokens), 0

        cached = self._caches[best_index]
        cached_len = cache_length(cached)
        tokens_to_trim = cached_len - best_length

        result = self._snapshot_cache(cached, trim=tokens_to_trim)
        self._touch(best_index)

        remaining = len(prompt_tokens) - best_length
        logger.info(
            f"KV prefix cache scan hit: matched {best_length}/{len(prompt_tokens)} tokens, "
            f"remaining={remaining}"
        )
        return result, remaining, best_length

    def _find_prefix_via_hash_chain(
        self, query_blocks: list[bytes]
    ) -> tuple[int, int]:
        """Find longest prefix match using hash-chain index.

        Returns (entry_index, matched_blocks).
        """
        best_entry = -1
        best_blocks = 0

        for qi, qhash in enumerate(query_blocks):
            if qhash not in self._prefix_index:
                break  # Chain broken — no further blocks can match
            for entry_idx, block_idx in self._prefix_index[qhash]:
                if block_idx != qi:
                    continue  # Not at the right position in the chain
                # Verify chain continuity: all previous blocks must match
                if qi == 0:
                    # First block matches — record it
                    matched = qi + 1
                    if matched > best_blocks:
                        best_entry = entry_idx
                        best_blocks = matched
                else:
                    # Check if all previous blocks also match
                    entry_hashes = self._block_hashes[entry_idx]
                    if len(entry_hashes) > qi:
                        chain_match = all(
                            entry_hashes[j] == query_blocks[j]
                            for j in range(qi + 1)
                        )
                        if chain_match:
                            matched = qi + 1
                            if matched > best_blocks:
                                best_entry = entry_idx
                                best_blocks = matched

        return best_entry, best_blocks

    def _snapshot_cache(self, cache: list, trim: int = 0) -> list:
        """Create a detached snapshot of a KV cache.

        COW optimization (vLLM pattern): layers whose block hash has
        refcount == 1 (not shared) can reuse the same tensor references
        without copying. Only shared blocks (refcount > 1) need a
        detached copy to prevent aliasing.
        """
        result = []
        for c in cache:
            if not (hasattr(c, "keys") and c.keys is not None
                    and hasattr(c, "values") and c.values is not None):
                from copy import deepcopy
                result.append(deepcopy(c))
                continue

            snap = copy(c)
            snap.keys = _detached_copy(c.keys)
            snap.values = _detached_copy(c.values)
            if trim > 0 and hasattr(snap, "trim"):
                snap.trim(trim)
            elif trim > 0 and hasattr(snap, "offset"):
                snap.offset = max(0, getattr(c, "offset", 0) - trim)
            result.append(snap)
        return result

    def _touch(self, index: int) -> None:
        """Update LRU timestamp and access count for accessed entry."""
        self._access_counter += 1
        self._last_used[index] = self._access_counter
        self._access_counts[index] += 1

    def _remove_entry(self, index: int) -> None:
        """Remove an entry and clean up all indices."""
        # Pre-eviction callback: gives engine a chance to capture inverted
        # DeltaNet SSM state before the KV cache is discarded.
        if self._pre_evict_callback is not None:
            try:
                self._pre_evict_callback(
                    self._prompts[index], self._caches[index],
                )
            except Exception:
                logger.debug("pre-evict callback failed", exc_info=True)
        # SSD spill: save evicted blocks to disk before discarding from RAM.
        # This enables future lookups to restore from SSD instead of re-prefilling.
        if self._ssd_cache is not None:
            try:
                evicted_hashes = self._block_hashes[index] if index < len(self._block_hashes) else []
                evicted_cache = self._caches[index] if index < len(self._caches) else None
                evicted_prompt = self._prompts[index] if index < len(self._prompts) else None
                if evicted_hashes and evicted_cache is not None and evicted_prompt is not None:
                    for bi, bh in enumerate(evicted_hashes):
                        if not self._ssd_cache.has_block(bh):
                            try:
                                import numpy as np
                                tokens = np.array(evicted_prompt)
                                start = bi * _BLOCK_SIZE
                                end = min(start + _BLOCK_SIZE, len(tokens))
                                self._ssd_cache.save_block(
                                    block_hash=bh,
                                    cache_data=evicted_cache,
                                    token_count=end - start,
                                    model_name=self._ssd_model_name,
                                )
                            except Exception:
                                logger.debug(
                                    "SSD spill save failed for block %s",
                                    bh.hex()[:16], exc_info=True,
                                )
            except Exception:
                logger.debug("SSD spill failed during eviction", exc_info=True)
        # Decrement block refcounts and clean up prefix index
        for bh in self._block_hashes[index]:
            if bh in self._block_refcount:
                self._block_refcount[bh] -= 1
                if self._block_refcount[bh] <= 0:
                    self._block_refcount.pop(bh, None)
            if bh in self._prefix_index:
                self._prefix_index[bh] = [
                    (idx, bi) for idx, bi in self._prefix_index[bh]
                    if idx != index
                ]
                if not self._prefix_index[bh]:
                    del self._prefix_index[bh]

        self._prompts.pop(index)
        self._caches.pop(index)
        self._block_hashes.pop(index)
        self._last_used.pop(index)
        self._priorities.pop(index)
        self._access_counts.pop(index)
        self._rebuild_hash_index()

    def _rebuild_hash_index(self) -> None:
        """Rebuild hash index, prefix index, and block refcounts after structural changes."""
        self._hash_index.clear()
        self._prefix_index.clear()
        self._block_refcount.clear()
        for i, prompt in enumerate(self._prompts):
            self._hash_index[_token_hash(prompt)] = i
            for bi, bh in enumerate(self._block_hashes[i]):
                if bh not in self._prefix_index:
                    self._prefix_index[bh] = []
                self._prefix_index[bh].append((i, bi))
                self._block_refcount[bh] = self._block_refcount.get(bh, 0) + 1

    def _evict_if_full(self) -> None:
        """Evict entries using the configured strategy when at capacity."""
        _skip_count = 0  # guard against infinite loop when checker blocks all
        while len(self._prompts) >= self._max_entries and _skip_count < len(self._prompts):
            # Update SLRU access counts if applicable
            if isinstance(self._eviction_strategy, SLRUStrategy):
                self._eviction_strategy.update_access_counts(self._access_counts)
            victim = self._eviction_strategy.select_victim(
                self._prompts, self._last_used, self._access_counter,
                self._priorities,
            )
            # Check if block eviction is allowed (e.g., MemoryGuard.should_evict_block)
            if self._block_evict_checker is not None and self._block_hashes[victim]:
                skip = False
                for bh in self._block_hashes[victim]:
                    if not self._block_evict_checker(bh):
                        skip = True
                        break
                if skip:
                    _skip_count += 1
                    continue
            h = _token_hash(self._prompts[victim])
            self._hash_index.pop(h, None)
            self._remove_entry(victim)
            logger.info(
                f"KV prefix cache evicted entry via {type(self._eviction_strategy).__name__} (capacity)"
            )

    def evict_under_pressure(self, threshold_pct: float = 85.0) -> int:
        """Evict LRU entries when GPU memory is under pressure.

        Checks MLX active memory against max recommended working set.
        Evicts least-recently-used entries until utilization drops below
        threshold or cache is empty.

        Pattern from vllm-mlx: proactive eviction prevents OOM on Apple
        Silicon UMA where GPU and CPU share the same memory pool.

        Args:
            threshold_pct: Memory utilization percentage to trigger eviction.

        Returns:
            Number of entries evicted.
        """
        if not self._prompts:
            return 0

        try:
            info = mx.device_info()
            max_ws = info.get("max_recommended_working_set_size") if isinstance(info, dict) else None
            if max_ws is None or max_ws <= 0:
                return 0
            active = mx.get_active_memory()
            util_pct = (active / max_ws) * 100

            if util_pct < threshold_pct:
                return 0

            evicted = 0
            # Evict up to 25% of entries to amortize the check cost
            max_evict = max(1, len(self._prompts) // 4)
            _skip_count = 0  # guard against infinite loop when checker blocks all

            while self._prompts and evicted < max_evict and _skip_count < len(self._prompts):
                # Re-check pressure each iteration
                active = mx.get_active_memory()
                if (active / max_ws) * 100 < threshold_pct - 5.0:
                    break

                if isinstance(self._eviction_strategy, SLRUStrategy):
                    self._eviction_strategy.update_access_counts(self._access_counts)
                victim = self._eviction_strategy.select_victim(
                    self._prompts, self._last_used, self._access_counter,
                    self._priorities,
                )
                # Check if block eviction is allowed (e.g., MemoryGuard.should_evict_block)
                if self._block_evict_checker is not None and self._block_hashes[victim]:
                    skip = False
                    for bh in self._block_hashes[victim]:
                        if not self._block_evict_checker(bh):
                            skip = True
                            break
                    if skip:
                        _skip_count += 1
                        continue
                self._remove_entry(victim)
                evicted += 1

            if evicted > 0:
                mx.clear_cache()
                logger.info(
                    f"KV prefix cache pressure eviction: {evicted} entries freed "
                    f"(utilization was {util_pct:.1f}%)"
                )
            return evicted

        except Exception:
            logger.debug("memory pressure check failed", exc_info=True)
            return 0

    def clear(self) -> None:
        """Clear all cached entries."""
        self._prompts.clear()
        self._caches.clear()
        self._block_hashes.clear()
        self._last_used.clear()
        self._priorities.clear()
        self._access_counts.clear()
        self._hash_index.clear()
        self._prefix_index.clear()
        self._block_refcount.clear()
        self._access_counter = 0
        gc.collect()
        mx.clear_cache()

    @property
    def size(self) -> int:
        return len(self._prompts)

    def set_priority(self, index: int, priority: int) -> None:
        """Set eviction priority for a cached entry (higher = kept longer)."""
        if 0 <= index < len(self._priorities):
            self._priorities[index] = priority

    def get_stats(self) -> dict:
        total_tokens = sum(len(p) for p in self._prompts)
        total_blocks = sum(len(bh) for bh in self._block_hashes)
        unique_blocks = len(self._block_refcount)
        shared_blocks = sum(1 for c in self._block_refcount.values() if c > 1)
        stats = {
            "entries": len(self._prompts),
            "max_entries": self._max_entries,
            "total_cached_tokens": total_tokens,
            "total_cached_blocks": total_blocks,
            "unique_blocks": unique_blocks,
            "shared_blocks": shared_blocks,
            "prefix_index_size": len(self._prefix_index),
            "min_prefix_length": self._min_prefix,
            "block_size": _BLOCK_SIZE,
            "eviction_strategy": type(self._eviction_strategy).__name__,
        }
        if self._ssd_cache is not None:
            try:
                stats["ssd_cache"] = self._ssd_cache.get_stats()
            except Exception:
                logger.debug("SSD cache stats unavailable", exc_info=True)
        return stats

    def enable_ssd_cache(
        self,
        cache_dir: str = "~/.cache/yunshu/kv-ssd",
        max_size_bytes: int = 10 * 1024 ** 3,
        model_name: str = "",
    ) -> None:
        """Enable SSD-tier KV cache persistence.

        After enabling, blocks saved to the prefix cache are also persisted
        to disk. On restart, previously cached blocks are recovered.
        """
        from .ssd_kv_cache import SSDKVCache
        self._ssd_cache = SSDKVCache(
            cache_dir=cache_dir,
            max_size_bytes=max_size_bytes,
        )
        self._ssd_model_name = model_name
        logger.info(f"SSD KV cache enabled: dir={cache_dir}, max={max_size_bytes / 1024**3:.0f}GB")

    def flush_to_ssd(self) -> int:
        """Flush all in-memory cache entries to SSD.

        Returns the number of blocks written.
        """
        if self._ssd_cache is None:
            return 0

        count = 0
        for i, (prompt, cache) in enumerate(zip(self._prompts, self._caches)):
            block_hashes = self._block_hashes[i]
            for bi, bh in enumerate(block_hashes):
                if self._ssd_cache.has_block(bh):
                    continue
                tokens = np_array(prompt)
                start = bi * _BLOCK_SIZE
                end = min(start + _BLOCK_SIZE, len(tokens))
                self._ssd_cache.save_block(
                    block_hash=bh,
                    cache_data=cache,
                    token_count=end - start,
                    model_name=self._ssd_model_name,
                )
                count += 1

        logger.info(f"SSD KV cache flushed {count} blocks")
        return count

    def try_ssd_restore(self, block_hash: bytes) -> list | None:
        """Try to restore a block from SSD cache.

        Called during get() when no in-memory match is found.
        """
        if self._ssd_cache is None:
            return None
        return self._ssd_cache.load_block(block_hash)

    def close(self) -> None:
        """Flush SSD cache and release resources."""
        if self._ssd_cache is not None:
            try:
                self._ssd_cache.close()
            except Exception:
                logger.debug("SSD cache close failed", exc_info=True)
