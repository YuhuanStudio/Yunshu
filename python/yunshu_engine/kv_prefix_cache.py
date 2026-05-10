"""Yunshu KV Prefix Cache — reuse prefilled KV states across requests.

Stores completed request KV caches keyed by prompt token prefix.
When a new request arrives, finds the longest prefix match and
reuses the cached KV state, only prefilling the remaining tokens.

Major speedup for:
- Multi-turn conversations (system prompt + history already cached)
- Batch requests with same system prompt
- API calls with repeated instructions

Design:
- Full-prefix matching (block-level hashing is planned for Phase 5)
- LRU eviction when at capacity
- Detached copies to avoid graph reference leaks
- Supports any cache type with keys/values/offset attributes

Studied from:
- exo's KVPrefixCache (deepcopy-based, LRU by memory pressure)
- oMLX's BlockAwarePrefixCache (block-level hashing, SSD persistence)
"""
from __future__ import annotations

import gc
import hashlib
import logging
from copy import copy
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


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
    import numpy as np
    if a.dtype == mx.bfloat16:
        return mx.array(np.array(a.astype(mx.float32))).astype(mx.bfloat16)
    return mx.array(np.array(a))


def _token_hash(tokens: mx.array) -> str:
    """Hash token array for fast cache key lookup."""
    import numpy as np
    return hashlib.blake2b(np.array(tokens).tobytes(), digest_size=16).hexdigest()


class KVPrefixCache:
    """Cache prefilled KV states keyed by prompt token prefix.

    LRU eviction when capacity is reached. Uses vectorized prefix
    matching (mx.equal + mx.cumprod) for fast lookup.
    """

    def __init__(self, max_entries: int = 64, min_prefix_length: int = 32):
        self._prompts: list[mx.array] = []
        self._caches: list[list] = []
        self._last_used: list[int] = []
        self._access_counter: int = 0
        self._max_entries = max_entries
        self._min_prefix = min_prefix_length
        # Fast lookup: hash of full prompt tokens → index
        self._hash_index: dict[str, int] = {}

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
            self._prompts.pop(old_idx)
            self._caches.pop(old_idx)
            self._last_used.pop(old_idx)
            # Rebuild index after removal shifts indices
            self._rebuild_hash_index()

        self._prompts.append(prompt_copy)
        self._caches.append(cache_copy)
        self._access_counter += 1
        self._last_used.append(self._access_counter)
        self._hash_index[h] = len(self._prompts) - 1

        logger.info(
            f"KV prefix cache added: {len(prompt_tokens)} tokens, "
            f"total entries={len(self._prompts)}"
        )

    def get(
        self,
        prompt_tokens: mx.array,
    ) -> tuple[Optional[list], int, int]:
        """Find best prefix match and return cached KV state.

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

        # Slow path: scan for longest prefix match
        best_index = -1
        best_length = 0

        for i, cached_prompt in enumerate(self._prompts):
            # Skip if cached prompt can't beat current best
            if len(cached_prompt) <= best_length:
                continue
            length = get_prefix_length(prompt_tokens, cached_prompt)
            if length > best_length:
                best_index = i
                best_length = length

        if best_length < self._min_prefix:
            return None, len(prompt_tokens), 0

        # Trim cache to match prefix length
        cached = self._caches[best_index]
        cached_len = cache_length(cached)
        tokens_to_trim = cached_len - best_length

        result = self._snapshot_cache(cached, trim=tokens_to_trim)

        self._touch(best_index)

        remaining = len(prompt_tokens) - best_length
        logger.info(
            f"KV prefix cache hit: matched {best_length}/{len(prompt_tokens)} tokens, "
            f"remaining={remaining}"
        )
        return result, remaining, best_length

    def _snapshot_cache(self, cache: list, trim: int = 0) -> list:
        """Create a detached snapshot of a KV cache."""
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
        """Update LRU timestamp for accessed entry."""
        self._access_counter += 1
        self._last_used[index] = self._access_counter

    def _rebuild_hash_index(self) -> None:
        """Rebuild hash index after structural changes."""
        self._hash_index.clear()
        for i, prompt in enumerate(self._prompts):
            self._hash_index[_token_hash(prompt)] = i

    def _evict_if_full(self) -> None:
        """Evict LRU entries when at capacity."""
        while len(self._prompts) >= self._max_entries:
            lru_index = self._last_used.index(min(self._last_used))
            h = _token_hash(self._prompts[lru_index])
            self._hash_index.pop(h, None)
            self._prompts.pop(lru_index)
            self._caches.pop(lru_index)
            self._last_used.pop(lru_index)
            self._rebuild_hash_index()
            logger.info("KV prefix cache evicted LRU entry (capacity)")

    def clear(self) -> None:
        """Clear all cached entries."""
        self._prompts.clear()
        self._caches.clear()
        self._last_used.clear()
        self._hash_index.clear()
        self._access_counter = 0
        gc.collect()
        mx.clear_cache()

    @property
    def size(self) -> int:
        return len(self._prompts)

    def get_stats(self) -> dict:
        total_tokens = sum(len(p) for p in self._prompts)
        return {
            "entries": len(self._prompts),
            "max_entries": self._max_entries,
            "total_cached_tokens": total_tokens,
            "min_prefix_length": self._min_prefix,
        }
