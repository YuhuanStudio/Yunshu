from __future__ import annotations
from collections import OrderedDict
"""Yunshu N-gram Speculative Decoding — model-free draft token proposal.

Three proposer modes:
1. **LPS mode**: KMP-based O(n) matching via LPS array (vLLM pattern)
2. **HashPool mode**: dict-based O(1) lookup with FIFO eviction
3. **LCGHashPool mode** ("ngram-mod"): O(1) circular-buffer pool using LCG
   hashing for both insertion and eviction (llama.cpp ngram-mod pattern)

The LCGHashPool mode is preferred for code/reasoning tasks where repeated
patterns are common. It uses a fixed-capacity circular buffer with LCG
hashing for O(1) operations at every layer: insert, lookup, and eviction.

Integration:
  - BatchedEngine fast path calls propose() before each decode step
  - Scheduler spec decode loop calls propose() per active request
  - Draft tokens are verified by the target model
  - Accepted tokens are kept; rejected tokens trigger resample

References:
  - vLLM NgramProposer (vllm/v1/spec_decode/ngram_proposer.py)
  - llama.cpp ngram-mod (ggml/src/ggml-common/speculative.h)
  - "Fast Inference from Transformers via Autoregressive Diffusion"
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class NgramConfig:
    """Configuration for N-gram speculative decoding."""
    # Minimum N-gram length to match
    min_n: int = 1
    # Maximum N-gram length to match
    max_n: int = 5
    # Number of draft tokens to propose per step
    k: int = 5
    # Maximum model context length
    max_model_len: int = 32768
    # Proposer mode: "lps" (KMP), "hashpool" (dict lookup), or "lcg" (LCG circular buffer)
    mode: str = "lps"
    # HashPool / LCG: max number of ngram entries before eviction
    hashpool_capacity: int = 1 << 17  # 131072 entries (~16MB for int keys)


def _find_longest_ngram_and_propose(
    token_ids: list[int],
    min_n: int,
    max_n: int,
    max_model_len: int,
    k: int,
) -> list[int]:
    """Find longest N-gram match and propose K tokens following it.

    Searches from longest to shortest ngram length. For each length n,
    takes the suffix of the sequence and scans for an earlier occurrence.
    When found, returns up to K tokens following that occurrence.

    Args:
        token_ids: Full context (prompt + generated tokens).
        min_n: Minimum N-gram length to match.
        max_n: Maximum N-gram length to match.
        max_model_len: Maximum model context length.
        k: Number of draft tokens to propose.

    Returns:
        List of proposed token IDs (may be empty if no match).
    """
    total = len(token_ids)
    if total < min_n + 1:
        return []

    k = min(k, max_model_len - total)
    if k <= 0:
        return []

    for n in range(min(max_n, total - 1), min_n - 1, -1):
        suffix = token_ids[total - n:]
        for i in range(total - n):
            if token_ids[i:i + n] == suffix:
                cont_start = i + n
                cont_end = min(cont_start + k, total)
                return token_ids[cont_start:cont_end]

    return []


class NgramHashPool:
    """Hash-based ngram pool for O(1) draft token lookup (llama.cpp pattern).

    Maintains a dict mapping ngram tuples to continuation token lists.
    For each position in the sequence, stores ngrams of lengths [min_n, max_n]
    and their continuations (up to k tokens following).

    When proposing, looks up the current suffix ngram in the pool and returns
    the stored continuation. O(1) per proposal regardless of context length.

    The pool has a capacity limit with LRU-like eviction to bound memory.
    """

    def __init__(self, config: NgramConfig) -> None:
        self.config = config
        self._pool: OrderedDict[tuple[int, ...], list[int]] = OrderedDict()
        self._capacity = config.hashpool_capacity
        self._total_inserts = 0
        self._total_evictions = 0

    def update(self, token_ids: list[int]) -> None:
        """Index ngrams from the token sequence into the pool.

        Tracks previously indexed length to avoid re-scanning the entire
        sequence on every call. Only new positions are indexed.
        """
        min_n = self.config.min_n
        max_n = self.config.max_n
        k = self.config.k
        total = len(token_ids)

        if total < min_n + 1:
            return

        start = max(0, getattr(self, '_indexed_len', 0) - max_n)
        self._indexed_len = total

        for n in range(min_n, max_n + 1):
            for i in range(max(0, start), total - n):
                ngram = tuple(token_ids[i:i + n])
                cont_start = i + n
                cont_end = min(cont_start + k, total)
                if cont_start < total:
                    self._pool[ngram] = token_ids[cont_start:cont_end]
                    self._pool.move_to_end(ngram)
                    self._total_inserts += 1

        # Evict oldest entries if over capacity (LRU via OrderedDict)
        if len(self._pool) > self._capacity:
            excess = len(self._pool) - self._capacity
            keys_to_evict = list(self._pool.keys())[:excess]
            for key in keys_to_evict:
                del self._pool[key]
            self._total_evictions += excess

    def propose(self, token_ids: list[int]) -> list[int]:
        """Propose draft tokens by looking up current suffix ngrams.

        Tries ngrams from max_n down to min_n, returns the longest match's
        continuation. O(1) per lookup.
        """
        min_n = self.config.min_n
        max_n = self.config.max_n
        total = len(token_ids)

        if total < min_n:
            return []

        for n in range(min(max_n, total), min_n - 1, -1):
            suffix = tuple(token_ids[-n:])
            continuation = self._pool.get(suffix)
            if continuation:
                k = min(self.config.k, self.config.max_model_len - total)
                return continuation[:max(k, 0)] if k > 0 else []

        return []

    def clear(self) -> None:
        """Clear the pool."""
        self._pool.clear()
        self._indexed_len = 0

    def get_stats(self) -> dict:
        return {
            "pool_size": len(self._pool),
            "capacity": self._capacity,
            "total_inserts": self._total_inserts,
            "total_evictions": self._total_evictions,
        }


class LCGHashPool:
    """Fixed-capacity circular-buffer hash pool using LCG hashing (llama.cpp ngram-mod).

    Uses a Linear Congruential Generator for slot indexing, providing O(1)
    insert, lookup, and eviction without any linked list or dict overhead.
    The circular buffer overwrites the oldest slot when full.

    Two parallel arrays:
      - _keys: ngram hash (0 = empty slot)
      - _values: continuation tokens packed as bytes or tuple

    LCG parameters (Knuth, TAOCP Vol 2): a=6364136223846793005, c=1442695040888963407
    """

    __slots__ = (
        "_capacity", "_mask", "_keys", "_ngrams", "_values",
        "_total_inserts", "_total_lookups", "_total_hits", "_total_evictions",
        "_min_n", "_max_n", "_k", "_max_model_len",
        "_indexed_len",
    )

    def __init__(self, config: NgramConfig) -> None:
        self._min_n = config.min_n
        self._max_n = config.max_n
        self._k = config.k
        self._max_model_len = config.max_model_len
        # Round capacity up to power of 2 for bitmask modulo
        cap = config.hashpool_capacity
        cap = 1 << (cap - 1).bit_length() if cap > 0 else 256
        self._capacity = cap
        self._mask = cap - 1
        self._keys = [0] * cap
        self._ngrams: list[tuple[int, ...] | None] = [None] * cap
        self._values: list[tuple[int, ...] | None] = [None] * cap
        self._total_inserts = 0
        self._total_lookups = 0
        self._total_hits = 0
        self._total_evictions = 0

    @staticmethod
    def _hash_ngram(ngram: tuple[int, ...]) -> int:
        """FNV-1a hash for ngram tuples — fast and good distribution."""
        h = 2166136261
        for v in ngram:
            h ^= v & 0xFFFFFFFF
            h = (h * 16777619) & 0xFFFFFFFFFFFFFFFF
        return h if h != 0 else 1  # 0 marks empty

    def _slot(self, h: int, probe: int) -> int:
        """LCG probe sequence: (h + probe * LCG_a) & mask."""
        return (h + probe * 6364136223846793005) & self._mask

    def update(self, token_ids: list[int]) -> None:
        """Index ngrams into the circular buffer (O(1) per insert).

        Tracks previously indexed length to avoid re-scanning the entire
        sequence on every call. Only new positions are indexed.
        """
        min_n = self._min_n
        max_n = self._max_n
        k = self._k
        total = len(token_ids)
        if total < min_n + 1:
            return

        start = max(0, getattr(self, '_indexed_len', 0) - max_n)
        self._indexed_len = total

        for n in range(min_n, max_n + 1):
            for i in range(max(0, start), total - n):
                cont_start = i + n
                if cont_start >= total:
                    break
                ngram = tuple(token_ids[i:i + n])
                cont_end = min(cont_start + k, total)
                continuation = tuple(token_ids[cont_start:cont_end])
                h = self._hash_ngram(ngram)
                self._insert(h, ngram, continuation)
                self._total_inserts += 1

    def _insert(self, h: int, ngram: tuple[int, ...], value: tuple[int, ...]) -> None:
        """Open-addressing insert with LCG probing."""
        for probe in range(8):
            slot = self._slot(h, probe)
            if self._keys[slot] == 0 or (self._keys[slot] == h and self._ngrams[slot] == ngram):
                self._keys[slot] = h
                self._ngrams[slot] = ngram
                self._values[slot] = value
                return
        # All probes occupied — overwrite last probe slot (oldest by collision)
        slot = self._slot(h, 7)
        self._keys[slot] = h
        self._ngrams[slot] = ngram
        self._values[slot] = value
        self._total_evictions += 1

    def propose(self, token_ids: list[int]) -> list[int]:
        """Propose draft tokens by looking up current suffix ngrams (O(1))."""
        min_n = self._min_n
        max_n = self._max_n
        total = len(token_ids)
        self._total_lookups += 1

        if total < min_n:
            return []

        for n in range(min(max_n, total), min_n - 1, -1):
            suffix = tuple(token_ids[-n:])
            h = self._hash_ngram(suffix)
            result = self._lookup(h, suffix)
            if result is not None:
                self._total_hits += 1
                k = min(self._k, self._max_model_len - total)
                if k <= 0:
                    return []
                return list(result[:k])

        return []

    def _lookup(self, h: int, ngram: tuple[int, ...]) -> tuple[int, ...] | None:
        """Open-addressing lookup with LCG probing and ngram verification."""
        for probe in range(8):
            slot = self._slot(h, probe)
            if self._keys[slot] == 0:
                return None
            if self._keys[slot] == h and self._ngrams[slot] == ngram:
                return self._values[slot]
        return None

    def clear(self) -> None:
        """Clear the pool in O(capacity)."""
        for i in range(self._capacity):
            self._keys[i] = 0
            self._ngrams[i] = None
            self._values[i] = None
        self._indexed_len = 0

    def get_stats(self) -> dict:
        occupied = sum(1 for k in self._keys if k != 0)
        return {
            "pool_size": occupied,
            "capacity": self._capacity,
            "load_factor": round(occupied / self._capacity, 3) if self._capacity > 0 else 0.0,
            "total_inserts": self._total_inserts,
            "total_lookups": self._total_lookups,
            "total_hits": self._total_hits,
            "hit_rate": round(self._total_hits / self._total_lookups, 3) if self._total_lookups > 0 else 0.0,
            "total_evictions": self._total_evictions,
        }


class NgramProposer:
    """N-gram based speculative decoding proposer.

    Two modes:
    - "lps": KMP-based O(n) matching (vLLM pattern, good for short contexts)
    - "hashpool": O(1) hash lookup (llama.cpp pattern, good for long contexts)

    Usage:
        proposer = NgramProposer(NgramConfig(min_n=1, max_n=5, k=5))
        draft_tokens = proposer.propose(token_ids)
    """

    def __init__(self, config: NgramConfig) -> None:
        self.config = config
        self._hashpool: Optional[NgramHashPool] = None
        self._lcg_pool: Optional[LCGHashPool] = None
        if config.mode == "hashpool":
            self._hashpool = NgramHashPool(config)
        elif config.mode == "lcg":
            self._lcg_pool = LCGHashPool(config)

    def propose(self, token_ids: list[int]) -> list[int]:
        """Propose draft tokens for a single request.

        Args:
            token_ids: Full context (prompt + generated tokens so far).

        Returns:
            List of proposed draft token IDs.
        """
        # LCG hash pool: O(1) circular-buffer lookup (llama.cpp ngram-mod)
        if self._lcg_pool is not None:
            self._lcg_pool.update(token_ids)
            result = self._lcg_pool.propose(token_ids)
            if result:
                return result
            # Fall through to LPS if LCG pool finds nothing

        # Dict-based hash pool: O(1) dict lookup
        if self._hashpool is not None:
            self._hashpool.update(token_ids)
            result = self._hashpool.propose(token_ids)
            if result:
                return result
            # Fall through to LPS if hashpool finds nothing

        return _find_longest_ngram_and_propose(
            token_ids=token_ids,
            min_n=self.config.min_n,
            max_n=self.config.max_n,
            max_model_len=self.config.max_model_len,
            k=self.config.k,
        )

    def batch_propose(self, batch_token_ids: list[list[int]]) -> list[list[int]]:
        """Propose draft tokens for a batch of requests."""
        return [self.propose(token_ids) for token_ids in batch_token_ids]

    def get_stats(self) -> dict:
        if self._lcg_pool is not None:
            return {"mode": "lcg", **self._lcg_pool.get_stats()}
        if self._hashpool is not None:
            return {"mode": "hashpool", **self._hashpool.get_stats()}
        return {"mode": "lps"}
