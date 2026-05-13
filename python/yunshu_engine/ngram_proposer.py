"""Yunshu N-gram Speculative Decoding — model-free draft token proposal.

Two proposer modes:
1. **LPS mode** (default): KMP-based O(n) matching via LPS array (vLLM pattern)
2. **HashPool mode**: O(1) lookup via hash map of ngram→continuation (llama.cpp pattern)

The HashPool mode (ngram-mod) is preferred for code/reasoning tasks where
repeated patterns are common across different positions. It maintains a
dict mapping (ngram tuple) → list of continuation tokens, enabling O(1)
lookup for the current suffix ngram.

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
from __future__ import annotations

import logging
from dataclasses import dataclass, field
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
    # Proposer mode: "lps" (KMP) or "hashpool" (O(1) dict lookup)
    mode: str = "lps"
    # HashPool: max number of ngram entries before eviction
    hashpool_capacity: int = 1 << 17  # 131072 entries (~16MB for int keys)


def _find_longest_ngram_and_propose(
    token_ids: list[int],
    min_n: int,
    max_n: int,
    max_model_len: int,
    k: int,
) -> list[int]:
    """Find longest N-gram match and propose K tokens following it.

    Uses KMP LPS algorithm on reversed tokens for O(n) matching.
    When a suffix of the sequence matches a previous N-gram (length
    in [min_n, max_n]), the tokens following that match are proposed
    as draft tokens.

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
    if total < min_n:
        return []

    k = min(k, max_model_len - total)
    if k <= 0:
        return []

    # Reverse tokens — matching suffix becomes matching prefix
    tokens = token_ids[::-1]

    # LPS array: longest proper prefix which is also a suffix
    # for each prefix of the reversed sequence
    lps = [0] * min(max_n, total)

    longest_ngram = 0
    position = 0

    prev_lps = 0
    i = 1
    while i < total:
        if tokens[prev_lps] == tokens[i]:
            prev_lps += 1
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_n:
                lps[i] = prev_lps
            if prev_lps == max_n:
                prev_lps = lps[max_n - 1]
            i += 1
        elif prev_lps != 0:
            prev_lps = lps[prev_lps - 1]
        else:
            i += 1

    if longest_ngram < min_n:
        return []

    # Convert back to original order
    start_position = total - 1 - position + longest_ngram
    end = min(start_position + k, total)
    return token_ids[start_position:end]


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
        self._pool: dict[tuple[int, ...], list[int]] = {}
        self._capacity = config.hashpool_capacity
        self._total_inserts = 0
        self._total_evictions = 0

    def update(self, token_ids: list[int]) -> None:
        """Index all ngrams from the token sequence into the pool.

        Only indexes new tokens — call with the full sequence each time;
        repeated entries are harmless (dict overwrites).
        """
        min_n = self.config.min_n
        max_n = self.config.max_n
        k = self.config.k
        total = len(token_ids)

        if total < min_n + 1:
            return

        for n in range(min_n, max_n + 1):
            for i in range(total - n):
                ngram = tuple(token_ids[i:i + n])
                cont_start = i + n
                cont_end = min(cont_start + k, total)
                if cont_start < total:
                    self._pool[ngram] = token_ids[cont_start:cont_end]
                    self._total_inserts += 1

        # Evict oldest entries if over capacity
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

    def get_stats(self) -> dict:
        return {
            "pool_size": len(self._pool),
            "capacity": self._capacity,
            "total_inserts": self._total_inserts,
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
        if config.mode == "hashpool":
            self._hashpool = NgramHashPool(config)

    def propose(self, token_ids: list[int]) -> list[int]:
        """Propose draft tokens for a single request.

        Args:
            token_ids: Full context (prompt + generated tokens so far).

        Returns:
            List of proposed draft token IDs.
        """
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
        if self._hashpool is not None:
            return {"mode": "hashpool", **self._hashpool.get_stats()}
        return {"mode": "lps"}
