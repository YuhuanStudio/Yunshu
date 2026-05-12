"""Yunshu N-gram Speculative Decoding — model-free draft token proposal.

Studied from vLLM's NgramProposer, adapted for pure Python (no numba):
- Finds longest N-gram match in the prompt/output history
- Proposes K tokens following the match as draft tokens
- No draft model required — pure CPU pattern matching
- Uses KMP (Knuth-Morris-Pratt) LPS array for O(n) matching

Algorithm:
  1. Reverse the token sequence
  2. Build LPS (longest proper prefix which is also suffix) array
  3. Find longest match of suffix in [min_n, max_n] range
  4. Extract K tokens following the match as draft proposals

Integration:
  - Scheduler calls propose() before each decode step
  - Draft tokens are verified by the target model
  - Accepted tokens are kept; rejected tokens trigger resample

References:
  - vLLM NgramProposer (vllm/v1/spec_decode/ngram_proposer.py)
  - "Fast Inference from Transformers via Autoregressive Diffusion"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

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


class NgramProposer:
    """N-gram based speculative decoding proposer (vLLM pattern).

    Proposes draft tokens by finding repeated N-gram patterns in
    the prompt + output history. No draft model required.

    Usage:
        proposer = NgramProposer(NgramConfig(min_n=1, max_n=5, k=5))
        draft_tokens = proposer.propose(token_ids)
    """

    def __init__(self, config: NgramConfig) -> None:
        self.config = config

    def propose(self, token_ids: list[int]) -> list[int]:
        """Propose draft tokens for a single request.

        Args:
            token_ids: Full context (prompt + generated tokens so far).

        Returns:
            List of proposed draft token IDs.
        """
        return _find_longest_ngram_and_propose(
            token_ids=token_ids,
            min_n=self.config.min_n,
            max_n=self.config.max_n,
            max_model_len=self.config.max_model_len,
            k=self.config.k,
        )

    def batch_propose(self, batch_token_ids: list[list[int]]) -> list[list[int]]:
        """Propose draft tokens for a batch of requests.

        Args:
            batch_token_ids: List of token ID lists, one per request.

        Returns:
            List of draft token lists (empty list if no proposal).
        """
        results = []
        for token_ids in batch_token_ids:
            results.append(self.propose(token_ids))
        return results
