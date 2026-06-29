from __future__ import annotations

"""GPU-accelerated N-gram Speculative Decoding — MLX vectorized lookup.

Stores n-gram → m-gram mappings as MLX arrays, enabling GPU-side matching
via mx.equal / mx.all for vectorized comparison. Much faster than Python
dict lookup for large tables because the comparison work happens on the
GPU/ANE instead of the CPU.

Architecture:
  - GPUNgramTable: stores n-gram keys and m-gram values as mx.array matrices
  - GPUNgramProposer: wraps GPUNgramTable with propose/add_sequence/get_stats
  - Falls back to CPU NgramProposer when the table is empty

References:
  - vLLM NgramProposer (vllm/v1/spec_decode/ngram_proposer.py)
  - vLLM GPU batch propose (vllm/v1/spec_decode/ngram_proposer_gpu.py)
"""

import logging
from dataclasses import dataclass

import mlx.core as mx

from .ngram_proposer import NgramConfig, NgramProposer

logger = logging.getLogger(__name__)


@dataclass
class GPUNgramConfig:
    """Configuration for GPU N-gram speculative decoding."""

    min_n: int = 1
    max_n: int = 5
    k: int = 5  # max draft tokens per proposal
    max_model_len: int = 32768
    max_table_entries: int = 1 << 16  # 65536 max entries in GPU table
    gpu_fallback: bool = True  # fall back to CPU NgramProposer on miss


class GPUNgramTable:
    """GPU-accelerated N-gram lookup table using MLX arrays.

    Stores n-gram keys and their continuation tokens as MLX arrays,
    enabling GPU-side vectorized comparison for matching.

    The keys array has shape (num_entries, max_n) and the values array
    has shape (num_entries, k). Matching is done by:
      1. Extract last N tokens from context as a query vector
      2. Compare query against all key rows with mx.equal + mx.all
      3. Return the continuation from the matching row

    For tables with multiple n-gram lengths, we maintain separate
    sub-tables per length for efficiency.
    """

    def __init__(self, config: GPUNgramConfig) -> None:
        self.config = config
        self._min_n = config.min_n
        self._max_n = config.max_n
        self._k = config.k

        # Per-n-length storage
        # _keys[n] = mx.array of shape (num_entries, n) — int32
        # _values[n] = mx.array of shape (num_entries, k) — int32
        self._keys: dict[int, mx.array] = {}
        self._values: dict[int, mx.array] = {}
        # Python-side bookkeeping for incremental builds
        self._pending_keys: dict[int, list[list[int]]] = {}
        self._pending_values: dict[int, list[list[int]]] = {}

        self._total_entries: int = 0
        self._total_gpu_lookups: int = 0
        self._total_gpu_hits: int = 0

    @property
    def total_entries(self) -> int:
        return self._total_entries

    def build_from_sequences(
        self,
        sequences: list[list[int]],
        n: int | None = None,
        m: int | None = None,
    ) -> None:
        """Build the GPU lookup table from token sequences.

        Extracts n-grams of lengths [min_n, max_n] from each sequence
        and stores them with their continuations (up to k tokens).

        Args:
            sequences: List of token ID sequences to index.
            n: N-gram length (defaults to all lengths in [min_n, max_n]).
            m: Unused, kept for API symmetry. Continuation length is k.
        """
        min_n = self._min_n if n is None else n
        max_n = self._max_n if n is None else n
        k = self._k

        for seq in sequences:
            seq_len = len(seq)
            if seq_len < min_n + 1:
                continue
            for ngram_len in range(min_n, max_n + 1):
                for i in range(seq_len - ngram_len):
                    key = seq[i : i + ngram_len]
                    cont_start = i + ngram_len
                    cont_end = min(cont_start + k, seq_len)
                    if cont_start >= seq_len:
                        continue
                    value = seq[cont_start:cont_end]
                    # Pad value to length k with -1
                    value = value + [-1] * (k - len(value))
                    self._pending_keys.setdefault(ngram_len, []).append(key)
                    self._pending_values.setdefault(ngram_len, []).append(value)
                    self._total_entries += 1

        self._flush_to_gpu()

    def _flush_to_gpu(self) -> None:
        """Convert pending Python lists to MLX arrays."""
        max_entries = self.config.max_table_entries
        for n in self._pending_keys:
            if not self._pending_keys[n]:
                continue
            keys_list = self._pending_keys[n]
            vals_list = self._pending_values[n]

            # Truncate to max entries
            if len(keys_list) > max_entries:
                keys_list = keys_list[-max_entries:]
                vals_list = vals_list[-max_entries:]

            new_keys = mx.array(keys_list, dtype=mx.int32)
            new_vals = mx.array(vals_list, dtype=mx.int32)

            if n in self._keys:
                # Concatenate with existing arrays
                old_keys = self._keys[n]
                old_vals = self._values[n]
                combined_keys = mx.concatenate([old_keys, new_keys], axis=0)
                combined_vals = mx.concatenate([old_vals, new_vals], axis=0)
                # Keep only the last max_entries
                if combined_keys.shape[0] > max_entries:
                    start = combined_keys.shape[0] - max_entries
                    combined_keys = combined_keys[start:]
                    combined_vals = combined_vals[start:]
                self._keys[n] = combined_keys
                self._values[n] = combined_vals
            else:
                self._keys[n] = new_keys
                self._values[n] = new_vals

            self._pending_keys[n] = []
            self._pending_values[n] = []

    def lookup_gpu(self, context: mx.array, n: int) -> mx.array | None:
        """GPU-accelerated context match for n-grams of length n.

        Compares the last n tokens of context against all stored n-gram
        keys of length n. Returns the continuation tokens from the first
        matching row.

        Args:
            context: Token IDs as mx.array (1D, int32).
            n: N-gram length to match.

        Returns:
            Continuation tokens as mx.array, or None if no match.
        """
        if n not in self._keys:
            return None

        total_len = context.shape[0]
        if total_len < n:
            return None

        # Extract query: last n tokens
        query = context[total_len - n :]  # shape: (n,)
        keys_n = self._keys[n]  # shape: (num_entries, n)

        # Vectorized comparison: broadcast query across all key rows
        # keys_n == query broadcasts (num_entries, n) vs (n,) -> (num_entries, n)
        matches = mx.all(keys_n == query, axis=-1)  # shape: (num_entries,)

        self._total_gpu_lookups += 1

        # MLX does not support boolean indexing, so use argmax to find
        # the first match. argmax on a boolean array returns the index of
        # the first True value (or 0 if all False).
        first_hit = int(mx.argmax(matches))
        if not bool(matches[first_hit]):
            return None

        self._total_gpu_hits += 1
        return self._values[n][first_hit]  # shape: (k,)

    def lookup_all_n(self, context: mx.array) -> mx.array | None:
        """Try all n-gram lengths from max_n down to min_n.

        Returns the longest match's continuation.

        Args:
            context: Token IDs as mx.array (1D, int32).

        Returns:
            Continuation tokens as mx.array, or None if no match.
        """
        total_len = context.shape[0]
        max_n = min(self._max_n, total_len)

        for n in range(max_n, self._min_n - 1, -1):
            result = self.lookup_gpu(context, n)
            if result is not None:
                return result
        return None

    def clear(self) -> None:
        """Clear all stored n-grams."""
        self._keys.clear()
        self._values.clear()
        self._pending_keys.clear()
        self._pending_values.clear()
        self._total_entries = 0
        self._total_gpu_lookups = 0
        self._total_gpu_hits = 0

    def get_stats(self) -> dict:
        """Return table statistics."""
        per_n_entries = {}
        for n, keys in self._keys.items():
            per_n_entries[str(n)] = keys.shape[0]

        return {
            "total_entries": self._total_entries,
            "gpu_arrays": len(self._keys),
            "per_n_entries": per_n_entries,
            "total_gpu_lookups": self._total_gpu_lookups,
            "total_gpu_hits": self._total_gpu_hits,
            "gpu_hit_rate": (
                round(self._total_gpu_hits / self._total_gpu_lookups, 4)
                if self._total_gpu_lookups > 0
                else 0.0
            ),
        }


class GPUNgramProposer:
    """GPU-accelerated N-gram speculative decoding proposer.

    Wraps GPUNgramTable with the propose/add_sequence lifecycle.
    Falls back to CPU NgramProposer when the GPU table has no matches.
    """

    def __init__(self, config: GPUNgramConfig | None = None) -> None:
        if config is None:
            config = GPUNgramConfig()
        self.config = config
        self._table = GPUNgramTable(config)
        self._cpu_fallback: NgramProposer | None = None
        if config.gpu_fallback:
            cpu_config = NgramConfig(
                min_n=config.min_n,
                max_n=config.max_n,
                k=config.k,
                max_model_len=config.max_model_len,
                mode="lps",
            )
            self._cpu_fallback = NgramProposer(cpu_config)

        self._total_proposals: int = 0
        self._total_gpu_used: int = 0
        self._total_cpu_used: int = 0
        self._total_tokens_proposed: int = 0

    @property
    def table(self) -> GPUNgramTable:
        """Access the underlying GPUNgramTable."""
        return self._table

    def build_from_sequences(self, sequences: list[list[int]]) -> None:
        """Build the GPU table from token sequences."""
        self._table.build_from_sequences(sequences)

    def add_sequence(self, tokens: list[int]) -> None:
        """Add a newly generated sequence to the table.

        Args:
            tokens: Token IDs from a completed generation.
        """
        self._table.build_from_sequences([tokens])

    def propose(self, context_tokens: list[int], n_draft: int = 0) -> list[int]:
        """Propose draft tokens using GPU-accelerated n-gram lookup.

        Tries GPU lookup first. If no match found and CPU fallback is
        enabled, falls back to CPU NgramProposer.

        Args:
            context_tokens: Full context (prompt + generated so far).
            n_draft: Max draft tokens (0 = use config.k).

        Returns:
            List of proposed draft token IDs (may be empty).
        """
        k = n_draft if n_draft > 0 else self.config.k
        total = len(context_tokens)
        k = min(k, self.config.max_model_len - total)
        if k <= 0 or total < self.config.min_n:
            return []

        self._total_proposals += 1

        # GPU lookup
        context_mx = mx.array(context_tokens, dtype=mx.int32)
        result = self._table.lookup_all_n(context_mx)

        if result is not None:
            # Filter padding (-1) and trim to k
            draft = []
            for i in range(result.shape[0]):
                val = int(result[i])
                if val == -1:
                    break
                draft.append(val)
                if len(draft) >= k:
                    break
            if draft:
                self._total_gpu_used += 1
                self._total_tokens_proposed += len(draft)
                return draft

        # CPU fallback
        if self._cpu_fallback is not None:
            cpu_draft = self._cpu_fallback.propose(context_tokens)
            cpu_draft = cpu_draft[:k]
            if cpu_draft:
                self._total_cpu_used += 1
                self._total_tokens_proposed += len(cpu_draft)
                return cpu_draft

        return []

    def batch_propose(
        self,
        batch_tokens: list[list[int]],
        n_draft: int = 0,
    ) -> list[list[int]]:
        """Propose draft tokens for a batch of requests.

        Args:
            batch_tokens: List of context token lists.
            n_draft: Max draft tokens per request (0 = use config.k).

        Returns:
            List of proposed draft token lists.
        """
        return [self.propose(tokens, n_draft) for tokens in batch_tokens]

    def get_stats(self) -> dict:
        """Return proposer statistics."""
        table_stats = self._table.get_stats()
        return {
            "mode": "gpu_ngram",
            "total_proposals": self._total_proposals,
            "total_gpu_used": self._total_gpu_used,
            "total_cpu_used": self._total_cpu_used,
            "total_tokens_proposed": self._total_tokens_proposed,
            "gpu_ratio": (
                round(self._total_gpu_used / self._total_proposals, 4)
                if self._total_proposals > 0
                else 0.0
            ),
            "avg_match_length": (
                round(self._total_tokens_proposed / self._total_proposals, 2)
                if self._total_proposals > 0
                else 0.0
            ),
            "table": table_stats,
        }
