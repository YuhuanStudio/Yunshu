from __future__ import annotations
"""Suffix-based Speculative Decoding — pattern reuse from own generation.

Finds common suffixes in the request's own generated text and reuses them
as draft tokens. Very effective for repetitive outputs (code, JSON,
structured text, agentic loops).

Architecture:
  - SuffixTrie: trie structure for efficient suffix matching
  - SuffixProposer: per-request rolling window + suffix lookup
  - SuffixStrategy: SpecStrategy wrapper for CompositeStrategy compatibility

Unlike N-gram matching which looks at the entire context, suffix decoding
focuses on the most recent tokens (configurable window) and finds the
longest matching suffix from earlier in the sequence. This is especially
effective for:
  - Code generation (repeated patterns like indentation + keywords)
  - JSON output (repeated key-value structure)
  - Structured text (repeated formatting patterns)
  - Agentic loops (self-reflection, self-consistency patterns)

References:
  - Suffix Decoding (https://arxiv.org/abs/2411.04975)
  - vLLM SuffixDecodingProposer (vllm/v1/spec_decode/suffix_decoding.py)
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SuffixConfig:
    """Configuration for suffix-based speculative decoding."""
    # Minimum suffix length to match
    min_suffix_length: int = 3
    # Maximum rolling window size per request
    max_window: int = 512
    # Maximum draft tokens per proposal
    max_draft: int = 5
    # Maximum model context length
    max_model_len: int = 32768
    # Maximum trie depth (limits memory usage)
    max_trie_depth: int = 64


class SuffixTrieNode:
    """A node in the suffix trie.

    Each node maps a token ID to child nodes and stores the continuation
    tokens (and their frequency) for paths that pass through this node.
    """

    __slots__ = ("children", "continuations", "total_count")

    def __init__(self) -> None:
        self.children: dict[int, SuffixTrieNode] = {}
        # Maps continuation token sequence (as tuple) to frequency
        self.continuations: dict[tuple[int, ...], int] = {}
        self.total_count: int = 0


class SuffixTrie:
    """Trie structure for efficient suffix matching.

    Builds a trie from all suffixes of a token sequence. Each node stores
    the continuation tokens that follow the matched prefix, enabling fast
    lookup of the most common continuation for any suffix.

    For example, given [1, 2, 3, 1, 2, 3, 4]:
      - The trie stores suffixes: [1,2,3,1,2,3,4], [2,3,1,2,3,4], ...
      - Matching suffix [1,2,3] returns continuation [1,2,3,4]
    """

    def __init__(self, max_depth: int = 64) -> None:
        self.max_depth = max_depth
        self.root = SuffixTrieNode()
        self._total_inserts: int = 0

    def insert(self, tokens: list[int]) -> None:
        """Insert all suffixes of a token sequence into the trie.

        For each starting position i, inserts the suffix tokens[i:]
        up to max_depth tokens deep. Also records the continuation
        (up to max_depth tokens after the suffix) at each node.

        Args:
            tokens: Token sequence to index.
        """
        n = len(tokens)
        if n < 2:
            return

        for start in range(n):
            node = self.root
            depth = 0
            for j in range(start, min(start + self.max_depth, n)):
                tok = tokens[j]
                if tok not in node.children:
                    node.children[tok] = SuffixTrieNode()
                node = node.children[tok]
                depth += 1

                # Record continuation: tokens after this suffix position
                cont_start = j + 1
                if cont_start < n:
                    # Store continuation of up to a few tokens
                    cont_end = min(cont_start + 5, n)
                    cont = tuple(tokens[cont_start:cont_end])
                    node.continuations[cont] = node.continuations.get(cont, 0) + 1
                    node.total_count += 1

            self._total_inserts += 1

    def find_longest_match(
        self,
        suffix: list[int],
        max_len: int = 5,
    ) -> list[int]:
        """Find the longest matching suffix and return continuation tokens.

        Walks the trie following the suffix tokens. At the deepest
        matching node, returns the most frequent continuation.

        Args:
            suffix: Token suffix to match (searched from end).
            max_len: Maximum continuation length to return.

        Returns:
            Continuation tokens, or empty list if no match.
        """
        if not suffix:
            return []

        # Walk the trie with the suffix
        node = self.root
        last_match_node: SuffixTrieNode | None = None
        matched_depth = 0

        for i, tok in enumerate(suffix):
            if tok in node.children:
                node = node.children[tok]
                if node.continuations:
                    last_match_node = node
                    matched_depth = i + 1
            else:
                break

        if last_match_node is None or not last_match_node.continuations:
            return []

        # Find the most frequent continuation
        best_cont = max(
            last_match_node.continuations,
            key=lambda c: last_match_node.continuations[c],
        )
        return list(best_cont[:max_len])

    def clear(self) -> None:
        """Reset the trie."""
        self.root = SuffixTrieNode()
        self._total_inserts = 0

    def get_stats(self) -> dict:
        """Return trie statistics."""
        def _count_nodes(node: SuffixTrieNode) -> int:
            count = 1
            for child in node.children.values():
                count += _count_nodes(child)
            return count

        return {
            "total_inserts": self._total_inserts,
            "total_nodes": _count_nodes(self.root),
            "max_depth": self.max_depth,
        }


class SuffixProposer:
    """Suffix-based speculative decoding proposer.

    Maintains a rolling window of recently generated tokens per request.
    When asked for drafts, searches for suffix matches in the rolling
    window and proposes continuation tokens.

    Per-request lifecycle:
      1. begin(request_id) — initializes per-request trie and window
      2. draft(context_tokens, n_draft) — proposes suffix-matched tokens
      3. accept(draft_tokens, verified_up_to) — records accepted tokens
      4. end(request_id) — cleans up per-request state
    """

    def __init__(self, config: SuffixConfig | None = None) -> None:
        if config is None:
            config = SuffixConfig()
        self.config = config

        # Per-request state: request_id -> SuffixTrie
        self._tries: dict[str, SuffixTrie] = {}
        # Per-request rolling windows: request_id -> list[int]
        self._windows: dict[str, list[int]] = {}
        # Per-request generated token tracking
        self._generated: dict[str, list[int]] = {}

        # Global statistics
        self._total_proposals: int = 0
        self._total_hits: int = 0
        self._total_tokens_proposed: int = 0
        self._total_tokens_accepted: int = 0

    def begin(self, request_id: str) -> None:
        """Initialize per-request state.

        Args:
            request_id: Unique identifier for the request.
        """
        self._tries[request_id] = SuffixTrie(
            max_depth=self.config.max_trie_depth,
        )
        self._windows[request_id] = []
        self._generated[request_id] = []

    def draft(
        self,
        context_tokens: list[int],
        n_draft: int = 0,
    ) -> list[int]:
        """Propose draft tokens by finding suffix matches.

        Searches for the current suffix (last N tokens) appearing earlier
        in the generated token history. Returns the continuation tokens
        following the longest matching suffix.

        Args:
            context_tokens: Full context (prompt + generated so far).
            n_draft: Max draft tokens (0 = use config.max_draft).

        Returns:
            List of proposed draft token IDs.
        """
        max_draft = n_draft if n_draft > 0 else self.config.max_draft
        total = len(context_tokens)
        max_draft = min(max_draft, self.config.max_model_len - total)

        if max_draft <= 0:
            return []

        active_id = self._find_active_request()
        if active_id is None:
            return []

        gen = self._generated.get(active_id)
        if not gen:
            return []

        min_suffix = self.config.min_suffix_length
        if total < min_suffix:
            return []

        self._total_proposals += 1

        # Search for the longest suffix of context_tokens that appears
        # earlier in the generated token history
        max_search = min(total, self.config.max_trie_depth)
        best_continuation: list[int] = []

        for suffix_len in range(min(max_search, total), min_suffix - 1, -1):
            suffix = context_tokens[-suffix_len:]
            # Search for this suffix in the generated history
            cont = self._find_suffix_in_history(suffix, gen, max_draft)
            if cont:
                best_continuation = cont
                break

        if best_continuation:
            self._total_hits += 1
            self._total_tokens_proposed += len(best_continuation)

        return best_continuation

    def _find_suffix_in_history(
        self,
        suffix: list[int],
        history: list[int],
        max_len: int,
    ) -> list[int]:
        """Find suffix in history and return continuation tokens.

        Searches for the last occurrence of suffix in history (not at the
        very end). Returns the tokens following the match.

        Args:
            suffix: Suffix tokens to search for.
            history: Token history to search in.
            max_len: Maximum continuation length.

        Returns:
            Continuation tokens, or empty list if not found.
        """
        n = len(suffix)
        h = len(history)

        if h < n + 1:
            return []

        # Search from the end backwards for the suffix
        # Don't match at position h-n (would be the very end of history)
        for i in range(h - n - 1, -1, -1):
            match = True
            for j in range(n):
                if history[i + j] != suffix[j]:
                    match = False
                    break
            if match:
                cont_start = i + n
                cont_end = min(cont_start + max_len, h)
                if cont_start < h:
                    return history[cont_start:cont_end]

        return []

    def accept(self, draft_tokens: list[int], verified_up_to: int) -> None:
        """Record accepted draft tokens for future matching.

        Args:
            draft_tokens: The draft tokens that were proposed.
            verified_up_to: Number of tokens verified as correct.
        """
        self._total_tokens_accepted += verified_up_to

        # Add accepted tokens to the active request's generated history
        active_id = self._find_active_request()
        if active_id is None:
            return

        accepted = draft_tokens[:verified_up_to]
        gen_list = self._generated.get(active_id)
        if gen_list is None:
            return
        gen_list.extend(accepted)

        # Rebuild trie with updated generated tokens
        gen = self._generated[active_id]
        if len(gen) >= self.config.min_suffix_length:
            trie = self._tries[active_id]
            trie.clear()
            # Only index up to max_window tokens
            window_gen = gen[-self.config.max_window :]
            trie.insert(window_gen)

    def end(self, request_id: str) -> None:
        """Clean up per-request state.

        Args:
            request_id: The request identifier from begin().
        """
        self._tries.pop(request_id, None)
        self._windows.pop(request_id, None)
        self._generated.pop(request_id, None)

    def _find_active_request(self) -> str | None:
        """Find the most recently active request ID."""
        # Return the last request that was begun
        if self._tries:
            return next(reversed(self._tries))
        return None

    def get_stats(self) -> dict:
        """Return proposer statistics."""
        return {
            "mode": "suffix",
            "active_requests": len(self._tries),
            "total_proposals": self._total_proposals,
            "total_hits": self._total_hits,
            "total_tokens_proposed": self._total_tokens_proposed,
            "total_tokens_accepted": self._total_tokens_accepted,
            "hit_rate": (
                round(self._total_hits / self._total_proposals, 4)
                if self._total_proposals > 0
                else 0.0
            ),
            "avg_proposed_length": (
                round(self._total_tokens_proposed / self._total_hits, 2)
                if self._total_hits > 0
                else 0.0
            ),
        }
