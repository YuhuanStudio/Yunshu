from __future__ import annotations
"""Unified speculative decoding interface — begin()/draft()/accept() lifecycle.

Inspired by llama.cpp's common_speculative_state pattern. Each proposer type
(N-gram, EAGLE, MTP) implements this ABC so they can be composed or swapped.

Lifecycle:
  1. proposer.begin(all_token_ids)  — initialize per-request state
  2. proposer.draft(all_token_ids)   — propose K draft tokens
  3. proposer.accept(n_accepted)     — inform proposer how many were accepted
  4. (repeat 2-3 until generation ends)

Usage in BatchedEngine:
  proposer = NgramProposer(config)
  proposer.begin(prompt_ids)
  drafts = proposer.draft(current_ids)
  ... verify against model ...
  proposer.accept(n_verified)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SpecProposal:
    """Result of a draft proposal."""
    token_ids: list[int]
    proposer_type: str  # "ngram", "eagle", "mtp"
    metadata: dict[str, Any] = field(default_factory=dict)


class SpecProposer(ABC):
    """Abstract base class for speculative decoding proposers.

    All proposers must implement begin/draft/accept lifecycle.
    """

    @abstractmethod
    def begin(self, all_token_ids: list[int]) -> None:
        """Initialize per-request proposer state."""

    @abstractmethod
    def draft(self, all_token_ids: list[int], k: int = 5) -> SpecProposal:
        """Propose up to K draft tokens."""

    @abstractmethod
    def accept(self, n_accepted: int) -> None:
        """Inform proposer how many draft tokens were accepted."""

    @property
    @abstractmethod
    def proposer_type(self) -> str:
        """Return the proposer type identifier."""

    def get_stats(self) -> dict[str, int]:
        """Return proposer-specific statistics."""
        return {}


class NgramSpecProposer(SpecProposer):
    """N-gram based spec proposer wrapping the existing NgramProposer."""

    def __init__(self, config: Any) -> None:
        from .ngram_proposer import NgramProposer
        self._proposer = NgramProposer(config)
        self._stats = {"proposals": 0, "accepted": 0, "total_draft": 0}

    def begin(self, all_token_ids: list[int]) -> None:
        self._proposer.reset()

    def draft(self, all_token_ids: list[int], k: int = 5) -> SpecProposal:
        config = self._proposer.config
        effective_k = min(k, config.k)
        tokens = self._proposer.propose(all_token_ids)
        draft_ids = tokens[:effective_k]
        self._stats["proposals"] += 1
        self._stats["total_draft"] += len(draft_ids)
        return SpecProposal(
            token_ids=draft_ids,
            proposer_type="ngram",
            metadata={"max_n": config.max_n},
        )

    def accept(self, n_accepted: int) -> None:
        self._stats["accepted"] += n_accepted

    @property
    def proposer_type(self) -> str:
        return "ngram"

    def get_stats(self) -> dict[str, int]:
        return dict(self._stats)


class CompositeSpecProposer(SpecProposer):
    """Combine multiple proposers, taking the best proposal.

    Runs all proposers and picks the one with the most draft tokens.
    This allows combining N-gram (CPU) + EAGLE (GPU) for maximum coverage.
    """

    def __init__(self, proposers: list[SpecProposer]) -> None:
        self._proposers = proposers
        self._last_winning_proposer: SpecProposer | None = None

    def begin(self, all_token_ids: list[int]) -> None:
        for p in self._proposers:
            p.begin(all_token_ids)

    def draft(self, all_token_ids: list[int], k: int = 5) -> SpecProposal:
        best = SpecProposal(token_ids=[], proposer_type="composite")
        self._last_winning_proposer = None
        for p in self._proposers:
            proposal = p.draft(all_token_ids, k)
            if len(proposal.token_ids) > len(best.token_ids):
                best = proposal
                self._last_winning_proposer = p
        return best

    def accept(self, n_accepted: int) -> None:
        if self._last_winning_proposer is not None:
            self._last_winning_proposer.accept(n_accepted)
            # Non-winning proposers still called with 0 so they can
            # maintain their internal statistics (e.g. proposal accuracy).
            for p in self._proposers:
                if p is not self._last_winning_proposer:
                    p.accept(0)
        else:
            for p in self._proposers:
                p.accept(n_accepted)

    @property
    def proposer_type(self) -> str:
        return "composite"

    def get_stats(self) -> dict[str, int]:
        stats: dict[str, int] = {}
        for p in self._proposers:
            for k, v in p.get_stats().items():
                stats[f"{p.proposer_type}_{k}"] = v
        return stats
