"""Unified Speculative Decoding Interface (C7) — llama.cpp begin/draft/accept pattern.

All speculative decoding strategies share a common lifecycle:
1. begin(request_id) — initialize state for a new generation
2. draft(tokens, n) — propose N draft tokens given current context
3. accept(draft_tokens, verified_up_to) — accept verified tokens, update state
4. stats() — return strategy-specific statistics

This enables composition: e.g., ngram-mod + draft model simultaneously.

Strategy classes:
  - NgramStrategy: wraps NgramProposer (model-free, LPS or HashPool)
  - CrossModelStrategy: wraps SpeculativeDecoder (draft+target model pair)
  - MTPStrategy: wraps MTPDecoder (multi-token prediction heads)
  - CompositeStrategy: combines multiple strategies (first non-empty draft wins)
  - SpecStrategyFactory: creates the right strategy from config dict
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class DraftProposal:
    """Result from a draft() call.

    Attributes:
        tokens: Proposed draft token IDs (may be empty if no proposal).
        strategy_name: Name of the strategy that produced this proposal.
        metadata: Optional strategy-specific metadata (e.g., ngram match length).
    """
    tokens: list[int] = field(default_factory=list)
    strategy_name: str = ""
    metadata: dict = field(default_factory=dict)


class SpecStrategy(ABC):
    """Abstract base class for speculative decoding strategies.

    Lifecycle:
        strategy.begin("req-123")
        proposal = strategy.draft([1, 2, 3, ...], n=5)
        # ... verify proposal.tokens against target model ...
        strategy.accept(proposal.tokens, verified_up_to=3)
        stats = strategy.stats()
        strategy.end("req-123")
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name (e.g., 'ngram', 'cross_model', 'mtp')."""
        ...

    @abstractmethod
    def begin(self, request_id: str) -> None:
        """Initialize state for a new generation request.

        Args:
            request_id: Unique identifier for the request.
        """
        ...

    @abstractmethod
    def draft(self, tokens: list[int], n: int) -> DraftProposal:
        """Propose up to N draft tokens given the current context.

        Args:
            tokens: Full context (prompt + generated tokens so far).
            n: Maximum number of draft tokens to propose.

        Returns:
            DraftProposal with proposed tokens (may be empty).
        """
        ...

    @abstractmethod
    def accept(self, draft_tokens: list[int], verified_up_to: int) -> None:
        """Accept verified tokens and update internal state.

        Called after the target model has verified the draft tokens.
        The strategy may use this feedback to improve future proposals.

        Args:
            draft_tokens: The draft tokens that were proposed.
            verified_up_to: Number of tokens verified as correct (0 = all rejected).
        """
        ...

    @abstractmethod
    def stats(self) -> dict:
        """Return strategy-specific statistics.

        Returns:
            Dict with at least 'name', 'total_drafts', 'total_accepted' keys.
        """
        ...

    def end(self, request_id: str) -> None:
        """Clean up state after a generation request completes.

        Default implementation is a no-op. Strategies that hold per-request
        state should override this to release it.

        Args:
            request_id: The request identifier from begin().
        """
        pass

    def reset(self) -> None:
        """Reset all accumulated statistics and state."""
        pass


class NgramStrategy(SpecStrategy):
    """Wraps NgramProposer as a SpecStrategy.

    The N-gram proposer is model-free: it looks up repeated patterns in
    the token history to predict what comes next. Three modes:
      - "lps": KMP-based O(n) matching (vLLM pattern)
      - "hashpool": O(1) dict lookup (llama.cpp pattern)
      - "lcg": O(1) circular-buffer LCG hash pool (llama.cpp ngram-mod pattern)
    """

    def __init__(self, config: Any = None) -> None:
        from .ngram_proposer import NgramProposer, NgramConfig
        if config is None:
            config = NgramConfig()
        elif isinstance(config, dict):
            config = NgramConfig(**config)
        self._config = config
        self._proposer = NgramProposer(config)
        self._request_id: Optional[str] = None
        self._total_drafts = 0
        self._total_draft_tokens = 0
        self._total_accepted = 0
        self._total_accepted_tokens = 0

    @property
    def name(self) -> str:
        return "ngram"

    def begin(self, request_id: str) -> None:
        self._request_id = request_id

    def draft(self, tokens: list[int], n: int) -> DraftProposal:
        proposed = self._proposer.propose(tokens)
        # Trim to requested max
        proposed = proposed[:n]
        self._total_drafts += 1
        self._total_draft_tokens += len(proposed)
        return DraftProposal(
            tokens=proposed,
            strategy_name=self.name,
            metadata={"mode": self._config.mode},
        )

    def accept(self, draft_tokens: list[int], verified_up_to: int) -> None:
        self._total_accepted += 1
        self._total_accepted_tokens += verified_up_to

    def stats(self) -> dict:
        proposer_stats = self._proposer.get_stats()
        return {
            "name": self.name,
            "total_drafts": self._total_drafts,
            "total_draft_tokens": self._total_draft_tokens,
            "total_accepted": self._total_accepted,
            "total_accepted_tokens": self._total_accepted_tokens,
            "acceptance_rate": (
                self._total_accepted_tokens / self._total_draft_tokens
                if self._total_draft_tokens > 0 else 0.0
            ),
            **proposer_stats,
        }

    def end(self, request_id: str) -> None:
        self._request_id = None

    def reset(self) -> None:
        self._total_drafts = 0
        self._total_draft_tokens = 0
        self._total_accepted = 0
        self._total_accepted_tokens = 0
        if self._proposer._hashpool is not None:
            self._proposer._hashpool.clear()
        if self._proposer._lcg_pool is not None:
            self._proposer._lcg_pool.clear()


class CrossModelStrategy(SpecStrategy):
    """Wraps SpeculativeDecoder as a SpecStrategy.

    The cross-model strategy uses a smaller draft model to propose K tokens,
    then verifies them against the target model. This is the EAGLE-3 pattern.

    Note: This strategy requires a loaded draft model. The draft() method
    returns an empty proposal if no draft model is available. The actual
    draft generation and verification is handled by SpeculativeDecoder.
    This wrapper provides lifecycle management and stats tracking.
    """

    def __init__(self, decoder: Any = None, config: Any = None) -> None:
        self._decoder = decoder
        self._config = config
        self._request_id: Optional[str] = None
        self._total_drafts = 0
        self._total_draft_tokens = 0
        self._total_accepted = 0
        self._total_accepted_tokens = 0

    @property
    def name(self) -> str:
        return "cross_model"

    @property
    def decoder(self) -> Any:
        """Access the underlying SpeculativeDecoder (may be None)."""
        return self._decoder

    def begin(self, request_id: str) -> None:
        self._request_id = request_id

    def draft(self, tokens: list[int], n: int) -> DraftProposal:
        if self._decoder is None:
            return DraftProposal(tokens=[], strategy_name=self.name)
        K = min(n, self._decoder.config.draft_length)
        self._total_drafts += 1
        self._total_draft_tokens += K
        return DraftProposal(
            tokens=[],  # Actual tokens filled by decoder's generate_draft
            strategy_name=self.name,
            metadata={"draft_length": K},
        )

    def accept(self, draft_tokens: list[int], verified_up_to: int) -> None:
        self._total_accepted += 1
        self._total_accepted_tokens += verified_up_to

    def stats(self) -> dict:
        result = {
            "name": self.name,
            "total_drafts": self._total_drafts,
            "total_draft_tokens": self._total_draft_tokens,
            "total_accepted": self._total_accepted,
            "total_accepted_tokens": self._total_accepted_tokens,
            "acceptance_rate": (
                self._total_accepted_tokens / self._total_draft_tokens
                if self._total_draft_tokens > 0 else 0.0
            ),
        }
        if self._decoder is not None:
            result["decoder_stats"] = self._decoder.get_stats()
        return result

    def end(self, request_id: str) -> None:
        self._request_id = None

    def reset(self) -> None:
        self._total_drafts = 0
        self._total_draft_tokens = 0
        self._total_accepted = 0
        self._total_accepted_tokens = 0


class MTPStrategy(SpecStrategy):
    """Wraps MTPDecoder as a SpecStrategy.

    The MTP (Multi-Token Prediction) strategy uses additional prediction
    heads built into the model to propose draft tokens. These heads share
    the model's backbone but add extra layers for next-token prediction.

    Note: This strategy requires a model with MTP heads. The draft() method
    returns an empty proposal if the model lacks MTP support. The actual
    draft generation is handled by MTPDecoder.
    """

    def __init__(self, decoder: Any = None, config: Any = None) -> None:
        self._decoder = decoder
        self._config = config
        self._request_id: Optional[str] = None
        self._total_drafts = 0
        self._total_draft_tokens = 0
        self._total_accepted = 0
        self._total_accepted_tokens = 0

    @property
    def name(self) -> str:
        return "mtp"

    @property
    def decoder(self) -> Any:
        """Access the underlying MTPDecoder (may be None)."""
        return self._decoder

    def begin(self, request_id: str) -> None:
        self._request_id = request_id

    def draft(self, tokens: list[int], n: int) -> DraftProposal:
        if self._decoder is None:
            return DraftProposal(tokens=[], strategy_name=self.name)
        # MTP always proposes exactly 1 draft token per step
        self._total_drafts += 1
        self._total_draft_tokens += 1
        return DraftProposal(
            tokens=[],  # Actual token filled by decoder's _mtp_draft
            strategy_name=self.name,
            metadata={"draft_length": 1},
        )

    def accept(self, draft_tokens: list[int], verified_up_to: int) -> None:
        self._total_accepted += 1
        self._total_accepted_tokens += verified_up_to

    def stats(self) -> dict:
        result = {
            "name": self.name,
            "total_drafts": self._total_drafts,
            "total_draft_tokens": self._total_draft_tokens,
            "total_accepted": self._total_accepted,
            "total_accepted_tokens": self._total_accepted_tokens,
            "acceptance_rate": (
                self._total_accepted_tokens / self._total_draft_tokens
                if self._total_draft_tokens > 0 else 0.0
            ),
        }
        if self._decoder is not None:
            s = self._decoder.stats
            result["mtp_accepts"] = s.accepts
            result["mtp_rejects"] = s.rejects
            result["mtp_cooldowns"] = s.cooldowns
            result["mtp_tokens_generated"] = s.tokens_generated
            result["mtp_total_cycles"] = s.total_cycles
        return result

    def end(self, request_id: str) -> None:
        self._request_id = None

    def reset(self) -> None:
        self._total_drafts = 0
        self._total_draft_tokens = 0
        self._total_accepted = 0
        self._total_accepted_tokens = 0


class CompositeStrategy(SpecStrategy):
    """Combines multiple strategies: first non-empty draft wins.

    Draft proposals are tried in order. The first strategy that returns
    a non-empty proposal is used. This enables fallback chains like:
      ngram-hashpool -> ngram-lps -> cross_model

    Stats are aggregated across all child strategies.
    """

    def __init__(self, strategies: list[SpecStrategy]) -> None:
        if not strategies:
            raise ValueError("CompositeStrategy requires at least one strategy")
        self._strategies = strategies
        self._request_id: Optional[str] = None
        self._total_drafts = 0
        self._total_used: dict[str, int] = {}  # strategy name -> times used

    @property
    def name(self) -> str:
        return "composite(" + "+".join(s.name for s in self._strategies) + ")"

    @property
    def strategies(self) -> list[SpecStrategy]:
        """Access child strategies."""
        return list(self._strategies)

    def begin(self, request_id: str) -> None:
        self._request_id = request_id
        for s in self._strategies:
            s.begin(request_id)

    def draft(self, tokens: list[int], n: int) -> DraftProposal:
        self._total_drafts += 1
        for s in self._strategies:
            proposal = s.draft(tokens, n)
            if proposal.tokens:
                self._total_used[s.name] = self._total_used.get(s.name, 0) + 1
                return proposal
        # All strategies returned empty
        return DraftProposal(tokens=[], strategy_name="composite(empty)")

    def accept(self, draft_tokens: list[int], verified_up_to: int) -> None:
        # Forward accept to all children so they can update their state
        for s in self._strategies:
            s.accept(draft_tokens, verified_up_to)

    def stats(self) -> dict:
        child_stats = [s.stats() for s in self._strategies]
        total_draft_tokens = sum(
            cs.get("total_draft_tokens", 0) for cs in child_stats
        )
        total_accepted_tokens = sum(
            cs.get("total_accepted_tokens", 0) for cs in child_stats
        )
        return {
            "name": self.name,
            "total_drafts": self._total_drafts,
            "total_draft_tokens": total_draft_tokens,
            "total_accepted_tokens": total_accepted_tokens,
            "acceptance_rate": (
                total_accepted_tokens / total_draft_tokens
                if total_draft_tokens > 0 else 0.0
            ),
            "strategy_usage": dict(self._total_used),
            "children": child_stats,
        }

    def end(self, request_id: str) -> None:
        self._request_id = None
        for s in self._strategies:
            s.end(request_id)

    def reset(self) -> None:
        self._total_drafts = 0
        self._total_used = {}
        for s in self._strategies:
            s.reset()


class SpecStrategyFactory:
    """Factory for creating SpecStrategy instances from configuration dicts.

    Supported strategy types:
      - "ngram": NgramStrategy with optional mode ("lps" or "hashpool")
      - "cross_model": CrossModelStrategy (requires draft model)
      - "mtp": MTPStrategy (requires model with MTP heads)
      - "composite": CompositeStrategy combining multiple strategies

    Usage:
        strategy = SpecStrategyFactory.create({
            "type": "ngram",
            "mode": "hashpool",
            "max_n": 5,
            "k": 5,
        })
    """

    @staticmethod
    def create(config: dict) -> SpecStrategy:
        """Create a SpecStrategy from a config dict.

        Args:
            config: Configuration dict. Must contain "type" key with one of:
                    "ngram", "cross_model", "mtp", "composite".

        Returns:
            A SpecStrategy instance.

        Raises:
            ValueError: If config is missing "type" or type is unknown.
        """
        strategy_type = config.get("type", "").lower()
        if not strategy_type:
            raise ValueError("Config must contain a 'type' key")

        if strategy_type == "ngram":
            return SpecStrategyFactory._create_ngram(config)
        elif strategy_type == "cross_model":
            return SpecStrategyFactory._create_cross_model(config)
        elif strategy_type == "mtp":
            return SpecStrategyFactory._create_mtp(config)
        elif strategy_type == "composite":
            return SpecStrategyFactory._create_composite(config)
        else:
            raise ValueError(
                f"Unknown spec strategy type: {strategy_type!r}. "
                f"Supported: ngram, cross_model, mtp, composite"
            )

    @staticmethod
    def _create_ngram(config: dict) -> NgramStrategy:
        from .ngram_proposer import NgramConfig
        ngram_kwargs = {}
        if "mode" in config:
            ngram_kwargs["mode"] = config["mode"]
        if "min_n" in config:
            ngram_kwargs["min_n"] = int(config["min_n"])
        if "max_n" in config:
            ngram_kwargs["max_n"] = int(config["max_n"])
        if "k" in config:
            ngram_kwargs["k"] = int(config["k"])
        if "max_model_len" in config:
            ngram_kwargs["max_model_len"] = int(config["max_model_len"])
        if "hashpool_capacity" in config:
            ngram_kwargs["hashpool_capacity"] = int(config["hashpool_capacity"])
        ngram_config = NgramConfig(**ngram_kwargs)
        return NgramStrategy(config=ngram_config)

    @staticmethod
    def _create_cross_model(config: dict) -> CrossModelStrategy:
        decoder = config.get("decoder")
        dec_config = config.get("decoder_config")
        return CrossModelStrategy(decoder=decoder, config=dec_config)

    @staticmethod
    def _create_mtp(config: dict) -> MTPStrategy:
        decoder = config.get("decoder")
        dec_config = config.get("decoder_config")
        return MTPStrategy(decoder=decoder, config=dec_config)

    @staticmethod
    def _create_composite(config: dict) -> CompositeStrategy:
        children_config = config.get("strategies")
        if not children_config or not isinstance(children_config, list):
            raise ValueError(
                "CompositeStrategy config must contain a 'strategies' list"
            )
        children = [SpecStrategyFactory.create(c) for c in children_config]
        return CompositeStrategy(children)

    @staticmethod
    def from_env() -> Optional[SpecStrategy]:
        """Create a strategy from environment variables.

        Reads YUNSHU_SPEC_STRATEGY env var for the strategy type.
        Additional config from YUNSHU_NGRAM_* env vars for ngram.

        Returns:
            A SpecStrategy, or None if no strategy is configured.
        """
        import os
        strategy_type = os.environ.get("YUNSHU_SPEC_STRATEGY", "").strip().lower()
        if not strategy_type:
            return None
        config = {"type": strategy_type}
        if strategy_type == "ngram":
            config["mode"] = os.environ.get("YUNSHU_NGRAM_MODE", "lps")
            config["max_n"] = int(os.environ.get("YUNSHU_NGRAM_MAX_N", "5"))
            config["k"] = int(os.environ.get("YUNSHU_NGRAM_K", "5"))
            cap = os.environ.get("YUNSHU_NGRAM_CAPACITY")
            if cap:
                config["hashpool_capacity"] = int(cap)
        return SpecStrategyFactory.create(config)
