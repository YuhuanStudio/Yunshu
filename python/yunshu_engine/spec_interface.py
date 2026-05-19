from __future__ import annotations
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
  - MedusaStrategy: wraps MedusaProposer (multi-head prediction on hidden state)
  - LLMStrategy: wraps LLMProposer (smaller LLM as draft model)
  - Gemma4Strategy: wraps Gemma4SpecProposer (Gemma4 built-in spec decode)
  - DeltaNetInversionStrategy: wraps DeltaNetInverter (SSM state inversion for KV recovery)
  - CompositeStrategy: combines multiple strategies (first non-empty draft wins)
  - SpecStrategyFactory: creates the right strategy from config dict
"""

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
        self._proposer.reset()  # Clear cross-request pool state

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
        _hashpool = getattr(self._proposer, '_hashpool', None)
        if _hashpool is not None:
            _hashpool.clear()
        _lcg_pool = getattr(self._proposer, '_lcg_pool', None)
        if _lcg_pool is not None:
            _lcg_pool.clear()


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
        # NOTE: Do not inflate _total_draft_tokens here — actual tokens are
        # filled by the decoder's generate_draft.  Only count real drafts.
        self._total_drafts += 1
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
        # MTP always proposes exactly 1 draft token per step.
        # NOTE: Do not inflate _total_draft_tokens here — actual token is
        # filled by the decoder's _mtp_draft.  Only count real drafts.
        self._total_drafts += 1
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


class MedusaStrategy(SpecStrategy):
    """Wraps MedusaProposer as a SpecStrategy.

    The Medusa strategy adds K prediction heads on top of the base model's
    hidden state. Each head predicts a token at a different future position.
    No separate draft model is needed.

    Tree-based verification evaluates multiple candidate paths from the
    top-K logits of each head, then verifies the best paths against the
    target model.

    Note: This strategy requires a model with Medusa heads attached via
    MedusaProposer.attach(model). Returns empty proposals if not attached.
    """

    def __init__(self, proposer: Any = None, config: Any = None) -> None:
        self._proposer = proposer
        self._config = config
        self._request_id: Optional[str] = None
        self._total_drafts = 0
        self._total_draft_tokens = 0
        self._total_accepted = 0
        self._total_accepted_tokens = 0

    @property
    def name(self) -> str:
        return "medusa"

    @property
    def proposer(self) -> Any:
        """Access the underlying MedusaProposer (may be None)."""
        return self._proposer

    def begin(self, request_id: str) -> None:
        self._request_id = request_id

    def draft(self, tokens: list[int], n: int) -> DraftProposal:
        if self._proposer is None or not self._proposer.is_attached:
            return DraftProposal(tokens=[], strategy_name=self.name)
        # Medusa proposes via tree — actual tokens come from propose()
        # which requires hidden_states (provided by the engine).
        # NOTE: Do not inflate _total_draft_tokens here — actual tokens are
        # filled by proposer.propose(hidden_states).  Only count real drafts.
        self._total_drafts += 1
        num_heads = self._proposer.config.num_heads
        draft_len = min(n, num_heads)
        return DraftProposal(
            tokens=[],  # Actual tokens filled by proposer.propose(hidden_states)
            strategy_name=self.name,
            metadata={"num_heads": num_heads, "draft_length": draft_len},
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
        if self._proposer is not None:
            result["proposer_stats"] = self._proposer.get_stats()
        return result

    def end(self, request_id: str) -> None:
        self._request_id = None

    def reset(self) -> None:
        self._total_drafts = 0
        self._total_draft_tokens = 0
        self._total_accepted = 0
        self._total_accepted_tokens = 0
        if self._proposer is not None:
            self._proposer.reset_stats()


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
        self._last_proposer: Optional[str] = None  # name of strategy that proposed

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
        self._last_proposer = None
        for s in self._strategies:
            proposal = s.draft(tokens, n)
            if proposal.tokens:
                self._total_used[s.name] = self._total_used.get(s.name, 0) + 1
                self._last_proposer = s.name
                return proposal
        # All strategies returned empty
        return DraftProposal(tokens=[], strategy_name="composite(empty)")

    def accept(self, draft_tokens: list[int], verified_up_to: int) -> None:
        # Only forward accept to the strategy that actually proposed the draft.
        # Forwarding to ALL children would inflate acceptance stats for
        # non-participating strategies.
        if self._last_proposer is not None:
            for s in self._strategies:
                if s.name == self._last_proposer:
                    s.accept(draft_tokens, verified_up_to)
                    break
        self._last_proposer = None

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


class GPUNgramStrategy(SpecStrategy):
    """Wraps GPUNgramProposer as a SpecStrategy.

    GPU-accelerated N-gram matching using MLX arrays for vectorized
    comparison. Falls back to CPU NgramProposer on miss.
    """

    def __init__(self, config: Any = None) -> None:
        from .gpu_ngram import GPUNgramConfig, GPUNgramProposer
        if config is None:
            config = GPUNgramConfig()
        elif isinstance(config, dict):
            config = GPUNgramConfig(**config)
        self._config = config
        self._proposer = GPUNgramProposer(config)
        self._request_id: Optional[str] = None
        self._total_drafts = 0
        self._total_draft_tokens = 0
        self._total_accepted = 0
        self._total_accepted_tokens = 0

    @property
    def name(self) -> str:
        return "gpu_ngram"

    def begin(self, request_id: str) -> None:
        self._request_id = request_id

    def draft(self, tokens: list[int], n: int) -> DraftProposal:
        proposed = self._proposer.propose(tokens, n_draft=n)
        self._total_drafts += 1
        self._total_draft_tokens += len(proposed)
        return DraftProposal(
            tokens=proposed,
            strategy_name=self.name,
            metadata={"mode": "gpu_ngram"},
        )

    def accept(self, draft_tokens: list[int], verified_up_to: int) -> None:
        self._total_accepted += 1
        self._total_accepted_tokens += verified_up_to
        # Add accepted tokens to the GPU table for future matching
        if verified_up_to > 0:
            self._proposer.add_sequence(draft_tokens[:verified_up_to])

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
        table = getattr(self._proposer, 'table', None)
        if table is not None:
            table.clear()


class SuffixStrategy(SpecStrategy):
    """Wraps SuffixProposer as a SpecStrategy.

    Suffix-based speculative decoding finds common suffixes in the
    request's own generated text and reuses them as draft tokens.
    Very effective for repetitive outputs (code, JSON, structured text).
    """

    def __init__(self, config: Any = None) -> None:
        from .suffix_proposer import SuffixConfig, SuffixProposer
        if config is None:
            config = SuffixConfig()
        elif isinstance(config, dict):
            config = SuffixConfig(**config)
        self._config = config
        self._proposer = SuffixProposer(config)
        self._total_drafts = 0
        self._total_draft_tokens = 0
        self._total_accepted = 0
        self._total_accepted_tokens = 0

    @property
    def name(self) -> str:
        return "suffix"

    def begin(self, request_id: str) -> None:
        self._proposer.begin(request_id)

    def draft(self, tokens: list[int], n: int) -> DraftProposal:
        proposed = self._proposer.draft(tokens, n_draft=n)
        self._total_drafts += 1
        self._total_draft_tokens += len(proposed)
        return DraftProposal(
            tokens=proposed,
            strategy_name=self.name,
            metadata={"mode": "suffix"},
        )

    def accept(self, draft_tokens: list[int], verified_up_to: int) -> None:
        self._total_accepted += 1
        self._total_accepted_tokens += verified_up_to
        self._proposer.accept(draft_tokens, verified_up_to)

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
        self._proposer.end(request_id)

    def reset(self) -> None:
        self._total_drafts = 0
        self._total_draft_tokens = 0
        self._total_accepted = 0
        self._total_accepted_tokens = 0


class LLMStrategy(SpecStrategy):
    """Wraps LLMProposer as a SpecStrategy.

    The LLM strategy uses a separate smaller LLM as the draft model.
    The draft model generates token proposals that are verified against
    the target model. Unlike cross_model (EAGLE-3), this provides a
    general interface for any draft LLM.

    Note: This strategy requires a loaded draft model via
    LLMProposer.load_draft_model(). Returns empty proposals if not loaded.
    """

    def __init__(self, proposer: Any = None, config: Any = None) -> None:
        self._proposer = proposer
        self._config = config
        self._request_id: Optional[str] = None
        self._total_drafts = 0
        self._total_draft_tokens = 0
        self._total_accepted = 0
        self._total_accepted_tokens = 0

    @property
    def name(self) -> str:
        return "llm"

    @property
    def proposer(self) -> Any:
        """Access the underlying LLMProposer (may be None)."""
        return self._proposer

    def begin(self, request_id: str) -> None:
        self._request_id = request_id

    def draft(self, tokens: list[int], n: int) -> DraftProposal:
        if self._proposer is None or not self._proposer.is_loaded:
            return DraftProposal(tokens=[], strategy_name=self.name)
        draft_tokens = self._proposer.propose(tokens, n_draft=n)
        self._total_drafts += 1
        self._total_draft_tokens += len(draft_tokens)
        return DraftProposal(
            tokens=draft_tokens,
            strategy_name=self.name,
            metadata={
                "draft_model": self._proposer.config.draft_model_name,
                "draft_length": len(draft_tokens),
            },
        )

    def accept(self, draft_tokens: list[int], verified_up_to: int) -> None:
        self._total_accepted += 1
        self._total_accepted_tokens += verified_up_to
        if self._proposer is not None:
            self._proposer._stats.total_accepted_tokens += verified_up_to

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
        if self._proposer is not None:
            result["proposer_stats"] = self._proposer.get_stats()
        return result

    def end(self, request_id: str) -> None:
        self._request_id = None

    def reset(self) -> None:
        self._total_drafts = 0
        self._total_draft_tokens = 0
        self._total_accepted = 0
        self._total_accepted_tokens = 0
        if self._proposer is not None:
            self._proposer.reset_stats()


class Gemma4Strategy(SpecStrategy):
    """Wraps Gemma4SpecProposer as a SpecStrategy.

    The Gemma4 strategy leverages Gemma4's built-in speculative decoding
    capability where intermediate attention layers generate draft token
    predictions. No separate draft model is needed.

    Note: This strategy requires a Gemma4 model with spec capability
    detected via Gemma4SpecProposer.detect(). Returns empty proposals
    if the model is not Gemma4 or lacks spec layers.
    """

    def __init__(self, proposer: Any = None, config: Any = None) -> None:
        self._proposer = proposer
        self._config = config
        self._request_id: Optional[str] = None
        self._total_drafts = 0
        self._total_draft_tokens = 0
        self._total_accepted = 0
        self._total_accepted_tokens = 0

    @property
    def name(self) -> str:
        return "gemma4"

    @property
    def proposer(self) -> Any:
        """Access the underlying Gemma4SpecProposer (may be None)."""
        return self._proposer

    def begin(self, request_id: str) -> None:
        self._request_id = request_id

    def draft(self, tokens: list[int], n: int) -> DraftProposal:
        if self._proposer is None or not self._proposer.is_detected:
            return DraftProposal(tokens=[], strategy_name=self.name)
        # Gemma4 proposes via hidden_states — actual tokens come from propose()
        # which requires hidden_states (provided by the engine).
        # NOTE: Do not inflate _total_draft_tokens here — actual tokens are
        # filled by proposer.propose(hidden_states).  Only count real drafts.
        self._total_drafts += 1
        draft_len = min(n, self._proposer.config.draft_length)
        return DraftProposal(
            tokens=[],  # Actual tokens filled by proposer.propose(hidden_states)
            strategy_name=self.name,
            metadata={
                "draft_length": draft_len,
                "spec_layers": self._proposer.spec_layers,
            },
        )

    def accept(self, draft_tokens: list[int], verified_up_to: int) -> None:
        self._total_accepted += 1
        self._total_accepted_tokens += verified_up_to
        if self._proposer is not None:
            self._proposer._stats.total_accepted_tokens += verified_up_to

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
        if self._proposer is not None:
            result["proposer_stats"] = self._proposer.get_stats()
        return result

    def end(self, request_id: str) -> None:
        self._request_id = None

    def reset(self) -> None:
        self._total_drafts = 0
        self._total_draft_tokens = 0
        self._total_accepted = 0
        self._total_accepted_tokens = 0
        if self._proposer is not None:
            self._proposer.reset_stats()


class DeltaNetInversionStrategy(SpecStrategy):
    """DeltaNet state inversion as a speculative decoding strategy.

    Uses analytical inversion of the GatedDeltaNet SSM recurrence to recover
    pre-step state on rejection. This is a novel technique (first-ever DeltaNet
    state inversion, 75x less overhead than checkpoint/restore).

    NOTE: This strategy is experimental. The inversion introduces roundtrip
    error (~0.016 in BF16) which may be too high for exact token reproduction.
    It is most useful as a KV cache eviction recovery mechanism, but is wired
    here as a selectable spec decode strategy for research and future
    mixed-precision inference.

    Enable via YUNSHU_DELTANET_SPEC=1 env var or SpecStrategyFactory.create({"type": "deltanet"}).
    """

    def __init__(self, inverter: Any = None, config: Any = None) -> None:
        self._inverter = inverter
        self._config = config
        self._request_id: Optional[str] = None
        self._total_drafts = 0
        self._total_draft_tokens = 0
        self._total_accepted = 0
        self._total_accepted_tokens = 0
        self._total_inversions = 0
        self._total_inversion_failures = 0

    @property
    def name(self) -> str:
        return "deltanet_inversion"

    @property
    def inverter(self) -> Any:
        """Access the underlying DeltaNetInverter (may be None)."""
        return self._inverter

    def begin(self, request_id: str) -> None:
        self._request_id = request_id
        if self._inverter is not None:
            try:
                self._inverter.start_capture()
            except Exception as e:
                logger.debug("DeltaNet start_capture failed in begin(): %s", e)

    def draft(self, tokens: list[int], n: int) -> DraftProposal:
        if self._inverter is None:
            return DraftProposal(tokens=[], strategy_name=self.name)
        # DeltaNet inversion does not propose new tokens — it recovers state
        # after rejection. Return empty proposal; the value is in accept()
        # where the inverter is used to recover pre-verify state.
        self._total_drafts += 1
        return DraftProposal(
            tokens=[],
            strategy_name=self.name,
            metadata={"inversion_available": True},
        )

    def accept(self, draft_tokens: list[int], verified_up_to: int) -> None:
        self._total_accepted += 1
        self._total_accepted_tokens += verified_up_to
        # If fewer tokens accepted than proposed, trigger inversion recovery
        if verified_up_to < len(draft_tokens) and self._inverter is not None:
            try:
                recovered = self._inverter.invert_all()
                self._total_inversions += len(recovered)
            except Exception as e:
                self._total_inversion_failures += 1
                logger.debug("DeltaNet inversion in accept() failed: %s", e)

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
            "total_inversions": self._total_inversions,
            "total_inversion_failures": self._total_inversion_failures,
        }
        return result

    def end(self, request_id: str) -> None:
        self._request_id = None

    def reset(self) -> None:
        self._total_drafts = 0
        self._total_draft_tokens = 0
        self._total_accepted = 0
        self._total_accepted_tokens = 0
        self._total_inversions = 0
        self._total_inversion_failures = 0


class SpecStrategyFactory:
    """Factory for creating SpecStrategy instances from configuration dicts.

    Supported strategy types:
      - "ngram": NgramStrategy with optional mode ("lps" or "hashpool")
      - "gpu_ngram": GPUNgramStrategy with MLX-accelerated lookup
      - "suffix": SuffixStrategy for self-pattern reuse
      - "cross_model": CrossModelStrategy (requires draft model)
      - "mtp": MTPStrategy (requires model with MTP heads)
      - "medusa": MedusaStrategy (requires MedusaProposer with attached heads)
      - "llm": LLMStrategy (smaller LLM as draft model)
      - "gemma4": Gemma4Strategy (Gemma4 built-in spec decode)
      - "dflash": DFlashStrategy (DFlash coarse pass as draft proposer)
      - "deltanet": DeltaNetInversionStrategy (SSM state inversion for KV recovery)
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
        elif strategy_type == "gpu_ngram":
            return SpecStrategyFactory._create_gpu_ngram(config)
        elif strategy_type == "suffix":
            return SpecStrategyFactory._create_suffix(config)
        elif strategy_type == "cross_model":
            return SpecStrategyFactory._create_cross_model(config)
        elif strategy_type == "mtp":
            return SpecStrategyFactory._create_mtp(config)
        elif strategy_type == "medusa":
            return SpecStrategyFactory._create_medusa(config)
        elif strategy_type == "llm":
            return SpecStrategyFactory._create_llm(config)
        elif strategy_type == "gemma4":
            return SpecStrategyFactory._create_gemma4(config)
        elif strategy_type == "dflash":
            return SpecStrategyFactory._create_dflash(config)
        elif strategy_type == "deltanet":
            return SpecStrategyFactory._create_deltanet(config)
        elif strategy_type == "composite":
            return SpecStrategyFactory._create_composite(config)
        else:
            raise ValueError(
                f"Unknown spec strategy type: {strategy_type!r}. "
                f"Supported: ngram, gpu_ngram, suffix, cross_model, mtp, "
                f"medusa, llm, gemma4, dflash, deltanet, composite"
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
    def _create_gpu_ngram(config: dict) -> GPUNgramStrategy:
        from .gpu_ngram import GPUNgramConfig
        kwargs: dict[str, Any] = {}
        if "min_n" in config:
            kwargs["min_n"] = int(config["min_n"])
        if "max_n" in config:
            kwargs["max_n"] = int(config["max_n"])
        if "k" in config:
            kwargs["k"] = int(config["k"])
        if "max_model_len" in config:
            kwargs["max_model_len"] = int(config["max_model_len"])
        if "max_table_entries" in config:
            kwargs["max_table_entries"] = int(config["max_table_entries"])
        gpu_config = GPUNgramConfig(**kwargs)
        return GPUNgramStrategy(config=gpu_config)

    @staticmethod
    def _create_suffix(config: dict) -> SuffixStrategy:
        from .suffix_proposer import SuffixConfig
        kwargs: dict[str, Any] = {}
        if "min_suffix_length" in config:
            kwargs["min_suffix_length"] = int(config["min_suffix_length"])
        if "max_window" in config:
            kwargs["max_window"] = int(config["max_window"])
        if "max_draft" in config:
            kwargs["max_draft"] = int(config["max_draft"])
        if "max_model_len" in config:
            kwargs["max_model_len"] = int(config["max_model_len"])
        if "max_trie_depth" in config:
            kwargs["max_trie_depth"] = int(config["max_trie_depth"])
        suffix_config = SuffixConfig(**kwargs)
        return SuffixStrategy(config=suffix_config)

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
    def _create_medusa(config: dict) -> MedusaStrategy:
        from .medusa_proposer import MedusaConfig, MedusaProposer
        proposer = config.get("proposer")
        medusa_config = config.get("medusa_config")
        if proposer is None:
            # Create proposer from config params
            kwargs = {}
            if "num_heads" in config:
                kwargs["num_heads"] = int(config["num_heads"])
            if "tree_size" in config:
                kwargs["tree_size"] = int(config["tree_size"])
            if "top_k_per_head" in config:
                kwargs["top_k_per_head"] = int(config["top_k_per_head"])
            medusa_config = MedusaConfig(**kwargs)
            proposer = MedusaProposer(medusa_config)
        return MedusaStrategy(proposer=proposer, config=medusa_config)

    @staticmethod
    def _create_llm(config: dict) -> LLMStrategy:
        from .llm_proposer import LLMProposerConfig, LLMProposer
        proposer = config.get("proposer")
        llm_config = config.get("llm_config")
        if proposer is None:
            kwargs: dict[str, Any] = {}
            if "draft_model_name" in config:
                kwargs["draft_model_name"] = config["draft_model_name"]
            if "max_draft_length" in config:
                kwargs["max_draft_length"] = int(config["max_draft_length"])
            if "temperature" in config:
                kwargs["temperature"] = float(config["temperature"])
            if "top_k" in config:
                kwargs["top_k"] = int(config["top_k"])
            if "compiled" in config:
                kwargs["compiled"] = bool(config["compiled"])
            llm_config = LLMProposerConfig(**kwargs)
            proposer = LLMProposer(llm_config)
        return LLMStrategy(proposer=proposer, config=llm_config)

    @staticmethod
    def _create_gemma4(config: dict) -> Gemma4Strategy:
        from .gemma4_spec import Gemma4SpecConfig, Gemma4SpecProposer
        proposer = config.get("proposer")
        gemma4_config = config.get("gemma4_config")
        if proposer is None:
            kwargs: dict[str, Any] = {}
            if "enabled" in config:
                kwargs["enabled"] = bool(config["enabled"])
            if "draft_length" in config:
                kwargs["draft_length"] = int(config["draft_length"])
            if "acceptance_threshold" in config:
                kwargs["acceptance_threshold"] = float(config["acceptance_threshold"])
            gemma4_config = Gemma4SpecConfig(**kwargs)
            proposer = Gemma4SpecProposer(gemma4_config)
        return Gemma4Strategy(proposer=proposer, config=gemma4_config)

    @staticmethod
    def _create_dflash(config: dict) -> "DFlashStrategy":
        from .dflash_proposer import DFlashProposer, DFlashProposerConfig, DFlashStrategy
        proposer = config.get("proposer")
        dflash_config = config.get("dflash_config")
        if proposer is None:
            kwargs: dict[str, Any] = {}
            if "enabled" in config:
                kwargs["enabled"] = bool(config["enabled"])
            if "coarse_draft_length" in config:
                kwargs["coarse_draft_length"] = int(config["coarse_draft_length"])
            if "temperature" in config:
                kwargs["temperature"] = float(config["temperature"])
            if "max_draft_length" in config:
                kwargs["max_draft_length"] = int(config["max_draft_length"])
            dflash_config = DFlashProposerConfig(**kwargs)
            proposer = DFlashProposer(dflash_config)
        return DFlashStrategy(proposer=proposer)

    @staticmethod
    def _create_deltanet(config: dict) -> DeltaNetInversionStrategy:
        inverter = config.get("inverter")
        if inverter is None:
            from .deltanet_inversion import DeltaNetInverter
            inverter = DeltaNetInverter()
        return DeltaNetInversionStrategy(inverter=inverter, config=config)

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
        elif strategy_type == "gpu_ngram":
            config["max_n"] = int(os.environ.get("YUNSHU_NGRAM_MAX_N", "5"))
            config["k"] = int(os.environ.get("YUNSHU_NGRAM_K", "5"))
        elif strategy_type == "suffix":
            config["min_suffix_length"] = int(
                os.environ.get("YUNSHU_SUFFIX_MIN_LEN", "3"))
            config["max_window"] = int(
                os.environ.get("YUNSHU_SUFFIX_MAX_WINDOW", "512"))
            config["max_draft"] = int(
                os.environ.get("YUNSHU_SUFFIX_MAX_DRAFT", "5"))
        elif strategy_type == "medusa":
            config["num_heads"] = int(os.environ.get("YUNSHU_MEDUSA_HEADS", "4"))
            config["tree_size"] = int(os.environ.get("YUNSHU_MEDUSA_TREE_SIZE", "5"))
        elif strategy_type == "llm":
            config["draft_model_name"] = os.environ.get(
                "YUNSHU_LLM_DRAFT_MODEL", "")
            config["max_draft_length"] = int(
                os.environ.get("YUNSHU_LLM_MAX_DRAFT", "5"))
        elif strategy_type == "gemma4":
            config["enabled"] = True
            config["draft_length"] = int(
                os.environ.get("YUNSHU_GEMMA4_DRAFT_LENGTH", "4"))
        elif strategy_type == "dflash":
            config["enabled"] = True
            config["coarse_draft_length"] = int(
                os.environ.get("YUNSHU_DFLASH_DRAFT_LENGTH", "5"))
        elif strategy_type == "deltanet":
            config["enabled"] = True
        return SpecStrategyFactory.create(config)
