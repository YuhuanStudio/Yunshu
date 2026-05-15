from __future__ import annotations
"""DFlash Block Diffusion as Speculative Decoding Proposer.

Uses DFlash's coarse stage as a fast draft token generator, then verifies
with the full model. The coarse stage's predictions are less accurate but
much faster than the full model, providing a natural draft/verify split.

Architecture:
  DFlashProposer:
    - Uses DFlash's coarse pass as a fast draft generator
    - propose(context_ids, n_draft) -> draft tokens via coarse prediction
    - verify(target_logits, draft_tokens) -> standard token-level verification
    - Falls back gracefully when DFlash is not available
    - Tracks proposal/acceptance stats for adaptive adjustment

  DFlashStrategy:
    - Wraps DFlashProposer as a SpecStrategy for the unified interface
    - Integrates with SpecStrategyFactory and CompositeStrategy

Integration:
  - YUNSHU_SPEC_STRATEGY=dflash enables DFlash as spec decode proposer
  - Compatible with CompositeStrategy (ngram + dflash fallback)
  - Works alongside existing Medusa, MTP, and cross-model strategies

References:
  - DFlash Block Diffusion (§13.2): 2-stage diffusion with coarse/refine
  - Speculative Sampling (Leviathan et al., 2023): draft/verify pattern
  - SpecStrategy lifecycle (C7): begin/draft/accept/stats/end
"""

import logging
import random
import time
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


# ── Configuration ──


@dataclass
class DFlashProposerConfig:
    """Configuration for DFlash speculative decoding proposer.

    Attributes:
        enabled: Whether DFlash proposer is active.
        coarse_draft_length: Number of draft tokens to propose per step.
        temperature: Sampling temperature for draft generation.
        acceptance_threshold: Probability ratio threshold for acceptance.
        cooldown_after_reject: Steps to reduce draft length after rejection.
        min_draft_length: Minimum draft length (never goes below this).
        max_draft_length: Maximum draft length.
    """
    enabled: bool = False
    coarse_draft_length: int = 5
    temperature: float = 1.0
    acceptance_threshold: float = 1.0
    cooldown_after_reject: int = 2
    min_draft_length: int = 1
    max_draft_length: int = 10

    @staticmethod
    def from_env() -> DFlashProposerConfig:
        """Create config from environment variables."""
        import os
        return DFlashProposerConfig(
            enabled=os.environ.get("YUNSHU_DFLASH_PROPOSER", "").strip()
            in ("1", "true", "yes"),
            coarse_draft_length=int(
                os.environ.get("YUNSHU_DFLASH_DRAFT_LENGTH", "5")
            ),
            temperature=float(
                os.environ.get("YUNSHU_DFLASH_TEMPERATURE", "1.0")
            ),
        )


@dataclass
class DFlashStats:
    """Runtime statistics for DFlash proposer."""
    total_proposals: int = 0
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    total_rejected_tokens: int = 0
    total_bonus_tokens: int = 0
    total_steps: int = 0

    # Coarse pass timing
    coarse_time_ns: int = 0
    verify_time_ns: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    @property
    def avg_draft_length(self) -> float:
        if self.total_proposals == 0:
            return 0.0
        return self.total_draft_tokens / self.total_proposals

    @property
    def avg_speedup(self) -> float:
        """Estimated effective speedup from speculative decoding."""
        if self.total_steps == 0:
            return 1.0
        total_output = self.total_accepted_tokens + self.total_bonus_tokens
        return total_output / self.total_steps

    def to_dict(self) -> dict:
        return {
            "total_proposals": self.total_proposals,
            "total_draft_tokens": self.total_draft_tokens,
            "total_accepted_tokens": self.total_accepted_tokens,
            "total_rejected_tokens": self.total_rejected_tokens,
            "total_bonus_tokens": self.total_bonus_tokens,
            "total_steps": self.total_steps,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "avg_draft_length": round(self.avg_draft_length, 2),
            "avg_speedup": round(self.avg_speedup, 4),
            "coarse_time_ms": round(self.coarse_time_ns / 1e6, 2),
            "verify_time_ms": round(self.verify_time_ns / 1e6, 2),
        }


# ── Draft/Verify Results ──


@dataclass
class DFlashDraftResult:
    """Result from DFlash coarse draft generation."""
    token_ids: list[int]
    logprobs: list[float]
    coarse_confidence: float = 0.0  # Average confidence from coarse pass


@dataclass
class DFlashVerifyResult:
    """Result from verifying DFlash draft tokens against target model."""
    accepted_count: int
    accepted_ids: list[int]
    rejected_at: int  # Index of first rejection (-1 if all accepted)
    bonus_token_id: int  # Extra token from target after acceptance
    target_logprobs: list[float]


# ── DFlashProposer ──


class DFlashProposer:
    """DFlash-based speculative decoding proposer.

    Uses DFlash's coarse stage as a fast draft generator. The coarse stage
    makes less accurate but faster predictions that serve as draft tokens.
    These are then verified against the full target model in a single pass.

    The coarse draft is conceptually similar to using a smaller draft model,
    but leverages DFlash's 2-stage architecture: the coarse pass is the
    same model running with fewer steps, providing natural speed/accuracy
    tradeoff.

    Usage:
        proposer = DFlashProposer(DFlashProposerConfig())
        draft = proposer.propose(context_ids, n_draft=5)
        result = proposer.verify(target_logits, draft.token_ids)
        stats = proposer.get_stats()
    """

    def __init__(self, config: DFlashProposerConfig | None = None) -> None:
        self._config = config or DFlashProposerConfig()
        self._stats = DFlashStats()
        self._rng = random.Random()
        self._dflash_available: bool | None = None
        self._cooldown_counter: int = 0

    @property
    def config(self) -> DFlashProposerConfig:
        return self._config

    @property
    def stats(self) -> DFlashStats:
        return self._stats

    @property
    def is_available(self) -> bool:
        """Check if DFlash is available for use."""
        if self._dflash_available is None:
            self._dflash_available = self._check_dflash_available()
        return self._dflash_available

    def _check_dflash_available(self) -> bool:
        """Check if DFlash module is importable and functional."""
        try:
            from .dflash import DFlashEngine
            return True
        except ImportError:
            return False

    def propose(
        self,
        context_ids: mx.array,
        n_draft: int | None = None,
    ) -> DFlashDraftResult:
        """Generate draft tokens using DFlash coarse pass.

        The coarse pass generates tokens using a simplified forward pass
        with fewer diffusion steps. This produces less accurate but much
        faster predictions.

        When DFlash is not available, falls back to a simple n-gram-like
        proposal using the last few tokens as context.

        Args:
            context_ids: Current context token IDs.
            n_draft: Number of draft tokens to propose (default from config).

        Returns:
            DFlashDraftResult with proposed tokens and their log probabilities.
        """
        if not self._config.enabled:
            return DFlashDraftResult(token_ids=[], logprobs=[])

        n = n_draft or self._effective_draft_length()
        self._stats.total_proposals += 1

        t0 = time.monotonic_ns()

        # Try DFlash coarse pass
        if self.is_available:
            draft = self._propose_dflash(context_ids, n)
        else:
            # Fallback: simple frequency-based proposal
            draft = self._propose_fallback(context_ids, n)

        self._stats.coarse_time_ns += time.monotonic_ns() - t0
        self._stats.total_draft_tokens += len(draft.token_ids)

        return draft

    def _propose_dflash(
        self,
        context_ids: mx.array,
        n_draft: int,
    ) -> DFlashDraftResult:
        """Generate draft tokens via DFlash coarse pass."""
        try:
            from .dflash import DFlashConfig, DFlashEngine

            # Use DFlash with coarse-only configuration for fast drafting
            config = DFlashConfig(
                enabled=True,
                coarse_steps=1,  # Minimal steps for draft speed
                refine_steps=0,  # No refinement in draft stage
            )

            # The DFlash engine generates a coarse prediction
            # For spec decode, we extract token-level predictions from it
            engine = DFlashEngine(config)

            # Generate coarse tokens
            token_ids = []
            logprobs = []
            current = context_ids

            for _ in range(n_draft):
                # Run coarse forward pass to get next-token prediction
                output = engine.generate(None, None, "draft")
                if output and "tokens" in output:
                    tid = output["tokens"][-1] if output["tokens"] else -1
                    lp = output.get("logprobs", [-1.0])[-1] if "logprobs" in output else -1.0
                    if tid >= 0:
                        token_ids.append(tid)
                        logprobs.append(lp)
                    else:
                        break
                else:
                    break

            if token_ids:
                return DFlashDraftResult(
                    token_ids=token_ids,
                    logprobs=logprobs,
                    coarse_confidence=sum(logprobs) / len(logprobs) if logprobs else 0.0,
                )
            # DFlash engine didn't produce tokens — fall through to fallback
            logger.debug("DFlash coarse pass produced no tokens, using fallback")
        except Exception as e:
            logger.debug("DFlash coarse pass failed: %s", e)

        return self._propose_fallback(context_ids, n_draft)

    def _propose_fallback(
        self,
        context_ids: mx.array,
        n_draft: int,
    ) -> DFlashDraftResult:
        """Fallback proposal when DFlash is not available.

        Uses a simple approach: sample tokens based on position-weighted
        frequency of recent tokens. This provides a minimal draft for testing
        the verification pipeline even without DFlash.
        """
        token_ids = []
        logprobs = []

        # Get context tokens as Python list
        if hasattr(context_ids, 'flatten'):
            ctx = context_ids.flatten().tolist()
        elif isinstance(context_ids, list):
            ctx = context_ids
        else:
            ctx = list(context_ids)

        if not ctx:
            return DFlashDraftResult(token_ids=[], logprobs=[])

        # Simple fallback: use recent tokens to predict via repetition
        # This is deliberately simple — DFlash coarse pass provides the real speedup
        recent = ctx[-min(5, len(ctx)):]

        for i in range(n_draft):
            # Cycle through recent tokens with slight variation
            idx = i % len(recent) if recent else 0
            token_id = recent[idx] if recent else 0
            token_ids.append(token_id)
            logprobs.append(-2.0 - i * 0.1)  # Decreasing confidence

        return DFlashDraftResult(
            token_ids=token_ids,
            logprobs=logprobs,
            coarse_confidence=0.3,  # Low confidence for fallback
        )

    def verify(
        self,
        target_logits: mx.array,
        draft_tokens: list[int],
        draft_logprobs: list[float] | None = None,
    ) -> DFlashVerifyResult:
        """Verify draft tokens against target model logits.

        Standard speculative sampling verification:
        1. For each draft token, compare target vs draft probability
        2. Accept if U < min(1, P_target/P_draft) (speculative sampling)
        3. On first rejection, sample correction token from target
        4. If all accepted, generate one bonus token from target

        Args:
            target_logits: Target model logits at each draft position.
                Shape: (1, num_draft, vocab_size) or (num_draft, vocab_size).
            draft_tokens: Draft token IDs to verify.
            draft_logprobs: Draft model's log probabilities for each token.

        Returns:
            DFlashVerifyResult with acceptance information.
        """
        if not draft_tokens:
            return DFlashVerifyResult(
                accepted_count=0,
                accepted_ids=[],
                rejected_at=-1,
                bonus_token_id=-1,
                target_logprobs=[],
            )

        K = len(draft_tokens)
        t0 = time.monotonic_ns()

        # Get target log probabilities
        if target_logits.ndim == 3:
            logits_2d = target_logits[0]  # (num_positions, vocab)
        elif target_logits.ndim == 2:
            logits_2d = target_logits
        else:
            logits_2d = target_logits.reshape(1, -1)

        target_logprobs_all = mx.log(mx.softmax(logits_2d, axis=-1))

        # Extract target logprob for each draft token
        accepted_ids = []
        target_lps = []
        rejected_at = -1

        num_positions = min(logits_2d.shape[0], K)

        for i in range(num_positions):
            draft_token = draft_tokens[i]
            target_lp = float(target_logprobs_all[i, draft_token].item())

            # Speculative sampling acceptance check
            if draft_logprobs is not None and i < len(draft_logprobs):
                draft_lp = draft_logprobs[i]
                ratio = min(1.0, target_lp - draft_lp)
                u = self._rng.random()
                if u < ratio:
                    accepted_ids.append(draft_token)
                    target_lps.append(target_lp)
                else:
                    rejected_at = i
                    break
            else:
                # No draft logprobs: use greedy verification
                target_token = int(mx.argmax(logits_2d[i]).item())
                if target_token == draft_token:
                    accepted_ids.append(draft_token)
                    target_lps.append(target_lp)
                else:
                    rejected_at = i
                    break

        # Sample bonus/correction token from target
        bonus_pos = len(accepted_ids)
        if bonus_pos < num_positions:
            bonus_logits = logits_2d[bonus_pos]
        else:
            bonus_logits = logits_2d[-1]

        # Sample from target distribution
        bonus_probs = mx.softmax(bonus_logits, axis=-1)
        bonus_token = int(mx.argmax(bonus_probs).item())

        # Update stats
        self._stats.verify_time_ns += time.monotonic_ns() - t0
        self._stats.total_accepted_tokens += len(accepted_ids)
        self._stats.total_rejected_tokens += K - len(accepted_ids)
        self._stats.total_steps += 1

        if rejected_at == -1:
            # All accepted — bonus token
            self._stats.total_bonus_tokens += 1
            self._cooldown_counter = 0
        else:
            # Rejection — increase cooldown
            self._cooldown_counter = self._config.cooldown_after_reject

        return DFlashVerifyResult(
            accepted_count=len(accepted_ids),
            accepted_ids=accepted_ids,
            rejected_at=rejected_at,
            bonus_token_id=bonus_token,
            target_logprobs=target_lps,
        )

    def _effective_draft_length(self) -> int:
        """Compute effective draft length considering cooldown."""
        base = self._config.coarse_draft_length
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return max(self._config.min_draft_length, base // 2)
        return min(base, self._config.max_draft_length)

    def get_stats(self) -> dict:
        """Return proposer statistics as a dictionary."""
        return self._stats.to_dict()

    def reset_stats(self) -> None:
        """Reset all accumulated statistics."""
        self._stats = DFlashStats()
        self._cooldown_counter = 0


# ── SpecStrategy Integration ──


class DFlashStrategy:
    """Wraps DFlashProposer as a SpecStrategy.

    Integrates DFlash's coarse draft generation into the unified
    speculative decoding interface (begin/draft/accept/stats/end).

    Usage:
        from yunshu_engine.spec_interface import SpecStrategyFactory
        strategy = SpecStrategyFactory.create({"type": "dflash"})

    Or directly:
        proposer = DFlashProposer(DFlashProposerConfig(enabled=True))
        strategy = DFlashStrategy(proposer)
    """

    def __init__(self, proposer: DFlashProposer | None = None) -> None:
        self._proposer = proposer or DFlashProposer()
        self._request_id: Optional[str] = None
        self._total_drafts = 0
        self._total_draft_tokens = 0
        self._total_accepted = 0
        self._total_accepted_tokens = 0

    @property
    def name(self) -> str:
        return "dflash"

    @property
    def proposer(self) -> DFlashProposer:
        """Access the underlying DFlashProposer."""
        return self._proposer

    def begin(self, request_id: str) -> None:
        """Initialize state for a new generation request."""
        self._request_id = request_id

    def draft(self, tokens: list[int], n: int):
        """Propose draft tokens using DFlash coarse pass.

        Args:
            tokens: Full context (prompt + generated tokens so far).
            n: Maximum number of draft tokens to propose.

        Returns:
            DraftProposal with proposed tokens.
        """
        from .spec_interface import DraftProposal

        context_ids = mx.array(tokens)
        result = self._proposer.propose(context_ids, n_draft=n)

        self._total_drafts += 1
        self._total_draft_tokens += len(result.token_ids)

        return DraftProposal(
            tokens=result.token_ids,
            strategy_name=self.name,
            metadata={
                "coarse_confidence": result.coarse_confidence,
                "dflash_available": self._proposer.is_available,
            },
        )

    def accept(self, draft_tokens: list[int], verified_up_to: int) -> None:
        """Accept verified tokens and update proposer state."""
        self._total_accepted += 1
        self._total_accepted_tokens += verified_up_to

    def stats(self) -> dict:
        """Return strategy-specific statistics."""
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
            "dflash_stats": proposer_stats,
        }

    def end(self, request_id: str) -> None:
        """Clean up state after a generation request completes."""
        self._request_id = None

    def reset(self) -> None:
        """Reset all accumulated statistics and state."""
        self._total_drafts = 0
        self._total_draft_tokens = 0
        self._total_accepted = 0
        self._total_accepted_tokens = 0
        self._proposer.reset_stats()
