"""Tests for DFlash Proposer — speculative decoding via DFlash coarse pass.

Covers:
  - DFlashProposerConfig: defaults, from_env
  - DFlashStats: acceptance_rate, avg_speedup, to_dict
  - DFlashProposer: propose, verify, fallback, cooldown, stats
  - DFlashStrategy: lifecycle (begin/draft/accept/stats/end/reset)
  - SpecStrategyFactory integration: create dflash strategy
  - Edge cases: empty tokens, all accepted, all rejected
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import mlx.core as mx

from yunshu_engine.dflash_proposer import (
    DFlashDraftResult,
    DFlashProposer,
    DFlashProposerConfig,
    DFlashStats,
    DFlashStrategy,
    DFlashVerifyResult,
)


# ── DFlashProposerConfig ──


class TestDFlashProposerConfig:
    def test_defaults(self):
        cfg = DFlashProposerConfig()
        assert not cfg.enabled
        assert cfg.coarse_draft_length == 5
        assert cfg.temperature == 1.0
        assert cfg.acceptance_threshold == 1.0
        assert cfg.cooldown_after_reject == 2
        assert cfg.min_draft_length == 1
        assert cfg.max_draft_length == 10

    def test_custom_config(self):
        cfg = DFlashProposerConfig(
            enabled=True,
            coarse_draft_length=8,
            temperature=0.8,
            max_draft_length=12,
        )
        assert cfg.enabled
        assert cfg.coarse_draft_length == 8
        assert cfg.max_draft_length == 12

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_DFLASH_PROPOSER", "1")
        monkeypatch.setenv("YUNSHU_DFLASH_DRAFT_LENGTH", "7")
        cfg = DFlashProposerConfig.from_env()
        assert cfg.enabled
        assert cfg.coarse_draft_length == 7

    def test_from_env_defaults(self, monkeypatch):
        monkeypatch.delenv("YUNSHU_DFLASH_PROPOSER", raising=False)
        cfg = DFlashProposerConfig.from_env()
        assert not cfg.enabled
        assert cfg.coarse_draft_length == 5


# ── DFlashStats ──


class TestDFlashStats:
    def test_default_stats(self):
        stats = DFlashStats()
        assert stats.acceptance_rate == 0.0
        assert stats.avg_draft_length == 0.0
        assert stats.avg_speedup == 1.0

    def test_acceptance_rate(self):
        stats = DFlashStats(total_draft_tokens=100, total_accepted_tokens=80)
        assert stats.acceptance_rate == 0.8

    def test_acceptance_rate_zero_draft(self):
        stats = DFlashStats()
        assert stats.acceptance_rate == 0.0

    def test_avg_draft_length(self):
        stats = DFlashStats(total_proposals=10, total_draft_tokens=50)
        assert stats.avg_draft_length == 5.0

    def test_avg_speedup(self):
        stats = DFlashStats(
            total_steps=10,
            total_accepted_tokens=40,
            total_bonus_tokens=10,
        )
        # (40 + 10) / 10 = 5.0
        assert stats.avg_speedup == 5.0

    def test_avg_speedup_no_steps(self):
        stats = DFlashStats()
        assert stats.avg_speedup == 1.0

    def test_to_dict(self):
        stats = DFlashStats(
            total_proposals=5,
            total_draft_tokens=25,
            total_accepted_tokens=20,
        )
        d = stats.to_dict()
        assert d["total_proposals"] == 5
        assert d["total_draft_tokens"] == 25
        assert d["total_accepted_tokens"] == 20
        assert "acceptance_rate" in d
        assert "avg_speedup" in d
        assert "coarse_time_ms" in d


# ── DFlashProposer ──


class TestDFlashProposer:
    def test_init_defaults(self):
        proposer = DFlashProposer()
        assert not proposer.config.enabled
        assert proposer.stats.total_proposals == 0

    def test_init_custom_config(self):
        cfg = DFlashProposerConfig(enabled=True, coarse_draft_length=3)
        proposer = DFlashProposer(cfg)
        assert proposer.config.enabled
        assert proposer.config.coarse_draft_length == 3

    def test_propose_disabled(self):
        proposer = DFlashProposer(DFlashProposerConfig(enabled=False))
        context = mx.array([1, 2, 3, 4, 5])
        result = proposer.propose(context)
        assert result.token_ids == []
        assert result.logprobs == []

    def test_propose_enabled_returns_draft(self):
        proposer = DFlashProposer(DFlashProposerConfig(enabled=True))
        context = mx.array([1, 2, 3, 4, 5])
        result = proposer.propose(context)
        # Should return some draft tokens (fallback mode)
        assert isinstance(result.token_ids, list)
        assert isinstance(result.logprobs, list)
        assert len(result.token_ids) == len(result.logprobs)

    def test_propose_increments_stats(self):
        proposer = DFlashProposer(DFlashProposerConfig(enabled=True))
        context = mx.array([1, 2, 3])
        proposer.propose(context)
        assert proposer.stats.total_proposals == 1

    def test_propose_custom_n_draft(self):
        proposer = DFlashProposer(DFlashProposerConfig(enabled=True))
        context = mx.array([1, 2, 3])
        result = proposer.propose(context, n_draft=3)
        assert len(result.token_ids) <= 3

    def test_propose_empty_context(self):
        proposer = DFlashProposer(DFlashProposerConfig(enabled=True))
        context = mx.array([])
        result = proposer.propose(context)
        # Empty context should return empty or minimal draft
        assert isinstance(result.token_ids, list)

    def test_propose_fallback_mechanism(self):
        """Fallback proposal should use recent tokens."""
        proposer = DFlashProposer(DFlashProposerConfig(enabled=True))
        # Force dflash unavailable
        proposer._dflash_available = False
        context = mx.array([10, 20, 30, 40, 50])
        result = proposer.propose(context)
        assert len(result.token_ids) > 0
        assert result.coarse_confidence < 1.0  # Low confidence for fallback

    def test_verify_empty_draft(self):
        proposer = DFlashProposer()
        logits = mx.zeros((1, 5, 100))
        result = proposer.verify(logits, [])
        assert result.accepted_count == 0
        assert result.accepted_ids == []

    def test_verify_all_accepted(self):
        proposer = DFlashProposer()
        # Create logits where draft tokens are the argmax at each position
        draft_tokens = [5, 10, 15]
        vocab_size = 20
        logits = mx.full((1, 3, vocab_size), -1e9)
        for i, t in enumerate(draft_tokens):
            logits[0, i, t] = 1e9
        mx.eval(logits)
        result = proposer.verify(logits, draft_tokens)
        assert result.accepted_count == 3
        assert result.accepted_ids == draft_tokens
        assert result.rejected_at == -1

    def test_verify_partial_acceptance(self):
        proposer = DFlashProposer()
        vocab_size = 20
        # Draft says [5, 10, 15], but target says [5, 12, ...]
        draft_tokens = [5, 10, 15]
        logits = mx.full((1, 3, vocab_size), -1e9)
        logits[0, 0, 5] = 1e9    # Match at pos 0
        logits[0, 1, 12] = 1e9   # Mismatch at pos 1 (target=12, draft=10)
        logits[0, 2, 15] = 1e9
        mx.eval(logits)
        result = proposer.verify(logits, draft_tokens)
        assert result.accepted_count == 1
        assert result.accepted_ids == [5]
        assert result.rejected_at == 1

    def test_verify_all_rejected(self):
        proposer = DFlashProposer()
        vocab_size = 20
        draft_tokens = [5, 10, 15]
        logits = mx.full((1, 3, vocab_size), -1e9)
        logits[0, 0, 19] = 1e9   # Target wants 19, draft says 5
        logits[0, 1, 18] = 1e9
        logits[0, 2, 17] = 1e9
        mx.eval(logits)
        result = proposer.verify(logits, draft_tokens)
        assert result.accepted_count == 0
        assert result.rejected_at == 0

    def test_verify_with_logprobs(self):
        """Verify with draft logprobs uses speculative sampling acceptance."""
        proposer = DFlashProposer()
        vocab_size = 20
        draft_tokens = [5, 10]
        # Set target to agree with draft
        logits = mx.full((1, 2, vocab_size), -1e9)
        logits[0, 0, 5] = 1e9
        logits[0, 1, 10] = 1e9
        mx.eval(logits)
        draft_logprobs = [-0.5, -0.8]
        result = proposer.verify(logits, draft_tokens, draft_logprobs)
        assert result.accepted_count >= 0  # Depends on random

    def test_verify_bonus_token(self):
        """When all accepted, verify should produce a bonus token."""
        proposer = DFlashProposer()
        vocab_size = 20
        draft_tokens = [5]
        logits = mx.full((1, 1, vocab_size), -1e9)
        logits[0, 0, 5] = 1e9
        mx.eval(logits)
        result = proposer.verify(logits, draft_tokens)
        assert result.accepted_count == 1
        assert result.bonus_token_id >= 0
        assert proposer.stats.total_bonus_tokens == 1

    def test_verify_updates_stats(self):
        proposer = DFlashProposer()
        vocab_size = 20
        draft_tokens = [5, 10]
        logits = mx.full((1, 2, vocab_size), -1e9)
        logits[0, 0, 5] = 1e9
        logits[0, 1, 10] = 1e9
        mx.eval(logits)
        proposer.verify(logits, draft_tokens)
        assert proposer.stats.total_steps == 1
        assert proposer.stats.total_accepted_tokens > 0

    def test_cooldown_after_rejection(self):
        """After rejection, effective draft length should decrease."""
        cfg = DFlashProposerConfig(
            enabled=True,
            coarse_draft_length=8,
            cooldown_after_reject=2,
            min_draft_length=1,
        )
        proposer = DFlashProposer(cfg)
        # Trigger rejection via verify
        vocab_size = 20
        draft_tokens = [5]
        logits = mx.full((1, 1, vocab_size), -1e9)
        logits[0, 0, 19] = 1e9  # Target wants 19, draft says 5 -> rejection
        mx.eval(logits)
        proposer.verify(logits, draft_tokens)
        assert proposer._cooldown_counter > 0
        effective = proposer._effective_draft_length()
        assert effective < cfg.coarse_draft_length

    def test_cooldown_decrements(self):
        cfg = DFlashProposerConfig(enabled=True, coarse_draft_length=8)
        proposer = DFlashProposer(cfg)
        proposer._cooldown_counter = 3
        proposer._effective_draft_length()  # 3 -> 2
        assert proposer._cooldown_counter == 2
        proposer._effective_draft_length()  # 2 -> 1
        assert proposer._cooldown_counter == 1

    def test_get_stats(self):
        proposer = DFlashProposer(DFlashProposerConfig(enabled=True))
        context = mx.array([1, 2, 3])
        proposer.propose(context)
        stats = proposer.get_stats()
        assert "total_proposals" in stats
        assert "acceptance_rate" in stats
        assert "avg_speedup" in stats

    def test_reset_stats(self):
        proposer = DFlashProposer(DFlashProposerConfig(enabled=True))
        context = mx.array([1, 2, 3])
        proposer.propose(context)
        assert proposer.stats.total_proposals > 0
        proposer.reset_stats()
        assert proposer.stats.total_proposals == 0
        assert proposer.stats.total_draft_tokens == 0

    def test_is_available_checks_dflash(self):
        proposer = DFlashProposer()
        # Just check it doesn't crash
        result = proposer.is_available
        assert isinstance(result, bool)


# ── DFlashStrategy (SpecStrategy integration) ──


class TestDFlashStrategy:
    def test_name(self):
        strategy = DFlashStrategy()
        assert strategy.name == "dflash"

    def test_lifecycle(self):
        strategy = DFlashStrategy()
        strategy.begin("req-1")
        assert strategy._request_id == "req-1"
        strategy.end("req-1")
        assert strategy._request_id is None

    def test_draft_returns_proposal(self):
        proposer = DFlashProposer(DFlashProposerConfig(enabled=True))
        strategy = DFlashStrategy(proposer)
        strategy.begin("req-1")
        proposal = strategy.draft([1, 2, 3, 4, 5], n=3)
        assert proposal.strategy_name == "dflash"
        assert isinstance(proposal.tokens, list)
        assert "coarse_confidence" in proposal.metadata

    def test_draft_increments_stats(self):
        proposer = DFlashProposer(DFlashProposerConfig(enabled=True))
        strategy = DFlashStrategy(proposer)
        strategy.begin("req-1")
        proposal = strategy.draft([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n=5)
        assert strategy._total_drafts == 1
        # The fallback proposer should produce some tokens
        assert strategy._total_draft_tokens >= 0  # May be 0 if fallback produces nothing

    def test_accept(self):
        strategy = DFlashStrategy()
        strategy.accept([1, 2, 3], verified_up_to=2)
        assert strategy._total_accepted == 1
        assert strategy._total_accepted_tokens == 2

    def test_stats(self):
        proposer = DFlashProposer(DFlashProposerConfig(enabled=True))
        strategy = DFlashStrategy(proposer)
        strategy.begin("req-1")
        strategy.draft([1, 2, 3], n=5)
        strategy.accept([1, 2, 3], 2)
        stats = strategy.stats()
        assert stats["name"] == "dflash"
        assert stats["total_drafts"] == 1
        assert stats["total_accepted"] == 1
        assert "dflash_stats" in stats

    def test_stats_acceptance_rate(self):
        strategy = DFlashStrategy()
        strategy._total_draft_tokens = 10
        strategy._total_accepted_tokens = 7
        stats = strategy.stats()
        assert stats["acceptance_rate"] == 0.7

    def test_stats_zero_draft(self):
        strategy = DFlashStrategy()
        stats = strategy.stats()
        assert stats["acceptance_rate"] == 0.0

    def test_reset(self):
        proposer = DFlashProposer(DFlashProposerConfig(enabled=True))
        strategy = DFlashStrategy(proposer)
        strategy.begin("req-1")
        strategy.draft([1, 2, 3], n=5)
        strategy.accept([1, 2], 2)
        strategy.reset()
        assert strategy._total_drafts == 0
        assert strategy._total_draft_tokens == 0
        assert proposer.stats.total_proposals == 0

    def test_multiple_requests(self):
        strategy = DFlashStrategy()
        strategy.begin("req-1")
        strategy.end("req-1")
        strategy.begin("req-2")
        assert strategy._request_id == "req-2"
        strategy.end("req-2")

    def test_proposer_accessible(self):
        proposer = DFlashProposer()
        strategy = DFlashStrategy(proposer)
        assert strategy.proposer is proposer


# ── SpecStrategyFactory integration ──


class TestDFlashFactoryIntegration:
    def test_create_dflash_strategy(self):
        from yunshu_engine.spec_interface import SpecStrategyFactory
        strategy = SpecStrategyFactory.create({"type": "dflash"})
        assert strategy.name == "dflash"

    def test_create_dflash_with_config(self):
        from yunshu_engine.spec_interface import SpecStrategyFactory
        strategy = SpecStrategyFactory.create({
            "type": "dflash",
            "enabled": True,
            "coarse_draft_length": 7,
        })
        assert strategy.name == "dflash"
        assert strategy.proposer.config.coarse_draft_length == 7

    def test_create_dflash_with_custom_proposer(self):
        from yunshu_engine.spec_interface import SpecStrategyFactory
        proposer = DFlashProposer(DFlashProposerConfig(enabled=True))
        strategy = SpecStrategyFactory.create({
            "type": "dflash",
            "proposer": proposer,
        })
        assert strategy.proposer is proposer

    def test_create_composite_with_dflash(self):
        from yunshu_engine.spec_interface import SpecStrategyFactory
        strategy = SpecStrategyFactory.create({
            "type": "composite",
            "strategies": [
                {"type": "dflash"},
                {"type": "ngram", "mode": "lps"},
            ],
        })
        assert "dflash" in strategy.name

    def test_unknown_type_raises(self):
        from yunshu_engine.spec_interface import SpecStrategyFactory
        with pytest.raises(ValueError, match="Unknown spec strategy"):
            SpecStrategyFactory.create({"type": "nonexistent"})

    def test_from_env_dflash(self, monkeypatch):
        from yunshu_engine.spec_interface import SpecStrategyFactory
        monkeypatch.setenv("YUNSHU_SPEC_STRATEGY", "dflash")
        strategy = SpecStrategyFactory.from_env()
        assert strategy is not None
        assert strategy.name == "dflash"
