"""Tests for Medusa speculative decoding proposer.

Covers:
  - MedusaConfig defaults and custom values
  - MedusaHead forward pass and weight initialization
  - MedusaProposer attach/detach lifecycle
  - Tree-based proposal generation
  - Greedy proposal (single path)
  - Verify against target logits
  - Load/save Medusa weights
  - MedusaStats tracking and reset
  - MedusaStrategy in spec_interface
  - SpecStrategyFactory "medusa" type
  - CompositeStrategy with Medusa
  - Edge cases: no model, no heads, empty proposals
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import mlx.core as mx
import mlx.nn as nn

from yunshu_engine.medusa_proposer import (
    MedusaConfig,
    MedusaHead,
    MedusaProposer,
    MedusaStats,
    _MedusaTreeNode,
)
from yunshu_engine.spec_interface import (
    CompositeStrategy,
    MedusaStrategy,
    SpecStrategyFactory,
)


# ── Helpers ──


class FakeModel(nn.Module):
    """Fake model with lm_head for testing Medusa attachment."""

    def __init__(self, hidden_size: int = 64, vocab_size: int = 256):
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)


class FakeModelWithLanguageModel(nn.Module):
    """Fake model with language_model wrapper."""

    def __init__(self, hidden_size: int = 64, vocab_size: int = 256):
        super().__init__()
        self.language_model = FakeModel(hidden_size, vocab_size)


class FakeModelNoHead(nn.Module):
    """Fake model without lm_head."""

    def __init__(self):
        super().__init__()
        self.some_layer = nn.Linear(32, 32)


# ── MedusaConfig Tests ──


class TestMedusaConfig:
    def test_defaults(self):
        config = MedusaConfig()
        assert config.num_heads == 4
        assert config.tree_size == 5
        assert config.enabled is False
        assert config.top_k_per_head == 5
        assert config.residual_length == 0

    def test_custom_values(self):
        config = MedusaConfig(num_heads=8, tree_size=10, enabled=True, top_k_per_head=3)
        assert config.num_heads == 8
        assert config.tree_size == 10
        assert config.enabled is True
        assert config.top_k_per_head == 3


# ── MedusaHead Tests ──


class TestMedusaHead:
    def test_forward_shape(self):
        head = MedusaHead(hidden_size=64, vocab_size=256)
        x = mx.random.normal((1, 1, 64))
        out = head(x)
        assert out.shape == (1, 1, 256)

    def test_forward_batch(self):
        head = MedusaHead(hidden_size=64, vocab_size=256)
        x = mx.random.normal((2, 5, 64))
        out = head(x)
        assert out.shape == (2, 5, 256)

    def test_single_token(self):
        head = MedusaHead(hidden_size=32, vocab_size=128)
        x = mx.random.normal((1, 1, 32))
        out = head(x)
        assert out.shape == (1, 1, 128)
        assert out.dtype == mx.float32


# ── _MedusaTreeNode Tests ──


class TestMedusaTreeNode:
    def test_root_path_tokens(self):
        root = _MedusaTreeNode(token_id=-1)
        assert root.path_tokens() == []

    def test_single_child_path(self):
        root = _MedusaTreeNode(token_id=-1)
        child = root.add_child(42, -0.5)
        assert child.token_id == 42
        assert child.cumulative_logprob == -0.5
        assert child.depth == 1
        assert child.path_tokens() == [42]

    def test_multi_level_path(self):
        root = _MedusaTreeNode(token_id=-1)
        c1 = root.add_child(1, -0.1)
        c2 = c1.add_child(2, -0.2)
        c3 = c2.add_child(3, -0.3)
        assert c3.path_tokens() == [1, 2, 3]
        assert abs(c3.cumulative_logprob - (-0.6)) < 1e-6
        assert c3.depth == 3

    def test_multiple_children(self):
        root = _MedusaTreeNode(token_id=-1)
        c1 = root.add_child(10, -1.0)
        c2 = root.add_child(20, -0.5)
        c3 = root.add_child(30, -0.1)
        assert len(root.children) == 3
        assert c1.cumulative_logprob == -1.0
        assert c2.cumulative_logprob == -0.5
        assert c3.cumulative_logprob == -0.1


# ── MedusaStats Tests ──


class TestMedusaStats:
    def test_defaults(self):
        stats = MedusaStats()
        assert stats.total_proposals == 0
        assert stats.total_draft_tokens == 0
        assert stats.total_accepted_tokens == 0
        assert stats.acceptance_rate == 0.0
        assert stats.avg_draft_length == 0.0

    def test_acceptance_rate(self):
        stats = MedusaStats(total_draft_tokens=100, total_accepted_tokens=65)
        assert abs(stats.acceptance_rate - 0.65) < 1e-6

    def test_avg_draft_length(self):
        stats = MedusaStats(total_proposals=10, total_draft_tokens=50)
        assert abs(stats.avg_draft_length - 5.0) < 1e-6

    def test_to_dict(self):
        stats = MedusaStats(
            total_proposals=5,
            total_draft_tokens=20,
            total_accepted_tokens=10,
            heads_attached=4,
        )
        d = stats.to_dict()
        assert d["total_proposals"] == 5
        assert d["total_draft_tokens"] == 20
        assert d["total_accepted_tokens"] == 10
        assert d["heads_attached"] == 4
        assert "acceptance_rate" in d
        assert "avg_draft_length" in d


# ── MedusaProposer Tests ──


class TestMedusaProposer:
    def test_init_defaults(self):
        proposer = MedusaProposer()
        assert proposer.config.num_heads == 4
        assert not proposer.is_attached
        assert proposer.heads == []

    def test_attach_model(self):
        model = FakeModel(hidden_size=64, vocab_size=256)
        proposer = MedusaProposer(MedusaConfig(num_heads=3))
        proposer.attach(model)
        assert proposer.is_attached
        assert len(proposer.heads) == 3

    def test_attach_wrapped_model(self):
        model = FakeModelWithLanguageModel(hidden_size=64, vocab_size=256)
        proposer = MedusaProposer(MedusaConfig(num_heads=2))
        proposer.attach(model)
        assert proposer.is_attached
        assert len(proposer.heads) == 2

    def test_attach_model_no_head(self):
        model = FakeModelNoHead()
        proposer = MedusaProposer(MedusaConfig(num_heads=2))
        proposer.attach(model)
        # Should not attach (no lm_head found)
        assert not proposer.is_attached

    def test_attach_idempotent(self):
        model = FakeModel(hidden_size=64, vocab_size=256)
        proposer = MedusaProposer(MedusaConfig(num_heads=2))
        proposer.attach(model)
        heads_count = len(proposer.heads)
        proposer.attach(model)  # Second attach should be a no-op
        assert len(proposer.heads) == heads_count

    def test_detach(self):
        model = FakeModel(hidden_size=64, vocab_size=256)
        proposer = MedusaProposer(MedusaConfig(num_heads=2))
        proposer.attach(model)
        assert proposer.is_attached
        proposer.detach()
        assert not proposer.is_attached
        assert proposer.heads == []

    def test_propose_without_attach(self):
        proposer = MedusaProposer()
        hidden = mx.random.normal((1, 1, 64))
        result = proposer.propose(hidden)
        assert result == []

    def test_propose_returns_candidates(self):
        model = FakeModel(hidden_size=64, vocab_size=256)
        proposer = MedusaProposer(MedusaConfig(num_heads=3, tree_size=3, top_k_per_head=2))
        proposer.attach(model)
        hidden = mx.random.normal((1, 1, 64))
        candidates = proposer.propose(hidden, n_draft=3)
        assert isinstance(candidates, list)
        # Each candidate should be a list of token IDs
        for c in candidates:
            assert isinstance(c, list)
            for t in c:
                assert isinstance(t, int)
                assert 0 <= t < 256

    def test_propose_updates_stats(self):
        model = FakeModel(hidden_size=64, vocab_size=256)
        proposer = MedusaProposer(MedusaConfig(num_heads=2, tree_size=2))
        proposer.attach(model)
        hidden = mx.random.normal((1, 1, 64))
        proposer.propose(hidden)
        assert proposer.stats.total_proposals == 1

    def test_greedy_propose(self):
        model = FakeModel(hidden_size=64, vocab_size=256)
        proposer = MedusaProposer(MedusaConfig(num_heads=4))
        proposer.attach(model)
        hidden = mx.random.normal((1, 1, 64))
        tokens = proposer.greedy_propose(hidden)
        assert len(tokens) == 4
        for t in tokens:
            assert isinstance(t, int)
            assert 0 <= t < 256

    def test_greedy_propose_without_attach(self):
        proposer = MedusaProposer()
        hidden = mx.random.normal((1, 1, 64))
        tokens = proposer.greedy_propose(hidden)
        assert tokens == []

    def test_verify_empty_candidates(self):
        proposer = MedusaProposer()
        logits = mx.random.normal((1, 5, 256))
        count, tokens = proposer.verify(logits, [])
        assert count == 0
        assert tokens == []

    def test_verify_single_match(self):
        model = FakeModel(hidden_size=64, vocab_size=256)
        proposer = MedusaProposer(MedusaConfig(num_heads=3))
        proposer.attach(model)

        # Create target logits that match the first candidate token
        target_logits = mx.zeros((1, 3, 256))
        candidate = [[42, 100, 200]]
        # Set argmax for position 0 to match
        target_logits[0, 0, 42] = 10.0

        count, tokens = proposer.verify(target_logits, candidate)
        assert count == 1
        assert tokens == [42]

    def test_verify_all_match(self):
        model = FakeModel(hidden_size=64, vocab_size=256)
        proposer = MedusaProposer(MedusaConfig(num_heads=3))
        proposer.attach(model)

        candidate = [[42, 100, 200]]
        target_logits = mx.zeros((1, 3, 256))
        target_logits[0, 0, 42] = 10.0
        target_logits[0, 1, 100] = 10.0
        target_logits[0, 2, 200] = 10.0

        count, tokens = proposer.verify(target_logits, candidate)
        assert count == 3
        assert tokens == [42, 100, 200]

    def test_verify_no_match(self):
        model = FakeModel(hidden_size=64, vocab_size=256)
        proposer = MedusaProposer(MedusaConfig(num_heads=2))
        proposer.attach(model)

        candidate = [[42, 100]]
        target_logits = mx.zeros((1, 2, 256))
        # Set argmax to different tokens
        target_logits[0, 0, 99] = 10.0

        count, tokens = proposer.verify(target_logits, candidate)
        assert count == 0
        assert tokens == []

    def test_get_stats(self):
        model = FakeModel(hidden_size=64, vocab_size=256)
        proposer = MedusaProposer(MedusaConfig(num_heads=2))
        proposer.attach(model)
        stats = proposer.get_stats()
        assert "total_proposals" in stats
        assert "heads_attached" in stats
        assert stats["heads_attached"] == 2

    def test_reset_stats(self):
        model = FakeModel(hidden_size=64, vocab_size=256)
        proposer = MedusaProposer(MedusaConfig(num_heads=2))
        proposer.attach(model)
        hidden = mx.random.normal((1, 1, 64))
        proposer.greedy_propose(hidden)
        assert proposer.stats.total_proposals > 0
        proposer.reset_stats()
        assert proposer.stats.total_proposals == 0
        assert proposer.stats.heads_attached == 2  # Preserved

    def test_forward_heads(self):
        model = FakeModel(hidden_size=64, vocab_size=256)
        proposer = MedusaProposer(MedusaConfig(num_heads=3))
        proposer.attach(model)
        hidden = mx.random.normal((1, 1, 64))
        head_outputs = proposer.forward_heads(hidden)
        assert len(head_outputs) == 3
        for out in head_outputs:
            assert out.shape[-1] == 256

    def test_forward_heads_without_attach(self):
        proposer = MedusaProposer()
        hidden = mx.random.normal((1, 1, 64))
        head_outputs = proposer.forward_heads(hidden)
        assert head_outputs == []

    def test_load_weights_nonexistent(self):
        model = FakeModel(hidden_size=64, vocab_size=256)
        proposer = MedusaProposer(MedusaConfig(num_heads=2))
        proposer.attach(model)
        result = proposer.load_medusa_weights("/nonexistent/path")
        assert result is False

    def test_load_weights_without_attach(self):
        proposer = MedusaProposer()
        result = proposer.load_medusa_weights("/tmp")
        assert result is False


# ── MedusaStrategy Tests ──


class TestMedusaStrategy:
    def test_name(self):
        strategy = MedusaStrategy()
        assert strategy.name == "medusa"

    def test_begin_end_lifecycle(self):
        strategy = MedusaStrategy()
        strategy.begin("req-1")
        strategy.end("req-1")
        # No error = success

    def test_draft_without_proposer(self):
        strategy = MedusaStrategy()
        proposal = strategy.draft([1, 2, 3], n=5)
        assert proposal.tokens == []
        assert proposal.strategy_name == "medusa"

    def test_draft_with_proposer(self):
        model = FakeModel(hidden_size=64, vocab_size=256)
        proposer = MedusaProposer(MedusaConfig(num_heads=4))
        proposer.attach(model)
        strategy = MedusaStrategy(proposer=proposer)
        proposal = strategy.draft([1, 2, 3], n=5)
        assert proposal.strategy_name == "medusa"
        assert proposal.metadata["num_heads"] == 4

    def test_accept_updates_stats(self):
        strategy = MedusaStrategy()
        strategy.accept([1, 2, 3], verified_up_to=2)
        stats = strategy.stats()
        assert stats["total_accepted"] == 1
        assert stats["total_accepted_tokens"] == 2

    def test_stats_without_proposer(self):
        strategy = MedusaStrategy()
        stats = strategy.stats()
        assert stats["name"] == "medusa"
        assert stats["acceptance_rate"] == 0.0

    def test_stats_with_proposer(self):
        model = FakeModel(hidden_size=64, vocab_size=256)
        proposer = MedusaProposer(MedusaConfig(num_heads=2))
        proposer.attach(model)
        strategy = MedusaStrategy(proposer=proposer)
        stats = strategy.stats()
        assert "proposer_stats" in stats
        assert stats["proposer_stats"]["heads_attached"] == 2

    def test_reset(self):
        strategy = MedusaStrategy()
        strategy.accept([1, 2], verified_up_to=2)
        strategy.reset()
        stats = strategy.stats()
        assert stats["total_accepted"] == 0


# ── SpecStrategyFactory Tests ──


class TestSpecStrategyFactoryMedusa:
    def test_create_medusa(self):
        strategy = SpecStrategyFactory.create({"type": "medusa"})
        assert isinstance(strategy, MedusaStrategy)
        assert strategy.name == "medusa"

    def test_create_medusa_with_params(self):
        strategy = SpecStrategyFactory.create({
            "type": "medusa",
            "num_heads": 8,
            "tree_size": 10,
        })
        assert isinstance(strategy, MedusaStrategy)
        assert strategy._proposer is not None
        assert strategy._proposer.config.num_heads == 8
        assert strategy._proposer.config.tree_size == 10

    def test_create_medusa_with_proposer(self):
        proposer = MedusaProposer(MedusaConfig(num_heads=2))
        strategy = SpecStrategyFactory.create({
            "type": "medusa",
            "proposer": proposer,
        })
        assert strategy.proposer is proposer

    def test_composite_with_medusa_and_ngram(self):
        strategy = SpecStrategyFactory.create({
            "type": "composite",
            "strategies": [
                {"type": "ngram", "mode": "lps"},
                {"type": "medusa", "num_heads": 4},
            ],
        })
        assert isinstance(strategy, CompositeStrategy)
        assert "ngram" in strategy.name
        assert "medusa" in strategy.name

    def test_unknown_type_still_raises(self):
        with pytest.raises(ValueError, match="Unknown spec strategy type"):
            SpecStrategyFactory.create({"type": "unknown"})


# ── CompositeStrategy with Medusa ──


class TestCompositeWithMedusa:
    def test_composite_medusa_fallback(self):
        """Medusa without attach returns empty, so ngram should be tried."""
        ngram = SpecStrategyFactory.create({"type": "ngram", "mode": "hashpool"})
        medusa = MedusaStrategy()  # No proposer attached
        composite = CompositeStrategy([medusa, ngram])
        composite.begin("req-1")

        # With enough repeating tokens, ngram should produce a proposal
        tokens = [1, 2, 3, 4, 5] * 20
        proposal = composite.draft(tokens, n=5)
        # Either medusa returns empty (ngram fills in) or both empty
        assert proposal.strategy_name in ("ngram", "composite(empty)")

    def test_composite_all_empty(self):
        medusa = MedusaStrategy()  # No proposer
        composite = CompositeStrategy([medusa])
        composite.begin("req-1")
        proposal = composite.draft([1, 2], n=5)
        assert proposal.tokens == []
        assert "empty" in proposal.strategy_name
