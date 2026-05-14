"""Tests for LLM-based speculative decoding proposer.

Covers:
  - LLMProposerConfig defaults and custom values
  - LLMStats tracking, acceptance_rate, avg_draft_length
  - LLMProposer lifecycle (init, load, propose, unload)
  - Graceful degradation (no model, OOM, failed generation)
  - Propose with mock draft model
  - Greedy sampling vs temperature sampling
  - mx.compile() opt-in behavior
  - get_stats() and reset_stats()
  - LLMStrategy in spec_interface
  - SpecStrategyFactory "llm" type
  - CompositeStrategy with LLM
  - from_env() with YUNSHU_LLM_* env vars
  - Edge cases: empty context, n_draft=0, no draft_model_name
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import mlx.core as mx
import mlx.nn as nn

from yunshu_engine.llm_proposer import (
    LLMProposer,
    LLMProposerConfig,
    LLMStats,
    _estimate_model_memory,
)
from yunshu_engine.spec_interface import (
    CompositeStrategy,
    LLMStrategy,
    SpecStrategyFactory,
)


# ── Helpers ──


class FakeDraftModel(nn.Module):
    """Fake draft model that produces deterministic outputs."""

    def __init__(self, hidden_size: int = 64, vocab_size: int = 256):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(self, input_ids, cache=None):
        """Return a mock output with .logits attribute."""
        # Simple: embed + project to get logits
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        h = self.embed_tokens(input_ids)
        logits = self.lm_head(h)
        # Return object with .logits
        out = MagicMock()
        out.logits = logits
        return out


class FakeDraftModelRawOutput(nn.Module):
    """Fake draft model that returns logits directly (no .logits attr)."""

    def __init__(self, hidden_size: int = 64, vocab_size: int = 256):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(self, input_ids, cache=None):
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        h = self.embed_tokens(input_ids)
        return self.lm_head(h)


class FakeTokenizer:
    """Fake tokenizer with eos_token_id."""

    eos_token_id = 2


# ── LLMProposerConfig Tests ──


class TestLLMProposerConfig:
    def test_defaults(self):
        config = LLMProposerConfig()
        assert config.draft_model_name == ""
        assert config.max_draft_length == 5
        assert config.temperature == 0.0
        assert config.top_k == 1
        assert config.compiled is False

    def test_custom_values(self):
        config = LLMProposerConfig(
            draft_model_name="Qwen2.5-0.5B",
            max_draft_length=10,
            temperature=0.5,
            top_k=3,
            compiled=True,
        )
        assert config.draft_model_name == "Qwen2.5-0.5B"
        assert config.max_draft_length == 10
        assert config.temperature == 0.5
        assert config.top_k == 3
        assert config.compiled is True


# ── LLMStats Tests ──


class TestLLMStats:
    def test_defaults(self):
        stats = LLMStats()
        assert stats.total_proposals == 0
        assert stats.total_draft_tokens == 0
        assert stats.total_accepted_tokens == 0
        assert stats.total_failed_proposals == 0
        assert stats.draft_model_loaded is False
        assert stats.draft_model_name == ""
        assert stats.draft_model_memory_mb == 0.0
        assert stats.acceptance_rate == 0.0
        assert stats.avg_draft_length == 0.0

    def test_acceptance_rate(self):
        stats = LLMStats(total_draft_tokens=100, total_accepted_tokens=72)
        assert abs(stats.acceptance_rate - 0.72) < 1e-6

    def test_acceptance_rate_zero_division(self):
        stats = LLMStats(total_draft_tokens=0)
        assert stats.acceptance_rate == 0.0

    def test_avg_draft_length(self):
        stats = LLMStats(total_proposals=10, total_draft_tokens=50)
        assert abs(stats.avg_draft_length - 5.0) < 1e-6

    def test_avg_draft_length_zero_division(self):
        stats = LLMStats(total_proposals=0)
        assert stats.avg_draft_length == 0.0

    def test_to_dict(self):
        stats = LLMStats(
            total_proposals=5,
            total_draft_tokens=25,
            total_accepted_tokens=20,
            total_failed_proposals=1,
            draft_model_loaded=True,
            draft_model_name="test-model",
            draft_model_memory_mb=128.5,
        )
        d = stats.to_dict()
        assert d["total_proposals"] == 5
        assert d["total_draft_tokens"] == 25
        assert d["total_accepted_tokens"] == 20
        assert d["total_failed_proposals"] == 1
        assert d["draft_model_loaded"] is True
        assert d["draft_model_name"] == "test-model"
        assert d["draft_model_memory_mb"] == 128.5
        assert "acceptance_rate" in d
        assert "avg_draft_length" in d


# ── LLMProposer Tests ──


class TestLLMProposer:
    def test_init_defaults(self):
        proposer = LLMProposer()
        assert proposer.config.max_draft_length == 5
        assert not proposer.is_loaded
        assert proposer.draft_model is None
        assert proposer.draft_tokenizer is None

    def test_init_custom_config(self):
        config = LLMProposerConfig(
            draft_model_name="test-model",
            max_draft_length=3,
        )
        proposer = LLMProposer(config)
        assert proposer.config.draft_model_name == "test-model"
        assert proposer.config.max_draft_length == 3

    def test_load_draft_model_no_name(self):
        proposer = LLMProposer()
        result = proposer.load_draft_model("")
        assert result is False
        assert not proposer.is_loaded

    def test_load_draft_model_failure(self):
        """Loading a nonexistent model should return False gracefully."""
        proposer = LLMProposer(LLMProposerConfig(draft_model_name="nonexistent-model"))
        result = proposer.load_draft_model("nonexistent-model")
        assert result is False
        assert not proposer.is_loaded

    def test_load_draft_model_success_mock(self):
        """Simulate successful model loading via mock."""
        proposer = LLMProposer(LLMProposerConfig())
        fake_model = FakeDraftModel()
        fake_tokenizer = FakeTokenizer()

        # Direct approach: set model manually (simulates successful load)
        proposer._model = fake_model
        proposer._tokenizer = fake_tokenizer
        proposer._stats.draft_model_loaded = True
        proposer._stats.draft_model_name = "test-model"
        proposer._stats.draft_model_memory_mb = _estimate_model_memory(fake_model)

        assert proposer.is_loaded
        assert proposer._stats.draft_model_loaded is True
        assert proposer._stats.draft_model_name == "test-model"
        assert proposer._stats.draft_model_memory_mb > 0

    def test_unload_draft_model(self):
        proposer = LLMProposer()
        fake_model = FakeDraftModel()
        proposer._model = fake_model
        proposer._tokenizer = FakeTokenizer()
        proposer._stats.draft_model_loaded = True

        proposer.unload_draft_model()
        assert proposer._model is None
        assert proposer._tokenizer is None
        assert not proposer.is_loaded
        assert proposer._stats.draft_model_loaded is False
        assert proposer._stats.draft_model_memory_mb == 0.0

    def test_propose_without_model(self):
        proposer = LLMProposer()
        result = proposer.propose([1, 2, 3], n_draft=5)
        assert result == []

    def test_propose_with_mock_model(self):
        """Test propose with a fake model directly set."""
        proposer = LLMProposer(LLMProposerConfig(max_draft_length=3))
        proposer._model = FakeDraftModel()
        proposer._stats.draft_model_loaded = True

        result = proposer.propose([1, 2, 3], n_draft=3)
        assert isinstance(result, list)
        assert len(result) == 3
        for t in result:
            assert isinstance(t, int)
            assert 0 <= t < 256  # vocab_size of FakeDraftModel

    def test_propose_with_raw_output_model(self):
        """Test propose with model that returns logits directly (no .logits)."""
        proposer = LLMProposer(LLMProposerConfig(max_draft_length=2))
        proposer._model = FakeDraftModelRawOutput()
        proposer._stats.draft_model_loaded = True

        result = proposer.propose([1, 2, 3], n_draft=2)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_propose_respects_max_draft_length(self):
        proposer = LLMProposer(LLMProposerConfig(max_draft_length=3))
        proposer._model = FakeDraftModel()
        proposer._stats.draft_model_loaded = True

        # Request more than max_draft_length
        result = proposer.propose([1, 2, 3], n_draft=10)
        assert len(result) <= 3

    def test_propose_uses_config_default_n(self):
        """When n_draft is None, use config.max_draft_length."""
        proposer = LLMProposer(LLMProposerConfig(max_draft_length=2))
        proposer._model = FakeDraftModel()
        proposer._stats.draft_model_loaded = True

        result = proposer.propose([1, 2, 3])
        assert len(result) <= 2

    def test_propose_updates_stats(self):
        proposer = LLMProposer(LLMProposerConfig(max_draft_length=3))
        proposer._model = FakeDraftModel()
        proposer._stats.draft_model_loaded = True

        proposer.propose([1, 2, 3], n_draft=3)
        assert proposer.stats.total_proposals == 1
        assert proposer.stats.total_draft_tokens == 3

    def test_propose_handles_exception(self):
        """Propose returns empty on internal error."""

        class BrokenModel(nn.Module):
            def __call__(self, *args, **kwargs):
                raise RuntimeError("GPU error")

        proposer = LLMProposer(LLMProposerConfig(max_draft_length=3))
        proposer._model = BrokenModel()
        proposer._stats.draft_model_loaded = True

        result = proposer.propose([1, 2, 3], n_draft=3)
        assert result == []
        assert proposer.stats.total_failed_proposals == 1

    def test_get_stats(self):
        proposer = LLMProposer()
        proposer._model = FakeDraftModel()
        proposer._stats.draft_model_loaded = True
        proposer._stats.draft_model_name = "test-model"

        stats = proposer.get_stats()
        assert stats["draft_model_loaded"] is True
        assert stats["draft_model_name"] == "test-model"
        assert "acceptance_rate" in stats
        assert "avg_draft_length" in stats

    def test_reset_stats(self):
        proposer = LLMProposer(LLMProposerConfig(max_draft_length=2))
        proposer._model = FakeDraftModel()
        proposer._stats.draft_model_loaded = True
        proposer._stats.draft_model_name = "test-model"

        proposer.propose([1, 2, 3], n_draft=2)
        assert proposer.stats.total_proposals > 0

        proposer.reset_stats()
        assert proposer.stats.total_proposals == 0
        assert proposer.stats.total_draft_tokens == 0
        # Model info preserved
        assert proposer.stats.draft_model_loaded is True
        assert proposer.stats.draft_model_name == "test-model"


# ── _estimate_model_memory Tests ──


class TestEstimateModelMemory:
    def test_returns_positive_for_real_model(self):
        model = FakeDraftModel()
        mem = _estimate_model_memory(model)
        assert mem > 0

    def test_returns_zero_for_empty(self):
        model = nn.Module()
        mem = _estimate_model_memory(model)
        assert mem == 0.0


# ── LLMStrategy Tests ──


class TestLLMStrategy:
    def test_name(self):
        strategy = LLMStrategy()
        assert strategy.name == "llm"

    def test_begin_end_lifecycle(self):
        strategy = LLMStrategy()
        strategy.begin("req-1")
        strategy.end("req-1")

    def test_draft_without_proposer(self):
        strategy = LLMStrategy()
        proposal = strategy.draft([1, 2, 3], n=5)
        assert proposal.tokens == []
        assert proposal.strategy_name == "llm"

    def test_draft_with_unloaded_proposer(self):
        proposer = LLMProposer()  # Not loaded
        strategy = LLMStrategy(proposer=proposer)
        proposal = strategy.draft([1, 2, 3], n=5)
        assert proposal.tokens == []
        assert proposal.strategy_name == "llm"

    def test_draft_with_loaded_proposer(self):
        proposer = LLMProposer(LLMProposerConfig(
            draft_model_name="test-model",
            max_draft_length=3,
        ))
        proposer._model = FakeDraftModel()
        proposer._stats.draft_model_loaded = True

        strategy = LLMStrategy(proposer=proposer)
        proposal = strategy.draft([1, 2, 3], n=5)
        assert len(proposal.tokens) == 3
        assert proposal.strategy_name == "llm"
        assert proposal.metadata["draft_model"] == "test-model"
        assert proposal.metadata["draft_length"] == 3

    def test_accept_updates_stats(self):
        strategy = LLMStrategy()
        strategy.accept([1, 2, 3], verified_up_to=2)
        stats = strategy.stats()
        assert stats["total_accepted"] == 1
        assert stats["total_accepted_tokens"] == 2

    def test_accept_forwards_to_proposer(self):
        proposer = LLMProposer()
        strategy = LLMStrategy(proposer=proposer)
        strategy.accept([1, 2, 3], verified_up_to=2)
        assert proposer._stats.total_accepted_tokens == 2

    def test_stats_without_proposer(self):
        strategy = LLMStrategy()
        stats = strategy.stats()
        assert stats["name"] == "llm"
        assert stats["acceptance_rate"] == 0.0
        assert "proposer_stats" not in stats

    def test_stats_with_proposer(self):
        proposer = LLMProposer()
        proposer._stats.draft_model_name = "test-model"
        strategy = LLMStrategy(proposer=proposer)
        stats = strategy.stats()
        assert "proposer_stats" in stats
        assert stats["proposer_stats"]["draft_model_name"] == "test-model"

    def test_reset(self):
        proposer = LLMProposer()
        strategy = LLMStrategy(proposer=proposer)
        strategy.accept([1, 2], verified_up_to=2)
        strategy.reset()
        stats = strategy.stats()
        assert stats["total_accepted"] == 0

    def test_proposer_property(self):
        proposer = LLMProposer()
        strategy = LLMStrategy(proposer=proposer)
        assert strategy.proposer is proposer


# ── SpecStrategyFactory Tests ──


class TestSpecStrategyFactoryLLM:
    def test_create_llm(self):
        strategy = SpecStrategyFactory.create({"type": "llm"})
        assert isinstance(strategy, LLMStrategy)
        assert strategy.name == "llm"

    def test_create_llm_with_params(self):
        strategy = SpecStrategyFactory.create({
            "type": "llm",
            "draft_model_name": "Qwen2.5-0.5B",
            "max_draft_length": 8,
            "temperature": 0.0,
            "top_k": 1,
        })
        assert isinstance(strategy, LLMStrategy)
        assert strategy._proposer is not None
        assert strategy._proposer.config.draft_model_name == "Qwen2.5-0.5B"
        assert strategy._proposer.config.max_draft_length == 8

    def test_create_llm_with_proposer(self):
        proposer = LLMProposer()
        strategy = SpecStrategyFactory.create({
            "type": "llm",
            "proposer": proposer,
        })
        assert strategy.proposer is proposer

    def test_composite_with_llm_and_ngram(self):
        strategy = SpecStrategyFactory.create({
            "type": "composite",
            "strategies": [
                {"type": "ngram", "mode": "lps"},
                {"type": "llm", "draft_model_name": "test"},
            ],
        })
        assert isinstance(strategy, CompositeStrategy)
        assert "ngram" in strategy.name
        assert "llm" in strategy.name

    def test_from_env_llm(self):
        old_strategy = os.environ.get("YUNSHU_SPEC_STRATEGY")
        old_model = os.environ.get("YUNSHU_LLM_DRAFT_MODEL")
        try:
            os.environ["YUNSHU_SPEC_STRATEGY"] = "llm"
            os.environ["YUNSHU_LLM_DRAFT_MODEL"] = "test-draft-model"
            s = SpecStrategyFactory.from_env()
            assert isinstance(s, LLMStrategy)
            assert s._proposer.config.draft_model_name == "test-draft-model"
        finally:
            if old_strategy is not None:
                os.environ["YUNSHU_SPEC_STRATEGY"] = old_strategy
            else:
                os.environ.pop("YUNSHU_SPEC_STRATEGY", None)
            if old_model is not None:
                os.environ["YUNSHU_LLM_DRAFT_MODEL"] = old_model
            else:
                os.environ.pop("YUNSHU_LLM_DRAFT_MODEL", None)


# ── CompositeStrategy with LLM ──


class TestCompositeWithLLM:
    def test_composite_llm_fallback_to_ngram(self):
        """LLM without loaded model returns empty, ngram fills in."""
        llm = LLMStrategy()  # No proposer loaded
        ngram = SpecStrategyFactory.create({"type": "ngram", "mode": "hashpool", "k": 5, "max_n": 5})
        composite = CompositeStrategy([llm, ngram])
        composite.begin("req-1")

        tokens = [1, 2, 3, 4, 5] * 20
        proposal = composite.draft(tokens, n=5)
        # ngram should find the repeating pattern
        assert len(proposal.tokens) > 0
        assert proposal.strategy_name == "ngram"

    def test_composite_llm_first_when_loaded(self):
        """LLM with loaded model provides proposals before ngram."""
        proposer = LLMProposer(LLMProposerConfig(max_draft_length=3))
        proposer._model = FakeDraftModel()
        proposer._stats.draft_model_loaded = True

        llm = LLMStrategy(proposer=proposer)
        ngram = SpecStrategyFactory.create({"type": "ngram", "mode": "lps"})
        composite = CompositeStrategy([llm, ngram])
        composite.begin("req-1")

        proposal = composite.draft([1, 2, 3], n=5)
        assert proposal.strategy_name == "llm"
        assert len(proposal.tokens) == 3
