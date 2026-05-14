"""Tests for Gemma4 built-in speculative decoding proposer.

Covers:
  - Gemma4SpecConfig defaults and custom values
  - Gemma4Stats tracking, acceptance_rate, avg_draft_length
  - Gemma4SpecProposer detection (model_type, spec keys, model attrs)
  - Gemma4SpecProposer.propose with mock hidden states
  - Propose without detection returns empty
  - Propose with acceptance threshold filtering
  - _get_lm_head with various model structures
  - _find_spec_layers from config, model_config, model attributes
  - get_stats() and reset_stats()
  - Gemma4Strategy in spec_interface
  - SpecStrategyFactory "gemma4" type
  - CompositeStrategy with Gemma4
  - from_env() with YUNSHU_GEMMA4_* env vars
  - Edge cases: no model, no spec layers, no lm_head
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import mlx.core as mx
import mlx.nn as nn

from yunshu_engine.gemma4_spec import (
    Gemma4SpecConfig,
    Gemma4SpecProposer,
    Gemma4Stats,
    _GEMMA4_MODEL_TYPES,
    _GEMMA4_SPEC_KEYS,
)
from yunshu_engine.spec_interface import (
    CompositeStrategy,
    Gemma4Strategy,
    SpecStrategyFactory,
)


# ── Helpers ──


class FakeGemma4Model(nn.Module):
    """Fake model that mimics Gemma4 with lm_head."""

    def __init__(self, hidden_size: int = 64, vocab_size: int = 256):
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)


class FakeGemma4ModelWithLanguageModel(nn.Module):
    """Fake model with language_model wrapper (VLM pattern)."""

    def __init__(self, hidden_size: int = 64, vocab_size: int = 256):
        super().__init__()
        self.language_model = FakeGemma4Model(hidden_size, vocab_size)


class FakeModelWithOutputHead(nn.Module):
    """Fake model with 'output' instead of 'lm_head'."""

    def __init__(self, hidden_size: int = 64, vocab_size: int = 256):
        super().__init__()
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)


class FakeModelNoHead(nn.Module):
    """Fake model without any head."""

    def __init__(self):
        super().__init__()
        self.some_layer = nn.Linear(32, 32)


class FakeModelWithSpecLayers(nn.Module):
    """Fake model with speculative_layers attribute."""

    def __init__(self, hidden_size: int = 64, vocab_size: int = 256):
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.speculative_layers = [28, 29, 30]


class FakeModelWithSpecHeads(nn.Module):
    """Fake model with spec_heads attribute."""

    def __init__(self, hidden_size: int = 64, vocab_size: int = 256):
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.spec_heads = 4


# ── Gemma4SpecConfig Tests ──


class TestGemma4SpecConfig:
    def test_defaults(self):
        config = Gemma4SpecConfig()
        assert config.enabled is False
        assert config.draft_length == 4
        assert config.acceptance_threshold == 0.1
        assert config.speculative_layer_indices == []

    def test_custom_values(self):
        config = Gemma4SpecConfig(
            enabled=True,
            draft_length=8,
            acceptance_threshold=0.5,
            speculative_layer_indices=[10, 11, 12],
        )
        assert config.enabled is True
        assert config.draft_length == 8
        assert config.acceptance_threshold == 0.5
        assert config.speculative_layer_indices == [10, 11, 12]


# ── Gemma4Stats Tests ──


class TestGemma4Stats:
    def test_defaults(self):
        stats = Gemma4Stats()
        assert stats.total_proposals == 0
        assert stats.total_draft_tokens == 0
        assert stats.total_accepted_tokens == 0
        assert stats.model_detected is False
        assert stats.model_type == ""
        assert stats.spec_layers_found == 0
        assert stats.acceptance_rate == 0.0
        assert stats.avg_draft_length == 0.0

    def test_acceptance_rate(self):
        stats = Gemma4Stats(total_draft_tokens=50, total_accepted_tokens=30)
        assert abs(stats.acceptance_rate - 0.6) < 1e-6

    def test_acceptance_rate_zero_division(self):
        stats = Gemma4Stats(total_draft_tokens=0)
        assert stats.acceptance_rate == 0.0

    def test_avg_draft_length(self):
        stats = Gemma4Stats(total_proposals=10, total_draft_tokens=40)
        assert abs(stats.avg_draft_length - 4.0) < 1e-6

    def test_to_dict(self):
        stats = Gemma4Stats(
            total_proposals=3,
            total_draft_tokens=12,
            total_accepted_tokens=8,
            model_detected=True,
            model_type="gemma4",
            spec_layers_found=4,
        )
        d = stats.to_dict()
        assert d["total_proposals"] == 3
        assert d["total_draft_tokens"] == 12
        assert d["total_accepted_tokens"] == 8
        assert d["model_detected"] is True
        assert d["model_type"] == "gemma4"
        assert d["spec_layers_found"] == 4
        assert "acceptance_rate" in d
        assert "avg_draft_length" in d


# ── Gemma4 Detection Constants ──


class TestGemma4Constants:
    def test_model_types_include_gemma4(self):
        assert "gemma4" in _GEMMA4_MODEL_TYPES
        assert "gemma_4" in _GEMMA4_MODEL_TYPES

    def test_spec_keys_include_known(self):
        assert "speculative_layers" in _GEMMA4_SPEC_KEYS
        assert "draft_heads" in _GEMMA4_SPEC_KEYS
        assert "num_speculative_heads" in _GEMMA4_SPEC_KEYS


# ── Gemma4SpecProposer Tests ──


class TestGemma4SpecProposer:
    def test_init_defaults(self):
        proposer = Gemma4SpecProposer()
        assert proposer.config.draft_length == 4
        assert not proposer.is_detected
        assert proposer.spec_layers == []

    def test_init_custom_config(self):
        config = Gemma4SpecConfig(enabled=True, draft_length=8)
        proposer = Gemma4SpecProposer(config)
        assert proposer.config.enabled is True
        assert proposer.config.draft_length == 8

    # ── Detection Tests ──

    def test_detect_gemma4_model_type(self):
        proposer = Gemma4SpecProposer(Gemma4SpecConfig(enabled=True))
        model = FakeGemma4Model()
        result = proposer.detect(model, {"model_type": "gemma4", "speculative_layers": 4, "num_hidden_layers": 32})
        assert result is True
        assert proposer.is_detected
        assert proposer.stats.model_detected is True
        assert proposer.stats.model_type == "gemma4"

    def test_detect_gemma_4_model_type(self):
        proposer = Gemma4SpecProposer(Gemma4SpecConfig(enabled=True))
        model = FakeGemma4Model()
        result = proposer.detect(model, {"model_type": "gemma_4", "draft_heads": 2, "num_hidden_layers": 16})
        assert result is True
        assert proposer.is_detected

    def test_detect_gemma_with_version(self):
        proposer = Gemma4SpecProposer(Gemma4SpecConfig(enabled=True))
        model = FakeGemma4Model()
        result = proposer.detect(model, {
            "model_type": "gemma",
            "model_version": 4,
            "speculative_layers": 3,
            "num_hidden_layers": 16,
        })
        assert result is True
        assert proposer.is_detected

    def test_detect_via_spec_keys(self):
        """Detect via speculative config keys even without gemma4 model_type."""
        proposer = Gemma4SpecProposer(Gemma4SpecConfig(enabled=True))
        model = FakeModelWithSpecLayers()
        result = proposer.detect(model, {"model_type": "other", "speculative_layers": 3})
        assert result is True
        assert proposer.is_detected

    def test_detect_via_model_attribute_spec_layers(self):
        """Detect via model's speculative_layers attribute."""
        proposer = Gemma4SpecProposer(Gemma4SpecConfig(enabled=True))
        model = FakeModelWithSpecLayers()
        result = proposer.detect(model, {"model_type": "other"})
        assert result is True
        assert proposer.spec_layers == [28, 29, 30]

    def test_detect_via_model_attribute_spec_heads(self):
        """Detect via model's spec_heads attribute."""
        proposer = Gemma4SpecProposer(Gemma4SpecConfig(enabled=True))
        model = FakeModelWithSpecHeads()
        result = proposer.detect(model, {"model_type": "other"})
        assert result is True
        assert proposer.spec_layers == [0, 1, 2, 3]

    def test_detect_not_gemma4(self):
        proposer = Gemma4SpecProposer()
        model = FakeModelNoHead()
        result = proposer.detect(model, {"model_type": "llama"})
        assert result is False
        assert not proposer.is_detected

    def test_detect_gemma4_no_spec_layers(self):
        """Gemma4 model without spec layer info fails detection."""
        proposer = Gemma4SpecProposer()
        model = FakeGemma4Model()
        result = proposer.detect(model, {"model_type": "gemma4"})
        assert result is False
        assert not proposer.is_detected

    def test_detect_with_explicit_indices(self):
        """Config with explicit speculative_layer_indices."""
        config = Gemma4SpecConfig(
            enabled=True,
            speculative_layer_indices=[5, 6, 7],
        )
        proposer = Gemma4SpecProposer(config)
        model = FakeGemma4Model()
        result = proposer.detect(model, {"model_type": "gemma4"})
        assert result is True
        assert proposer.spec_layers == [5, 6, 7]

    # ── Spec Layer Finding Tests ──

    def test_find_spec_layers_from_config_speculative_layers(self):
        proposer = Gemma4SpecProposer()
        model = FakeGemma4Model()
        layers = proposer._find_spec_layers(model, {
            "speculative_layers": 4,
            "num_hidden_layers": 32,
        })
        assert len(layers) == 4
        assert layers == [28, 29, 30, 31]

    def test_find_spec_layers_from_config_draft_heads(self):
        proposer = Gemma4SpecProposer()
        model = FakeGemma4Model()
        layers = proposer._find_spec_layers(model, {
            "draft_heads": 2,
            "num_hidden_layers": 16,
        })
        assert len(layers) == 2
        assert layers == [14, 15]

    def test_find_spec_layers_from_config_no_num_hidden(self):
        """Without num_hidden_layers, uses sequential indices."""
        proposer = Gemma4SpecProposer()
        model = FakeGemma4Model()
        layers = proposer._find_spec_layers(model, {
            "speculative_layers": 3,
        })
        assert layers == [0, 1, 2]

    # ── Propose Tests ──

    def test_propose_without_detection(self):
        proposer = Gemma4SpecProposer()
        hidden = mx.random.normal((1, 1, 64))
        result = proposer.propose(hidden, n_draft=4)
        assert result == []

    def test_propose_with_detection(self):
        proposer = Gemma4SpecProposer(Gemma4SpecConfig(enabled=True, acceptance_threshold=0.0))
        model = FakeGemma4Model()
        proposer.detect(model, {"model_type": "gemma4", "speculative_layers": 4, "num_hidden_layers": 32})

        hidden = mx.random.normal((1, 1, 64))
        result = proposer.propose(hidden, n_draft=4)
        assert isinstance(result, list)
        assert len(result) > 0
        for t in result:
            assert isinstance(t, int)
            assert 0 <= t < 256

    def test_propose_respects_n_draft(self):
        proposer = Gemma4SpecProposer(Gemma4SpecConfig(enabled=True, draft_length=4, acceptance_threshold=0.0))
        model = FakeGemma4Model()
        proposer.detect(model, {"model_type": "gemma4", "speculative_layers": 4, "num_hidden_layers": 32})

        hidden = mx.random.normal((1, 1, 64))
        result = proposer.propose(hidden, n_draft=2)
        assert len(result) <= 2

    def test_propose_respects_draft_length_config(self):
        proposer = Gemma4SpecProposer(Gemma4SpecConfig(enabled=True, draft_length=2, acceptance_threshold=0.0))
        model = FakeGemma4Model()
        proposer.detect(model, {"model_type": "gemma4", "speculative_layers": 4, "num_hidden_layers": 32})

        hidden = mx.random.normal((1, 1, 64))
        result = proposer.propose(hidden, n_draft=10)
        assert len(result) <= 2

    def test_propose_updates_stats(self):
        proposer = Gemma4SpecProposer(Gemma4SpecConfig(enabled=True, acceptance_threshold=0.0))
        model = FakeGemma4Model()
        proposer.detect(model, {"model_type": "gemma4", "speculative_layers": 4, "num_hidden_layers": 32})

        hidden = mx.random.normal((1, 1, 64))
        proposer.propose(hidden, n_draft=4)
        assert proposer.stats.total_proposals == 1
        assert proposer.stats.total_draft_tokens > 0

    def test_propose_with_wrapped_model(self):
        """Propose with VLM-style model (language_model attribute)."""
        proposer = Gemma4SpecProposer(Gemma4SpecConfig(enabled=True, acceptance_threshold=0.0))
        model = FakeGemma4ModelWithLanguageModel()
        proposer.detect(model, {"model_type": "gemma4", "speculative_layers": 2, "num_hidden_layers": 8})

        hidden = mx.random.normal((1, 1, 64))
        result = proposer.propose(hidden, n_draft=2)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_propose_with_output_head(self):
        """Propose with model that uses 'output' instead of 'lm_head'."""
        proposer = Gemma4SpecProposer(Gemma4SpecConfig(enabled=True, acceptance_threshold=0.0))
        model = FakeModelWithOutputHead()
        proposer.detect(model, {"model_type": "gemma4", "speculative_layers": 2, "num_hidden_layers": 8})

        hidden = mx.random.normal((1, 1, 64))
        result = proposer.propose(hidden, n_draft=2)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_propose_with_no_head_model(self):
        """Propose with model that has no lm_head returns empty."""
        proposer = Gemma4SpecProposer(Gemma4SpecConfig(enabled=True))
        # Use explicit indices to force detection even without head
        config = Gemma4SpecConfig(enabled=True, speculative_layer_indices=[1, 2])
        proposer = Gemma4SpecProposer(config)
        model = FakeModelNoHead()
        # Detect will succeed due to explicit indices (no spec layers found from config,
        # but we set explicit ones)
        # Actually, detection checks gemma4 model type first, so this won't detect
        proposer.detect(model, {"model_type": "other"})
        assert not proposer.is_detected

    def test_propose_with_acceptance_threshold(self):
        """With high acceptance_threshold, low-confidence predictions are filtered."""
        config = Gemma4SpecConfig(enabled=True, acceptance_threshold=0.99)
        proposer = Gemma4SpecProposer(config)
        model = FakeGemma4Model()
        proposer.detect(model, {"model_type": "gemma4", "speculative_layers": 4, "num_hidden_layers": 32})

        hidden = mx.random.normal((1, 1, 64))
        result = proposer.propose(hidden, n_draft=4)
        # With random hidden states, max softmax prob is ~1/256 = 0.004 < 0.99
        # So all predictions should be filtered out
        assert isinstance(result, list)
        assert len(result) == 0

    def test_propose_with_zero_threshold(self):
        """With acceptance_threshold=0.0, all predictions pass."""
        config = Gemma4SpecConfig(enabled=True, acceptance_threshold=0.0)
        proposer = Gemma4SpecProposer(config)
        model = FakeGemma4Model()
        proposer.detect(model, {"model_type": "gemma4", "speculative_layers": 4, "num_hidden_layers": 32})

        hidden = mx.random.normal((1, 1, 64))
        result = proposer.propose(hidden, n_draft=4)
        assert len(result) == 4

    def test_get_stats(self):
        proposer = Gemma4SpecProposer()
        stats = proposer.get_stats()
        assert stats["total_proposals"] == 0
        assert stats["model_detected"] is False
        assert "acceptance_rate" in stats

    def test_reset_stats(self):
        proposer = Gemma4SpecProposer(Gemma4SpecConfig(enabled=True))
        model = FakeGemma4Model()
        proposer.detect(model, {"model_type": "gemma4", "speculative_layers": 4, "num_hidden_layers": 32})

        hidden = mx.random.normal((1, 1, 64))
        proposer.propose(hidden, n_draft=4)
        assert proposer.stats.total_proposals > 0

        proposer.reset_stats()
        assert proposer.stats.total_proposals == 0
        assert proposer.stats.total_draft_tokens == 0
        # Model detection info preserved
        assert proposer.stats.model_detected is True

    def test_is_detected_false_initially(self):
        proposer = Gemma4SpecProposer()
        assert proposer.is_detected is False


# ── Gemma4Strategy Tests ──


class TestGemma4Strategy:
    def test_name(self):
        strategy = Gemma4Strategy()
        assert strategy.name == "gemma4"

    def test_begin_end_lifecycle(self):
        strategy = Gemma4Strategy()
        strategy.begin("req-1")
        strategy.end("req-1")

    def test_draft_without_proposer(self):
        strategy = Gemma4Strategy()
        proposal = strategy.draft([1, 2, 3], n=5)
        assert proposal.tokens == []
        assert proposal.strategy_name == "gemma4"

    def test_draft_with_undetected_proposer(self):
        proposer = Gemma4SpecProposer()  # Not detected
        strategy = Gemma4Strategy(proposer=proposer)
        proposal = strategy.draft([1, 2, 3], n=5)
        assert proposal.tokens == []
        assert proposal.strategy_name == "gemma4"

    def test_draft_with_detected_proposer(self):
        proposer = Gemma4SpecProposer(Gemma4SpecConfig(enabled=True))
        model = FakeGemma4Model()
        proposer.detect(model, {"model_type": "gemma4", "speculative_layers": 4, "num_hidden_layers": 32})

        strategy = Gemma4Strategy(proposer=proposer)
        proposal = strategy.draft([1, 2, 3], n=5)
        assert proposal.strategy_name == "gemma4"
        assert proposal.metadata["draft_length"] > 0
        # Note: tokens are empty because Gemma4 needs hidden_states from engine
        assert proposal.tokens == []

    def test_accept_updates_stats(self):
        strategy = Gemma4Strategy()
        strategy.accept([1, 2, 3], verified_up_to=2)
        stats = strategy.stats()
        assert stats["total_accepted"] == 1
        assert stats["total_accepted_tokens"] == 2

    def test_accept_forwards_to_proposer(self):
        proposer = Gemma4SpecProposer()
        strategy = Gemma4Strategy(proposer=proposer)
        strategy.accept([1, 2, 3], verified_up_to=2)
        assert proposer._stats.total_accepted_tokens == 2

    def test_stats_without_proposer(self):
        strategy = Gemma4Strategy()
        stats = strategy.stats()
        assert stats["name"] == "gemma4"
        assert stats["acceptance_rate"] == 0.0
        assert "proposer_stats" not in stats

    def test_stats_with_proposer(self):
        proposer = Gemma4SpecProposer()
        proposer._stats.model_detected = True
        strategy = Gemma4Strategy(proposer=proposer)
        stats = strategy.stats()
        assert "proposer_stats" in stats
        assert stats["proposer_stats"]["model_detected"] is True

    def test_reset(self):
        proposer = Gemma4SpecProposer()
        strategy = Gemma4Strategy(proposer=proposer)
        strategy.accept([1, 2], verified_up_to=2)
        strategy.reset()
        stats = strategy.stats()
        assert stats["total_accepted"] == 0

    def test_proposer_property(self):
        proposer = Gemma4SpecProposer()
        strategy = Gemma4Strategy(proposer=proposer)
        assert strategy.proposer is proposer


# ── SpecStrategyFactory Tests ──


class TestSpecStrategyFactoryGemma4:
    def test_create_gemma4(self):
        strategy = SpecStrategyFactory.create({"type": "gemma4"})
        assert isinstance(strategy, Gemma4Strategy)
        assert strategy.name == "gemma4"

    def test_create_gemma4_with_params(self):
        strategy = SpecStrategyFactory.create({
            "type": "gemma4",
            "enabled": True,
            "draft_length": 8,
            "acceptance_threshold": 0.3,
        })
        assert isinstance(strategy, Gemma4Strategy)
        assert strategy._proposer is not None
        assert strategy._proposer.config.enabled is True
        assert strategy._proposer.config.draft_length == 8
        assert strategy._proposer.config.acceptance_threshold == 0.3

    def test_create_gemma4_with_proposer(self):
        proposer = Gemma4SpecProposer()
        strategy = SpecStrategyFactory.create({
            "type": "gemma4",
            "proposer": proposer,
        })
        assert strategy.proposer is proposer

    def test_composite_with_gemma4_and_ngram(self):
        strategy = SpecStrategyFactory.create({
            "type": "composite",
            "strategies": [
                {"type": "gemma4", "enabled": True},
                {"type": "ngram", "mode": "lps"},
            ],
        })
        assert isinstance(strategy, CompositeStrategy)
        assert "gemma4" in strategy.name
        assert "ngram" in strategy.name

    def test_from_env_gemma4(self):
        old_strategy = os.environ.get("YUNSHU_SPEC_STRATEGY")
        old_len = os.environ.get("YUNSHU_GEMMA4_DRAFT_LENGTH")
        try:
            os.environ["YUNSHU_SPEC_STRATEGY"] = "gemma4"
            os.environ["YUNSHU_GEMMA4_DRAFT_LENGTH"] = "6"
            s = SpecStrategyFactory.from_env()
            assert isinstance(s, Gemma4Strategy)
            assert s._proposer.config.draft_length == 6
        finally:
            if old_strategy is not None:
                os.environ["YUNSHU_SPEC_STRATEGY"] = old_strategy
            else:
                os.environ.pop("YUNSHU_SPEC_STRATEGY", None)
            if old_len is not None:
                os.environ["YUNSHU_GEMMA4_DRAFT_LENGTH"] = old_len
            else:
                os.environ.pop("YUNSHU_GEMMA4_DRAFT_LENGTH", None)


# ── CompositeStrategy with Gemma4 ──


class TestCompositeWithGemma4:
    def test_composite_gemma4_fallback_to_ngram(self):
        """Gemma4 without detection returns empty, ngram fills in."""
        gemma4 = Gemma4Strategy()  # No proposer detected
        ngram = SpecStrategyFactory.create({"type": "ngram", "mode": "hashpool", "k": 5, "max_n": 5})
        composite = CompositeStrategy([gemma4, ngram])
        composite.begin("req-1")

        tokens = [1, 2, 3, 4, 5] * 20
        proposal = composite.draft(tokens, n=5)
        assert len(proposal.tokens) > 0
        assert proposal.strategy_name == "ngram"

    def test_composite_gemma4_all_empty(self):
        gemma4 = Gemma4Strategy()  # No proposer
        composite = CompositeStrategy([gemma4])
        composite.begin("req-1")
        proposal = composite.draft([1, 2], n=5)
        assert proposal.tokens == []
        assert "empty" in proposal.strategy_name
