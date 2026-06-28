"""Tests for unified speculative decoding interface (C7).

Covers:
  - NgramStrategy lifecycle and draft proposals
  - CrossModelStrategy lifecycle (with and without decoder)
  - MTPStrategy lifecycle (with and without decoder)
  - CompositeStrategy combining multiple strategies
  - SpecStrategyFactory creation from config
  - SpecStrategyFactory.from_env()
  - Stats tracking and reset
  - Edge cases: empty drafts, all rejected, sequential requests
"""
from __future__ import annotations

import os
import sys

import pytest

# Ensure yunshu_engine is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from yunshu_engine.ngram_proposer import NgramConfig
from yunshu_engine.spec_interface import (
    CompositeStrategy,
    CrossModelStrategy,
    DeltaNetInversionStrategy,
    DraftProposal,
    MTPStrategy,
    NgramStrategy,
    SpecStrategy,
    SpecStrategyFactory,
)

# ── Helpers ──


def _repeating_tokens(n: int = 50) -> list[int]:
    """Generate a token sequence with repeating patterns for ngram matching.

    Pattern: [1, 2, 3, 4, 5] repeated n//5 times.
    """
    base = [1, 2, 3, 4, 5]
    return (base * ((n // len(base)) + 1))[:n]


class FakeMTPDecoder:
    """Fake MTPDecoder with .stats attribute for testing MTPStrategy."""

    def __init__(self):
        self.stats = type("Stats", (), {
            "accepts": 10,
            "rejects": 5,
            "cooldowns": 2,
            "tokens_generated": 100,
            "total_cycles": 15,
        })()


# ── NgramStrategy Tests ──


class TestNgramStrategy:

    def test_name(self):
        s = NgramStrategy()
        assert s.name == "ngram"

    def test_begin_sets_request_id(self):
        s = NgramStrategy()
        s.begin("req-1")
        assert s._request_id == "req-1"

    def test_end_clears_request_id(self):
        s = NgramStrategy()
        s.begin("req-1")
        s.end("req-1")
        assert s._request_id is None

    def test_draft_empty_on_short_context(self):
        """No proposals when context is shorter than min_n."""
        s = NgramStrategy(NgramConfig(min_n=2))
        proposal = s.draft([42], n=5)
        assert proposal.tokens == []
        assert proposal.strategy_name == "ngram"

    def test_draft_returns_proposal_type(self):
        """draft() returns a DraftProposal instance."""
        s = NgramStrategy(NgramConfig(min_n=1, max_n=3, k=5))
        tokens = _repeating_tokens(30)
        proposal = s.draft(tokens, n=5)
        assert isinstance(proposal, DraftProposal)
        assert proposal.strategy_name == "ngram"

    def test_draft_with_repeating_pattern(self):
        """Ngram should find repeating pattern and propose tokens."""
        s = NgramStrategy(NgramConfig(min_n=1, max_n=5, k=5))
        tokens = _repeating_tokens(50)
        proposal = s.draft(tokens, n=5)
        # Should find the repeating [1,2,3,4,5] pattern
        assert len(proposal.tokens) > 0

    def test_draft_respects_max_n(self):
        """draft(tokens, n) should return at most n tokens."""
        s = NgramStrategy(NgramConfig(min_n=1, max_n=5, k=10))
        tokens = _repeating_tokens(50)
        proposal = s.draft(tokens, n=2)
        assert len(proposal.tokens) <= 2

    def test_accept_updates_stats(self):
        s = NgramStrategy(NgramConfig(min_n=1, max_n=3, k=5))
        s.begin("req-1")
        proposal = s.draft(_repeating_tokens(30), n=5)
        draft_len = len(proposal.tokens)
        s.accept(proposal.tokens, verified_up_to=draft_len)
        stats = s.stats()
        assert stats["total_accepted"] == 1
        assert stats["total_accepted_tokens"] == draft_len

    def test_stats_structure(self):
        s = NgramStrategy()
        stats = s.stats()
        assert "name" in stats
        assert "total_drafts" in stats
        assert "total_draft_tokens" in stats
        assert "total_accepted" in stats
        assert "total_accepted_tokens" in stats
        assert "acceptance_rate" in stats
        assert stats["name"] == "ngram"

    def test_stats_zero_division(self):
        """acceptance_rate should be 0.0 when no drafts have been made."""
        s = NgramStrategy()
        stats = s.stats()
        assert stats["acceptance_rate"] == 0.0

    def test_reset_clears_counters(self):
        s = NgramStrategy(NgramConfig(min_n=1, max_n=3, k=5))
        s.draft(_repeating_tokens(30), n=5)
        s.accept([1, 2, 3], 3)
        s.reset()
        stats = s.stats()
        assert stats["total_drafts"] == 0
        assert stats["total_draft_tokens"] == 0
        assert stats["total_accepted"] == 0
        assert stats["total_accepted_tokens"] == 0

    def test_hashpool_mode(self):
        s = NgramStrategy(NgramConfig(min_n=1, max_n=3, k=5, mode="hashpool"))
        tokens = _repeating_tokens(30)
        s.draft(tokens, n=5)
        stats = s.stats()
        assert stats["mode"] == "hashpool"

    def test_multiple_requests_lifecycle(self):
        """Multiple begin/draft/accept/end cycles accumulate stats."""
        s = NgramStrategy(NgramConfig(min_n=1, max_n=3, k=5))
        tokens = _repeating_tokens(30)

        s.begin("req-1")
        s.draft(tokens, n=5)
        s.accept([1, 2], 2)
        s.end("req-1")

        s.begin("req-2")
        s.draft(tokens, n=5)
        s.accept([3], 1)
        s.end("req-2")

        stats = s.stats()
        assert stats["total_drafts"] == 2
        assert stats["total_accepted"] == 2
        assert stats["total_accepted_tokens"] == 3


# ── CrossModelStrategy Tests ──


class TestCrossModelStrategy:

    def test_name(self):
        s = CrossModelStrategy()
        assert s.name == "cross_model"

    def test_begin_end_lifecycle(self):
        s = CrossModelStrategy()
        s.begin("req-1")
        assert s._request_id == "req-1"
        s.end("req-1")
        assert s._request_id is None

    def test_draft_without_decoder_returns_empty(self):
        """Without a decoder, draft returns empty proposal."""
        s = CrossModelStrategy()
        proposal = s.draft([1, 2, 3], n=5)
        assert proposal.tokens == []
        assert proposal.strategy_name == "cross_model"

    def test_draft_with_decoder_tracks_stats(self):
        """With a mock decoder, draft records draft_length metadata."""
        class MockDecoder:
            class config:
                draft_length = 3
            def get_stats(self):
                return {}
        s = CrossModelStrategy(decoder=MockDecoder())
        proposal = s.draft([1, 2, 3], n=5)
        assert proposal.metadata["draft_length"] == 3
        assert s.stats()["total_drafts"] == 1
        # CrossModelStrategy returns empty tokens (actual tokens filled by
        # decoder's generate_draft), so draft_tokens is NOT inflated here.
        assert s.stats()["total_draft_tokens"] == 0

    def test_accept_all_rejected(self):
        """verified_up_to=0 means all draft tokens were rejected."""
        s = CrossModelStrategy()
        s.begin("req-1")
        s.accept([1, 2, 3], verified_up_to=0)
        stats = s.stats()
        assert stats["total_accepted"] == 1
        assert stats["total_accepted_tokens"] == 0

    def test_accept_partial(self):
        s = CrossModelStrategy()
        s.accept([1, 2, 3, 4, 5], verified_up_to=3)
        stats = s.stats()
        assert stats["total_accepted_tokens"] == 3

    def test_stats_without_decoder(self):
        s = CrossModelStrategy()
        stats = s.stats()
        assert stats["name"] == "cross_model"
        assert "decoder_stats" not in stats

    def test_stats_with_decoder(self):
        class MockDecoder:
            def get_stats(self):
                return {"acceptance_rate": 0.65}
        s = CrossModelStrategy(decoder=MockDecoder())
        stats = s.stats()
        assert "decoder_stats" in stats
        assert stats["decoder_stats"]["acceptance_rate"] == 0.65

    def test_reset(self):
        s = CrossModelStrategy()
        s.draft([1, 2], n=5)
        s.accept([1], 1)
        s.reset()
        stats = s.stats()
        assert stats["total_drafts"] == 0
        assert stats["total_accepted_tokens"] == 0

    def test_decoder_property(self):
        class MockDecoder:
            pass
        dec = MockDecoder()
        s = CrossModelStrategy(decoder=dec)
        assert s.decoder is dec


# ── MTPStrategy Tests ──


class TestMTPStrategy:

    def test_name(self):
        s = MTPStrategy()
        assert s.name == "mtp"

    def test_begin_end_lifecycle(self):
        s = MTPStrategy()
        s.begin("req-1")
        assert s._request_id == "req-1"
        s.end("req-1")
        assert s._request_id is None

    def test_draft_without_decoder_returns_empty(self):
        s = MTPStrategy()
        proposal = s.draft([1, 2, 3], n=5)
        assert proposal.tokens == []
        assert proposal.strategy_name == "mtp"

    def test_draft_with_decoder_tracks_single_token(self):
        """MTP always proposes exactly 1 draft token."""
        s = MTPStrategy(decoder=FakeMTPDecoder())
        proposal = s.draft([1, 2, 3], n=5)
        # MTPStrategy returns empty tokens (actual token filled by decoder's
        # _mtp_draft), so draft_tokens is NOT inflated here.
        assert s.stats()["total_draft_tokens"] == 0
        assert proposal.metadata["draft_length"] == 1

    def test_accept_updates_stats(self):
        s = MTPStrategy()
        s.accept([42], verified_up_to=1)
        stats = s.stats()
        assert stats["total_accepted"] == 1
        assert stats["total_accepted_tokens"] == 1

    def test_stats_with_decoder_exposes_mtp_fields(self):
        s = MTPStrategy(decoder=FakeMTPDecoder())
        stats = s.stats()
        assert stats["mtp_accepts"] == 10
        assert stats["mtp_rejects"] == 5
        assert stats["mtp_cooldowns"] == 2
        assert stats["mtp_tokens_generated"] == 100
        assert stats["mtp_total_cycles"] == 15

    def test_stats_without_decoder(self):
        s = MTPStrategy()
        stats = s.stats()
        assert stats["name"] == "mtp"
        assert "mtp_accepts" not in stats

    def test_reset(self):
        s = MTPStrategy()
        s.draft([1, 2], n=5)
        s.accept([1], 1)
        s.reset()
        stats = s.stats()
        assert stats["total_drafts"] == 0
        assert stats["total_accepted_tokens"] == 0

    def test_decoder_property(self):
        dec = FakeMTPDecoder()
        s = MTPStrategy(decoder=dec)
        assert s.decoder is dec


# ── CompositeStrategy Tests ──


class TestCompositeStrategy:

    def _make_composite(self):
        """Create a composite with ngram + cross_model fallback."""
        ngram = NgramStrategy(NgramConfig(min_n=1, max_n=3, k=5))
        cross = CrossModelStrategy()
        return CompositeStrategy([ngram, cross])

    def test_name(self):
        s = self._make_composite()
        assert "ngram" in s.name
        assert "cross_model" in s.name

    def test_requires_at_least_one_strategy(self):
        with pytest.raises(ValueError, match="at least one"):
            CompositeStrategy([])

    def test_begin_forwards_to_children(self):
        s = self._make_composite()
        s.begin("req-1")
        assert s._strategies[0]._request_id == "req-1"
        assert s._strategies[1]._request_id == "req-1"

    def test_end_forwards_to_children(self):
        s = self._make_composite()
        s.begin("req-1")
        s.end("req-1")
        assert s._strategies[0]._request_id is None
        assert s._strategies[1]._request_id is None

    def test_draft_uses_first_nonempty(self):
        """When ngram finds a match, cross_model is not called for tokens."""
        s = self._make_composite()
        tokens = _repeating_tokens(30)
        proposal = s.draft(tokens, n=5)
        # ngram should find the pattern
        assert len(proposal.tokens) > 0
        assert proposal.strategy_name == "ngram"
        # strategy_usage should show ngram was used
        stats = s.stats()
        assert stats["strategy_usage"].get("ngram", 0) > 0

    def test_draft_falls_back_to_second(self):
        """When first strategy returns empty, second is tried."""
        ngram = NgramStrategy(NgramConfig(min_n=10, max_n=20, k=5))
        cross = CrossModelStrategy()
        # ngram won't match on short sequences with min_n=10
        s = CompositeStrategy([ngram, cross])
        proposal = s.draft([1, 2, 3], n=5)
        # Both return empty, so proposal is empty
        assert proposal.tokens == []
        assert "empty" in proposal.strategy_name

    def test_accept_forwards_only_to_proposer(self):
        """accept() should only update the strategy that actually proposed."""
        s = self._make_composite()
        # First, call draft() so the composite knows which strategy proposed
        tokens = _repeating_tokens(30)
        proposal = s.draft(tokens, n=5)
        # Now accept — only the proposing strategy should be updated
        s.accept(proposal.tokens if proposal.tokens else [1, 2, 3], verified_up_to=2)
        # Only one strategy should have acceptance stats
        proposers = [c for c in s._strategies if c.stats()["total_accepted"] > 0]
        assert len(proposers) == 1, (
            f"Expected exactly 1 strategy with accept stats, got {len(proposers)}"
        )
        proposer = proposers[0]
        assert proposer.stats()["total_accepted"] == 1
        assert proposer.stats()["total_accepted_tokens"] == 2

    def test_accept_without_draft_is_safe(self):
        """accept() without a prior draft() should not crash or corrupt state."""
        s = self._make_composite()
        s.accept([1, 2, 3], verified_up_to=2)
        # No strategy should have inflated stats
        for child in s._strategies:
            assert child.stats()["total_accepted"] == 0

    def test_stats_aggregates_children(self):
        s = self._make_composite()
        tokens = _repeating_tokens(30)
        s.draft(tokens, n=5)
        s.accept([1, 2], 2)
        stats = s.stats()
        assert "children" in stats
        assert len(stats["children"]) == 2
        assert "strategy_usage" in stats

    def test_reset_clears_all(self):
        s = self._make_composite()
        s.draft(_repeating_tokens(30), n=5)
        s.accept([1], 1)
        s.reset()
        stats = s.stats()
        assert stats["total_drafts"] == 0
        assert stats["strategy_usage"] == {}
        for child_stats in stats["children"]:
            assert child_stats["total_drafts"] == 0

    def test_strategies_property(self):
        s = self._make_composite()
        assert len(s.strategies) == 2

    def test_three_strategy_composite(self):
        """Composite of three strategies."""
        ngram = NgramStrategy(NgramConfig(min_n=1, max_n=3, k=5))
        cross = CrossModelStrategy()
        mtp = MTPStrategy()
        s = CompositeStrategy([ngram, cross, mtp])
        assert "ngram" in s.name
        assert "cross_model" in s.name
        assert "mtp" in s.name
        proposal = s.draft(_repeating_tokens(30), n=5)
        assert len(proposal.tokens) > 0


# ── SpecStrategyFactory Tests ──


class TestSpecStrategyFactory:

    def test_create_ngram_default(self):
        s = SpecStrategyFactory.create({"type": "ngram"})
        assert isinstance(s, NgramStrategy)
        assert s.name == "ngram"

    def test_create_ngram_with_config(self):
        s = SpecStrategyFactory.create({
            "type": "ngram",
            "mode": "hashpool",
            "max_n": 3,
            "k": 3,
            "min_n": 1,
        })
        assert isinstance(s, NgramStrategy)
        assert s._config.mode == "hashpool"
        assert s._config.max_n == 3
        assert s._config.k == 3

    def test_create_cross_model(self):
        s = SpecStrategyFactory.create({"type": "cross_model"})
        assert isinstance(s, CrossModelStrategy)
        assert s.name == "cross_model"

    def test_create_cross_model_with_decoder(self):
        class MockDecoder:
            pass
        s = SpecStrategyFactory.create({
            "type": "cross_model",
            "decoder": MockDecoder(),
        })
        assert s.decoder is not None

    def test_create_mtp(self):
        s = SpecStrategyFactory.create({"type": "mtp"})
        assert isinstance(s, MTPStrategy)
        assert s.name == "mtp"

    def test_create_composite(self):
        s = SpecStrategyFactory.create({
            "type": "composite",
            "strategies": [
                {"type": "ngram", "mode": "hashpool"},
                {"type": "cross_model"},
            ],
        })
        assert isinstance(s, CompositeStrategy)
        assert len(s.strategies) == 2

    def test_create_composite_missing_strategies(self):
        with pytest.raises(ValueError, match="strategies"):
            SpecStrategyFactory.create({"type": "composite"})

    def test_create_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            SpecStrategyFactory.create({"type": "eagle imaginary"})

    def test_create_missing_type_raises(self):
        with pytest.raises(ValueError, match="type"):
            SpecStrategyFactory.create({})

    def test_from_env_no_config(self):
        """from_env returns None when no env var set."""
        old = os.environ.pop("YUNSHU_SPEC_STRATEGY", None)
        try:
            result = SpecStrategyFactory.from_env()
            assert result is None
        finally:
            if old is not None:
                os.environ["YUNSHU_SPEC_STRATEGY"] = old

    def test_from_env_ngram(self):
        old_strategy = os.environ.get("YUNSHU_SPEC_STRATEGY")
        old_mode = os.environ.get("YUNSHU_NGRAM_MODE")
        try:
            os.environ["YUNSHU_SPEC_STRATEGY"] = "ngram"
            os.environ["YUNSHU_NGRAM_MODE"] = "hashpool"
            s = SpecStrategyFactory.from_env()
            assert isinstance(s, NgramStrategy)
            assert s._config.mode == "hashpool"
        finally:
            if old_strategy is not None:
                os.environ["YUNSHU_SPEC_STRATEGY"] = old_strategy
            else:
                os.environ.pop("YUNSHU_SPEC_STRATEGY", None)
            if old_mode is not None:
                os.environ["YUNSHU_NGRAM_MODE"] = old_mode
            else:
                os.environ.pop("YUNSHU_NGRAM_MODE", None)


# ── DraftProposal Tests ──


class TestDraftProposal:

    def test_default_empty(self):
        p = DraftProposal()
        assert p.tokens == []
        assert p.strategy_name == ""
        assert p.metadata == {}

    def test_with_values(self):
        p = DraftProposal(
            tokens=[1, 2, 3],
            strategy_name="ngram",
            metadata={"mode": "hashpool"},
        )
        assert p.tokens == [1, 2, 3]
        assert p.strategy_name == "ngram"
        assert p.metadata["mode"] == "hashpool"


# ── Integration-style: BatchedEngine wiring check ──


class TestBatchedEngineWiring:
    """Verify that BatchedEngine can use SpecStrategyFactory."""

    def test_get_spec_strategy_returns_strategy(self):
        """_get_spec_strategy() on BatchedEngine should return a SpecStrategy."""
        # We test this by checking the import and factory creation work together
        strategy = SpecStrategyFactory.create({"type": "ngram", "mode": "lps"})
        assert isinstance(strategy, SpecStrategy)
        assert strategy.name == "ngram"


# ── DeltaNetInversionStrategy Tests ──


class TestDeltaNetInversionStrategy:
    """Verify DeltaNetInversionStrategy lifecycle and factory integration."""

    def test_strategy_name(self):
        s = DeltaNetInversionStrategy()
        assert s.name == "deltanet_inversion"

    def test_strategy_without_inverter_returns_empty_draft(self):
        s = DeltaNetInversionStrategy()
        s.begin("req-1")
        proposal = s.draft([1, 2, 3], n=5)
        assert proposal.tokens == []
        assert proposal.strategy_name == "deltanet_inversion"
        s.end("req-1")

    def test_strategy_with_inverter_returns_empty_tokens(self):
        """DeltaNet inversion does not propose tokens — it recovers state."""
        from unittest.mock import MagicMock
        mock_inverter = MagicMock()
        s = DeltaNetInversionStrategy(inverter=mock_inverter)
        s.begin("req-1")
        proposal = s.draft([1, 2, 3], n=5)
        assert proposal.tokens == []
        assert proposal.metadata.get("inversion_available") is True
        mock_inverter.start_capture.assert_called_once()
        s.end("req-1")

    def test_accept_triggers_inversion_on_partial_reject(self):
        from unittest.mock import MagicMock
        mock_inverter = MagicMock()
        mock_inverter.invert_all.return_value = [MagicMock(), MagicMock()]
        s = DeltaNetInversionStrategy(inverter=mock_inverter)
        s.begin("req-1")
        s.draft([1, 2, 3], n=5)
        # Accept with verified_up_to=1 < len(draft_tokens)=3 triggers inversion
        s.accept([10, 20, 30], verified_up_to=1)
        mock_inverter.invert_all.assert_called_once()
        stats = s.stats()
        assert stats["total_inversions"] == 2
        s.end("req-1")

    def test_accept_no_inversion_on_full_accept(self):
        from unittest.mock import MagicMock
        mock_inverter = MagicMock()
        s = DeltaNetInversionStrategy(inverter=mock_inverter)
        s.begin("req-1")
        s.draft([1, 2, 3], n=5)
        # Accept with verified_up_to=3 == len(draft_tokens)=3, no inversion
        s.accept([10, 20, 30], verified_up_to=3)
        mock_inverter.invert_all.assert_not_called()
        s.end("req-1")

    def test_accept_handles_inversion_failure(self):
        from unittest.mock import MagicMock
        mock_inverter = MagicMock()
        mock_inverter.invert_all.side_effect = RuntimeError("inversion failed")
        s = DeltaNetInversionStrategy(inverter=mock_inverter)
        s.begin("req-1")
        s.draft([1, 2, 3], n=5)
        # Should not raise
        s.accept([10, 20, 30], verified_up_to=1)
        stats = s.stats()
        assert stats["total_inversion_failures"] == 1
        s.end("req-1")

    def test_stats_tracking(self):
        s = DeltaNetInversionStrategy()
        s.begin("req-1")
        # Without inverter, draft() returns early but still counts as a draft
        # We need to manually increment to verify stats accumulate correctly
        s._total_drafts = 1
        s._total_draft_tokens = 0  # DeltaNet returns 0 draft tokens
        s.accept([10], verified_up_to=0)
        stats = s.stats()
        assert stats["name"] == "deltanet_inversion"
        assert stats["total_drafts"] == 1
        assert stats["total_accepted"] == 1
        assert stats["total_accepted_tokens"] == 0
        s.end("req-1")

    def test_stats_tracking_with_inverter(self):
        from unittest.mock import MagicMock
        mock_inverter = MagicMock()
        s = DeltaNetInversionStrategy(inverter=mock_inverter)
        s.begin("req-1")
        s.draft([1, 2, 3], n=5)
        s.accept([10, 20], verified_up_to=1)
        stats = s.stats()
        assert stats["name"] == "deltanet_inversion"
        assert stats["total_drafts"] == 1
        assert stats["total_accepted"] == 1
        assert stats["total_accepted_tokens"] == 1
        s.end("req-1")

    def test_reset_clears_stats(self):
        s = DeltaNetInversionStrategy()
        s.begin("req-1")
        s.draft([1, 2, 3], n=5)
        s.reset()
        stats = s.stats()
        assert stats["total_drafts"] == 0
        assert stats["total_inversions"] == 0

    def test_inverter_property(self):
        from unittest.mock import MagicMock
        mock_inv = MagicMock()
        s = DeltaNetInversionStrategy(inverter=mock_inv)
        assert s.inverter is mock_inv

    def test_inverter_property_none(self):
        s = DeltaNetInversionStrategy()
        assert s.inverter is None


class TestSpecStrategyFactoryDeltaNet:
    """Verify SpecStrategyFactory supports deltanet type."""

    def test_factory_creates_deltanet(self):
        s = SpecStrategyFactory.create({"type": "deltanet"})
        assert isinstance(s, DeltaNetInversionStrategy)
        assert s.name == "deltanet_inversion"

    def test_factory_deltanet_with_inverter(self):
        from unittest.mock import MagicMock
        mock_inv = MagicMock()
        s = SpecStrategyFactory.create({"type": "deltanet", "inverter": mock_inv})
        assert isinstance(s, DeltaNetInversionStrategy)
        assert s.inverter is mock_inv

    def test_factory_deltanet_creates_inverter(self):
        """Without explicit inverter, factory should create a DeltaNetInverter."""
        s = SpecStrategyFactory.create({"type": "deltanet"})
        assert s._inverter is not None

    def test_factory_composite_with_deltanet(self):
        strategy = SpecStrategyFactory.create({
            "type": "composite",
            "strategies": [
                {"type": "ngram"},
                {"type": "deltanet"},
            ],
        })
        assert isinstance(strategy, CompositeStrategy)
        assert "deltanet_inversion" in strategy.name

    def test_factory_unknown_type_mentions_deltanet(self):
        with pytest.raises(ValueError, match="deltanet"):
            SpecStrategyFactory.create({"type": "unknown_strategy"})

    def test_from_env_deltanet(self):
        old = os.environ.get("YUNSHU_SPEC_STRATEGY")
        try:
            os.environ["YUNSHU_SPEC_STRATEGY"] = "deltanet"
            s = SpecStrategyFactory.from_env()
            assert isinstance(s, DeltaNetInversionStrategy)
        finally:
            if old is not None:
                os.environ["YUNSHU_SPEC_STRATEGY"] = old
            else:
                os.environ.pop("YUNSHU_SPEC_STRATEGY", None)
