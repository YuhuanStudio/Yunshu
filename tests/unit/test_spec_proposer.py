"""Tests for SpecProposer — unified speculative decoding interface."""
import pytest

from yunshu_engine.spec_proposer import (
    CompositeSpecProposer,
    NgramSpecProposer,
    SpecProposal,
)
from yunshu_engine.ngram_proposer import NgramConfig


class TestSpecProposal:
    def test_defaults(self):
        p = SpecProposal(token_ids=[1, 2, 3], proposer_type="ngram")
        assert p.token_ids == [1, 2, 3]
        assert p.proposer_type == "ngram"
        assert p.metadata == {}

    def test_with_metadata(self):
        p = SpecProposal(token_ids=[], proposer_type="test", metadata={"k": 5})
        assert p.metadata["k"] == 5


class TestNgramSpecProposer:
    def test_proposer_type(self):
        proposer = NgramSpecProposer(NgramConfig())
        assert proposer.proposer_type == "ngram"

    def test_draft_returns_proposal(self):
        proposer = NgramSpecProposer(NgramConfig(min_n=1, max_n=5, k=3))
        proposer.begin([1, 2, 3])
        result = proposer.draft([1, 2, 3, 1, 2, 3])
        assert isinstance(result, SpecProposal)
        assert result.proposer_type == "ngram"

    def test_draft_limits_k(self):
        proposer = NgramSpecProposer(NgramConfig(min_n=1, max_n=5, k=10))
        result = proposer.draft([1, 2, 3, 1, 2, 3], k=3)
        assert len(result.token_ids) <= 3

    def test_accept_updates_stats(self):
        proposer = NgramSpecProposer(NgramConfig())
        proposer.draft([1, 2, 3, 1, 2, 3])
        proposer.accept(2)
        stats = proposer.get_stats()
        assert stats["accepted"] == 2
        assert stats["proposals"] == 1

    def test_stats(self):
        proposer = NgramSpecProposer(NgramConfig())
        stats = proposer.get_stats()
        assert stats["proposals"] == 0
        assert stats["accepted"] == 0
        assert stats["total_draft"] == 0

    def test_begin_is_noop(self):
        proposer = NgramSpecProposer(NgramConfig())
        # Should not raise
        proposer.begin([1, 2, 3])


class TestCompositeSpecProposer:
    def test_composite_type(self):
        p1 = NgramSpecProposer(NgramConfig(min_n=1, max_n=3, k=3))
        p2 = NgramSpecProposer(NgramConfig(min_n=2, max_n=5, k=5))
        composite = CompositeSpecProposer([p1, p2])
        assert composite.proposer_type == "composite"

    def test_picks_best_proposal(self):
        # p1: small ngram range, fewer matches likely
        # p2: larger range, potentially more matches
        p1 = NgramSpecProposer(NgramConfig(min_n=1, max_n=2, k=3))
        p2 = NgramSpecProposer(NgramConfig(min_n=1, max_n=5, k=5))
        composite = CompositeSpecProposer([p1, p2])
        composite.begin([1, 2, 3, 4, 5])
        result = composite.draft([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        assert result.proposer_type in ("ngram", "composite")

    def test_accept_propagates(self):
        p1 = NgramSpecProposer(NgramConfig())
        p2 = NgramSpecProposer(NgramConfig())
        composite = CompositeSpecProposer([p1, p2])
        composite.accept(3)
        assert p1.get_stats()["accepted"] == 3
        assert p2.get_stats()["accepted"] == 3

    def test_stats_combined(self):
        p1 = NgramSpecProposer(NgramConfig())
        p2 = NgramSpecProposer(NgramConfig())
        composite = CompositeSpecProposer([p1, p2])
        composite.draft([1, 2, 1, 2])
        stats = composite.get_stats()
        assert "ngram_proposals" in stats
