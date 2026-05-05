"""Tests for speculative decoding engine."""
import pytest

from yunshu_engine.speculative_decoder import (
    SpecDecodingConfig,
    DraftResult,
    VerifyResult,
    SpeculativeDecoder,
    LookaheadReasoning,
)


class TestSpecDecodingConfig:
    def test_defaults(self):
        config = SpecDecodingConfig()
        assert config.draft_length == 5
        assert config.acceptance_threshold == 1.0
        assert config.bonus_token is True

    def test_custom_config(self):
        config = SpecDecodingConfig(draft_length=8, draft_temperature=0.9)
        assert config.draft_length == 8
        assert config.draft_temperature == 0.9


class TestDraftResult:
    def test_creation(self):
        result = DraftResult(token_ids=[1, 2, 3], logprobs=[-0.1, -0.2, -0.3])
        assert len(result.token_ids) == 3
        assert len(result.logprobs) == 3


class TestVerifyResult:
    def test_all_accepted(self):
        result = VerifyResult(
            accepted_count=5,
            accepted_ids=[1, 2, 3, 4, 5],
            rejected_at=-1,
            bonus_token_id=6,
            target_logprobs=[-0.1, -0.2, -0.3, -0.1, -0.2],
        )
        assert result.accepted_count == 5
        assert result.rejected_at == -1

    def test_partial_rejection(self):
        result = VerifyResult(
            accepted_count=3,
            accepted_ids=[1, 2, 3],
            rejected_at=3,
            bonus_token_id=99,
            target_logprobs=[-0.1, -0.2, -0.3],
        )
        assert result.accepted_count == 3
        assert result.rejected_at == 3


class TestSpeculativeDecoderStats:
    def test_initial_stats(self):
        class MockModel:
            pass
        class MockTokenizer:
            eos_token_id = 2

        decoder = SpeculativeDecoder(MockModel(), MockModel(), MockTokenizer())
        stats = decoder.get_stats()
        assert stats["total_steps"] == 0
        assert stats["acceptance_rate"] == 0.0

    def test_acceptance_rate_zero_at_start(self):
        class MockModel:
            pass
        class MockTokenizer:
            pass
        decoder = SpeculativeDecoder(MockModel(), MockModel(), MockTokenizer())
        assert decoder.acceptance_rate == 0.0


class TestLookaheadReasoning:
    def test_initial_state(self):
        config = SpecDecodingConfig()
        class MockModel:
            pass
        class MockTokenizer:
            pass
        decoder = SpeculativeDecoder(MockModel(), MockModel(), MockTokenizer(), config)
        lookahead = LookaheadReasoning(decoder)
        assert lookahead._in_thinking is False

    def test_adjust_draft_length_normal(self):
        config = SpecDecodingConfig(draft_length=5)
        class MockModel:
            pass
        class MockTokenizer:
            pass
        decoder = SpeculativeDecoder(MockModel(), MockModel(), MockTokenizer(), config)
        lookahead = LookaheadReasoning(decoder)
        assert lookahead.adjust_draft_length() == 5

    def test_adjust_draft_length_thinking(self):
        config = SpecDecodingConfig(draft_length=5)
        class MockModel:
            pass
        class MockTokenizer:
            pass
        decoder = SpeculativeDecoder(MockModel(), MockModel(), MockTokenizer(), config)
        lookahead = LookaheadReasoning(decoder)
        lookahead._in_thinking = True
        assert lookahead.adjust_draft_length() == 10

    def test_get_stats(self):
        config = SpecDecodingConfig()
        class MockModel:
            pass
        class MockTokenizer:
            pass
        decoder = SpeculativeDecoder(MockModel(), MockModel(), MockTokenizer(), config)
        lookahead = LookaheadReasoning(decoder)
        stats = lookahead.get_stats()
        assert "in_thinking" in stats
        assert "decoder_stats" in stats
