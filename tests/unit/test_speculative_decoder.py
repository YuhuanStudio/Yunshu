"""Tests for speculative decoding engine."""

from unittest.mock import MagicMock

import mlx.core as mx

from yunshu_engine.speculative_decoder import (
    DraftResult,
    LookaheadReasoning,
    SpecDecodingConfig,
    SpeculativeDecoder,
    VerifyResult,
    auto_configure_speculative,
    detect_spec_heads,
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
        lookahead = LookaheadReasoning(base_draft_k=5, thinking_draft_k=10)
        assert lookahead.in_thinking is False
        assert lookahead.adjust_draft_k() == 5

    def test_adjust_draft_k_normal(self):
        lookahead = LookaheadReasoning(base_draft_k=5, thinking_draft_k=10)
        assert lookahead.adjust_draft_k() == 5

    def test_adjust_draft_k_thinking(self):
        lookahead = LookaheadReasoning(base_draft_k=5, thinking_draft_k=10)
        lookahead._in_thinking = True
        assert lookahead.adjust_draft_k() == 10

    def test_adjust_draft_k_thinking_with_acceptance(self):
        lookahead = LookaheadReasoning(base_draft_k=5, thinking_draft_k=10)
        lookahead._in_thinking = True
        # Low acceptance → slight boost
        for _ in range(5):
            lookahead.record_accept(2)
        assert lookahead.adjust_draft_k() == 7  # min(5+2, 10)
        # High acceptance → full boost
        lookahead._recent_accepts.clear()
        for _ in range(5):
            lookahead.record_accept(4)
        assert lookahead.adjust_draft_k() == 10  # 4 >= 5*0.7

    def test_get_stats(self):
        lookahead = LookaheadReasoning(base_draft_k=5, thinking_draft_k=10)
        stats = lookahead.get_stats()
        assert "in_thinking" in stats
        assert stats["base_draft_k"] == 5
        assert stats["thinking_draft_k"] == 10

    def test_with_decoder(self):
        config = SpecDecodingConfig()

        class MockModel:
            pass

        class MockTokenizer:
            pass

        decoder = SpeculativeDecoder(MockModel(), MockModel(), MockTokenizer(), config)
        lookahead = LookaheadReasoning(decoder=decoder)
        stats = lookahead.get_stats()
        assert "decoder_stats" in stats


# ---------------------------------------------------------------------------
# detect_spec_heads — config inspection
# ---------------------------------------------------------------------------


class TestDetectSpecHeads:
    def test_empty_config(self):
        info = detect_spec_heads({})
        assert info.head_type == "none"

    def test_non_dict_returns_none(self):
        info = detect_spec_heads("not a dict")
        assert info.head_type == "none"

    def test_none_returns_none(self):
        info = detect_spec_heads(None)
        assert info.head_type == "none"

    def test_mtp_via_num_nextn_predict_layers(self):
        info = detect_spec_heads({"num_nextn_predict_layers": 3})
        assert info.head_type == "mtp"
        assert info.num_heads == 3
        assert info.draft_length == 3

    def test_mtp_via_mtp_num_hidden_layers(self):
        info = detect_spec_heads({"mtp_num_hidden_layers": 2})
        assert info.head_type == "mtp"

    def test_mtp_via_model_type(self):
        info = detect_spec_heads({"model_type": "deepseek_mtp"})
        assert info.head_type == "mtp"

    def test_mtp_qwen35_with_n_predict(self):
        info = detect_spec_heads(
            {
                "model_type": "qwen3_5_mtp",
                "mtp_num_hidden_layers": 1,
                "n_predict": 4,
            }
        )
        assert info.head_type == "mtp"
        assert info.draft_length == 4

    def test_eagle3_via_key(self):
        info = detect_spec_heads({"eagle3": {"num_speculative_tokens": 5}})
        assert info.head_type == "eagle3"
        assert info.draft_length == 5

    def test_eagle3_via_model_type(self):
        info = detect_spec_heads({"model_type": "eagle3"})
        assert info.head_type == "eagle3"

    def test_eagle3_int_value(self):
        info = detect_spec_heads({"eagle3": 3})
        assert info.head_type == "eagle3"
        assert info.draft_length == 3

    def test_eagle_via_key(self):
        info = detect_spec_heads({"eagle": {"num_speculative_tokens": 4}})
        assert info.head_type == "eagle"

    def test_eagle_via_model_type(self):
        info = detect_spec_heads({"model_type": "eagle"})
        assert info.head_type == "eagle"

    def test_eagle_via_draft_model_path(self):
        info = detect_spec_heads({"draft_model_path": "models/eagle-draft-v2"})
        assert info.head_type == "eagle"

    def test_mlp_speculator(self):
        info = detect_spec_heads(
            {
                "model_type": "mlp_speculator",
                "num_speculative_tokens": 5,
            }
        )
        assert info.head_type == "mlp_speculator"

    def test_medusa_via_model_type(self):
        info = detect_spec_heads({"model_type": "medusa", "medusa_num_heads": 4})
        assert info.head_type == "medusa"
        assert info.num_heads == 4

    def test_medusa_via_num_draft_tokens(self):
        info = detect_spec_heads({"num_draft_tokens": 5})
        assert info.head_type == "medusa"

    def test_medusa_via_num_speculative_tokens(self):
        info = detect_spec_heads({"num_speculative_tokens": 3})
        assert info.head_type == "medusa"

    def test_unknown_model_type(self):
        info = detect_spec_heads({"model_type": "llama"})
        assert info.head_type == "none"

    def test_mtp_priority_over_eagle(self):
        info = detect_spec_heads(
            {
                "model_type": "deepseek_mtp",
                "eagle": {"num_speculative_tokens": 3},
            }
        )
        assert info.head_type == "mtp"

    def test_eagle3_priority_over_eagle(self):
        info = detect_spec_heads(
            {
                "eagle3": 3,
                "eagle": {"num_speculative_tokens": 5},
            }
        )
        assert info.head_type == "eagle3"


# ---------------------------------------------------------------------------
# auto_configure_speculative
# ---------------------------------------------------------------------------


class TestAutoConfigureSpeculative:
    def test_no_heads_disables(self):
        config = auto_configure_speculative({})
        assert config.draft_length == 0
        assert config.bonus_token is False

    def test_eagle3_gets_higher_draft_length(self):
        config = auto_configure_speculative({"eagle3": 2})
        assert config.draft_length >= 5
        assert config.bonus_token is True

    def test_mtp_configured(self):
        config = auto_configure_speculative(
            {"mtp_num_hidden_layers": 2, "n_predict": 3}
        )
        assert config.draft_length >= 1

    def test_draft_length_capped_at_10(self):
        config = auto_configure_speculative(
            {
                "model_type": "medusa",
                "num_speculative_tokens": 20,
            }
        )
        assert config.draft_length <= 10

    def test_none_input_disables(self):
        config = auto_configure_speculative(None)
        assert config.draft_length == 0


# ---------------------------------------------------------------------------
# Cache snapshot / restore
# ---------------------------------------------------------------------------


class TestCacheSnapshotRestore:
    def test_snapshot_empty(self):
        snapshot = SpeculativeDecoder._snapshot_cache([])
        assert snapshot == []

    def test_snapshot_kv_cache(self):
        cache = MagicMock()
        cache.offset = 42
        del cache.cache
        snapshot = SpeculativeDecoder._snapshot_cache([cache])
        assert snapshot[0] == ("kv", 42)

    def test_snapshot_arrays_cache(self):
        cache = MagicMock()
        cache.cache = [mx.zeros((2, 3)), mx.ones((2, 3))]
        snapshot = SpeculativeDecoder._snapshot_cache([cache])
        assert snapshot[0][0] == "arrays"
        assert len(snapshot[0][1]) == 2

    def test_restore_kv_offset(self):
        cache = MagicMock()
        cache.offset = 99
        del cache.cache
        snapshot = SpeculativeDecoder._snapshot_cache([cache])
        cache.offset = 200
        SpeculativeDecoder._restore_cache([cache], snapshot)
        assert cache.offset == 99

    def test_restore_arrays(self):
        cache = MagicMock()
        cache.cache = [mx.zeros((2, 3))]
        snapshot = SpeculativeDecoder._snapshot_cache([cache])
        cache.cache = [mx.ones((2, 3))]
        SpeculativeDecoder._restore_cache([cache], snapshot)
        assert cache.cache is snapshot[0][1]

    def test_restore_empty(self):
        SpeculativeDecoder._restore_cache([], [])  # should not raise

    def test_mixed_types(self):
        kv = MagicMock()
        kv.offset = 10
        del kv.cache
        arr = MagicMock()
        arr.cache = [mx.zeros((1, 1))]
        snapshot = SpeculativeDecoder._snapshot_cache([kv, arr])
        assert snapshot[0] == ("kv", 10)
        assert snapshot[1][0] == "arrays"


# ---------------------------------------------------------------------------
# generate_draft (mocked model)
# ---------------------------------------------------------------------------


class TestGenerateDraft:
    def test_returns_correct_length(self):
        draft = MagicMock()
        draft.return_value = mx.zeros((1, 1, 100))
        config = SpecDecodingConfig(draft_length=3)
        decoder = SpeculativeDecoder(MagicMock(), draft, MagicMock(), config=config)
        result = decoder.generate_draft(mx.array([[1, 2, 3]]), cache=[])
        assert isinstance(result, DraftResult)
        assert len(result.token_ids) == 3
        assert len(result.logprobs) == 3

    def test_single_draft_token(self):
        draft = MagicMock()
        draft.return_value = mx.zeros((1, 1, 50))
        config = SpecDecodingConfig(draft_length=1)
        decoder = SpeculativeDecoder(MagicMock(), draft, MagicMock(), config=config)
        result = decoder.generate_draft(mx.array([[1]]), cache=[])
        assert len(result.token_ids) == 1


# ---------------------------------------------------------------------------
# verify_draft (mocked model)
# ---------------------------------------------------------------------------


class TestVerifyDraft:
    def test_returns_verify_result(self):
        target = MagicMock()
        K = 3
        target.return_value = mx.zeros((1, K, 100))
        config = SpecDecodingConfig(draft_length=K)
        decoder = SpeculativeDecoder(target, MagicMock(), MagicMock(), config=config)
        draft = DraftResult(token_ids=[10, 20, 30], logprobs=[-1.0, -1.0, -1.0])
        result = decoder.verify_draft(draft, mx.array([[1, 2, 3]]), cache=[])
        assert isinstance(result, VerifyResult)
        assert result.accepted_count <= K
        assert isinstance(result.bonus_token_id, int)

    def test_with_logits_attr(self):
        target = MagicMock()
        K = 2
        output = MagicMock()
        output.logits = mx.zeros((1, K, 100))
        target.return_value = output
        config = SpecDecodingConfig(draft_length=K)
        decoder = SpeculativeDecoder(target, MagicMock(), MagicMock(), config=config)
        draft = DraftResult(token_ids=[5, 10], logprobs=[-0.5, -0.5])
        result = decoder.verify_draft(draft, mx.array([[1]]), cache=[])
        assert isinstance(result, VerifyResult)


# ---------------------------------------------------------------------------
# Acceptance rate calculation
# ---------------------------------------------------------------------------


class TestAcceptanceRate:
    def test_zero_at_start(self):
        decoder = SpeculativeDecoder(MagicMock(), MagicMock(), MagicMock())
        assert decoder.acceptance_rate == 0.0

    def test_with_stats(self):
        decoder = SpeculativeDecoder(MagicMock(), MagicMock(), MagicMock())
        decoder._stats["total_draft_tokens"] = 100
        decoder._stats["total_accepted_tokens"] = 72
        assert abs(decoder.acceptance_rate - 0.72) < 1e-6

    def test_get_stats_computed_fields(self):
        decoder = SpeculativeDecoder(MagicMock(), MagicMock(), MagicMock())
        decoder._stats["total_steps"] = 10
        decoder._stats["total_draft_tokens"] = 50
        decoder._stats["total_accepted_tokens"] = 35
        decoder._stats["total_bonus_tokens"] = 8
        stats = decoder.get_stats()
        assert stats["avg_accepted_per_step"] == 3.5
        assert stats["acceptance_rate"] == 0.7
        assert stats["effective_speedup"] > 0
