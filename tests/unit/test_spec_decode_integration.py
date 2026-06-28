"""Tests for speculative decoding integration with scheduler and batched engine.

Phase 4 integration tests:
- SchedulerConfig spec decode fields
- Scheduler spec decoder detection and initialization
- BatchedEngine spec_decode parameter handling
- Single-request speculative decoding path
"""
from unittest.mock import MagicMock

from yunshu_engine.scheduler import Scheduler, SchedulerConfig
from yunshu_engine.speculative_decoder import (
    auto_configure_speculative,
    detect_spec_heads,
)


class TestSchedulerConfigSpecDecode:
    """Test SchedulerConfig speculative decoding fields."""

    def test_spec_decode_defaults(self):
        config = SchedulerConfig()
        assert config.enable_spec_decode is False
        assert config.draft_model == ""
        assert config.spec_draft_length == 5

    def test_spec_decode_enabled(self):
        config = SchedulerConfig(
            enable_spec_decode=True,
            draft_model="Qwen2.5-0.5B-Instruct",
            spec_draft_length=8,
        )
        assert config.enable_spec_decode is True
        assert config.draft_model == "Qwen2.5-0.5B-Instruct"
        assert config.spec_draft_length == 8

    def test_spec_decode_fields_preserved_in_scheduler(self):
        config = SchedulerConfig(enable_spec_decode=True, spec_draft_length=10)
        scheduler = Scheduler(model=MagicMock(), tokenizer=MagicMock(), config=config)
        assert scheduler.config.enable_spec_decode is True
        assert scheduler.config.spec_draft_length == 10


class TestSchedulerSpecDetection:
    """Test speculative decoding head detection in Scheduler."""

    def test_spec_decoder_initially_none(self):
        scheduler = Scheduler(model=MagicMock(), tokenizer=MagicMock())
        assert scheduler._spec_decoder is None
        assert scheduler._spec_head_info is None

    def test_get_spec_head_info_none_by_default(self):
        scheduler = Scheduler(model=MagicMock(), tokenizer=MagicMock())
        assert scheduler.get_spec_head_info() is None

    def test_try_init_spec_decoder_no_heads(self):
        """Model without spec heads should not create a decoder."""
        model = MagicMock()
        model.config = MagicMock()
        model.config.to_dict.return_value = {"model_type": "llama"}

        config = SchedulerConfig(enable_spec_decode=True)
        scheduler = Scheduler(model=model, tokenizer=MagicMock(), config=config)
        scheduler._try_init_spec_decoder()

        head_info = scheduler.get_spec_head_info()
        assert head_info is not None
        assert head_info.head_type == "none"

    def test_try_init_spec_decoder_mtp_detected(self):
        """Model with MTP heads should be detected."""
        model = MagicMock()
        model.config = MagicMock()
        model.config.to_dict.return_value = {
            "model_type": "deepseek_mtp",
            "num_nextn_predict_layers": 1,
        }

        config = SchedulerConfig(enable_spec_decode=True)
        scheduler = Scheduler(model=model, tokenizer=MagicMock(), config=config)
        scheduler._try_init_spec_decoder()

        head_info = scheduler.get_spec_head_info()
        assert head_info is not None
        assert head_info.head_type == "mtp"
        assert head_info.num_heads == 1

    def test_try_init_spec_decoder_disabled_in_config(self):
        """Model with spec heads but enable_spec_decode=False should not activate."""
        model = MagicMock()
        model.config = MagicMock()
        model.config.to_dict.return_value = {
            "model_type": "deepseek_mtp",
            "num_nextn_predict_layers": 2,
        }

        config = SchedulerConfig(enable_spec_decode=False)
        scheduler = Scheduler(model=model, tokenizer=MagicMock(), config=config)
        scheduler._try_init_spec_decoder()

        head_info = scheduler.get_spec_head_info()
        assert head_info is not None
        assert head_info.head_type == "mtp"
        # spec_decoder should remain None because enable_spec_decode is False
        assert scheduler._spec_decoder is None

    def test_try_init_spec_decoder_eagle3_detected(self):
        """EAGLE-3 model config should be detected."""
        model = MagicMock()
        model.config = MagicMock()
        model.config.to_dict.return_value = {
            "model_type": "eagle3",
            "eagle3": {"num_speculative_tokens": 5},
        }

        config = SchedulerConfig(enable_spec_decode=True)
        scheduler = Scheduler(model=model, tokenizer=MagicMock(), config=config)
        scheduler._try_init_spec_decoder()

        head_info = scheduler.get_spec_head_info()
        assert head_info is not None
        assert head_info.head_type == "eagle3"

    def test_try_init_spec_decoder_no_config(self):
        """Model without config attribute should not crash."""
        model = MagicMock(spec=[])  # No config attribute

        config = SchedulerConfig(enable_spec_decode=True)
        scheduler = Scheduler(model=model, tokenizer=MagicMock(), config=config)
        scheduler._try_init_spec_decoder()

        # Should still create a head_info with head_type="none" (from empty config)
        assert scheduler._spec_head_info is not None
        assert scheduler._spec_head_info.head_type == "none"

    def test_deep_reset_clears_spec_decoder(self):
        model = MagicMock()
        model.config = MagicMock()
        model.config.to_dict.return_value = {"model_type": "deepseek_mtp", "num_nextn_predict_layers": 1}

        config = SchedulerConfig(enable_spec_decode=True)
        scheduler = Scheduler(model=model, tokenizer=MagicMock(), config=config)
        scheduler._try_init_spec_decoder()
        assert scheduler._spec_head_info is not None

        scheduler.deep_reset()
        assert scheduler._spec_decoder is None
        assert scheduler._spec_head_info is None


class TestSchedulerSpecHeadInfoInStats:
    """Test that spec head info is accessible via scheduler."""

    def test_spec_head_info_medusa(self):
        model = MagicMock()
        model.config = MagicMock()
        model.config.to_dict.return_value = {
            "model_type": "medusa",
            "medusa_num_heads": 4,
            "num_speculative_tokens": 5,
        }

        config = SchedulerConfig(enable_spec_decode=True)
        scheduler = Scheduler(model=model, tokenizer=MagicMock(), config=config)
        scheduler._try_init_spec_decoder()

        info = scheduler.get_spec_head_info()
        assert info.head_type == "medusa"
        assert info.num_heads == 4
        assert info.draft_length == 5


class TestBatchedEngineSpecDecode:
    """Test BatchedEngine speculative decoding integration."""

    def test_spec_decode_disabled_by_default(self):
        from yunshu_engine.batched_engine import BatchedEngine
        engine = BatchedEngine(model_name="test")
        assert engine._spec_decoder is None
        assert engine._spec_enabled is False

    def test_init_spec_decode_no_heads(self):
        """Model without spec heads should not enable spec decode."""
        from yunshu_engine.batched_engine import BatchedEngine

        engine = BatchedEngine(model_name="test")
        engine._model = MagicMock()
        engine._model.config = MagicMock()
        engine._model.config.to_dict.return_value = {"model_type": "llama"}
        engine._tokenizer = MagicMock()

        engine._init_spec_decode()
        assert engine._spec_enabled is False

    def test_init_spec_decode_with_mtp(self):
        """Model with MTP heads should enable spec decode."""
        from yunshu_engine.batched_engine import BatchedEngine

        engine = BatchedEngine(model_name="test")
        engine._model = MagicMock()
        engine._model.config = MagicMock()
        engine._model.config.to_dict.return_value = {
            "model_type": "deepseek_mtp",
            "num_nextn_predict_layers": 1,
            "n_predict": 3,
        }
        engine._tokenizer = MagicMock()

        engine._init_spec_decode()
        assert engine._spec_enabled is True

    def test_init_spec_decode_with_eagle(self):
        """Model with EAGLE heads should enable spec decode."""
        from yunshu_engine.batched_engine import BatchedEngine

        engine = BatchedEngine(model_name="test")
        engine._model = MagicMock()
        engine._model.config = MagicMock()
        engine._model.config.to_dict.return_value = {
            "model_type": "eagle",
            "eagle": {"num_speculative_tokens": 5},
        }
        engine._tokenizer = MagicMock()

        engine._init_spec_decode()
        assert engine._spec_enabled is True

    def test_init_spec_decode_no_config(self):
        """Model without config should not crash."""
        from yunshu_engine.batched_engine import BatchedEngine

        engine = BatchedEngine(model_name="test")
        engine._model = MagicMock(spec=[])
        engine._tokenizer = MagicMock()

        engine._init_spec_decode()
        assert engine._spec_enabled is False


class TestAutoConfigureSpeculative:
    """Test auto_configure_speculative function."""

    def test_no_heads_returns_disabled(self):
        config = auto_configure_speculative({"model_type": "llama"})
        assert config.draft_length == 0
        assert config.bonus_token is False

    def test_mtp_config(self):
        config = auto_configure_speculative({
            "model_type": "deepseek_mtp",
            "num_nextn_predict_layers": 2,
            "n_predict": 4,
        })
        assert config.draft_length >= 1
        assert config.bonus_token is True

    def test_eagle3_config(self):
        config = auto_configure_speculative({
            "model_type": "eagle3",
            "eagle3": {"num_speculative_tokens": 5},
        })
        assert config.draft_length >= 5
        assert config.bonus_token is True

    def test_medusa_config(self):
        config = auto_configure_speculative({
            "model_type": "medusa",
            "medusa_num_heads": 4,
            "num_speculative_tokens": 5,
        })
        assert config.draft_length >= 1
        assert config.bonus_token is True

    def test_mlp_speculator_config(self):
        config = auto_configure_speculative({
            "model_type": "mlp_speculator",
            "num_speculative_tokens": 5,
        })
        assert config.draft_length >= 1
        assert config.bonus_token is True


class TestDetectSpecHeadsEdgeCases:
    """Edge cases for detect_spec_heads."""

    def test_empty_config(self):
        info = detect_spec_heads({})
        assert info.head_type == "none"

    def test_none_config(self):
        info = detect_spec_heads(None)
        assert info.head_type == "none"

    def test_non_dict_config(self):
        info = detect_spec_heads("not a dict")
        assert info.head_type == "none"

    def test_eagle_with_draft_path(self):
        info = detect_spec_heads({
            "draft_model_path": "/models/eagle-draft-v1",
        })
        assert info.head_type == "eagle"

    def test_eagle_draft_path_no_match(self):
        info = detect_spec_heads({
            "draft_model_path": "/models/some-other-model",
        })
        assert info.head_type == "none"
