"""Tests for speculative decoding head auto-detection from model config.

Covers detect_spec_heads() and auto_configure_speculative() for all supported
head types: EAGLE, EAGLE-3, Medusa, MLPSpeculator, MTP, and no spec heads.
"""
import pytest

from yunshu_engine.speculative_decoder import (
    SpecDecodingConfig,
    SpecHeadInfo,
    auto_configure_speculative,
    detect_spec_heads,
)


# ── SpecHeadInfo ──


class TestSpecHeadInfo:
    def test_defaults(self):
        info = SpecHeadInfo()
        assert info.head_type == "none"
        assert info.num_heads == 0
        assert info.draft_length == 0
        assert info.head_config == {}

    def test_custom_values(self):
        info = SpecHeadInfo(
            head_type="medusa",
            num_heads=4,
            draft_length=5,
            head_config={"num_draft_tokens": 5},
        )
        assert info.head_type == "medusa"
        assert info.num_heads == 4
        assert info.draft_length == 5
        assert info.head_config["num_draft_tokens"] == 5


# ── detect_spec_heads: MTP ──


class TestDetectMTP:
    """MTP (Multi-Token Prediction) detection.

    Patterns from vLLM SpeculativeConfig:
    - DeepSeek-V3/R1: num_nextn_predict_layers
    - Qwen3.5: mtp_num_hidden_layers
    - General: n_predict, mtp_heads
    - model_type: deepseek_mtp, qwen3_5_mtp, etc.
    """

    def test_deepseek_v3_num_nextn_predict_layers(self):
        """DeepSeek-V3/R1 MTP detection via num_nextn_predict_layers."""
        config = {
            "model_type": "deepseek_v3",
            "num_nextn_predict_layers": 1,
            "n_predict": 1,
        }
        info = detect_spec_heads(config)
        assert info.head_type == "mtp"
        assert info.num_heads == 1
        assert info.draft_length == 1

    def test_deepseek_v3_multiple_mtp_layers(self):
        """Multiple MTP layers (DeepSeek-V3 typically uses 1–2)."""
        config = {
            "model_type": "deepseek_v3",
            "num_nextn_predict_layers": 3,
            "n_predict": 3,
        }
        info = detect_spec_heads(config)
        assert info.head_type == "mtp"
        assert info.num_heads == 3
        assert info.draft_length == 3

    def test_qwen35_mtp_num_hidden_layers(self):
        """Qwen3.5 MTP detection via mtp_num_hidden_layers."""
        config = {
            "model_type": "qwen3_5",
            "mtp_num_hidden_layers": 2,
        }
        info = detect_spec_heads(config)
        assert info.head_type == "mtp"
        assert info.num_heads == 2
        assert info.draft_length == 2

    def test_mtp_model_type_deepseek_mtp(self):
        """model_type='deepseek_mtp' is a known MTP type."""
        config = {
            "model_type": "deepseek_mtp",
        }
        info = detect_spec_heads(config)
        assert info.head_type == "mtp"
        assert info.num_heads == 1  # default when no layer count specified

    def test_mtp_model_type_qwen35_mtp(self):
        """model_type='qwen3_5_mtp' is a known MTP type."""
        config = {
            "model_type": "qwen3_5_mtp",
        }
        info = detect_spec_heads(config)
        assert info.head_type == "mtp"

    def test_mtp_heads_key(self):
        """mtp_heads key (alternative MTP config key)."""
        config = {
            "model_type": "llama",
            "mtp_heads": 2,
        }
        info = detect_spec_heads(config)
        assert info.head_type == "mtp"
        assert info.num_heads == 2
        assert info.draft_length == 2

    def test_mtp_with_n_predict_override(self):
        """n_predict sets draft_length, num_nextn_predict_layers sets num_heads."""
        config = {
            "model_type": "deepseek_v3",
            "num_nextn_predict_layers": 2,
            "n_predict": 5,
        }
        info = detect_spec_heads(config)
        assert info.head_type == "mtp"
        assert info.num_heads == 2
        assert info.draft_length == 5

    def test_mtp_head_config_populated(self):
        """head_config should contain raw config values for debugging."""
        config = {
            "model_type": "deepseek_v3",
            "num_nextn_predict_layers": 1,
            "n_predict": 1,
        }
        info = detect_spec_heads(config)
        assert info.head_config["num_nextn_predict_layers"] == 1
        assert info.head_config["model_type"] == "deepseek_v3"


# ── detect_spec_heads: EAGLE-3 ──


class TestDetectEagle3:
    """EAGLE-3 detection.

    EAGLE-3 uses auxiliary hidden states from the target model to predict
    multiple tokens via a tree-structured draft.
    """

    def test_eagle3_key_present(self):
        """eagle3 key in config."""
        config = {
            "model_type": "llama",
            "eagle3": True,
        }
        info = detect_spec_heads(config)
        assert info.head_type == "eagle3"
        assert info.num_heads == 1
        assert info.draft_length >= 1

    def test_eagle3_model_type(self):
        """model_type='eagle3'."""
        config = {
            "model_type": "eagle3",
        }
        info = detect_spec_heads(config)
        assert info.head_type == "eagle3"

    def test_eagle3_with_dict_value(self):
        """eagle3 as dict with num_speculative_tokens."""
        config = {
            "model_type": "llama",
            "eagle3": {
                "num_speculative_tokens": 5,
            },
        }
        info = detect_spec_heads(config)
        assert info.head_type == "eagle3"
        assert info.draft_length == 5

    def test_eagle3_with_int_value(self):
        """eagle3 as int (draft length)."""
        config = {
            "model_type": "llama",
            "eagle3": 3,
        }
        info = detect_spec_heads(config)
        assert info.head_type == "eagle3"
        assert info.draft_length == 3

    def test_eagle3_with_num_lookahead_tokens(self):
        """num_lookahead_tokens overrides draft length."""
        config = {
            "model_type": "llama",
            "eagle3": True,
            "num_lookahead_tokens": 8,
        }
        info = detect_spec_heads(config)
        assert info.head_type == "eagle3"
        assert info.draft_length == 8


# ── detect_spec_heads: EAGLE ──


class TestDetectEagle:
    """EAGLE (v1) detection.

    EAGLE uses the target model's hidden states + token embeddings to
    draft tokens with a single FC layer.
    """

    def test_eagle_key_present(self):
        """eagle key in config."""
        config = {
            "model_type": "llama",
            "eagle": True,
        }
        info = detect_spec_heads(config)
        assert info.head_type == "eagle"
        assert info.num_heads == 1

    def test_eagle_model_type(self):
        """model_type='eagle'."""
        config = {
            "model_type": "eagle",
        }
        info = detect_spec_heads(config)
        assert info.head_type == "eagle"

    def test_eagle_draft_model_path(self):
        """draft_model_path containing 'eagle' triggers eagle detection."""
        config = {
            "model_type": "llama",
            "draft_model_path": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
        }
        info = detect_spec_heads(config)
        assert info.head_type == "eagle"
        assert info.head_config["draft_model_path"] == "yuhuili/EAGLE-LLaMA3-Instruct-8B"

    def test_eagle_draft_model_path_case_insensitive(self):
        """draft_model_path is matched case-insensitively."""
        config = {
            "model_type": "llama",
            "draft_model_path": "models/Eagle-Qwen-7B",
        }
        info = detect_spec_heads(config)
        assert info.head_type == "eagle"

    def test_eagle_with_dict_value(self):
        """eagle as dict with num_speculative_tokens."""
        config = {
            "model_type": "llama",
            "eagle": {
                "num_speculative_tokens": 4,
            },
        }
        info = detect_spec_heads(config)
        assert info.head_type == "eagle"
        assert info.draft_length == 4

    def test_eagle_with_num_lookahead_tokens(self):
        """num_lookahead_tokens overrides draft length."""
        config = {
            "model_type": "llama",
            "eagle": True,
            "num_lookahead_tokens": 6,
        }
        info = detect_spec_heads(config)
        assert info.head_type == "eagle"
        assert info.draft_length == 6

    def test_draft_model_path_without_eagle_not_detected(self):
        """draft_model_path without 'eagle' does not trigger eagle detection."""
        config = {
            "model_type": "llama",
            "draft_model_path": "models/tiny-llama-1.1b",
        }
        info = detect_spec_heads(config)
        assert info.head_type == "none"


# ── detect_spec_heads: Medusa ──


class TestDetectMedusa:
    """Medusa head detection.

    Medusa adds multiple MLP heads on top of the target model's hidden states.
    Each head predicts a token independently, enabling parallel multi-token drafting.
    """

    def test_medusa_model_type(self):
        """model_type='medusa'."""
        config = {
            "model_type": "medusa",
            "medusa_num_heads": 4,
            "num_draft_tokens": 5,
        }
        info = detect_spec_heads(config)
        assert info.head_type == "medusa"
        assert info.num_heads == 4
        assert info.draft_length == 5

    def test_num_draft_tokens(self):
        """num_draft_tokens triggers medusa detection."""
        config = {
            "model_type": "llama",
            "num_draft_tokens": 5,
        }
        info = detect_spec_heads(config)
        assert info.head_type == "medusa"
        assert info.draft_length == 5

    def test_num_speculative_tokens(self):
        """num_speculative_tokens triggers medusa detection."""
        config = {
            "model_type": "llama",
            "num_speculative_tokens": 8,
        }
        info = detect_spec_heads(config)
        assert info.head_type == "medusa"
        assert info.draft_length == 8

    def test_medusa_with_num_heads(self):
        """medusa_num_heads sets the number of heads."""
        config = {
            "model_type": "medusa",
            "medusa_num_heads": 5,
        }
        info = detect_spec_heads(config)
        assert info.head_type == "medusa"
        assert info.num_heads == 5

    def test_medusa_defaults_when_only_model_type(self):
        """Only model_type='medusa', defaults num_heads=1, draft_length=1."""
        config = {
            "model_type": "medusa",
        }
        info = detect_spec_heads(config)
        assert info.head_type == "medusa"
        assert info.num_heads == 1
        assert info.draft_length == 1

    def test_medusa_num_draft_tokens_preferred_over_heads(self):
        """num_draft_tokens takes priority for draft_length over num_heads."""
        config = {
            "model_type": "medusa",
            "medusa_num_heads": 3,
            "num_draft_tokens": 7,
        }
        info = detect_spec_heads(config)
        assert info.head_type == "medusa"
        assert info.num_heads == 3
        assert info.draft_length == 7


# ── detect_spec_heads: MLPSpeculator ──


class TestDetectMLPSpeculator:
    """MLPSpeculator detection (IBM's speculative decoding approach)."""

    def test_mlp_speculator_model_type(self):
        """model_type='mlp_speculator'."""
        config = {
            "model_type": "mlp_speculator",
            "num_speculative_tokens": 5,
        }
        info = detect_spec_heads(config)
        assert info.head_type == "mlp_speculator"
        assert info.draft_length == 5

    def test_mlp_speculator_defaults(self):
        """MLPSpeculator with only model_type set."""
        config = {
            "model_type": "mlp_speculator",
        }
        info = detect_spec_heads(config)
        assert info.head_type == "mlp_speculator"
        assert info.draft_length == 5  # default

    def test_mlp_speculator_num_predict(self):
        """num_predict overrides default draft length."""
        config = {
            "model_type": "mlp_speculator",
            "num_predict": 3,
        }
        info = detect_spec_heads(config)
        assert info.head_type == "mlp_speculator"
        assert info.draft_length == 3


# ── detect_spec_heads: No spec heads ──


class TestDetectNoSpecHeads:
    """Config with no speculative decoding heads should return 'none'."""

    def test_plain_llama(self):
        """Standard LLaMA config has no spec heads."""
        config = {
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
        }
        info = detect_spec_heads(config)
        assert info.head_type == "none"
        assert info.num_heads == 0
        assert info.draft_length == 0
        assert info.head_config == {}

    def test_plain_qwen2(self):
        """Standard Qwen2 config has no spec heads."""
        config = {
            "model_type": "qwen2",
            "hidden_size": 3584,
        }
        info = detect_spec_heads(config)
        assert info.head_type == "none"

    def test_empty_config(self):
        """Empty dict returns 'none'."""
        info = detect_spec_heads({})
        assert info.head_type == "none"

    def test_none_config(self):
        """None input returns 'none'."""
        info = detect_spec_heads(None)
        assert info.head_type == "none"

    def test_string_config(self):
        """Non-dict input returns 'none'."""
        info = detect_spec_heads("not a dict")
        assert info.head_type == "none"

    def test_config_with_unrelated_keys(self):
        """Config with keys that don't match any spec pattern."""
        config = {
            "model_type": "gpt2",
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12,
        }
        info = detect_spec_heads(config)
        assert info.head_type == "none"


# ── detect_spec_heads: Priority ──


class TestDetectionPriority:
    """MTP has highest priority (matches first), then EAGLE-3, EAGLE, Medusa."""

    def test_mtp_takes_priority_over_medusa(self):
        """If both MTP and Medusa keys present, MTP wins (checked first)."""
        config = {
            "model_type": "deepseek_v3",
            "num_nextn_predict_layers": 1,
            "num_draft_tokens": 5,
        }
        info = detect_spec_heads(config)
        assert info.head_type == "mtp"

    def test_eagle3_takes_priority_over_eagle(self):
        """If both eagle3 and eagle keys present, EAGLE-3 wins."""
        config = {
            "model_type": "llama",
            "eagle3": True,
            "eagle": True,
        }
        info = detect_spec_heads(config)
        assert info.head_type == "eagle3"

    def test_eagle_takes_priority_over_medusa(self):
        """If both eagle and medusa keys present, EAGLE wins."""
        config = {
            "model_type": "llama",
            "eagle": True,
            "num_draft_tokens": 5,
        }
        info = detect_spec_heads(config)
        assert info.head_type == "eagle"


# ── auto_configure_speculative ──


class TestAutoConfigureSpeculative:
    """auto_configure_speculative() creates SpecDecodingConfig from model config."""

    def test_no_heads_returns_disabled_config(self):
        """No spec heads → draft_length=0, bonus_token=False."""
        config = {"model_type": "llama"}
        spec_config = auto_configure_speculative(config)
        assert isinstance(spec_config, SpecDecodingConfig)
        assert spec_config.draft_length == 0
        assert spec_config.bonus_token is False

    def test_eagle3_config(self):
        """EAGLE-3 → draft_length >= 5 (proven speedup baseline)."""
        config = {
            "model_type": "eagle3",
        }
        spec_config = auto_configure_speculative(config)
        assert spec_config.draft_length >= 5
        assert spec_config.bonus_token is True

    def test_eagle_config(self):
        """EAGLE → draft_length from head config."""
        config = {
            "model_type": "llama",
            "eagle": {"num_speculative_tokens": 4},
        }
        spec_config = auto_configure_speculative(config)
        assert spec_config.draft_length == 4
        assert spec_config.bonus_token is True

    def test_medusa_config(self):
        """Medusa → draft_length from num_draft_tokens."""
        config = {
            "model_type": "medusa",
            "medusa_num_heads": 4,
            "num_draft_tokens": 5,
        }
        spec_config = auto_configure_speculative(config)
        assert spec_config.draft_length == 5
        assert spec_config.bonus_token is True

    def test_mtp_config(self):
        """MTP → draft_length from num_nextn_predict_layers."""
        config = {
            "model_type": "deepseek_v3",
            "num_nextn_predict_layers": 1,
            "n_predict": 1,
        }
        spec_config = auto_configure_speculative(config)
        assert spec_config.draft_length == 1
        assert spec_config.bonus_token is True

    def test_mtp_config_multiple_layers(self):
        """MTP with multiple layers."""
        config = {
            "model_type": "deepseek_v3",
            "num_nextn_predict_layers": 3,
            "n_predict": 3,
        }
        spec_config = auto_configure_speculative(config)
        assert spec_config.draft_length == 3

    def test_draft_length_capped_at_10(self):
        """Draft length is capped at 10 (reasonable maximum)."""
        config = {
            "model_type": "medusa",
            "num_draft_tokens": 50,  # unreasonably high
        }
        spec_config = auto_configure_speculative(config)
        assert spec_config.draft_length == 10

    def test_mtp_with_zero_predict_falls_back_to_heads(self):
        """If n_predict is 0, fall back to num_heads for draft_length."""
        config = {
            "model_type": "deepseek_v3",
            "num_nextn_predict_layers": 2,
            "n_predict": 0,
        }
        spec_config = auto_configure_speculative(config)
        # draft_length should use max(num_heads, 1) = max(2, 1) = 2
        assert spec_config.draft_length == 2

    def test_empty_config(self):
        """Empty config returns disabled speculative decoding."""
        spec_config = auto_configure_speculative({})
        assert spec_config.draft_length == 0
        assert spec_config.bonus_token is False

    def test_mlp_speculator_config(self):
        """MLPSpeculator → draft_length from num_speculative_tokens."""
        config = {
            "model_type": "mlp_speculator",
            "num_speculative_tokens": 5,
        }
        spec_config = auto_configure_speculative(config)
        assert spec_config.draft_length == 5
        assert spec_config.bonus_token is True


# ── Integration: Detection → Config roundtrip ──


class TestDetectionRoundtrip:
    """Verify that detect_spec_heads output feeds into auto_configure correctly."""

    def test_eagle3_roundtrip(self):
        """EAGLE-3 detection → config with appropriate defaults."""
        config = {
            "model_type": "eagle3",
            "num_lookahead_tokens": 5,
        }
        info = detect_spec_heads(config)
        assert info.head_type == "eagle3"

        spec_config = auto_configure_speculative(config)
        assert spec_config.draft_length == 5
        assert spec_config.acceptance_threshold == 1.0

    def test_mtp_roundtrip_deepseek_v3(self):
        """DeepSeek-V3 MTP detection → config roundtrip."""
        config = {
            "model_type": "deepseek_v3",
            "architectures": ["DeepseekV3ForCausalLM"],
            "num_nextn_predict_layers": 1,
            "n_predict": 1,
            "hidden_size": 7168,
        }
        info = detect_spec_heads(config)
        assert info.head_type == "mtp"
        assert info.head_config["num_nextn_predict_layers"] == 1

        spec_config = auto_configure_speculative(config)
        assert spec_config.draft_length == 1
        assert spec_config.bonus_token is True

    def test_qwen35_mtp_roundtrip(self):
        """Qwen3.5 MTP detection → config roundtrip."""
        config = {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5MTP"],
            "mtp_num_hidden_layers": 2,
        }
        info = detect_spec_heads(config)
        assert info.head_type == "mtp"
        assert info.num_heads == 2

        spec_config = auto_configure_speculative(config)
        assert spec_config.draft_length == 2
