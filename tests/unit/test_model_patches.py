"""Tests for Model Patches — DeepSeek, Qwen 3.5, Gemma runtime patches."""
import pytest

from yunshu_engine.model_patches import (
    apply_model_patches,
    detect_model_family,
    get_model_capabilities,
)


class FakeConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class FakeModel:
    def __init__(self, config=None):
        self.config = config


class FakeTokenizer:
    def __init__(self, chat_template=None):
        self.chat_template = chat_template


class TestDetectModelFamily:
    def test_deepseek(self):
        assert detect_model_family("deepseek-v3") == "deepseek"

    def test_qwen(self):
        assert detect_model_family("Qwen2.5-7B") == "qwen"

    def test_gemma(self):
        assert detect_model_family("gemma-4-9b") == "gemma"

    def test_llama(self):
        assert detect_model_family("llama-3-8b") == "llama"

    def test_glm(self):
        assert detect_model_family("glm-4-9b") == "glm"

    def test_mistral(self):
        assert detect_model_family("mistral-7b") == "mistral"

    def test_phi(self):
        assert detect_model_family("phi-3") == "phi"

    def test_unknown(self):
        assert detect_model_family("unknown-model") == "generic"


class TestDeepSeekPatches:
    def test_mla_detection(self):
        config = FakeConfig(kv_lora_rank=512, num_hidden_layers=28)
        model = FakeModel(config)
        patches = apply_model_patches(model, None, "deepseek-v3")
        assert "deepseek_mla_cache" in patches
        assert model._yunshu_mla_mode is True

    def test_rope_scaling(self):
        config = FakeConfig(rope_scaling={"type": "yarn", "factor": 4.0})
        model = FakeModel(config)
        patches = apply_model_patches(model, None, "deepseek-v3")
        assert "deepseek_rope_scaling" in patches

    def test_chat_template_patch(self):
        tokenizer = FakeTokenizer(chat_template="{{ messages }}")
        model = FakeModel(None)
        patches = apply_model_patches(model, tokenizer, "deepseek-v3")
        assert "deepseek_chat_template" in patches


class TestQwen35Patches:
    def test_yarn_rope(self):
        config = FakeConfig(rope_scaling={"type": "yarn", "factor": 4.0})
        model = FakeModel(config)
        patches = apply_model_patches(model, None, "Qwen3.5-7B")
        assert "qwen35_yarn_rope" in patches

    def test_dual_chunk(self):
        config = FakeConfig(max_position_embeddings=131072)
        model = FakeModel(config)
        patches = apply_model_patches(model, None, "Qwen3.5-72B")
        assert "qwen35_dual_chunk" in patches

    def test_no_patch_for_qwen25(self):
        model = FakeModel(None)
        patches = apply_model_patches(model, None, "Qwen2.5-7B")
        assert len(patches) == 0


class TestGemmaPatches:
    def test_attn_softcap(self):
        config = FakeConfig(attn_logit_softcapping=50.0)
        model = FakeModel(config)
        patches = apply_model_patches(model, None, "gemma-4-9b")
        assert "gemma_attn_softcap" in patches

    def test_final_softcap(self):
        config = FakeConfig(final_logit_softcapping=30.0)
        model = FakeModel(config)
        patches = apply_model_patches(model, None, "gemma-4-27b")
        assert "gemma_final_softcap" in patches


class TestGetModelCapabilities:
    def test_basic(self):
        config = FakeConfig(
            num_hidden_layers=28,
            num_attention_heads=28,
            num_key_value_heads=4,
            hidden_size=3584,
            max_position_embeddings=131072,
            vocab_size=152064,
            model_type="qwen2",
        )
        model = FakeModel(config)
        caps = get_model_capabilities(model, "Qwen2.5-7B")
        assert caps["family"] == "qwen"
        assert caps["num_layers"] == 28
        assert caps["num_kv_heads"] == 4
        assert caps["head_dim"] == 128

    def test_moe(self):
        config = FakeConfig(
            num_hidden_layers=28,
            num_attention_heads=28,
            num_key_value_heads=4,
            hidden_size=3584,
            num_local_experts=64,
            num_experts_per_tok=8,
        )
        model = FakeModel(config)
        caps = get_model_capabilities(model, "deepseek-v3")
        assert caps["moe_experts"] == 64
        assert caps["moe_top_k"] == 8

    def test_no_config(self):
        model = FakeModel(None)
        caps = get_model_capabilities(model, "unknown")
        assert caps["family"] == "generic"
