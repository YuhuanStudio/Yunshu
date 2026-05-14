"""Tests for Model Optimizations — RoPE, Attention, MoE, Warmup."""
import math
import pytest
from unittest.mock import MagicMock

from yunshu_engine.model_optimizations import (
    AttentionConfig,
    AttentionOptimizer,
    AttentionType,
    ModelWarmupManager,
    MoEEfficiencyOptimizer,
    RoPEScalingOptimizer,
    RopeScalingConfig,
    RopeScalingType,
    WarmupResult,
    _get_config,
    _safe_int,
)


# ── Shared helpers ─────────────────────────────────────────────────────────────


class FakeConfig:
    """Mimics a model config object with arbitrary attributes."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class FakeModel:
    """Mimics a model object with optional config."""

    def __init__(self, config=None):
        self.config = config


# ══════════════════════════════════════════════════════════════════════════════
# RoPEScalingOptimizer tests
# ══════════════════════════════════════════════════════════════════════════════


class TestRoPEScalingType:
    def test_enum_values(self):
        assert RopeScalingType.LINEAR.value == "linear"
        assert RopeScalingType.DYNAMIC_NTK.value == "dynamic_ntk"
        assert RopeScalingType.YARN.value == "yarn"
        assert RopeScalingType.LLAMA3.value == "llama3"
        assert RopeScalingType.LONGROPE.value == "longrope"
        assert RopeScalingType.NONE.value == "none"


class TestRoPEScalingConfigure:
    def test_no_config_returns_default(self):
        opt = RoPEScalingOptimizer()
        model = FakeModel(None)
        cfg = opt.configure(model, 8192)
        assert cfg.scaling_type == RopeScalingType.NONE

    def test_linear_scaling_detected(self):
        config = FakeConfig(
            max_position_embeddings=8192,
            rope_theta=10000.0,
            hidden_size=4096,
            num_attention_heads=32,
            rope_scaling={"type": "linear", "factor": 2.0},
        )
        model = FakeModel(config)
        opt = RoPEScalingOptimizer()
        cfg = opt.configure(model, 16384)
        assert cfg.scaling_type == RopeScalingType.LINEAR
        assert cfg.scaling_factor == 2.0

    def test_yarn_scaling_from_config(self):
        config = FakeConfig(
            max_position_embeddings=8192,
            rope_theta=10000.0,
            hidden_size=4096,
            num_attention_heads=32,
            rope_scaling={
                "type": "yarn",
                "factor": 8.0,
                "attention_factor": 1.2,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
            },
        )
        model = FakeModel(config)
        opt = RoPEScalingOptimizer()
        cfg = opt.configure(model, 65536)
        assert cfg.scaling_type == RopeScalingType.YARN
        assert cfg.scaling_factor == 8.0
        assert cfg.yarn_attention_factor == 1.2

    def test_llama3_scaling(self):
        config = FakeConfig(
            max_position_embeddings=8192,
            rope_theta=500000.0,
            hidden_size=4096,
            num_attention_heads=32,
            rope_scaling={
                "type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
        )
        model = FakeModel(config)
        opt = RoPEScalingOptimizer()
        cfg = opt.configure(model, 131072)
        assert cfg.scaling_type == RopeScalingType.LLAMA3

    def test_auto_detect_no_scaling_when_context_fits(self):
        config = FakeConfig(
            max_position_embeddings=32768,
            rope_theta=10000.0,
            hidden_size=4096,
            num_attention_heads=32,
        )
        model = FakeModel(config)
        opt = RoPEScalingOptimizer()
        cfg = opt.configure(model, 32768)
        assert cfg.scaling_type == RopeScalingType.NONE

    def test_auto_detect_dynamic_ntk_for_medium_factor(self):
        config = FakeConfig(
            max_position_embeddings=8192,
            rope_theta=10000.0,
            hidden_size=4096,
            num_attention_heads=32,
        )
        model = FakeModel(config)
        opt = RoPEScalingOptimizer()
        cfg = opt.configure(model, 32768)
        # factor = 4.0 -> dynamic NTK
        assert cfg.scaling_type == RopeScalingType.DYNAMIC_NTK

    def test_auto_detect_yarn_for_large_factor(self):
        config = FakeConfig(
            max_position_embeddings=8192,
            rope_theta=10000.0,
            hidden_size=4096,
            num_attention_heads=32,
        )
        model = FakeModel(config)
        opt = RoPEScalingOptimizer()
        cfg = opt.configure(model, 131072)
        # factor = 16.0 -> YaRN
        assert cfg.scaling_type == RopeScalingType.YARN

    def test_head_dim_derived_from_hidden_and_heads(self):
        config = FakeConfig(
            hidden_size=3584,
            num_attention_heads=28,
            max_position_embeddings=4096,
        )
        model = FakeModel(config)
        opt = RoPEScalingOptimizer()
        cfg = opt.configure(model, 4096)
        assert cfg.head_dim == 128  # 3584 / 28

    def test_layer_overrides_from_rope_scaling(self):
        config = FakeConfig(
            max_position_embeddings=8192,
            rope_theta=10000.0,
            hidden_size=4096,
            num_attention_heads=32,
            num_hidden_layers=28,
            rope_scaling={
                "type": "yarn",
                "factor": 4.0,
                "layers": {
                    "0-4": {"type": "linear"},
                    "24-28": {"type": "dynamic_ntk"},
                },
            },
        )
        model = FakeModel(config)
        opt = RoPEScalingOptimizer()
        cfg = opt.configure(model, 32768)
        assert 0 in cfg.layer_overrides
        assert 24 in cfg.layer_overrides

    def test_rope_scaling_config_via_rope_type_key(self):
        """Some configs use 'rope_type' instead of 'type'."""
        config = FakeConfig(
            max_position_embeddings=8192,
            rope_theta=10000.0,
            hidden_size=4096,
            num_attention_heads=32,
            rope_scaling={"rope_type": "dynamic", "factor": 2.0},
        )
        model = FakeModel(config)
        opt = RoPEScalingOptimizer()
        cfg = opt.configure(model, 16384)
        assert cfg.scaling_type == RopeScalingType.DYNAMIC_NTK


class TestComputeRopeFreqs:
    def test_none_scaling_returns_base_freqs(self):
        freqs = RoPEScalingOptimizer.compute_rope_freqs(
            10000.0, 128, 8192, RopeScalingType.NONE
        )
        assert len(freqs) == 64  # 128 // 2
        # base_freq^(2i/head_dim) is 1.0 at i=0, grows with i
        assert freqs[0] < freqs[-1]  # increasing (inverse wavelengths)

    def test_linear_scaling_divides_by_factor(self):
        base = RoPEScalingOptimizer.compute_rope_freqs(
            10000.0, 64, 8192, RopeScalingType.NONE
        )
        scaled = RoPEScalingOptimizer.compute_rope_freqs(
            10000.0, 64, 8192, RopeScalingType.LINEAR, scaling_factor=2.0
        )
        for b, s in zip(base, scaled):
            assert abs(s - b / 2.0) < 1e-6

    def test_dynamic_ntk_changes_base_freq(self):
        base = RoPEScalingOptimizer.compute_rope_freqs(
            10000.0, 64, 8192, RopeScalingType.NONE
        )
        ntk = RoPEScalingOptimizer.compute_rope_freqs(
            10000.0, 64, 8192, RopeScalingType.DYNAMIC_NTK, scaling_factor=4.0
        )
        # NTK should produce different frequencies than base
        assert ntk != base

    def test_llama3_preserves_high_freq(self):
        """Llama-3 scaling should keep high-frequency components unchanged."""
        freqs = RoPEScalingOptimizer.compute_rope_freqs(
            500000.0, 128, 131072, RopeScalingType.LLAMA3, scaling_factor=8.0
        )
        assert len(freqs) == 64

    def test_yarn_produces_valid_freqs(self):
        freqs = RoPEScalingOptimizer.compute_rope_freqs(
            10000.0, 64, 32768, RopeScalingType.YARN, scaling_factor=4.0
        )
        assert len(freqs) == 32
        # All freqs should be positive
        assert all(f > 0 for f in freqs)

    def test_scaling_factor_leq_1_no_change(self):
        base = RoPEScalingOptimizer.compute_rope_freqs(
            10000.0, 64, 8192, RopeScalingType.LINEAR, scaling_factor=1.0
        )
        none = RoPEScalingOptimizer.compute_rope_freqs(
            10000.0, 64, 8192, RopeScalingType.NONE
        )
        assert base == none

    def test_longrope_produces_valid_freqs(self):
        freqs = RoPEScalingOptimizer.compute_rope_freqs(
            10000.0, 64, 65536, RopeScalingType.LONGROPE, scaling_factor=8.0
        )
        assert len(freqs) == 32
        assert all(f > 0 for f in freqs)


class TestRoPEScalingStats:
    def test_stats_unconfigured(self):
        opt = RoPEScalingOptimizer()
        stats = opt.get_stats()
        assert stats["configured"] is False
        assert stats["scaling_type"] == "none"

    def test_stats_after_configure(self):
        config = FakeConfig(
            max_position_embeddings=8192,
            rope_theta=10000.0,
            hidden_size=4096,
            num_attention_heads=32,
            rope_scaling={"type": "linear", "factor": 2.0},
        )
        model = FakeModel(config)
        opt = RoPEScalingOptimizer()
        opt.configure(model, 16384)
        stats = opt.get_stats()
        assert stats["configured"] is True
        assert stats["scaling_type"] == "linear"
        assert stats["original_context"] == 8192
        assert stats["effective_context"] == 16384


# ══════════════════════════════════════════════════════════════════════════════
# AttentionOptimizer tests
# ══════════════════════════════════════════════════════════════════════════════


class TestAttentionDetection:
    def test_mha_detection(self):
        config = FakeConfig(
            num_attention_heads=32,
            num_key_value_heads=32,
            hidden_size=4096,
        )
        model = FakeModel(config)
        opt = AttentionOptimizer()
        cfg = opt.detect_attention_type(model)
        assert cfg.attention_type == AttentionType.MHA

    def test_gqa_detection(self):
        config = FakeConfig(
            num_attention_heads=32,
            num_key_value_heads=4,
            hidden_size=4096,
        )
        model = FakeModel(config)
        opt = AttentionOptimizer()
        cfg = opt.detect_attention_type(model)
        assert cfg.attention_type == AttentionType.GQA
        assert cfg.kv_repeat_needed is True

    def test_mqa_detection(self):
        config = FakeConfig(
            num_attention_heads=32,
            num_key_value_heads=1,
            hidden_size=4096,
        )
        model = FakeModel(config)
        opt = AttentionOptimizer()
        cfg = opt.detect_attention_type(model)
        assert cfg.attention_type == AttentionType.MQA

    def test_mla_detection(self):
        config = FakeConfig(
            num_attention_heads=32,
            num_key_value_heads=4,
            hidden_size=4096,
            kv_lora_rank=512,
        )
        model = FakeModel(config)
        opt = AttentionOptimizer()
        cfg = opt.detect_attention_type(model)
        assert cfg.attention_type == AttentionType.MLA
        assert cfg.mla_mode is True

    def test_swa_detection(self):
        config = FakeConfig(
            num_attention_heads=32,
            num_key_value_heads=32,
            hidden_size=4096,
            sliding_window=4096,
        )
        model = FakeModel(config)
        opt = AttentionOptimizer()
        cfg = opt.detect_attention_type(model)
        assert cfg.sliding_window == 4096

    def test_no_config_returns_default(self):
        model = FakeModel(None)
        opt = AttentionOptimizer()
        cfg = opt.detect_attention_type(model)
        assert cfg.attention_type == AttentionType.MHA

    def test_head_dim_computed(self):
        config = FakeConfig(
            num_attention_heads=28,
            num_key_value_heads=4,
            hidden_size=3584,
        )
        model = FakeModel(config)
        opt = AttentionOptimizer()
        cfg = opt.detect_attention_type(model)
        assert cfg.head_dim == 128


class TestAttentionOptimization:
    def test_gqa_optimization_applied(self):
        model = FakeModel(None)
        opt = AttentionOptimizer()
        cfg = AttentionConfig(
            attention_type=AttentionType.GQA,
            num_heads=32,
            num_kv_heads=4,
            head_dim=128,
        )
        optimizations = opt.optimize_attention(model, cfg)
        assert "gqa_no_kv_expansion" in optimizations
        assert model._yunshu_gqa_mode is True
        assert model._yunshu_gqa_kv_heads == 4

    def test_mha_optimization_applied(self):
        model = FakeModel(None)
        opt = AttentionOptimizer()
        cfg = AttentionConfig(
            attention_type=AttentionType.MHA,
            num_heads=32,
            num_kv_heads=32,
            head_dim=128,
        )
        optimizations = opt.optimize_attention(model, cfg)
        assert "mha_batched_attention" in optimizations
        assert model._yunshu_mha_batched is True

    def test_mqa_optimization_applied(self):
        model = FakeModel(None)
        opt = AttentionOptimizer()
        cfg = AttentionConfig(
            attention_type=AttentionType.MQA,
            num_heads=32,
            num_kv_heads=1,
            head_dim=128,
        )
        optimizations = opt.optimize_attention(model, cfg)
        assert "mqa_single_kv" in optimizations

    def test_mla_optimization_applied(self):
        model = FakeModel(None)
        opt = AttentionOptimizer()
        cfg = AttentionConfig(
            attention_type=AttentionType.MLA,
            num_heads=32,
            num_kv_heads=4,
            head_dim=128,
            mla_mode=True,
        )
        optimizations = opt.optimize_attention(model, cfg)
        assert "mla_latent_cache" in optimizations

    def test_swa_optimization_applied(self):
        model = FakeModel(None)
        opt = AttentionOptimizer()
        cfg = AttentionConfig(
            attention_type=AttentionType.MHA,
            num_heads=32,
            num_kv_heads=32,
            head_dim=128,
            sliding_window=4096,
        )
        optimizations = opt.optimize_attention(model, cfg)
        assert "swa_token_skip" in optimizations
        assert model._yunshu_swa_window == 4096

    def test_optimize_uses_detected_config(self):
        """optimize_attention uses the internally detected config if none passed."""
        config = FakeConfig(
            num_attention_heads=32,
            num_key_value_heads=4,
            hidden_size=4096,
        )
        model = FakeModel(config)
        opt = AttentionOptimizer()
        opt.detect_attention_type(model)
        optimizations = opt.optimize_attention(model)
        assert "gqa_no_kv_expansion" in optimizations


class TestAttentionStats:
    def test_stats_unconfigured(self):
        opt = AttentionOptimizer()
        stats = opt.get_stats()
        assert stats["configured"] is False

    def test_stats_after_optimize(self):
        model = FakeModel(None)
        opt = AttentionOptimizer()
        cfg = AttentionConfig(
            attention_type=AttentionType.GQA,
            num_heads=32,
            num_kv_heads=4,
            head_dim=128,
        )
        opt.optimize_attention(model, cfg)
        stats = opt.get_stats()
        assert stats["configured"] is True
        assert stats["attention_type"] == "gqa"
        assert "gqa_no_kv_expansion" in stats["optimizations"]


# ══════════════════════════════════════════════════════════════════════════════
# MoEEfficiencyOptimizer tests
# ══════════════════════════════════════════════════════════════════════════════


class TestMoEConfigure:
    def test_basic_configure(self):
        model = FakeModel(None)
        opt = MoEEfficiencyOptimizer()
        result = opt.configure(model, 8, 2)
        assert result["num_experts"] == 8
        assert result["top_k"] == 2

    def test_top_k_capped_at_num_experts(self):
        model = FakeModel(None)
        opt = MoEEfficiencyOptimizer()
        result = opt.configure(model, 4, 8)
        assert result["top_k"] == 4  # capped

    def test_model_hints_set(self):
        model = FakeModel(None)
        opt = MoEEfficiencyOptimizer()
        opt.configure(model, 64, 8)
        assert model._yunshu_moe_optimized is True
        assert model._yunshu_moe_num_experts == 64
        assert model._yunshu_moe_top_k == 8

    def test_cache_experts_initialized(self):
        model = FakeModel(None)
        opt = MoEEfficiencyOptimizer()
        result = opt.configure(model, 8, 2)
        # top_k * 2 = 4 cached experts
        assert result["cached_experts"] == 4


class TestMoEExpertSelection:
    def test_select_experts_basic(self):
        model = FakeModel(None)
        opt = MoEEfficiencyOptimizer()
        opt.configure(model, 4, 2)

        # 2 tokens, 4 experts each — use peaked logits to avoid dynamic_k bump
        logits = [
            [0.0, 0.0, 10.0, 9.0],
            [1.0, 10.0, 0.0, 9.0],
        ]
        result = opt.select_experts(None, logits)
        assert len(result) == 2
        for indices, weights in result:
            assert len(indices) == 2
            assert len(weights) == 2
            assert abs(sum(weights) - 1.0) < 1e-6

    def test_select_experts_empty_logits(self):
        model = FakeModel(None)
        opt = MoEEfficiencyOptimizer()
        opt.configure(model, 4, 2)
        result = opt.select_experts(None, [])
        assert result == []

    def test_select_experts_unconfigured(self):
        opt = MoEEfficiencyOptimizer()
        result = opt.select_experts(None, [[1.0, 2.0]])
        assert result == []

    def test_dynamic_top_k_for_uncertain_tokens(self):
        model = FakeModel(None)
        opt = MoEEfficiencyOptimizer()
        opt.configure(model, 8, 2)

        # Uniform logits = high entropy = uncertain token
        uniform_logits = [[1.0] * 8]
        result = opt.select_experts(None, uniform_logits)
        indices, weights = result[0]
        # With high uncertainty, dynamic_k might bump to 3
        assert len(indices) >= 2

    def test_expert_counts_updated(self):
        model = FakeModel(None)
        opt = MoEEfficiencyOptimizer()
        opt.configure(model, 4, 2)

        logits = [[0.0, 0.0, 10.0, 9.0]]
        opt.select_experts(None, logits)
        stats = opt.get_stats()
        assert stats["total_tokens"] == 1
        assert sum(stats["expert_utilization"].values()) == 2  # top_k=2

    def test_cache_tracking(self):
        model = FakeModel(None)
        opt = MoEEfficiencyOptimizer()
        opt.configure(model, 4, 2)

        logits = [[0.0, 0.0, 10.0, 9.0]]
        opt.select_experts(None, logits)
        stats = opt.get_stats()
        # Experts 0, 1, 2, 3 are all cached (top_k*2=4)
        assert stats["cache_hits"] > 0


class TestMoELoadBalance:
    def test_perfect_balance(self):
        opt = MoEEfficiencyOptimizer()
        model = FakeModel(None)
        opt.configure(model, 4, 2)

        # Feed many tokens to distribute evenly
        for _ in range(100):
            opt.select_experts(None, [[1.0, 2.0, 3.0, 4.0]])

        score = opt.get_load_balance_score()
        # Top-2 always selects experts 3,2 -> not perfectly balanced
        assert 0.0 <= score <= 1.0

    def test_zero_tokens_score(self):
        opt = MoEEfficiencyOptimizer()
        score = opt.get_load_balance_score()
        assert score == 0.0


class TestMoEStats:
    def test_stats_unconfigured(self):
        opt = MoEEfficiencyOptimizer()
        stats = opt.get_stats()
        assert stats["configured"] is False
        assert stats["cache_hit_rate"] == 0.0

    def test_stats_after_selection(self):
        model = FakeModel(None)
        opt = MoEEfficiencyOptimizer()
        opt.configure(model, 4, 2)
        opt.select_experts(None, [[0.0, 0.0, 10.0, 9.0]])
        stats = opt.get_stats()
        assert stats["configured"] is True
        assert stats["total_tokens"] == 1
        assert stats["num_experts"] == 4
        assert stats["top_k"] == 2
        assert 0.0 <= stats["load_balance_score"] <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# ModelWarmupManager tests
# ══════════════════════════════════════════════════════════════════════════════


class TestModelWarmup:
    def test_basic_warmup(self):
        model = FakeModel(None)
        mgr = ModelWarmupManager()
        result = mgr.warmup(model, "generic")
        assert isinstance(result, WarmupResult)
        assert result.warmup_time_s >= 0
        assert result.model_type == "generic"

    def test_warmup_compile(self):
        model = FakeModel(None)
        mgr = ModelWarmupManager()
        cached = mgr.warmup_compile(model)
        assert isinstance(cached, bool)

    def test_warmup_kv_cache(self):
        model = FakeModel(None)
        mgr = ModelWarmupManager()
        count = mgr.warmup_kv_cache(model, ["Hello", "World"])
        assert count == 2

    def test_warmup_with_model_family(self):
        model = FakeModel(None)
        mgr = ModelWarmupManager()
        result = mgr.warmup(model, "qwen")
        assert result.prompts_warmed > 0

    def test_warmup_llama(self):
        model = FakeModel(None)
        mgr = ModelWarmupManager()
        result = mgr.warmup(model, "llama")
        assert result.prompts_warmed > 0

    def test_warmup_deepseek(self):
        model = FakeModel(None)
        mgr = ModelWarmupManager()
        result = mgr.warmup(model, "deepseek")
        assert result.prompts_warmed > 0

    def test_warmup_gemma(self):
        model = FakeModel(None)
        mgr = ModelWarmupManager()
        result = mgr.warmup(model, "gemma")
        assert result.prompts_warmed > 0

    def test_warmup_unknown_family_uses_generic(self):
        model = FakeModel(None)
        mgr = ModelWarmupManager()
        result = mgr.warmup(model, "unknown_model_xyz")
        assert result.prompts_warmed > 0

    def test_warmup_without_compile(self):
        model = FakeModel(None)
        mgr = ModelWarmupManager()
        result = mgr.warmup(model, "generic", compile=False)
        assert result.steps_run >= 0


class TestWarmupStats:
    def test_stats_unwarmed(self):
        mgr = ModelWarmupManager()
        stats = mgr.get_stats()
        assert stats["warmed"] is False

    def test_stats_after_warmup(self):
        model = FakeModel(None)
        mgr = ModelWarmupManager()
        mgr.warmup(model, "qwen")
        stats = mgr.get_stats()
        assert stats["warmed"] is True
        assert stats["model_type"] == "qwen"
        assert stats["prompts_warmed"] > 0


# ══════════════════════════════════════════════════════════════════════════════
# Helper tests
# ══════════════════════════════════════════════════════════════════════════════


class TestHelpers:
    def test_safe_int_valid(self):
        assert _safe_int(42, 0) == 42
        assert _safe_int("10", 0) == 10

    def test_safe_int_invalid(self):
        assert _safe_int(None, 5) == 5
        assert _safe_int("abc", 3) == 3

    def test_get_config_direct(self):
        config = FakeConfig(hidden_size=4096)
        model = FakeModel(config)
        assert _get_config(model) is config

    def test_get_config_via_args(self):
        config = FakeConfig(hidden_size=4096)
        model = MagicMock()
        model.config = None
        model.args = config
        assert _get_config(model) is config

    def test_get_config_nested(self):
        config = FakeConfig(hidden_size=4096)
        inner = MagicMock()
        inner.config = config
        model = MagicMock()
        model.config = None
        model.args = None
        model.model = inner
        assert _get_config(model) is config

    def test_get_config_none(self):
        assert _get_config(None) is None
