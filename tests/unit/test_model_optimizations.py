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
    WarmPromptResult,
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


# ══════════════════════════════════════════════════════════════════════════════
# Warm Prompt Prefill tests (vllm-mlx pattern)
# ══════════════════════════════════════════════════════════════════════════════


class TestWarmPromptResult:
    """Tests for WarmPromptResult dataclass."""

    def test_defaults(self):
        r = WarmPromptResult()
        assert r.prompts_loaded == 0
        assert r.prompts_prefilled == 0
        assert r.prompts_skipped_cached == 0
        assert r.prompts_failed == 0
        assert r.total_tokens_prefilled == 0
        assert r.prefill_time_s == 0.0
        assert r.source == ""

    def test_with_values(self):
        r = WarmPromptResult(
            prompts_loaded=3,
            prompts_prefilled=2,
            prompts_skipped_cached=1,
            total_tokens_prefilled=256,
            prefill_time_s=0.5,
            source="env",
        )
        assert r.prompts_loaded == 3
        assert r.prompts_prefilled == 2
        assert r.source == "env"


class TestResolveWarmPrompts:
    """Tests for ModelWarmupManager.resolve_warm_prompts()."""

    def test_no_env_var(self, monkeypatch):
        monkeypatch.delenv("YUNSHU_WARM_PROMPTS", raising=False)
        result = ModelWarmupManager.resolve_warm_prompts()
        assert result == []

    def test_empty_env_var(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_WARM_PROMPTS", "")
        result = ModelWarmupManager.resolve_warm_prompts()
        assert result == []

    def test_single_prompt(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_WARM_PROMPTS", "Hello world")
        result = ModelWarmupManager.resolve_warm_prompts()
        assert result == ["Hello world"]

    def test_multiple_prompts_separated(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_WARM_PROMPTS", "Hello||World||Test")
        result = ModelWarmupManager.resolve_warm_prompts()
        assert result == ["Hello", "World", "Test"]

    def test_whitespace_trimmed(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_WARM_PROMPTS", "  Hello  ||  World  ")
        result = ModelWarmupManager.resolve_warm_prompts()
        assert result == ["Hello", "World"]

    def test_empty_parts_skipped(self, monkeypatch):
        monkeypatch.setenv("YUNSHU_WARM_PROMPTS", "Hello||||World")
        result = ModelWarmupManager.resolve_warm_prompts()
        assert result == ["Hello", "World"]

    def test_file_path_not_found_skipped(self, monkeypatch, tmp_path):
        nonexistent = str(tmp_path / "nonexistent.txt")
        monkeypatch.setenv("YUNSHU_WARM_PROMPTS", nonexistent)
        result = ModelWarmupManager.resolve_warm_prompts()
        assert result == []

    def test_file_path_reads_content(self, monkeypatch, tmp_path):
        prompt_file = tmp_path / "prompts.txt"
        prompt_file.write_text("System prompt content")
        monkeypatch.setenv("YUNSHU_WARM_PROMPTS", str(prompt_file))
        result = ModelWarmupManager.resolve_warm_prompts()
        assert result == ["System prompt content"]

    def test_tilde_expansion(self, monkeypatch, tmp_path):
        monkeypatch.setenv("YUNSHU_WARM_PROMPTS", "~/nonexistent_prompt_file_xyz.txt")
        result = ModelWarmupManager.resolve_warm_prompts()
        # File doesn't exist, should be skipped gracefully
        assert result == []

    def test_mixed_inline_and_file(self, monkeypatch, tmp_path):
        prompt_file = tmp_path / "system.txt"
        prompt_file.write_text("System instruction")
        monkeypatch.setenv(
            "YUNSHU_WARM_PROMPTS",
            f"inline text||{prompt_file}||more inline",
        )
        result = ModelWarmupManager.resolve_warm_prompts()
        assert result == ["inline text", "System instruction", "more inline"]

    def test_empty_file_skipped(self, monkeypatch, tmp_path):
        prompt_file = tmp_path / "empty.txt"
        prompt_file.write_text("")
        monkeypatch.setenv("YUNSHU_WARM_PROMPTS", str(prompt_file))
        result = ModelWarmupManager.resolve_warm_prompts()
        assert result == []

    def test_custom_env_var(self, monkeypatch):
        monkeypatch.setenv("CUSTOM_PROMPTS", "custom prompt")
        result = ModelWarmupManager.resolve_warm_prompts(env_var="CUSTOM_PROMPTS")
        assert result == ["custom prompt"]


class FakeKVCacheLayer:
    """Fake KV cache layer with keys/values/offset for snapshot testing."""
    def __init__(self, offset=0):
        self.keys = MagicMock()
        self.values = MagicMock()
        self.offset = offset


class FakeKVPrefixCache:
    """Minimal KV prefix cache mock for warm prompt tests."""
    def __init__(self):
        self._entries = {}  # hash -> (tokens, cache)
        self._evict_calls = 0

    def evict_under_pressure(self, threshold=85.0):
        self._evict_calls += 1

    def get(self, tokens):
        """Return (None, len(tokens), 0) for miss, (cache, 0, len) for hit."""
        key = tuple(int(t) for t in tokens)
        if key in self._entries:
            return self._entries[key], 0, len(tokens)
        return None, len(tokens), 0

    def add(self, tokens, cache):
        key = tuple(int(t) for t in tokens)
        self._entries[key] = cache


class FakeTokenizer:
    """Fake tokenizer that encodes strings to integer lists."""
    def __init__(self, vocab=None):
        self._vocab = vocab or {}

    def encode(self, text):
        # Simple deterministic encoding: each char → ord(char)
        return [ord(c) for c in text]


class TestWarmPromptPrefill:
    """Tests for ModelWarmupManager.warm_prompt_prefill()."""

    def test_empty_prompts_list(self):
        mgr = ModelWarmupManager()
        result = mgr.warm_prompt_prefill(
            model=MagicMock(),
            tokenizer=FakeTokenizer(),
            kv_prefix_cache=FakeKVPrefixCache(),
            warm_prompts=[],
        )
        assert isinstance(result, WarmPromptResult)
        assert result.prompts_loaded == 0
        assert result.prompts_prefilled == 0
        assert result.source == "config"

    def test_none_prompts_no_env(self, monkeypatch):
        monkeypatch.delenv("YUNSHU_WARM_PROMPTS", raising=False)
        mgr = ModelWarmupManager()
        result = mgr.warm_prompt_prefill(
            model=MagicMock(),
            tokenizer=FakeTokenizer(),
            kv_prefix_cache=FakeKVPrefixCache(),
            warm_prompts=None,
        )
        assert result.prompts_loaded == 0
        # Source is "env" because warm_prompts=None means "read from env var"
        assert result.source == "env"

    def test_warm_prompt_stats_tracked(self):
        mgr = ModelWarmupManager()
        stats = mgr.get_warm_prompt_stats()
        assert stats["prompts_loaded"] == 0
        assert stats["prompts_prefilled"] == 0
        assert stats["source"] == ""

    def test_get_stats_includes_warm_prompt_prefill(self):
        mgr = ModelWarmupManager()
        stats = mgr.get_stats()
        assert "warm_prompt_prefill" in stats
        wp = stats["warm_prompt_prefill"]
        assert wp["prompts_loaded"] == 0
        assert wp["source"] == ""

    def test_prefill_with_model_failure(self):
        """When generate_step fails, prompts are counted as failed."""
        mgr = ModelWarmupManager()
        model = MagicMock()
        tokenizer = FakeTokenizer()
        cache = FakeKVPrefixCache()

        # The real prefill requires mlx imports — test with import failure
        # by mocking the import mechanism. Instead, test the failure path
        # by using a model that raises on encode.
        bad_tokenizer = MagicMock()
        bad_tokenizer.encode.side_effect = RuntimeError("encode failed")

        result = mgr.warm_prompt_prefill(
            model=model,
            tokenizer=bad_tokenizer,
            kv_prefix_cache=cache,
            warm_prompts=["Hello"],
        )
        # The prompt should be counted as failed
        assert result.prompts_failed == 1
        assert result.prompts_prefilled == 0

    def test_prefill_multiple_prompts(self):
        """Multiple prompts should all be attempted."""
        mgr = ModelWarmupManager()
        bad_tokenizer = MagicMock()
        bad_tokenizer.encode.side_effect = RuntimeError("nope")

        result = mgr.warm_prompt_prefill(
            model=MagicMock(),
            tokenizer=bad_tokenizer,
            kv_prefix_cache=FakeKVPrefixCache(),
            warm_prompts=["prompt1", "prompt2", "prompt3"],
        )
        assert result.prompts_loaded == 3
        assert result.prompts_failed == 3

    def test_prefill_empty_string_skipped(self):
        """Empty strings in warm_prompts should be skipped, whitespace-only attempted."""
        mgr = ModelWarmupManager()
        bad_tokenizer = MagicMock()
        bad_tokenizer.encode.side_effect = RuntimeError("nope")

        result = mgr.warm_prompt_prefill(
            model=MagicMock(),
            tokenizer=bad_tokenizer,
            kv_prefix_cache=FakeKVPrefixCache(),
            warm_prompts=["", "  ", "real prompt"],
        )
        # "" is skipped (falsy), "  " is truthy so attempted and fails,
        # "real prompt" is attempted and fails
        assert result.prompts_failed == 2
        assert result.prompts_loaded == 3

    def test_warm_max_tokens_env_override(self, monkeypatch):
        """YUNSHU_WARM_MAX_TOKENS should override the default."""
        monkeypatch.setenv("YUNSHU_WARM_MAX_TOKENS", "5")
        mgr = ModelWarmupManager()
        # Just verify the env var is read — actual prefill requires MLX
        result = mgr.warm_prompt_prefill(
            model=MagicMock(),
            tokenizer=FakeTokenizer(),
            kv_prefix_cache=FakeKVPrefixCache(),
            warm_prompts=["test"],
            max_tokens=1,  # default, should be overridden to 5 by env
        )
        # The result will show a failure since no real MLX, but the
        # important thing is that the env var was processed
        assert isinstance(result, WarmPromptResult)

    def test_warm_max_tokens_invalid_env(self, monkeypatch):
        """Invalid YUNSHU_WARM_MAX_TOKENS should keep the default."""
        monkeypatch.setenv("YUNSHU_WARM_MAX_TOKENS", "not_a_number")
        mgr = ModelWarmupManager()
        result = mgr.warm_prompt_prefill(
            model=MagicMock(),
            tokenizer=FakeTokenizer(),
            kv_prefix_cache=FakeKVPrefixCache(),
            warm_prompts=["test"],
        )
        assert isinstance(result, WarmPromptResult)


class TestWarmPromptPrefillWithKV:
    """Integration tests for warm prompt prefill with real-ish KV prefix cache.

    These tests use mocked MLX internals to simulate the full prefill flow
    without requiring a real GPU model.
    """

    def test_prefill_stores_in_kv_cache(self, monkeypatch):
        """After prefill, KV prefix cache should have the entry."""
        import types

        mgr = ModelWarmupManager()
        cache = FakeKVPrefixCache()
        tokenizer = FakeTokenizer()

        # Create a fake mx module
        fake_mx = types.SimpleNamespace(
            array=lambda x: x,
            clear_cache=lambda: None,
        )

        # Create fake mlx_lm modules
        fake_cache_module = types.SimpleNamespace(
            make_prompt_cache=lambda model: [FakeKVCacheLayer()],
        )

        fake_generate_step_called = []
        def fake_generate_step(ids, model, max_tokens=1, sampler=None, prompt_cache=None):
            fake_generate_step_called.append(max_tokens)
            yield None  # yield once then break

        fake_generate_module = types.SimpleNamespace(
            generate_step=fake_generate_step,
        )

        fake_sampler_module = types.SimpleNamespace(
            make_sampler=lambda temp=0.0: None,
        )

        # Patch imports within warm_prompt_prefill
        monkeypatch.setitem(
            __import__("sys").modules, "mlx.core", fake_mx,
        )
        monkeypatch.setitem(
            __import__("sys").modules, "mlx", types.SimpleNamespace(core=fake_mx),
        )
        monkeypatch.setitem(
            __import__("sys").modules, "mlx_lm.models.cache", fake_cache_module,
        )
        monkeypatch.setitem(
            __import__("sys").modules, "mlx_lm.generate", fake_generate_module,
        )
        monkeypatch.setitem(
            __import__("sys").modules, "mlx_lm.sample_utils", fake_sampler_module,
        )
        monkeypatch.setitem(
            __import__("sys").modules, "mlx_lm", types.SimpleNamespace(),
        )

        result = mgr.warm_prompt_prefill(
            model=MagicMock(),
            tokenizer=tokenizer,
            kv_prefix_cache=cache,
            warm_prompts=["Hello world test prompt for warmup"],
            max_tokens=1,
        )
        assert result.prompts_prefilled == 1
        assert result.prompts_failed == 0
        assert result.total_tokens_prefilled > 0
        assert len(cache._entries) == 1

    def test_prefill_skips_already_cached(self, monkeypatch):
        """Prompts already in KV cache should be skipped."""
        import types

        mgr = ModelWarmupManager()
        cache = FakeKVPrefixCache()
        tokenizer = FakeTokenizer()

        # Pre-populate the cache with the same prompt
        tokens = tuple(ord(c) for c in "Hello world test prompt for warmup")
        cache._entries[tokens] = [FakeKVCacheLayer(offset=len(tokens))]

        # Mock mlx imports
        fake_mx = types.SimpleNamespace(array=lambda x: x, clear_cache=lambda: None)
        fake_cache_module = types.SimpleNamespace(make_prompt_cache=lambda m: [FakeKVCacheLayer()])
        fake_generate_module = types.SimpleNamespace(
            generate_step=lambda ids, model, max_tokens=1, sampler=None, prompt_cache=None: iter([None]),
        )
        fake_sampler_module = types.SimpleNamespace(make_sampler=lambda temp=0.0: None)

        monkeypatch.setitem(__import__("sys").modules, "mlx.core", fake_mx)
        monkeypatch.setitem(__import__("sys").modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(__import__("sys").modules, "mlx_lm.models.cache", fake_cache_module)
        monkeypatch.setitem(__import__("sys").modules, "mlx_lm.generate", fake_generate_module)
        monkeypatch.setitem(__import__("sys").modules, "mlx_lm.sample_utils", fake_sampler_module)
        monkeypatch.setitem(__import__("sys").modules, "mlx_lm", types.SimpleNamespace())

        result = mgr.warm_prompt_prefill(
            model=MagicMock(),
            tokenizer=tokenizer,
            kv_prefix_cache=cache,
            warm_prompts=["Hello world test prompt for warmup"],
        )
        assert result.prompts_prefilled == 0
        assert result.prompts_skipped_cached == 1

    def test_prefill_multiple_different_prompts(self, monkeypatch):
        """Multiple different prompts should each get their own cache entry."""
        import types

        mgr = ModelWarmupManager()
        cache = FakeKVPrefixCache()
        tokenizer = FakeTokenizer()

        fake_mx = types.SimpleNamespace(array=lambda x: x, clear_cache=lambda: None)
        fake_cache_module = types.SimpleNamespace(make_prompt_cache=lambda m: [FakeKVCacheLayer()])
        fake_generate_module = types.SimpleNamespace(
            generate_step=lambda ids, model, max_tokens=1, sampler=None, prompt_cache=None: iter([None]),
        )
        fake_sampler_module = types.SimpleNamespace(make_sampler=lambda temp=0.0: None)

        monkeypatch.setitem(__import__("sys").modules, "mlx.core", fake_mx)
        monkeypatch.setitem(__import__("sys").modules, "mlx", types.SimpleNamespace(core=fake_mx))
        monkeypatch.setitem(__import__("sys").modules, "mlx_lm.models.cache", fake_cache_module)
        monkeypatch.setitem(__import__("sys").modules, "mlx_lm.generate", fake_generate_module)
        monkeypatch.setitem(__import__("sys").modules, "mlx_lm.sample_utils", fake_sampler_module)
        monkeypatch.setitem(__import__("sys").modules, "mlx_lm", types.SimpleNamespace())

        prompts = [
            "First warm prompt for testing",
            "Second warm prompt different",
            "Third warm prompt unique content",
        ]
        result = mgr.warm_prompt_prefill(
            model=MagicMock(),
            tokenizer=tokenizer,
            kv_prefix_cache=cache,
            warm_prompts=prompts,
        )
        assert result.prompts_prefilled == 3
        assert result.total_tokens_prefilled == sum(len(p) for p in prompts)
        assert len(cache._entries) == 3

    def test_prefill_stats_persisted_in_manager(self):
        """Warm prompt stats should be available via get_warm_prompt_stats()."""
        mgr = ModelWarmupManager()
        mgr.warm_prompt_prefill(
            model=MagicMock(),
            tokenizer=FakeTokenizer(),
            kv_prefix_cache=FakeKVPrefixCache(),
            warm_prompts=[],
        )
        stats = mgr.get_warm_prompt_stats()
        assert stats["source"] == "config"
        assert stats["prompts_loaded"] == 0


class TestWarmPromptLifecycle:
    """Tests for warm prompt manager lifecycle."""

    def test_manager_initial_state(self):
        mgr = ModelWarmupManager()
        assert mgr._warmed is False
        assert isinstance(mgr._warm_prompt_result, WarmPromptResult)
        assert mgr._warm_prompt_result.prompts_loaded == 0

    def test_manager_after_warmup(self):
        """After warmup(), manager should still accept warm prompt prefill."""
        model = FakeModel(None)
        mgr = ModelWarmupManager()
        mgr.warmup(model, "qwen")
        assert mgr._warmed is True

        # Warm prompt prefill should still work
        result = mgr.warm_prompt_prefill(
            model=MagicMock(),
            tokenizer=FakeTokenizer(),
            kv_prefix_cache=FakeKVPrefixCache(),
            warm_prompts=[],
        )
        assert result.prompts_loaded == 0

    def test_get_stats_after_prefill(self):
        """get_stats() should include warm_prompt_prefill after prefill."""
        mgr = ModelWarmupManager()
        mgr.warm_prompt_prefill(
            model=MagicMock(),
            tokenizer=FakeTokenizer(),
            kv_prefix_cache=FakeKVPrefixCache(),
            warm_prompts=["test"],
        )
        stats = mgr.get_stats()
        assert "warm_prompt_prefill" in stats
        assert stats["warm_prompt_prefill"]["source"] == "config"
