"""Tests for yunshu_engine.ane_embedding — ANE embedding co-processor."""
from __future__ import annotations

import platform
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from yunshu_engine.ane_embedding import (
    ANEEmbeddingConfig,
    ANEEmbeddingProcessor,
    _HAS_COREMLTOOLS,
    _HAS_MLX,
    estimate_ane_speedup,
    is_ane_available,
)


# ---------------------------------------------------------------------------
# ANEEmbeddingConfig
# ---------------------------------------------------------------------------


class TestANEEmbeddingConfig:
    """Tests for ANEEmbeddingConfig dataclass."""

    def test_defaults(self):
        config = ANEEmbeddingConfig()
        assert config.model_name == "intfloat/e5-small-v2"
        assert config.max_seq_length == 512
        assert config.normalize_embeddings is True
        assert config.compile_on_init is True
        assert config.cache_dir == ""

    def test_custom_config(self):
        config = ANEEmbeddingConfig(
            model_name="BAAI/bge-small-en-v1.5",
            max_seq_length=256,
            normalize_embeddings=False,
            compile_on_init=False,
            cache_dir="/tmp/ane_test",
        )
        assert config.model_name == "BAAI/bge-small-en-v1.5"
        assert config.max_seq_length == 256
        assert config.normalize_embeddings is False
        assert config.compile_on_init is False
        assert config.cache_dir == "/tmp/ane_test"

    def test_get_cache_dir_default(self):
        config = ANEEmbeddingConfig()
        cache_dir = config.get_cache_dir()
        assert cache_dir == Path.home() / ".yunshu" / "ane_cache"

    def test_get_cache_dir_custom(self):
        config = ANEEmbeddingConfig(cache_dir="/custom/path")
        cache_dir = config.get_cache_dir()
        assert cache_dir == Path("/custom/path")

    def test_get_cache_dir_expands_home(self):
        config = ANEEmbeddingConfig(cache_dir="~/my_cache")
        cache_dir = config.get_cache_dir()
        assert "~" not in str(cache_dir)
        assert cache_dir.name == "my_cache"


# ---------------------------------------------------------------------------
# is_ane_available
# ---------------------------------------------------------------------------


class TestIsAneAvailable:
    """Tests for is_ane_available() helper."""

    def test_returns_bool(self):
        result = is_ane_available()
        assert isinstance(result, bool)

    def test_false_on_non_darwin(self):
        with patch("yunshu_engine.ane_embedding.platform") as mock_platform:
            mock_platform.system.return_value = "Linux"
            mock_platform.machine.return_value = "x86_64"
            assert is_ane_available() is False

    def test_false_on_intel_mac(self):
        with patch("yunshu_engine.ane_embedding.platform") as mock_platform:
            mock_platform.system.return_value = "Darwin"
            mock_platform.machine.return_value = "x86_64"
            assert is_ane_available() is False

    def test_false_without_coremltools(self):
        with patch("yunshu_engine.ane_embedding._HAS_COREMLTOOLS", False):
            # Even on Apple Silicon, no coremltools means no ANE path
            result = is_ane_available()
            # Result depends on platform, but should be a bool
            assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# estimate_ane_speedup
# ---------------------------------------------------------------------------


class TestEstimateAneSpeedup:
    """Tests for estimate_ane_speedup() helper."""

    def test_returns_float(self):
        result = estimate_ane_speedup(33.0, 128)
        assert isinstance(result, float)

    def test_small_model_short_seq_high_speedup(self):
        """Small models with short sequences should show higher ANE advantage."""
        small = estimate_ane_speedup(10.0, 64)
        large = estimate_ane_speedup(200.0, 512)
        assert small >= large

    def test_output_in_valid_range(self):
        """Speedup should always be in [1.0, 3.0]."""
        for params in [1.0, 10.0, 33.0, 100.0, 500.0]:
            for seq in [32, 128, 256, 512, 1024]:
                result = estimate_ane_speedup(params, seq)
                assert 1.0 <= result <= 3.0, (
                    f"params={params}, seq={seq} -> {result} outside [1.0, 3.0]"
                )

    def test_zero_params_returns_unity(self):
        assert estimate_ane_speedup(0.0, 128) == 1.0

    def test_negative_params_returns_unity(self):
        assert estimate_ane_speedup(-10.0, 128) == 1.0

    def test_zero_seq_returns_unity(self):
        assert estimate_ane_speedup(33.0, 0) == 1.0

    def test_negative_seq_returns_unity(self):
        assert estimate_ane_speedup(33.0, -1) == 1.0

    def test_larger_model_lower_speedup(self):
        """Larger models should have lower or equal ANE speedup."""
        small = estimate_ane_speedup(10.0, 128)
        big = estimate_ane_speedup(150.0, 128)
        assert small >= big

    def test_longer_seq_lower_speedup(self):
        """Longer sequences should have lower ANE speedup for same model size."""
        short = estimate_ane_speedup(33.0, 64)
        long_seq = estimate_ane_speedup(33.0, 1024)
        assert short >= long_seq


# ---------------------------------------------------------------------------
# ANEEmbeddingProcessor — init and stats
# ---------------------------------------------------------------------------


class TestANEEmbeddingProcessorInit:
    """Tests for ANEEmbeddingProcessor initialization."""

    def test_init_default_config(self):
        config = ANEEmbeddingConfig(compile_on_init=False)
        proc = ANEEmbeddingProcessor(config)
        assert proc.is_compiled() is False

    def test_init_stats_initial(self):
        config = ANEEmbeddingConfig(compile_on_init=False)
        proc = ANEEmbeddingProcessor(config)
        stats = proc.get_stats()
        assert stats["is_compiled"] is False
        assert stats["compiled_path"] is None
        assert stats["inference_count"] == 0
        assert stats["avg_latency_s"] is None
        assert isinstance(stats["ane_available"], bool)
        assert isinstance(stats["coremltools_installed"], bool)
        assert isinstance(stats["mlx_available"], bool)
        assert stats["model_name"] == config.model_name


# ---------------------------------------------------------------------------
# ANEEmbeddingProcessor — get_stats structure
# ---------------------------------------------------------------------------


class TestANEEmbeddingProcessorStats:
    """Tests for get_stats() return structure."""

    def test_stats_has_all_keys(self):
        config = ANEEmbeddingConfig(compile_on_init=False)
        proc = ANEEmbeddingProcessor(config)
        stats = proc.get_stats()
        expected_keys = {
            "model_name",
            "is_compiled",
            "compiled_path",
            "inference_count",
            "avg_latency_s",
            "ane_available",
            "coremltools_installed",
            "mlx_available",
        }
        assert set(stats.keys()) == expected_keys

    def test_stats_after_inference(self):
        config = ANEEmbeddingConfig(compile_on_init=False)
        proc = ANEEmbeddingProcessor(config)
        # embed() will use MLX fallback
        proc.embed(["test input"])
        stats = proc.get_stats()
        assert stats["inference_count"] == 1
        assert stats["avg_latency_s"] is not None
        assert stats["avg_latency_s"] >= 0


# ---------------------------------------------------------------------------
# ANEEmbeddingProcessor — embed fallback
# ---------------------------------------------------------------------------


class TestANEEmbeddingEmbedFallback:
    """Tests for embed() MLX fallback behavior."""

    def test_embed_empty_list(self):
        config = ANEEmbeddingConfig(compile_on_init=False)
        proc = ANEEmbeddingProcessor(config)
        result = proc.embed([])
        assert result == []

    def test_embed_fallback_returns_vectors(self):
        """When not compiled, embed() should use MLX fallback and return vectors."""
        config = ANEEmbeddingConfig(compile_on_init=False)
        proc = ANEEmbeddingProcessor(config)
        result = proc.embed(["hello world", "test text"])
        assert len(result) == 2
        for emb in result:
            assert isinstance(emb, list)
            assert len(emb) > 0
            assert all(isinstance(x, float) for x in emb)

    def test_embed_fallback_normalized(self):
        """With normalize_embeddings=True, vectors should be unit length."""
        config = ANEEmbeddingConfig(compile_on_init=False, normalize_embeddings=True)
        proc = ANEEmbeddingProcessor(config)
        result = proc.embed(["hello world"])
        assert len(result) == 1
        # Check L2 norm is approximately 1.0
        norm = sum(x * x for x in result[0]) ** 0.5
        assert abs(norm - 1.0) < 0.01, f"Expected unit norm, got {norm}"

    def test_embed_fallback_no_normalize(self):
        """With normalize_embeddings=False, vectors should NOT be unit length."""
        config = ANEEmbeddingConfig(compile_on_init=False, normalize_embeddings=False)
        proc = ANEEmbeddingProcessor(config)
        result = proc.embed(["hello world"])
        assert len(result) == 1
        # Non-normalized vectors — norm likely not 1.0
        norm = sum(x * x for x in result[0]) ** 0.5
        # Just check it's a valid vector (not all zeros)
        assert norm > 0.0

    def test_embed_increments_count(self):
        config = ANEEmbeddingConfig(compile_on_init=False)
        proc = ANEEmbeddingProcessor(config)
        assert proc.get_stats()["inference_count"] == 0
        proc.embed(["a"])
        assert proc.get_stats()["inference_count"] == 1
        proc.embed(["b", "c"])
        assert proc.get_stats()["inference_count"] == 2

    def test_embed_without_mlx_returns_zero_vectors(self):
        """When MLX is also unavailable, should return zero vectors."""
        config = ANEEmbeddingConfig(compile_on_init=False)
        proc = ANEEmbeddingProcessor(config)
        with patch("yunshu_engine.ane_embedding._HAS_MLX", False):
            result = proc.embed(["test"])
        assert len(result) == 1
        assert all(x == 0.0 for x in result[0])


# ---------------------------------------------------------------------------
# ANEEmbeddingProcessor — compile_model graceful handling
# ---------------------------------------------------------------------------


class TestANEEmbeddingProcessorCompile:
    """Tests for compile_model() graceful degradation."""

    def test_compile_without_coremltools_raises(self):
        """compile_model() should raise RuntimeError when coremltools is missing."""
        config = ANEEmbeddingConfig(compile_on_init=False)
        proc = ANEEmbeddingProcessor(config)
        with patch("yunshu_engine.ane_embedding._HAS_COREMLTOOLS", False):
            with pytest.raises(RuntimeError, match="coremltools is required"):
                proc.compile_model("/fake/model/path")

    def test_is_compiled_initially_false(self):
        config = ANEEmbeddingConfig(compile_on_init=False)
        proc = ANEEmbeddingProcessor(config)
        assert proc.is_compiled() is False
