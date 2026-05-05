"""Tests for ANE Drafter Path B — draft model compilation and execution.

Phase 4 tests:
- compile_drafter_model function
- draft_token function (ANE path and GPU fallback)
- Edge cases: no CoreML, no MLX, empty inputs
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from yunshu_engine.ane_embedding import (
    compile_drafter_model,
    draft_token,
    _draft_token_gpu,
    is_ane_available,
    _HAS_COREMLTOOLS,
    _HAS_MLX,
)


class TestCompileDrafterModel:
    """Test compile_drafter_model function."""

    def test_returns_gpu_fallback_without_coreml(self):
        """Without CoreML tools, should return GPU fallback."""
        with patch("yunshu_engine.ane_embedding._HAS_COREMLTOOLS", False):
            result = compile_drafter_model("/fake/model_path")
            assert result["target"] == "gpu"
            assert "compiled_path" in result
            assert "compile_time_s" in result

    def test_returns_dict_with_required_keys(self):
        result = compile_drafter_model("/fake/model_path")
        assert "compiled_path" in result
        assert "model_size" in result
        assert "compile_time_s" in result
        assert "target" in result

    def test_custom_output_path(self, tmp_path):
        """Should use custom output path."""
        output_dir = str(tmp_path / "custom_cache")
        with patch("yunshu_engine.ane_embedding._HAS_COREMLTOOLS", False):
            result = compile_drafter_model("/fake/model_path", output_path=output_dir)
            assert result["target"] == "gpu"

    def test_model_name_in_compiled_path(self):
        """Compiled path should include the model name."""
        result = compile_drafter_model("/models/qwen-draft", output_path="/tmp/test_ane")
        assert "qwen-draft" in result["compiled_path"]

    @patch("yunshu_engine.ane_embedding._HAS_COREMLTOOLS", True)
    @patch("yunshu_engine.ane_embedding.is_ane_available", return_value=False)
    def test_ane_unavailable_returns_gpu_fallback(self, mock_ane):
        """If ANE is not available, should fallback to GPU."""
        result = compile_drafter_model("/fake/model_path")
        assert result["target"] == "gpu"

    @patch("yunshu_engine.ane_embedding._HAS_COREMLTOOLS", True)
    @patch("yunshu_engine.ane_embedding.is_ane_available", return_value=True)
    @patch("yunshu_engine.ane_embedding.ct")
    def test_coreml_compilation_error_fallback(self, mock_ct, mock_ane):
        """CoreML compilation error should fallback to GPU."""
        mock_ct.convert.side_effect = RuntimeError("Conversion failed")

        result = compile_drafter_model("/fake/model_path")
        assert result["target"] == "gpu"

    def test_already_compiled_returns_cached(self, tmp_path):
        """If mlmodelc already exists, should return cached result."""
        # Create the expected mlmodelc directory
        model_path = tmp_path / "my_draft_model"
        model_path.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mlmodelc = cache_dir / "my_draft_model_drafter.mlmodelc"
        mlmodelc.mkdir()
        # Create a file inside to have a non-zero size
        (mlmodelc / "model.mlmodel").write_bytes(b"\x00" * 100)

        with patch("yunshu_engine.ane_embedding._HAS_COREMLTOOLS", True):
            with patch("yunshu_engine.ane_embedding.is_ane_available", return_value=True):
                result = compile_drafter_model(
                    str(model_path), output_path=str(cache_dir)
                )
                assert result["compile_time_s"] == 0.0
                assert result["model_size"] > 0
                assert result["target"] == "ane"


class TestDraftToken:
    """Test draft_token function."""

    def test_empty_context_returns_empty(self):
        result = draft_token("/some/model", [])
        assert result == []

    def test_zero_draft_returns_empty(self):
        result = draft_token("/some/model", [1, 2, 3], num_draft=0)
        assert result == []

    def test_negative_draft_returns_empty(self):
        result = draft_token("/some/model", [1, 2, 3], num_draft=-1)
        assert result == []

    def test_gpu_fallback_returns_tokens(self):
        """GPU fallback should return requested number of tokens."""
        result = draft_token("/some/model", [1, 2, 3, 4, 5], num_draft=3)
        assert len(result) == 3
        assert all(isinstance(t, int) for t in result)

    def test_gpu_fallback_deterministic(self):
        """GPU fallback with same context should produce same results."""
        context = [100, 200, 300]
        result1 = draft_token("/some/model", context, num_draft=5)
        result2 = draft_token("/some/model", context, num_draft=5)
        assert result1 == result2

    def test_gpu_fallback_different_context_different_results(self):
        """Different context should produce different draft tokens."""
        result1 = draft_token("/some/model", [1, 2, 3], num_draft=5)
        result2 = draft_token("/some/model", [4, 5, 6], num_draft=5)
        assert result1 != result2

    def test_draft_token_count(self):
        """Should return exactly num_draft tokens."""
        for k in [1, 3, 5, 10]:
            result = draft_token("/some/model", [1, 2, 3], num_draft=k)
            assert len(result) == k


class TestDraftTokenGPU:
    """Test _draft_token_gpu internal function."""

    def test_returns_tokens(self):
        result = _draft_token_gpu("/some/model", [1, 2, 3], 5)
        assert len(result) == 5
        assert all(isinstance(t, int) for t in result)

    def test_tokens_in_vocab_range(self):
        """Tokens should be in reasonable vocab range."""
        result = _draft_token_gpu("/some/model", [1, 2, 3], 10)
        for t in result:
            assert 0 <= t < 32000

    @patch("yunshu_engine.ane_embedding._HAS_MLX", False)
    def test_no_mlx_returns_empty(self):
        result = _draft_token_gpu("/some/model", [1, 2, 3], 5)
        assert result == []


class TestDraftTokenCoreMLPath:
    """Test CoreML path in draft_token."""

    @patch("yunshu_engine.ane_embedding._HAS_COREMLTOOLS", False)
    def test_nonexistent_mlmodelc_falls_to_gpu(self):
        """Non-existent .mlmodelc path should fall back to GPU."""
        result = draft_token("/fake/model.mlmodelc", [1, 2, 3], num_draft=3)
        assert len(result) == 3  # GPU fallback still returns tokens

    @patch("yunshu_engine.ane_embedding._HAS_COREMLTOOLS", True)
    @patch("yunshu_engine.ane_embedding.ct")
    def test_coreml_load_failure_fallback(self, mock_ct):
        """CoreML load failure should fall back to GPU."""
        mock_ct.models.MLModel.side_effect = Exception("Load failed")

        # Create a temp mlmodelc dir to pass the path check
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            mlmodelc_path = Path(tmpdir) / "test.mlmodelc"
            mlmodelc_path.mkdir()

            result = draft_token(str(mlmodelc_path), [1, 2, 3], num_draft=3)
            assert len(result) == 3


class TestDrafterEdgeCases:
    """Edge cases for drafter functions."""

    def test_single_context_token(self):
        result = draft_token("/model", [42], num_draft=5)
        assert len(result) == 5

    def test_large_context(self):
        """Large context should still work (uses last 64 tokens)."""
        context = list(range(1000))
        result = draft_token("/model", context, num_draft=5)
        assert len(result) == 5

    def test_model_path_with_special_chars(self):
        result = draft_token("/model/path with spaces/draft", [1, 2, 3], num_draft=3)
        assert len(result) == 3
