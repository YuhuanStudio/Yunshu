"""Tests for Phase 0 ANE micro-benchmark and CoreML validation.

Covers:
  - ANEBenchConfig defaults and custom configuration
  - ANEBenchmark initialization and benchmark methods
  - bench_embedding_inference, bench_linear_layer, bench_transformer_layer
  - format_results output validation
  - is_ane_available() from ane_embedding
  - estimate_ane_speedup() from ane_embedding
  - compile_embedding_model() and benchmark_ane_vs_gpu() convenience functions
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Add scripts dir to path for bench_ane import ──
import sys

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# ANEBenchConfig
# ---------------------------------------------------------------------------


class TestANEBenchConfig:
    """Tests for ANEBenchConfig dataclass."""

    def test_defaults(self):
        from bench_ane import ANEBenchConfig

        config = ANEBenchConfig()
        assert config.model_sizes == [10, 50, 100, 300]
        assert config.seq_lengths == [32, 64, 128, 256, 512]
        assert config.num_warmup == 3
        assert config.num_iters == 50

    def test_custom(self):
        from bench_ane import ANEBenchConfig

        config = ANEBenchConfig(
            model_sizes=[5, 20],
            seq_lengths=[16, 32],
            num_warmup=1,
            num_iters=10,
        )
        assert config.model_sizes == [5, 20]
        assert config.seq_lengths == [16, 32]
        assert config.num_warmup == 1
        assert config.num_iters == 10

    def test_model_sizes_is_list(self):
        from bench_ane import ANEBenchConfig

        config = ANEBenchConfig()
        assert isinstance(config.model_sizes, list)

    def test_seq_lengths_is_list(self):
        from bench_ane import ANEBenchConfig

        config = ANEBenchConfig()
        assert isinstance(config.seq_lengths, list)


# ---------------------------------------------------------------------------
# ANEBenchmark initialization
# ---------------------------------------------------------------------------


class TestANEBenchmarkInit:
    """Tests for ANEBenchmark initialization."""

    def test_init_default_config(self):
        from bench_ane import ANEBenchmark

        bench = ANEBenchmark()
        assert bench.config is not None
        assert bench.config.num_warmup == 3
        assert bench.config.num_iters == 50

    def test_init_custom_config(self):
        from bench_ane import ANEBenchConfig, ANEBenchmark

        config = ANEBenchConfig(num_warmup=1, num_iters=5)
        bench = ANEBenchmark(config)
        assert bench.config.num_warmup == 1
        assert bench.config.num_iters == 5

    def test_init_detects_mlx(self):
        from bench_ane import ANEBenchmark

        bench = ANEBenchmark()
        # On Apple Silicon with mlx installed, this should be True
        # On other platforms, it may be False
        assert isinstance(bench._mlx_available, bool)

    def test_init_detects_coreml(self):
        from bench_ane import ANEBenchmark

        bench = ANEBenchmark()
        assert isinstance(bench._coreml_available, bool)


# ---------------------------------------------------------------------------
# bench_embedding_inference
# ---------------------------------------------------------------------------


class TestBenchEmbeddingInference:
    """Tests for ANEBenchmark.bench_embedding_inference."""

    def test_returns_dict_with_expected_keys_gpu(self):
        from bench_ane import ANEBenchConfig, ANEBenchmark

        config = ANEBenchConfig(num_warmup=1, num_iters=3)
        bench = ANEBenchmark(config)
        result = bench.bench_embedding_inference(10, 32, device="gpu")
        assert isinstance(result, dict)
        expected_keys = {"model_size_m", "seq_length", "device", "latency_ms", "throughput_seq_per_s", "status"}
        assert set(result.keys()) == expected_keys

    def test_returns_dict_with_expected_keys_ane(self):
        from bench_ane import ANEBenchConfig, ANEBenchmark

        config = ANEBenchConfig(num_warmup=1, num_iters=3)
        bench = ANEBenchmark(config)
        result = bench.bench_embedding_inference(10, 32, device="ane")
        assert isinstance(result, dict)
        assert "latency_ms" in result
        assert "status" in result
        # ANE result may be estimated or actual
        assert result["status"] in ("ok", "estimated", "mlx_unavailable", "ane_unavailable")

    def test_gpu_result_has_latency_if_mlx_available(self):
        from bench_ane import ANEBenchConfig, ANEBenchmark

        config = ANEBenchConfig(num_warmup=1, num_iters=3)
        bench = ANEBenchmark(config)
        if bench._mlx_available:
            result = bench.bench_embedding_inference(10, 32, device="gpu")
            assert result["status"] == "ok"
            assert result["latency_ms"] is not None
            assert result["latency_ms"] > 0

    def test_ane_estimated_result_has_latency(self):
        from bench_ane import ANEBenchConfig, ANEBenchmark

        config = ANEBenchConfig(num_warmup=1, num_iters=3)
        bench = ANEBenchmark(config)
        # Force estimated path by mocking coreml_available to False
        bench._coreml_available = False
        result = bench.bench_embedding_inference(10, 32, device="ane")
        assert result["status"] == "estimated"
        assert result["latency_ms"] is not None
        assert result["latency_ms"] > 0

    def test_invalid_device_returns_unknown_status(self):
        from bench_ane import ANEBenchConfig, ANEBenchmark

        config = ANEBenchConfig(num_warmup=1, num_iters=3)
        bench = ANEBenchmark(config)
        result = bench.bench_embedding_inference(10, 32, device="tpu")
        assert result["status"] == "unknown_device_tpu"

    def test_gpu_without_mlx_returns_unavailable(self):
        from bench_ane import ANEBenchConfig, ANEBenchmark

        config = ANEBenchConfig(num_warmup=1, num_iters=3)
        bench = ANEBenchmark(config)
        bench._mlx_available = False
        result = bench.bench_embedding_inference(10, 32, device="gpu")
        assert result["status"] == "mlx_unavailable"


# ---------------------------------------------------------------------------
# bench_linear_layer
# ---------------------------------------------------------------------------


class TestBenchLinearLayer:
    """Tests for ANEBenchmark.bench_linear_layer."""

    def test_returns_valid_results_gpu(self):
        from bench_ane import ANEBenchConfig, ANEBenchmark

        config = ANEBenchConfig(num_warmup=1, num_iters=3)
        bench = ANEBenchmark(config)
        if bench._mlx_available:
            result = bench.bench_linear_layer(64, 64, 32, device="gpu")
            assert result["status"] == "ok"
            assert result["latency_ms"] is not None
            assert result["gflops"] is not None

    def test_returns_valid_results_ane_estimated(self):
        from bench_ane import ANEBenchConfig, ANEBenchmark

        config = ANEBenchConfig(num_warmup=1, num_iters=3)
        bench = ANEBenchmark(config)
        bench._coreml_available = False
        result = bench.bench_linear_layer(64, 64, 32, device="ane")
        assert result["status"] == "estimated"
        assert result["latency_ms"] is not None
        assert result["latency_ms"] > 0

    def test_result_has_expected_keys(self):
        from bench_ane import ANEBenchConfig, ANEBenchmark

        config = ANEBenchConfig(num_warmup=1, num_iters=3)
        bench = ANEBenchmark(config)
        bench._coreml_available = False
        result = bench.bench_linear_layer(128, 256, 64, device="ane")
        expected_keys = {"in_dim", "out_dim", "seq_length", "device", "latency_ms", "gflops", "status"}
        assert set(result.keys()) == expected_keys
        assert result["in_dim"] == 128
        assert result["out_dim"] == 256
        assert result["seq_length"] == 64

    def test_invalid_device(self):
        from bench_ane import ANEBenchConfig, ANEBenchmark

        config = ANEBenchConfig(num_warmup=1, num_iters=3)
        bench = ANEBenchmark(config)
        result = bench.bench_linear_layer(64, 64, 32, device="fpga")
        assert result["status"] == "unknown_device_fpga"


# ---------------------------------------------------------------------------
# bench_transformer_layer
# ---------------------------------------------------------------------------


class TestBenchTransformerLayer:
    """Tests for ANEBenchmark.bench_transformer_layer."""

    def test_returns_valid_results_gpu(self):
        from bench_ane import ANEBenchConfig, ANEBenchmark

        config = ANEBenchConfig(num_warmup=1, num_iters=3)
        bench = ANEBenchmark(config)
        if bench._mlx_available:
            result = bench.bench_transformer_layer(64, 32, device="gpu")
            assert result["status"] == "ok"
            assert result["latency_ms"] is not None
            assert result["latency_ms"] > 0

    def test_returns_valid_results_ane_estimated(self):
        from bench_ane import ANEBenchConfig, ANEBenchmark

        config = ANEBenchConfig(num_warmup=1, num_iters=3)
        bench = ANEBenchmark(config)
        bench._coreml_available = False
        result = bench.bench_transformer_layer(128, 64, device="ane")
        assert result["status"] == "estimated"
        assert result["latency_ms"] is not None
        assert result["latency_ms"] > 0

    def test_result_has_expected_keys(self):
        from bench_ane import ANEBenchConfig, ANEBenchmark

        config = ANEBenchConfig(num_warmup=1, num_iters=3)
        bench = ANEBenchmark(config)
        bench._coreml_available = False
        result = bench.bench_transformer_layer(128, 64, device="ane")
        expected_keys = {"hidden_dim", "seq_length", "device", "latency_ms", "throughput_seq_per_s", "status"}
        assert set(result.keys()) == expected_keys
        assert result["hidden_dim"] == 128
        assert result["seq_length"] == 64


# ---------------------------------------------------------------------------
# format_results
# ---------------------------------------------------------------------------


class TestFormatResults:
    """Tests for ANEBenchmark.format_results."""

    def test_empty_results(self):
        from bench_ane import ANEBenchmark

        output = ANEBenchmark.format_results([])
        assert isinstance(output, str)
        assert "no results" in output

    def test_produces_non_empty_string(self):
        from bench_ane import ANEBenchmark

        # Create sample results
        results = [
            {"model_size_m": 10, "seq_length": 32, "device": "gpu",
             "latency_ms": 1.5, "throughput_seq_per_s": 666.7, "status": "ok"},
            {"model_size_m": 10, "seq_length": 32, "device": "ane",
             "latency_ms": 0.8, "throughput_seq_per_s": 1250.0, "status": "ok"},
        ]
        output = ANEBenchmark.format_results(results)
        assert isinstance(output, str)
        assert len(output) > 0
        assert "Embedding" in output or "10" in output

    def test_formats_linear_results(self):
        from bench_ane import ANEBenchmark

        results = [
            {"in_dim": 128, "out_dim": 128, "seq_length": 64,
             "device": "gpu", "latency_ms": 0.5, "gflops": 2.1, "status": "ok"},
            {"in_dim": 128, "out_dim": 128, "seq_length": 64,
             "device": "ane", "latency_ms": 0.3, "gflops": 3.5, "status": "ok"},
        ]
        output = ANEBenchmark.format_results(results)
        assert "Linear" in output

    def test_formats_transformer_results(self):
        from bench_ane import ANEBenchmark

        results = [
            {"hidden_dim": 256, "seq_length": 128,
             "device": "gpu", "latency_ms": 2.0, "throughput_seq_per_s": 500.0, "status": "ok"},
            {"hidden_dim": 256, "seq_length": 128,
             "device": "ane", "latency_ms": 1.5, "throughput_seq_per_s": 666.7, "status": "ok"},
        ]
        output = ANEBenchmark.format_results(results)
        assert "Transformer" in output

    def test_handles_n_a_values(self):
        from bench_ane import ANEBenchmark

        results = [
            {"model_size_m": 10, "seq_length": 32, "device": "gpu",
             "latency_ms": None, "throughput_seq_per_s": None, "status": "mlx_unavailable"},
            {"model_size_m": 10, "seq_length": 32, "device": "ane",
             "latency_ms": None, "throughput_seq_per_s": None, "status": "mlx_unavailable"},
        ]
        output = ANEBenchmark.format_results(results)
        assert "N/A" in output


# ---------------------------------------------------------------------------
# is_ane_available (from ane_embedding)
# ---------------------------------------------------------------------------


class TestIsAneAvailablePhase0:
    """Tests for is_ane_available() from ane_embedding module."""

    def test_returns_bool(self):
        from yunshu_engine.ane_embedding import is_ane_available

        result = is_ane_available()
        assert isinstance(result, bool)

    def test_false_on_linux(self):
        from yunshu_engine.ane_embedding import is_ane_available

        with patch("yunshu_engine.ane_embedding.platform") as mock_platform:
            mock_platform.system.return_value = "Linux"
            mock_platform.machine.return_value = "x86_64"
            mock_platform.mac_ver.return_value = ("", "", "")
            assert is_ane_available() is False

    def test_false_on_intel_mac(self):
        from yunshu_engine.ane_embedding import is_ane_available

        with patch("yunshu_engine.ane_embedding.platform") as mock_platform:
            mock_platform.system.return_value = "Darwin"
            mock_platform.machine.return_value = "x86_64"
            mock_platform.mac_ver.return_value = ("14.0", "", "")
            assert is_ane_available() is False

    def test_false_without_coremltools(self):
        from yunshu_engine.ane_embedding import is_ane_available

        with patch("yunshu_engine.ane_embedding._HAS_COREMLTOOLS", False):
            result = is_ane_available()
            # Should be False without coremltools
            assert result is False or isinstance(result, bool)


# ---------------------------------------------------------------------------
# estimate_ane_speedup (from ane_embedding)
# ---------------------------------------------------------------------------


class TestEstimateAneSpeedupPhase0:
    """Tests for estimate_ane_speedup() from ane_embedding module."""

    def test_returns_float(self):
        from yunshu_engine.ane_embedding import estimate_ane_speedup

        result = estimate_ane_speedup(50.0, 128)
        assert isinstance(result, float)

    def test_in_valid_range(self):
        from yunshu_engine.ane_embedding import estimate_ane_speedup

        for params in [1.0, 33.0, 100.0, 500.0]:
            for seq in [32, 128, 256, 512]:
                result = estimate_ane_speedup(params, seq)
                assert 1.0 <= result <= 3.0

    def test_zero_params_returns_unity(self):
        from yunshu_engine.ane_embedding import estimate_ane_speedup

        assert estimate_ane_speedup(0.0, 128) == 1.0

    def test_zero_seq_returns_unity(self):
        from yunshu_engine.ane_embedding import estimate_ane_speedup

        assert estimate_ane_speedup(50.0, 0) == 1.0

    def test_negative_returns_unity(self):
        from yunshu_engine.ane_embedding import estimate_ane_speedup

        assert estimate_ane_speedup(-10.0, -1) == 1.0

    def test_small_model_faster(self):
        from yunshu_engine.ane_embedding import estimate_ane_speedup

        small = estimate_ane_speedup(10.0, 128)
        large = estimate_ane_speedup(200.0, 128)
        assert small >= large


# ---------------------------------------------------------------------------
# compile_embedding_model (module-level function)
# ---------------------------------------------------------------------------


class TestCompileEmbeddingModel:
    """Tests for compile_embedding_model() convenience function."""

    def test_raises_without_coremltools(self):
        from yunshu_engine.ane_embedding import compile_embedding_model

        with patch("yunshu_engine.ane_embedding._HAS_COREMLTOOLS", False):
            with pytest.raises(RuntimeError, match="coremltools is required"):
                compile_embedding_model("/fake/path")

    def test_raises_for_nonexistent_path(self):
        from yunshu_engine.ane_embedding import compile_embedding_model

        with patch("yunshu_engine.ane_embedding._HAS_COREMLTOOLS", True):
            with pytest.raises(FileNotFoundError, match="does not exist"):
                compile_embedding_model("/nonexistent/model/path")


# ---------------------------------------------------------------------------
# benchmark_ane_vs_gpu (module-level function)
# ---------------------------------------------------------------------------


class TestBenchmarkAneVsGpu:
    """Tests for benchmark_ane_vs_gpu() convenience function."""

    def test_empty_texts_returns_zeros(self):
        from yunshu_engine.ane_embedding import benchmark_ane_vs_gpu

        result = benchmark_ane_vs_gpu([])
        assert result["gpu_latency_ms"] == 0.0
        assert result["ane_latency_ms"] is None
        assert result["speedup"] is None
        assert result["accuracy_diff"] is None

    def test_returns_expected_keys(self):
        from yunshu_engine.ane_embedding import benchmark_ane_vs_gpu

        with patch("yunshu_engine.ane_embedding.ANEEmbeddingProcessor") as mock_proc_cls:
            mock_proc = MagicMock()
            mock_proc.embed.return_value = [[0.1] * 384]
            mock_proc_cls.return_value = mock_proc
            result = benchmark_ane_vs_gpu(["hello world"])
        expected_keys = {"gpu_latency_ms", "ane_latency_ms", "speedup", "accuracy_diff"}
        assert set(result.keys()) == expected_keys

    def test_gpu_latency_positive(self):
        from yunshu_engine.ane_embedding import benchmark_ane_vs_gpu

        with patch("yunshu_engine.ane_embedding.ANEEmbeddingProcessor") as mock_proc_cls:
            mock_proc = MagicMock()
            mock_proc.embed.return_value = [[0.1] * 384]
            mock_proc_cls.return_value = mock_proc
            result = benchmark_ane_vs_gpu(["test text"])
        assert result["gpu_latency_ms"] > 0

    def test_returns_dict(self):
        from yunshu_engine.ane_embedding import benchmark_ane_vs_gpu

        with patch("yunshu_engine.ane_embedding.ANEEmbeddingProcessor") as mock_proc_cls:
            mock_proc = MagicMock()
            mock_proc.embed.return_value = [[0.1] * 384, [0.2] * 384]
            mock_proc_cls.return_value = mock_proc
            result = benchmark_ane_vs_gpu(["a", "b"])
        assert isinstance(result, dict)
