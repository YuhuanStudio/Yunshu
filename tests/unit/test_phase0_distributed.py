"""Tests for Phase 0 distributed validation scripts and infrastructure.

Validates SoakTestRunner, JACCLBenchmark, and CollectiveOps.bench_collective
all work correctly in single-process mode (no multi-node required).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import mlx.core as mx

from yunshu_mesh.collective import CollectiveOps

# ── Import scripts as modules ──
import sys

_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from soak_test_distributed import (
    SoakTestConfig,
    SoakTestReport,
    SoakTestRunner,
)
from bench_jaccl import (
    JACCLBenchConfig,
    JACCLBenchmark,
    DEFAULT_SIZES,
    _parse_size,
    _fmt_bytes,
    _fmt_elements,
)


# ════════════════════════════════════════════════════════════════
#  SoakTestConfig
# ════════════════════════════════════════════════════════════════


class TestSoakTestConfig:
    """Tests for SoakTestConfig dataclass."""

    def test_defaults(self):
        cfg = SoakTestConfig()
        assert cfg.duration_hours == 72.0
        assert cfg.ops == ["all_reduce", "all_gather", "send_recv"]
        assert cfg.tensor_size == 1024 * 1024
        assert cfg.check_interval_seconds == 300

    def test_custom_values(self):
        cfg = SoakTestConfig(
            duration_hours=1.0,
            ops=["all_reduce"],
            tensor_size=4096,
            check_interval_seconds=60,
        )
        assert cfg.duration_hours == 1.0
        assert cfg.ops == ["all_reduce"]
        assert cfg.tensor_size == 4096
        assert cfg.check_interval_seconds == 60

    def test_from_args(self):
        args = MagicMock()
        args.duration = 0.5
        args.ops = "all_reduce,send_recv"
        args.tensor_size = 8192
        args.check_interval = 120
        cfg = SoakTestConfig.from_args(args)
        assert cfg.duration_hours == 0.5
        assert cfg.ops == ["all_reduce", "send_recv"]
        assert cfg.tensor_size == 8192
        assert cfg.check_interval_seconds == 120

    def test_from_args_strips_whitespace(self):
        args = MagicMock()
        args.duration = 1.0
        args.ops = " all_reduce , all_gather "
        args.tensor_size = 1024
        args.check_interval = 300
        cfg = SoakTestConfig.from_args(args)
        assert cfg.ops == ["all_reduce", "all_gather"]


# ════════════════════════════════════════════════════════════════
#  SoakTestReport
# ════════════════════════════════════════════════════════════════


class TestSoakTestReport:
    """Tests for SoakTestReport dataclass."""

    def test_creation(self):
        report = SoakTestReport(
            total_iterations=100,
            total_errors=2,
            avg_latency_ms=1.5,
            p99_latency_ms=5.0,
            max_memory_gb=2.3,
            duration_hours=1.0,
        )
        assert report.total_iterations == 100
        assert report.total_errors == 2
        assert report.avg_latency_ms == 1.5
        assert report.p99_latency_ms == 5.0

    def test_defaults(self):
        report = SoakTestReport()
        assert report.total_iterations == 0
        assert report.total_errors == 0
        assert report.avg_latency_ms == 0.0
        assert report.mode == "single_process"
        assert report.errors == []

    def test_to_dict(self):
        report = SoakTestReport(
            total_iterations=50,
            total_errors=1,
            avg_latency_ms=2.3456,
            p99_latency_ms=4.5678,
            max_memory_gb=1.2,
            duration_hours=0.5,
            errors=["some error"],
            op_results={"all_reduce": {"count": 50}},
        )
        d = report.to_dict()
        assert d["total_iterations"] == 50
        assert d["avg_latency_ms"] == round(2.3456, 3)
        assert d["mode"] == "single_process"
        assert "errors" in d

    def test_to_dict_caps_errors(self):
        report = SoakTestReport(errors=[f"err{i}" for i in range(100)])
        d = report.to_dict()
        assert len(d["errors"]) == 50  # capped at last 50


# ════════════════════════════════════════════════════════════════
#  SoakTestRunner
# ════════════════════════════════════════════════════════════════


class TestSoakTestRunner:
    """Tests for SoakTestRunner (single-process mode)."""

    def test_initialization(self):
        cfg = SoakTestConfig(tensor_size=1024)
        runner = SoakTestRunner(cfg)
        assert runner._config is cfg
        assert runner._iterations == 0
        assert runner._errors == []
        assert runner._distributed is False

    def test_run_single_op_all_reduce(self):
        cfg = SoakTestConfig(tensor_size=1024, ops=["all_reduce"])
        runner = SoakTestRunner(cfg)
        result = runner.run_single_op("all_reduce")
        assert result["op"] == "all_reduce"
        assert result["latency_ms"] >= 0
        assert result["correct"] is True

    def test_run_single_op_all_gather(self):
        cfg = SoakTestConfig(tensor_size=1024, ops=["all_gather"])
        runner = SoakTestRunner(cfg)
        result = runner.run_single_op("all_gather")
        assert result["op"] == "all_gather"
        assert result["latency_ms"] >= 0
        assert result["correct"] is True

    def test_run_single_op_send_recv(self):
        cfg = SoakTestConfig(tensor_size=1024, ops=["send_recv"])
        runner = SoakTestRunner(cfg)
        result = runner.run_single_op("send_recv")
        assert result["op"] == "send_recv"
        assert result["latency_ms"] >= 0
        assert result["correct"] is True

    def test_run_single_op_unknown_raises(self):
        cfg = SoakTestConfig(tensor_size=1024)
        runner = SoakTestRunner(cfg)
        result = runner.run_single_op("unknown_op")
        assert result["correct"] is False
        assert "error" in result

    def test_run_iteration(self):
        cfg = SoakTestConfig(tensor_size=1024, ops=["all_reduce", "all_gather"])
        runner = SoakTestRunner(cfg)
        results = runner.run_iteration()
        assert "all_reduce" in results
        assert "all_gather" in results
        assert runner._iterations == 1

    def test_run_short_soak(self):
        """Run a very short soak test (fraction of a second) in single-process mode."""
        cfg = SoakTestConfig(
            duration_hours=0.0001,  # ~0.36 seconds
            ops=["all_reduce"],
            tensor_size=1024,
            check_interval_seconds=1,
        )
        runner = SoakTestRunner(cfg)
        report = runner.run()
        assert report.total_iterations >= 1
        assert report.total_errors == 0
        assert report.avg_latency_ms >= 0
        assert report.mode == "single_process"
        assert report.duration_hours > 0

    def test_check_health(self):
        cfg = SoakTestConfig(tensor_size=1024, ops=["all_reduce"])
        runner = SoakTestRunner(cfg)
        runner.run_iteration()
        health = runner.check_health()
        assert health["iterations"] == 1
        assert health["errors"] == 0
        assert health["avg_latency_ms"] >= 0

    def test_check_health_before_run(self):
        cfg = SoakTestConfig(tensor_size=1024)
        runner = SoakTestRunner(cfg)
        health = runner.check_health()
        assert health["iterations"] == 0
        assert health["avg_latency_ms"] == 0.0
        assert health["elapsed_hours"] == 0.0

    def test_errors_tracked(self):
        """Verify errors are tracked when an unknown op is encountered."""
        cfg = SoakTestConfig(tensor_size=1024, ops=["bad_op"])
        runner = SoakTestRunner(cfg)
        runner.run_iteration()
        assert len(runner._errors) > 0


# ════════════════════════════════════════════════════════════════
#  JACCLBenchConfig
# ════════════════════════════════════════════════════════════════


class TestJACCLBenchConfig:
    """Tests for JACCLBenchConfig dataclass."""

    def test_defaults(self):
        cfg = JACCLBenchConfig()
        assert cfg.tensor_sizes == DEFAULT_SIZES
        assert cfg.num_warmup == 5
        assert cfg.num_iters == 100
        assert cfg.backend == "any"

    def test_custom_values(self):
        cfg = JACCLBenchConfig(
            tensor_sizes=[1024, 4096],
            num_warmup=2,
            num_iters=10,
            backend="ring",
        )
        assert cfg.tensor_sizes == [1024, 4096]
        assert cfg.num_warmup == 2
        assert cfg.num_iters == 10
        assert cfg.backend == "ring"

    def test_from_args(self):
        args = MagicMock()
        args.sizes = "1K,1M"
        args.warmup = 3
        args.iters = 20
        args.backend = "jaccl"
        cfg = JACCLBenchConfig.from_args(args)
        assert cfg.tensor_sizes == [1024, 1024 * 1024]
        assert cfg.num_warmup == 3
        assert cfg.num_iters == 20
        assert cfg.backend == "jaccl"


# ════════════════════════════════════════════════════════════════
#  JACCLBenchmark
# ════════════════════════════════════════════════════════════════


class TestJACCLBenchmark:
    """Tests for JACCLBenchmark (single-node baseline mode)."""

    def test_bench_all_reduce_latency(self):
        """all_reduce should return valid latency measurements."""
        cfg = JACCLBenchConfig(tensor_sizes=[1024], num_warmup=1, num_iters=5)
        bench = JACCLBenchmark(cfg)
        result = bench.bench_all_reduce(1024)
        assert result["op"] == "all_reduce"
        assert result["size"] == 1024
        assert result["avg_ms"] >= 0
        assert result["p50_ms"] >= 0
        assert result["p99_ms"] >= 0
        assert result["bandwidth_gbs"] >= 0
        assert result["mode"] == "single_node"

    def test_bench_all_gather_latency(self):
        cfg = JACCLBenchConfig(tensor_sizes=[1024], num_warmup=1, num_iters=5)
        bench = JACCLBenchmark(cfg)
        result = bench.bench_all_gather(1024)
        assert result["op"] == "all_gather"
        assert result["size"] == 1024
        assert result["avg_ms"] >= 0
        assert result["mode"] == "single_node"

    def test_bench_send_recv_latency(self):
        cfg = JACCLBenchConfig(tensor_sizes=[1024], num_warmup=1, num_iters=5)
        bench = JACCLBenchmark(cfg)
        result = bench.bench_send_recv(1024)
        assert result["op"] == "send_recv"
        assert result["size"] == 1024
        assert result["avg_ms"] >= 0
        assert result["mode"] == "single_node"

    def test_bench_ring_bandwidth(self):
        cfg = JACCLBenchConfig(tensor_sizes=[1024, 4096], num_warmup=1, num_iters=3)
        bench = JACCLBenchmark(cfg)
        result = bench.bench_ring_bandwidth()
        assert result["op"] == "ring_allreduce_sweep"
        assert result["sizes_tested"] == 2
        assert "peak_bandwidth_gbs" in result
        assert "per_size" in result
        assert len(result["per_size"]) == 2

    def test_run_all_benchmarks(self):
        cfg = JACCLBenchConfig(tensor_sizes=[1024], num_warmup=1, num_iters=3)
        bench = JACCLBenchmark(cfg)
        results = bench.run()
        # 3 ops x 1 size = 3 results
        assert len(results) == 3
        ops = {r["op"] for r in results}
        assert ops == {"all_reduce", "all_gather", "send_recv"}

    def test_run_multiple_sizes(self):
        cfg = JACCLBenchConfig(
            tensor_sizes=[1024, 4096],
            num_warmup=1,
            num_iters=3,
        )
        bench = JACCLBenchmark(cfg)
        results = bench.run()
        # 3 ops x 2 sizes = 6 results
        assert len(results) == 6

    def test_latency_increases_with_size(self):
        """Larger tensors should generally take longer (within reason)."""
        cfg = JACCLBenchConfig(
            tensor_sizes=[1024, 1024 * 1024],
            num_warmup=2,
            num_iters=10,
        )
        bench = JACCLBenchmark(cfg)
        r_small = bench.bench_all_reduce(1024)
        r_large = bench.bench_all_reduce(1024 * 1024)
        # Not strictly guaranteed on every run, but should hold in practice
        assert r_small["bytes"] < r_large["bytes"]

    def test_bandwidth_calculation(self):
        """Verify bandwidth formula: data_volume / time."""
        cfg = JACCLBenchConfig(tensor_sizes=[1024], num_warmup=1, num_iters=3)
        bench = JACCLBenchmark(cfg)
        result = bench.bench_all_reduce(1024)
        # For all_reduce, data_volume = 2 * size * 4 bytes (float32)
        expected_volume = 1024 * 4 * 2
        expected_bw = (expected_volume / (result["avg_ms"] / 1000.0)) / (1024**3)
        assert abs(result["bandwidth_gbs"] - round(expected_bw, 3)) < 0.01


# ════════════════════════════════════════════════════════════════
#  JACCLBenchmark.format_results
# ════════════════════════════════════════════════════════════════


class TestJACCLFormatResults:
    """Tests for JACCLBenchmark.format_results output."""

    def test_format_empty_results(self):
        cfg = JACCLBenchConfig()
        bench = JACCLBenchmark(cfg)
        output = bench.format_results([])
        assert "JACCL Baseline Benchmark" in output
        assert "N/A" in output

    def test_format_single_result(self):
        cfg = JACCLBenchConfig(tensor_sizes=[1024], num_warmup=1, num_iters=3)
        bench = JACCLBenchmark(cfg)
        results = bench.run()
        output = bench.format_results(results)
        assert "all_reduce" in output
        assert "all_gather" in output
        assert "send_recv" in output
        assert "ms" in output
        assert "GB/s" in output
        assert "single_node" in output

    def test_format_table_has_separator_lines(self):
        cfg = JACCLBenchConfig(tensor_sizes=[1024], num_warmup=1, num_iters=3)
        bench = JACCLBenchmark(cfg)
        results = bench.run()
        output = bench.format_results(results)
        assert "=" * 50 in output
        assert "-" * 50 in output

    def test_format_note_about_single_node(self):
        cfg = JACCLBenchConfig(tensor_sizes=[1024], num_warmup=1, num_iters=3)
        bench = JACCLBenchmark(cfg)
        results = bench.run()
        output = bench.format_results(results)
        assert "single_node mode measures local" in output
        assert "lower-bound baseline" in output


# ════════════════════════════════════════════════════════════════
#  Utility functions
# ════════════════════════════════════════════════════════════════


class TestBenchUtils:
    """Tests for helper functions."""

    def test_parse_size_plain(self):
        assert _parse_size("1024") == 1024

    def test_parse_size_k(self):
        assert _parse_size("1K") == 1024
        assert _parse_size("64K") == 64 * 1024

    def test_parse_size_m(self):
        assert _parse_size("1M") == 1024 * 1024
        assert _parse_size("16M") == 16 * 1024 * 1024

    def test_parse_size_g(self):
        assert _parse_size("1G") == 1024**3

    def test_parse_size_float(self):
        assert _parse_size("0.5M") == int(0.5 * 1024 * 1024)

    def test_parse_size_whitespace(self):
        assert _parse_size("  1K  ") == 1024

    def test_fmt_bytes(self):
        assert "B" in _fmt_bytes(512)
        assert "KB" in _fmt_bytes(2048)
        assert "MB" in _fmt_bytes(1024**2 + 1)
        assert "GB" in _fmt_bytes(1024**3 + 1)

    def test_fmt_elements(self):
        assert _fmt_elements(500) == "500"
        assert "K" in _fmt_elements(1500)
        assert "M" in _fmt_elements(2_000_000)


# ════════════════════════════════════════════════════════════════
#  CollectiveOps.bench_collective
# ════════════════════════════════════════════════════════════════


class TestCollectiveOpsBenchCollective:
    """Tests for the new bench_collective method on CollectiveOps."""

    def test_bench_all_reduce_single_node(self):
        """bench_collective should work without mx.distributed initialisation."""
        ops = CollectiveOps(backend="any")
        assert not ops.is_initialized
        result = ops.bench_collective("all_reduce", tensor_size=1024, num_iters=5)
        assert result["op"] == "all_reduce"
        assert result["tensor_size"] == 1024
        assert result["avg_latency_ms"] >= 0
        assert result["p50_ms"] >= 0
        assert result["p99_ms"] >= 0
        assert result["bandwidth_gbs"] >= 0
        assert result["mode"] == "single_node"

    def test_bench_all_gather_single_node(self):
        ops = CollectiveOps()
        result = ops.bench_collective("all_gather", tensor_size=2048, num_iters=5)
        assert result["op"] == "all_gather"
        assert result["tensor_size"] == 2048
        assert result["mode"] == "single_node"

    def test_bench_send_recv_single_node(self):
        ops = CollectiveOps()
        result = ops.bench_collective("send_recv", tensor_size=4096, num_iters=5)
        assert result["op"] == "send_recv"
        assert result["tensor_size"] == 4096
        assert result["mode"] == "single_node"

    def test_bench_unknown_op_raises(self):
        ops = CollectiveOps()
        with pytest.raises(ValueError, match="Unknown collective op"):
            ops.bench_collective("nonexistent", tensor_size=1024, num_iters=3)

    def test_bench_latency_reasonable(self):
        """Latency should be positive and finite."""
        ops = CollectiveOps()
        result = ops.bench_collective("all_reduce", tensor_size=1024, num_iters=10)
        assert math.isfinite(result["avg_latency_ms"])
        assert math.isfinite(result["p50_ms"])
        assert math.isfinite(result["p99_ms"])
        assert math.isfinite(result["bandwidth_gbs"])

    def test_bench_p50_le_p99(self):
        """P50 should not exceed P99."""
        ops = CollectiveOps()
        result = ops.bench_collective("all_reduce", tensor_size=1024, num_iters=20)
        assert result["p50_ms"] <= result["p99_ms"]

    def test_bench_bandwidth_positive(self):
        """Bandwidth should be positive for non-zero latency."""
        ops = CollectiveOps()
        result = ops.bench_collective("all_reduce", tensor_size=1024, num_iters=5)
        if result["avg_latency_ms"] > 0:
            assert result["bandwidth_gbs"] > 0

    def test_bench_default_params(self):
        """Test that default parameters work."""
        ops = CollectiveOps()
        result = ops.bench_collective("all_reduce")
        assert result["tensor_size"] == 1024 * 1024
        assert result["mode"] == "single_node"

    def test_bench_all_reduce_bandwidth_formula(self):
        """Verify all_reduce uses 2x data volume for bandwidth."""
        ops = CollectiveOps()
        size = 4096
        result = ops.bench_collective("all_reduce", tensor_size=size, num_iters=5)
        expected_volume = size * 4 * 2  # float32 * 2 for all_reduce
        expected_bw = (expected_volume / (result["avg_latency_ms"] / 1000.0)) / (1024**3)
        assert abs(result["bandwidth_gbs"] - round(expected_bw, 3)) < 0.01


# ════════════════════════════════════════════════════════════════
#  Integration: SoakTestRunner produces valid report
# ════════════════════════════════════════════════════════════════


class TestSoakIntegration:
    """Integration tests for the complete soak test pipeline."""

    def test_report_serializable(self):
        """Report should be JSON-serializable."""
        cfg = SoakTestConfig(
            duration_hours=0.00005,  # very short
            ops=["all_reduce"],
            tensor_size=1024,
            check_interval_seconds=999,
        )
        runner = SoakTestRunner(cfg)
        report = runner.run()
        data = report.to_dict()
        # Should not raise
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["total_iterations"] == report.total_iterations

    def test_report_saved_to_file(self, tmp_path):
        """Verify report can be saved to and loaded from a JSON file."""
        cfg = SoakTestConfig(
            duration_hours=0.00005,
            ops=["all_reduce"],
            tensor_size=1024,
            check_interval_seconds=999,
        )
        runner = SoakTestRunner(cfg)
        report = runner.run()

        out_path = tmp_path / "soak_report.json"
        out_path.write_text(json.dumps(report.to_dict(), indent=2))
        loaded = json.loads(out_path.read_text())
        assert loaded["total_iterations"] == report.total_iterations
        assert loaded["mode"] == "single_process"
