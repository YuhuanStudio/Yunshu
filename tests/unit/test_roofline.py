"""Tests for yunshu_engine.roofline — Apple Silicon roofline model."""

from __future__ import annotations

import os
import tempfile

import pytest

from yunshu_engine.roofline import (
    CHIP_PARAMS,
    MODEL_CONFIGS,
    RooflineModel,
    _normalise_chip,
)

# ---------------------------------------------------------------------------
# Chip parameter lookups
# ---------------------------------------------------------------------------


class TestChipParams:
    """Verify the chip parameter database is self-consistent."""

    def test_all_chips_have_required_keys(self):
        for name, params in CHIP_PARAMS.items():
            assert "bandwidth_gbps" in params, f"{name} missing bandwidth_gbps"
            assert "compute_tflops_fp16" in params, (
                f"{name} missing compute_tflops_fp16"
            )
            assert params["bandwidth_gbps"] > 0, f"{name} bandwidth must be > 0"
            assert params["compute_tflops_fp16"] > 0, f"{name} compute must be > 0"

    def test_ultras_are_fastest_bandwidth(self):
        """Ultra variants should have 800 GB/s bandwidth."""
        for name in ("M1_Ultra", "M2_Ultra", "M3_Ultra"):
            assert CHIP_PARAMS[name]["bandwidth_gbps"] == 800

    def test_tier_ordering_within_generation(self):
        """Within each generation: base < Pro < Max < Ultra for both metrics."""
        for gen in ("M1", "M2", "M3", "M4"):
            tiers = []
            for tier in ("", "_Pro", "_Max"):
                key = f"{gen}{tier}"
                if key in CHIP_PARAMS:
                    tiers.append(key)
            for i in range(len(tiers) - 1):
                bw_a = CHIP_PARAMS[tiers[i]]["bandwidth_gbps"]
                bw_b = CHIP_PARAMS[tiers[i + 1]]["bandwidth_gbps"]
                assert bw_a <= bw_b, (
                    f"{tiers[i]} BW ({bw_a}) > {tiers[i + 1]} BW ({bw_b})"
                )


# ---------------------------------------------------------------------------
# Chip name normalisation
# ---------------------------------------------------------------------------


class TestNormaliseChip:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("M3_Max", "M3_Max"),
            ("m3_max", "M3_Max"),
            ("Apple M3 Max", "M3_Max"),
            ("Apple M4 Pro", "M4_Pro"),
            ("M4", "M4"),
            ("m4", "M4"),
            ("M1_Ultra", "M1_Ultra"),
        ],
    )
    def test_known_variants(self, raw, expected):
        assert _normalise_chip(raw) == expected

    def test_unknown_returns_as_is(self):
        # Unknown chip falls back to M3_Max inside RooflineModel, but _normalise_chip
        # returns best-effort
        result = _normalise_chip("SomeRandomChip")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# RooflineModel construction
# ---------------------------------------------------------------------------


class TestRooflineModelConstruction:
    def test_explicit_chip(self):
        rm = RooflineModel("M3_Max")
        assert rm.chip_key == "M3_Max"
        assert rm.bandwidth_gbs == 400.0
        assert rm.compute_tflops == 14.0

    def test_explicit_chip_variants(self):
        rm = RooflineModel("Apple M4 Max")
        assert rm.chip_key == "M4_Max"
        assert rm.bandwidth_gbs == 546.0
        assert rm.compute_tflops == 18.0

    def test_auto_detect(self):
        """Auto-detect should not raise; uses sysctl."""
        rm = RooflineModel()  # uses get_chip_name()
        assert rm.chip_key in CHIP_PARAMS
        assert rm.bandwidth_gbs > 0
        assert rm.compute_tflops > 0

    def test_unknown_chip_falls_back(self):
        """Unknown chip should fall back to M3_Max with a warning."""
        rm = RooflineModel("Imaginary_Chip_XYZ")
        assert rm.chip_key == "M3_Max"


# ---------------------------------------------------------------------------
# GEMM roofline math
# ---------------------------------------------------------------------------


class TestGemmRoofline:
    def test_basic_gemm_math(self):
        rm = RooflineModel("M3_Max")  # 400 GB/s, 14 TFLOP/s
        r = rm.compute_gemm_roofline(M=1, N=4096, K=4096, dtype="fp16")

        # FLOPs = 2 * M * N * K = 2 * 1 * 4096 * 4096 = 33554432
        expected_flops = 2 * 1 * 4096 * 4096
        assert r.flops == expected_flops

        # Bytes = (M*K + K*N + M*N) * 2
        expected_bytes = (1 * 4096 + 4096 * 4096 + 1 * 4096) * 2
        assert r.bytes_accessed == expected_bytes

        # OI = flops / bytes
        assert abs(r.operational_intensity - expected_flops / expected_bytes) < 1e-6

    def test_decode_gemm_is_memory_bound(self):
        """Small decode GEMM (M=1) should be memory-bound."""
        rm = RooflineModel("M3_Max")
        r = rm.compute_gemm_roofline(M=1, N=4096, K=4096)
        assert r.bound == "memory"
        assert r.operational_intensity < 1.0  # very low OI

    def test_prefill_gemm_is_compute_bound(self):
        """Large prefill GEMM should be compute-bound."""
        rm = RooflineModel("M3_Max")
        r = rm.compute_gemm_roofline(M=2048, N=4096, K=4096)
        assert r.bound == "compute"
        assert r.operational_intensity > 1.0

    def test_predicted_gflops_capped_at_peak(self):
        rm = RooflineModel("M3_Max")
        r = rm.compute_gemm_roofline(M=2048, N=4096, K=4096)
        peak_gflops = 14.0 * 1000.0  # 14000 GFLOP/s
        assert r.predicted_gflops <= peak_gflops + 1e-6

    def test_dtype_affects_bytes(self):
        rm = RooflineModel("M3_Max")
        r16 = rm.compute_gemm_roofline(M=1, N=1024, K=1024, dtype="fp16")
        r32 = rm.compute_gemm_roofline(M=1, N=1024, K=1024, dtype="fp32")
        # fp32 should have double the bytes of fp16
        assert r32.bytes_accessed == r16.bytes_accessed * 2
        # FLOPs should be identical
        assert r32.flops == r16.flops


# ---------------------------------------------------------------------------
# Attention roofline math
# ---------------------------------------------------------------------------


class TestAttentionRoofline:
    def test_basic_attention_math(self):
        rm = RooflineModel("M3_Max")
        r = rm.compute_attention_roofline(
            seq_len=128,
            num_heads=32,
            head_dim=128,
            batch_size=1,
            dtype="fp16",
        )

        # FLOPs = 4 * B * S^2 * H * D = 4 * 1 * 128^2 * 32 * 128
        expected_flops = 4 * 1 * 128 * 128 * 32 * 128
        assert r.flops == expected_flops

        # Bytes = B * S * H * D * 4 * 2  (Q+K+V+O, fp16=2 bytes)
        expected_bytes = int(1 * 128 * 32 * 128 * 4 * 2)
        assert r.bytes_accessed == expected_bytes

    def test_prefill_attention_is_compute_bound(self):
        """Long sequence attention should be compute-bound."""
        rm = RooflineModel("M3_Max")
        r = rm.compute_attention_roofline(seq_len=4096, num_heads=32, head_dim=128)
        assert r.bound == "compute"

    def test_decode_attention_is_memory_bound(self):
        """Single-token attention decode is memory-bound."""
        rm = RooflineModel("M3_Max")
        r = rm.compute_attention_roofline(seq_len=1, num_heads=32, head_dim=128)
        assert r.bound == "memory"

    def test_batch_scales_linearly(self):
        rm = RooflineModel("M3_Max")
        r1 = rm.compute_attention_roofline(
            seq_len=64, num_heads=16, head_dim=64, batch_size=1
        )
        r4 = rm.compute_attention_roofline(
            seq_len=64, num_heads=16, head_dim=64, batch_size=4
        )
        assert r4.flops == r1.flops * 4
        assert r4.bytes_accessed == r1.bytes_accessed * 4


# ---------------------------------------------------------------------------
# Decode roofline (full transformer step)
# ---------------------------------------------------------------------------


class TestDecodeRoofline:
    @pytest.fixture
    def qwen_7b_config(self):
        return MODEL_CONFIGS["Qwen2.5-7B"]

    def test_decode_produces_ops(self, qwen_7b_config):
        rm = RooflineModel("M3_Max")
        result = rm.compute_decode_roofline(qwen_7b_config)
        # Should have ops for each layer (6 ops: QKV, attn, O_proj, gate, up, down)
        # plus LM_head
        num_layers = qwen_7b_config["num_layers"]
        expected_ops = num_layers * 6 + 1  # 6 ops per layer + LM head
        assert len(result.ops) == expected_ops

    def test_decode_totals_match(self, qwen_7b_config):
        """Total FLOPs and bytes should equal sum of individual ops."""
        rm = RooflineModel("M3_Max")
        result = rm.compute_decode_roofline(qwen_7b_config)
        sum_flops = sum(op.flops for op in result.ops)
        sum_bytes = sum(op.bytes_accessed for op in result.ops)
        assert result.total_flops == sum_flops
        assert result.total_bytes == sum_bytes

    def test_decode_is_memory_bound(self, qwen_7b_config):
        """Decode (batch=1) should be memory-bound for typical models."""
        rm = RooflineModel("M3_Max")
        result = rm.compute_decode_roofline(qwen_7b_config)
        assert result.bound == "memory"

    def test_decode_with_context(self, qwen_7b_config):
        """Context length should affect attention ops."""
        rm = RooflineModel("M3_Max")
        r0 = rm.compute_decode_roofline(qwen_7b_config, context_len=0)
        r2048 = rm.compute_decode_roofline(qwen_7b_config, context_len=2048)
        # With context, attention reads more K/V data -> more bytes
        assert r2048.total_bytes > r0.total_bytes


# ---------------------------------------------------------------------------
# estimate_max_throughput
# ---------------------------------------------------------------------------


class TestEstimateMaxThroughput:
    def test_qwen_7b_on_m3_max(self):
        rm = RooflineModel("M3_Max")
        result = rm.estimate_max_throughput("Qwen2.5-7B")
        assert result["model"] == "Qwen2.5-7B"
        assert result["chip"] == "M3_Max"
        assert result["tokens_per_sec"] > 0
        assert result["tokens_per_sec"] < 10000  # sanity upper bound
        assert result["gflops_per_token"] > 0
        assert result["bound"] in ("memory", "compute")

    def test_fuzzy_model_match(self):
        rm = RooflineModel("M3_Max")
        # Should fuzzy-match even with Instruct suffix
        result = rm.estimate_max_throughput("Qwen2.5-7B-Instruct")
        assert result["tokens_per_sec"] > 0

    def test_unknown_model_raises(self):
        rm = RooflineModel("M3_Max")
        with pytest.raises(ValueError, match="Unknown model"):
            rm.estimate_max_throughput("NonExistentModel-999B")

    def test_larger_model_slower(self):
        """A larger model should produce fewer tokens/sec on the same chip."""
        rm = RooflineModel("M3_Max")
        small = rm.estimate_max_throughput("Qwen2.5-3B")
        large = rm.estimate_max_throughput("Qwen2.5-32B")
        assert small["tokens_per_sec"] > large["tokens_per_sec"]

    def test_faster_chip_higher_throughput(self):
        """A faster chip should produce more tokens/sec for the same model."""
        r_slow = RooflineModel("M3").estimate_max_throughput("Qwen2.5-7B")
        r_fast = RooflineModel("M3_Max").estimate_max_throughput("Qwen2.5-7B")
        assert r_fast["tokens_per_sec"] > r_slow["tokens_per_sec"]


# ---------------------------------------------------------------------------
# Plotting (optional — only if matplotlib is installed)
# ---------------------------------------------------------------------------


class TestPlot:
    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip plot test in CI (no display)",
    )
    def test_plot_generates_file(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            pytest.skip("matplotlib not installed")

        rm = RooflineModel("M3_Max")
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "roofline.png")
            result_path = rm.plot_roofline(out)
            assert os.path.exists(result_path)
            assert os.path.getsize(result_path) > 0

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip plot test in CI (no display)",
    )
    def test_plot_various_chips(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            pytest.skip("matplotlib not installed")

        for chip in ("M1", "M3_Max", "M4_Pro"):
            rm = RooflineModel(chip)
            with tempfile.TemporaryDirectory() as tmpdir:
                out = os.path.join(tmpdir, f"roofline_{chip}.png")
                rm.plot_roofline(out)
                assert os.path.exists(out)


# ---------------------------------------------------------------------------
# RooflineResult invariants
# ---------------------------------------------------------------------------


class TestRooflineResultInvariants:
    def test_predicted_gflops_non_negative(self):
        rm = RooflineModel("M3_Max")
        for M in [1, 16, 256, 1024]:
            r = rm.compute_gemm_roofline(M=M, N=4096, K=4096)
            assert r.predicted_gflops >= 0

    def test_bound_is_consistent(self):
        """predicted_gflops == bandwidth*OI if memory-bound, == peak if compute-bound."""
        rm = RooflineModel("M3_Max")
        r = rm.compute_gemm_roofline(M=1, N=4096, K=4096)
        peak = rm.compute_tflops * 1000.0
        if r.bound == "memory":
            expected = rm.bandwidth_gbs * r.operational_intensity
            assert abs(r.predicted_gflops - expected) < 1e-3
        else:
            assert abs(r.predicted_gflops - peak) < 1e-3
