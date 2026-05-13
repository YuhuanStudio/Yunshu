"""Tests for adaptive hardware defaults — oMLX pattern auto-tuning."""
import pytest

from yunshu_engine.utils.hardware import (
    HardwareInfo,
    compute_adaptive_defaults,
    get_hardware_profile,
    parse_chip_info,
)


class TestParseChipInfo:
    def test_m1(self):
        gen, tier = parse_chip_info("Apple M1")
        assert gen == "M1"
        assert tier == ""

    def test_m1_pro(self):
        gen, tier = parse_chip_info("Apple M1 Pro")
        assert gen == "M1"
        assert tier == "Pro"

    def test_m2_max(self):
        gen, tier = parse_chip_info("Apple M2 Max")
        assert gen == "M2"
        assert tier == "Max"

    def test_m3_ultra(self):
        gen, tier = parse_chip_info("Apple M3 Ultra")
        assert gen == "M3"
        assert tier == "Ultra"

    def test_unknown(self):
        gen, tier = parse_chip_info("Unknown CPU")
        assert gen == "M1"
        assert tier == ""


class TestComputeAdaptiveDefaults:
    def test_8gb_base(self):
        hw = HardwareInfo(
            chip_name="Apple M1",
            total_memory_gb=8.0,
            max_working_set_bytes=6 * 1024**3,
            gpu_cores=8,
        )
        overrides = compute_adaptive_defaults(hw)
        assert overrides["batch_size"] == 1
        assert overrides["memory_pressure_threshold"] == 70.0
        assert overrides["kv_cache_quant_bits"] == 4
        assert overrides["prefix_cache_max_entries"] == 16
        assert overrides["prefill_chunk_size"] == 1024
        assert overrides["ssd_cache_enabled"] is True

    def test_16gb_pro(self):
        hw = HardwareInfo(
            chip_name="Apple M2 Pro",
            total_memory_gb=16.0,
            max_working_set_bytes=12 * 1024**3,
            gpu_cores=12,
        )
        overrides = compute_adaptive_defaults(hw)
        assert overrides["batch_size"] == 4  # Pro=2, 16GB => *2
        assert overrides["memory_pressure_threshold"] == 80.0
        assert overrides["ngram_spec_enabled"] is True
        assert overrides["prefill_chunk_size"] == 2048
        assert overrides["prefix_cache_max_entries"] == 32

    def test_32gb_max(self):
        hw = HardwareInfo(
            chip_name="Apple M3 Max",
            total_memory_gb=36.0,
            max_working_set_bytes=27 * 1024**3,
            gpu_cores=30,
        )
        overrides = compute_adaptive_defaults(hw)
        # 36GB > 32 => Max=4, *4 = 16
        assert overrides["batch_size"] == 16
        assert overrides["prefill_chunk_size"] == 4096
        assert overrides["ngram_spec_enabled"] is True

    def test_64gb_ultra(self):
        hw = HardwareInfo(
            chip_name="Apple M4 Ultra",
            total_memory_gb=64.0,
            max_working_set_bytes=48 * 1024**3,
            gpu_cores=40,
        )
        overrides = compute_adaptive_defaults(hw)
        assert overrides["batch_size"] == 32  # Ultra=8, 32+ => *4
        assert overrides["prefix_cache_max_entries"] == 64
        assert overrides["prefill_chunk_size"] == 4096

    def test_max_kv_cache_memory(self):
        hw = HardwareInfo(
            chip_name="Apple M1",
            total_memory_gb=8.0,
            max_working_set_bytes=6 * 1024**3,
        )
        overrides = compute_adaptive_defaults(hw)
        # 60% of working set
        assert overrides["max_kv_cache_memory"] == int(6 * 1024**3 * 0.60)

    def test_ssd_cache_on_small_memory(self):
        hw = HardwareInfo(
            chip_name="Apple M1",
            total_memory_gb=8.0,
            max_working_set_bytes=6 * 1024**3,
        )
        overrides = compute_adaptive_defaults(hw)
        assert overrides["ssd_cache_enabled"] is True
        assert overrides["ssd_cache_max_gb"] <= 10

    def test_stream_keepalive_small(self):
        hw = HardwareInfo(
            chip_name="Apple M1",
            total_memory_gb=8.0,
            max_working_set_bytes=6 * 1024**3,
        )
        overrides = compute_adaptive_defaults(hw)
        assert overrides["stream_keepalive_interval"] == 10.0

    def test_no_quant_on_32gb(self):
        hw = HardwareInfo(
            chip_name="Apple M3 Max",
            total_memory_gb=36.0,
            max_working_set_bytes=27 * 1024**3,
        )
        overrides = compute_adaptive_defaults(hw)
        assert "kv_cache_quant_bits" not in overrides

    def test_returns_dict_only_overrides(self):
        hw = HardwareInfo(
            chip_name="Apple M1",
            total_memory_gb=8.0,
            max_working_set_bytes=6 * 1024**3,
        )
        overrides = compute_adaptive_defaults(hw)
        assert isinstance(overrides, dict)
        assert all(v is not None for v in overrides.values())


class TestGetHardwareProfile:
    def test_returns_dict(self):
        profile = get_hardware_profile()
        assert isinstance(profile, dict)
        assert "chip_name" in profile
        assert "chip_generation" in profile
        assert "chip_tier" in profile
        assert "total_memory_gb" in profile
        assert "working_set_gb" in profile
        assert "adaptive_defaults" in profile

    def test_adaptive_defaults_populated(self):
        profile = get_hardware_profile()
        assert isinstance(profile["adaptive_defaults"], dict)
        assert len(profile["adaptive_defaults"]) > 0
