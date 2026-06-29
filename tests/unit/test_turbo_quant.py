"""Tests for TurboQuant KV Cache — per-layer mixed-precision quantization."""

from yunshu_engine.turbo_quant import (
    TurboQuantConfig,
    TurboQuantManager,
)


class TestTurboQuantConfig:
    def test_defaults(self):
        cfg = TurboQuantConfig()
        assert not cfg.enabled
        assert cfg.fp16_end_layer == 4
        assert cfg.int8_end_layer == 16

    def test_layer_config_fp16(self):
        cfg = TurboQuantConfig(enabled=True, total_layers=32)
        layer_cfg = cfg.get_layer_config(0)
        assert layer_cfg is None  # FP16

    def test_layer_config_int8(self):
        cfg = TurboQuantConfig(enabled=True, total_layers=32)
        layer_cfg = cfg.get_layer_config(10)  # 5..16 => INT8
        assert layer_cfg is not None
        assert layer_cfg.bits == 8

    def test_layer_config_int4(self):
        cfg = TurboQuantConfig(enabled=True, total_layers=32)
        layer_cfg = cfg.get_layer_config(20)  # 17+ => INT4
        assert layer_cfg is not None
        assert layer_cfg.bits == 4

    def test_compression_ratio(self):
        cfg = TurboQuantConfig(enabled=True, total_layers=32)
        ratio = cfg.expected_compression_ratio
        assert ratio > 1.0
        assert ratio < 4.0  # Not as extreme as uniform 4-bit

    def test_compression_disabled(self):
        cfg = TurboQuantConfig(enabled=False)
        assert cfg.expected_compression_ratio == 1.0

    def test_to_dict(self):
        cfg = TurboQuantConfig(enabled=True, total_layers=32)
        d = cfg.to_dict()
        assert d["enabled"] is True
        assert "expected_compression_ratio" in d

    def test_from_model_settings_disabled(self):
        class FakeSettings:
            kv_cache_quant_bits = None

        cfg = TurboQuantManager.from_model_settings(FakeSettings())
        assert not cfg.enabled

    def test_from_model_settings_enabled(self):
        class FakeSettings:
            kv_cache_quant_bits = 4
            total_layers = 28
            kv_cache_quant_start_layer = 4
            kv_cache_quant_group_size = 64

        cfg = TurboQuantManager.from_model_settings(FakeSettings())
        assert cfg.enabled
        assert cfg.total_layers == 28


class TestTurboQuantManager:
    def test_disabled_passthrough(self):
        cfg = TurboQuantConfig(enabled=False)
        mgr = TurboQuantManager(cfg)
        data = [[1.0, 2.0, 3.0]]
        result, meta = mgr.quantize_layer(0, data)
        assert result == data
        assert meta is None

    def test_enabled_quantizes(self):
        cfg = TurboQuantConfig(
            enabled=True, total_layers=8, fp16_end_layer=1, int8_end_layer=3
        )
        mgr = TurboQuantManager(cfg)
        # Layer 0 => FP16 (no quantization)
        data = [[1.0, 2.0, 3.0]]
        result, meta = mgr.quantize_layer(0, data)
        assert result == data
        assert meta is None

        # Layer 5 => INT4
        result, meta = mgr.quantize_layer(5, data)
        assert meta is not None
        assert isinstance(result, bytes)

    def test_dequantize_fp16(self):
        cfg = TurboQuantConfig(enabled=True, total_layers=8, fp16_end_layer=1)
        mgr = TurboQuantManager(cfg)
        data = [[1.0, 2.0, 3.0]]
        result = mgr.dequantize_layer(0, data, None)
        assert result == data

    def test_dequantize_roundtrip(self):
        cfg = TurboQuantConfig(
            enabled=True, total_layers=8, fp16_end_layer=0, int8_end_layer=2
        )
        mgr = TurboQuantManager(cfg)
        # Layer 5 => INT4, use a larger tensor for meaningful quantization
        data = [[1.0] * 64, [2.0] * 64]
        packed, meta = mgr.quantize_layer(5, data)
        reconstructed = mgr.dequantize_layer(5, packed, meta)
        # Should be approximately the same
        assert len(reconstructed) == 2
        for i in range(2):
            for j in range(64):
                assert abs(reconstructed[i][j] - data[i][j]) < 0.5

    def test_stats(self):
        cfg = TurboQuantConfig(
            enabled=True, total_layers=8, fp16_end_layer=1, int8_end_layer=3
        )
        mgr = TurboQuantManager(cfg)
        stats = mgr.get_stats()
        assert stats["layers_fp16"] == 2  # 0, 1
        assert stats["layers_int8"] == 2  # 2, 3
        assert stats["layers_int4"] == 4  # 4, 5, 6, 7
        assert "config" in stats

    def test_stats_disabled(self):
        cfg = TurboQuantConfig(enabled=False)
        mgr = TurboQuantManager(cfg)
        stats = mgr.get_stats()
        assert stats["layers_fp16"] == 0
