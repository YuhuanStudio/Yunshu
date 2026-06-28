"""Tests for KV quantization module.

Covers:
- KVQuantConfig creation, validation, and properties
- KVQuantizer.quantize / dequantize roundtrip
- Packed byte format (4-bit nibbles)
- Asymmetric quantization mode
- estimate_compression ratio
- validate_accuracy metrics (MSE, max_error, cosine_similarity)
- Edge cases: empty tensors, single-element, power-of-2 sizes
"""


import pytest

from yunshu_engine.kv_quantization import (
    KVQuantConfig,
    KVQuantizer,
    _flatten_with_shape,
    _pack_nibbles,
    _unflatten_to_shape,
    _unpack_nibbles,
)

# ── KVQuantConfig ────────────────────────────────────────────────────────────


class TestKVQuantConfig:
    def test_defaults(self):
        config = KVQuantConfig()
        assert config.bits == 4
        assert config.group_size == 64
        assert config.symmetric is True

    def test_max_q(self):
        assert KVQuantConfig(bits=4).max_q == 15
        assert KVQuantConfig(bits=8).max_q == 255
        assert KVQuantConfig(bits=2).max_q == 3
        assert KVQuantConfig(bits=1).max_q == 1

    def test_compression_ratio(self):
        assert KVQuantConfig(bits=4).compression_ratio == 8.0
        assert KVQuantConfig(bits=8).compression_ratio == 4.0
        assert KVQuantConfig(bits=2).compression_ratio == 16.0
        assert KVQuantConfig(bits=1).compression_ratio == 32.0

    def test_invalid_bits_too_low(self):
        with pytest.raises(ValueError, match="bits must be in"):
            KVQuantConfig(bits=0)

    def test_invalid_bits_too_high(self):
        with pytest.raises(ValueError, match="bits must be in"):
            KVQuantConfig(bits=9)

    def test_invalid_group_size(self):
        with pytest.raises(ValueError, match="group_size must be >= 1"):
            KVQuantConfig(group_size=0)

    def test_asymmetric_mode(self):
        config = KVQuantConfig(symmetric=False)
        assert config.symmetric is False


# ── Quantize/Dequantize Roundtrip ────────────────────────────────────────────


class TestQuantizeDequantizeRoundtrip:
    def test_simple_1d_roundtrip(self):
        config = KVQuantConfig(bits=4, group_size=4)
        quantizer = KVQuantizer(config)
        original = [[[[1.0, 2.0, 3.0, 4.0]]]]
        packed, meta = quantizer.quantize(original)
        reconstructed = quantizer.dequantize(packed, meta)
        assert len(reconstructed) == 1
        assert len(reconstructed[0]) == 1
        assert len(reconstructed[0][0]) == 1
        assert len(reconstructed[0][0][0]) == 4
        for orig, recon in zip(
            [1.0, 2.0, 3.0, 4.0], reconstructed[0][0][0], strict=False
        ):
            assert abs(orig - recon) < 0.3, f"Expected ~{orig}, got {recon}"

    def test_4d_tensor_roundtrip(self):
        config = KVQuantConfig(bits=4, group_size=8)
        quantizer = KVQuantizer(config)
        # [2 layers, 3 heads, 4 seq, 16 dim] — use small range for 4-bit
        original = [
            [
                [[float(i * 0.1 + j * 0.2 + k * 0.3 + l * 0.1) for l in range(16)] for k in range(4)]
                for j in range(3)
            ]
            for i in range(2)
        ]
        packed, meta = quantizer.quantize(original)
        assert meta["shape"] == [2, 3, 4, 16]
        assert meta["num_elements"] == 2 * 3 * 4 * 16

        reconstructed = quantizer.dequantize(packed, meta)
        flat_orig, _ = _flatten_with_shape(original)
        flat_recon, _ = _flatten_with_shape(reconstructed)
        for o, r in zip(flat_orig, flat_recon, strict=False):
            assert abs(o - r) < 0.5, f"Original {o} vs reconstructed {r}"

    def test_packed_data_is_bytes(self):
        config = KVQuantConfig(bits=4, group_size=4)
        quantizer = KVQuantizer(config)
        original = [[[[1.0, 2.0, 3.0, 4.0]]]]
        packed, meta = quantizer.quantize(original)
        assert isinstance(packed, bytes)

    def test_metadata_fields(self):
        config = KVQuantConfig(bits=4, group_size=8, symmetric=True)
        quantizer = KVQuantizer(config)
        original = [[[[0.5] * 8]]]
        packed, meta = quantizer.quantize(original)
        assert "shape" in meta
        assert "bits" in meta
        assert "group_size" in meta
        assert "symmetric" in meta
        assert "scales" in meta
        assert "zero_points" in meta
        assert "num_elements" in meta
        assert meta["bits"] == 4
        assert meta["group_size"] == 8
        assert meta["symmetric"] is True
        assert len(meta["scales"]) == 1  # 1 group

    def test_large_group_size_covers_all(self):
        config = KVQuantConfig(bits=4, group_size=256)
        quantizer = KVQuantizer(config)
        original = [[[[float(i) for i in range(128)]]]]
        packed, meta = quantizer.quantize(original)
        assert len(meta["scales"]) == 1  # ceil(128/256) = 1
        reconstructed = quantizer.dequantize(packed, meta)
        flat_recon, _ = _flatten_with_shape(reconstructed)
        assert len(flat_recon) == 128


# ── Asymmetric Quantization ──────────────────────────────────────────────────


class TestAsymmetricQuantization:
    def test_asymmetric_roundtrip(self):
        config = KVQuantConfig(bits=4, group_size=4, symmetric=False)
        quantizer = KVQuantizer(config)
        original = [[[[0.1, 0.2, 0.3, 0.4]]]]
        packed, meta = quantizer.quantize(original)
        assert meta["symmetric"] is False
        assert len(meta["zero_points"]) > 0
        reconstructed = quantizer.dequantize(packed, meta)
        flat_orig, _ = _flatten_with_shape(original)
        flat_recon, _ = _flatten_with_shape(reconstructed)
        for o, r in zip(flat_orig, flat_recon, strict=False):
            assert abs(o - r) < 0.1, f"Asymmetric: expected ~{o}, got {r}"

    def test_asymmetric_negative_values(self):
        config = KVQuantConfig(bits=4, group_size=4, symmetric=False)
        quantizer = KVQuantizer(config)
        original = [[1.0, -1.0, 0.5, -0.5]]
        packed, meta = quantizer.quantize(original)
        reconstructed = quantizer.dequantize(packed, meta)
        flat_orig, _ = _flatten_with_shape(original)
        flat_recon, _ = _flatten_with_shape(reconstructed)
        for o, r in zip(flat_orig, flat_recon, strict=False):
            assert abs(o - r) < 0.3

    def test_asymmetric_has_zero_points(self):
        config = KVQuantConfig(bits=4, group_size=4, symmetric=False)
        quantizer = KVQuantizer(config)
        original = [[[[1.0, 2.0, 3.0, 4.0]]]]
        _, meta = quantizer.quantize(original)
        assert len(meta["zero_points"]) == len(meta["scales"])


# ── Edge Cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_tensor(self):
        config = KVQuantConfig()
        quantizer = KVQuantizer(config)
        packed, meta = quantizer.quantize([])
        assert packed == b""
        assert meta["num_elements"] == 0

    def test_single_element(self):
        config = KVQuantConfig(bits=4, group_size=1)
        quantizer = KVQuantizer(config)
        original = [[[[3.14]]]]
        packed, meta = quantizer.quantize(original)
        reconstructed = quantizer.dequantize(packed, meta)
        flat_recon, _ = _flatten_with_shape(reconstructed)
        assert abs(flat_recon[0] - 3.14) < 0.3

    def test_all_zeros(self):
        config = KVQuantConfig(bits=4, group_size=4)
        quantizer = KVQuantizer(config)
        original = [[[[0.0, 0.0, 0.0, 0.0]]]]
        packed, meta = quantizer.quantize(original)
        reconstructed = quantizer.dequantize(packed, meta)
        flat_recon, _ = _flatten_with_shape(reconstructed)
        for v in flat_recon:
            assert v == 0.0

    def test_group_boundary_alignment(self):
        """Test with non-aligned sizes (elements not a multiple of group_size)."""
        config = KVQuantConfig(bits=4, group_size=64)
        quantizer = KVQuantizer(config)
        # 100 elements -> 2 groups (64 + 36)
        original = [[[[float(i) for i in range(100)]]]]
        packed, meta = quantizer.quantize(original)
        assert len(meta["scales"]) == 2
        reconstructed = quantizer.dequantize(packed, meta)
        flat_recon, _ = _flatten_with_shape(reconstructed)
        assert len(flat_recon) == 100

    def test_8bit_quantization(self):
        config = KVQuantConfig(bits=8, group_size=4)
        quantizer = KVQuantizer(config)
        original = [[[[1.0, 2.0, 3.0, 4.0]]]]
        packed, meta = quantizer.quantize(original)
        reconstructed = quantizer.dequantize(packed, meta)
        flat_orig, _ = _flatten_with_shape(original)
        flat_recon, _ = _flatten_with_shape(reconstructed)
        # 8-bit should be much more precise
        for o, r in zip(flat_orig, flat_recon, strict=False):
            assert abs(o - r) < 0.05

    def test_2bit_quantization(self):
        config = KVQuantConfig(bits=2, group_size=4)
        quantizer = KVQuantizer(config)
        original = [[[[1.0, 2.0, 3.0, 4.0]]]]
        packed, meta = quantizer.quantize(original)
        reconstructed = quantizer.dequantize(packed, meta)
        # 2-bit is very coarse; just verify it runs without error
        flat_recon, _ = _flatten_with_shape(reconstructed)
        assert len(flat_recon) == 4


# ── estimate_compression ─────────────────────────────────────────────────────


class TestEstimateCompression:
    def test_4bit_ratio(self):
        config = KVQuantConfig(bits=4)
        quantizer = KVQuantizer(config)
        assert quantizer.estimate_compression(1024) == 8.0

    def test_8bit_ratio(self):
        config = KVQuantConfig(bits=8)
        quantizer = KVQuantizer(config)
        assert quantizer.estimate_compression(1024) == 4.0

    def test_1bit_ratio(self):
        config = KVQuantConfig(bits=1)
        quantizer = KVQuantizer(config)
        assert quantizer.estimate_compression(1024) == 32.0

    def test_zero_bytes(self):
        config = KVQuantConfig(bits=4)
        quantizer = KVQuantizer(config)
        assert quantizer.estimate_compression(0) == 0.0


# ── validate_accuracy ────────────────────────────────────────────────────────


class TestValidateAccuracy:
    def test_identical_tensors(self):
        config = KVQuantConfig()
        quantizer = KVQuantizer(config)
        original = [[[[1.0, 2.0, 3.0, 4.0]]]]
        metrics = quantizer.validate_accuracy(original, original)
        assert metrics["mse"] == 0.0
        assert metrics["max_error"] == 0.0
        assert metrics["cosine_similarity"] == pytest.approx(1.0)

    def test_different_tensors(self):
        config = KVQuantConfig()
        quantizer = KVQuantizer(config)
        original = [[[[1.0, 2.0, 3.0, 4.0]]]]
        reconstructed = [[[[1.1, 2.1, 3.1, 4.1]]]]
        metrics = quantizer.validate_accuracy(original, reconstructed)
        assert metrics["mse"] > 0
        assert metrics["max_error"] == pytest.approx(0.1, abs=0.01)
        assert metrics["cosine_similarity"] > 0.99

    def test_orthogonal_tensors(self):
        config = KVQuantConfig()
        quantizer = KVQuantizer(config)
        original = [[[[1.0, 0.0]]]]
        orthogonal = [[[[0.0, 1.0]]]]
        metrics = quantizer.validate_accuracy(original, orthogonal)
        assert metrics["cosine_similarity"] == pytest.approx(0.0)

    def test_empty_tensors(self):
        config = KVQuantConfig()
        quantizer = KVQuantizer(config)
        metrics = quantizer.validate_accuracy([], [])
        assert metrics["mse"] == 0.0
        assert metrics["max_error"] == 0.0
        assert metrics["cosine_similarity"] == 1.0

    def test_zero_vector(self):
        config = KVQuantConfig()
        quantizer = KVQuantizer(config)
        metrics = quantizer.validate_accuracy([[[[0.0, 0.0]]]], [[[[0.0, 0.0]]]])
        assert metrics["cosine_similarity"] == 1.0

    def test_large_tensor_metrics(self):
        config = KVQuantConfig(bits=4, group_size=16)
        quantizer = KVQuantizer(config)
        # [1, 2, 8, 32] = 512 elements
        original = [
            [
                [[float(i * 0.1 + j) for i in range(32)] for _ in range(8)]
                for j in range(2)
            ]
        ]
        packed, meta = quantizer.quantize(original)
        reconstructed = quantizer.dequantize(packed, meta)
        metrics = quantizer.validate_accuracy(original, reconstructed)
        assert metrics["mse"] >= 0
        assert metrics["max_error"] >= 0
        assert -1.0 <= metrics["cosine_similarity"] <= 1.0


# ── Packing / Unpacking ──────────────────────────────────────────────────────


class TestPacking:
    def test_pack_4bit_basic(self):
        values = [1, 2, 3, 4]
        packed = _pack_nibbles(values, 4)
        assert isinstance(packed, bytes)
        assert len(packed) == 2  # 4 values / 2 per byte

    def test_pack_4bit_odd_count(self):
        values = [1, 2, 3]
        packed = _pack_nibbles(values, 4)
        assert len(packed) == 2  # ceil(3/2) = 2 bytes

    def test_unpack_4bit_roundtrip(self):
        values = [1, 5, 10, 15]
        packed = _pack_nibbles(values, 4)
        unpacked = _unpack_nibbles(packed, 4, 4)
        assert unpacked == values

    def test_pack_8bit_identity(self):
        values = [0, 127, 255, 42]
        packed = _pack_nibbles(values, 8)
        assert packed == bytes(values)

    def test_unpack_8bit_identity(self):
        values = [0, 127, 255, 42]
        packed = _pack_nibbles(values, 8)
        unpacked = _unpack_nibbles(packed, 8, 4)
        assert unpacked == values

    def test_pack_unpack_2bit_roundtrip(self):
        values = [0, 1, 2, 3, 0, 1, 2, 3]
        packed = _pack_nibbles(values, 2)
        unpacked = _unpack_nibbles(packed, 2, len(values))
        assert unpacked == values

    def test_unpack_empty(self):
        assert _unpack_nibbles(b"", 4, 0) == []

    def test_pack_empty(self):
        assert _pack_nibbles([], 4) == b""


# ── Shape Utilities ──────────────────────────────────────────────────────────


class TestShapeUtilities:
    def test_flatten_1d(self):
        flat, shape = _flatten_with_shape([1.0, 2.0, 3.0])
        assert flat == [1.0, 2.0, 3.0]
        assert shape == [3]

    def test_flatten_2d(self):
        flat, shape = _flatten_with_shape([[1.0, 2.0], [3.0, 4.0]])
        assert flat == [1.0, 2.0, 3.0, 4.0]
        assert shape == [2, 2]

    def test_flatten_4d(self):
        flat, shape = _flatten_with_shape([[[[1.0, 2.0]]]])
        assert flat == [1.0, 2.0]
        assert shape == [1, 1, 1, 2]

    def test_unflatten_1d(self):
        result = _unflatten_to_shape([1.0, 2.0, 3.0], [3])
        assert result == [1.0, 2.0, 3.0]

    def test_unflatten_2d(self):
        result = _unflatten_to_shape([1.0, 2.0, 3.0, 4.0], [2, 2])
        assert result == [[1.0, 2.0], [3.0, 4.0]]

    def test_unflatten_4d(self):
        result = _unflatten_to_shape([1.0, 2.0], [1, 1, 1, 2])
        assert result == [[[[1.0, 2.0]]]]

    def test_flatten_unflatten_roundtrip(self):
        original = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        flat, shape = _flatten_with_shape(original)
        result = _unflatten_to_shape(flat, shape)
        assert result == original

    def test_flatten_empty(self):
        flat, shape = _flatten_with_shape([])
        assert flat == []
        assert shape == []
