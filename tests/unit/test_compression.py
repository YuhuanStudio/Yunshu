"""Tests for KV compression."""

import numpy as np
import pytest

from yunshu_kv.compression import (
    compute_compression_ratio,
    dequantize_kv_4bit,
    quantize_kv_4bit,
)

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestKVCompression:
    def test_quantize_dequantize_roundtrip(self):
        original = mx.array(
            np.random.randn(4, 64, 128).astype(np.float16) * 0.5,
            dtype=mx.float16,
        )
        packed, scales = quantize_kv_4bit(original, group_size=64)
        recovered = dequantize_kv_4bit(packed, scales, head_dim=128)

        orig_np = np.array(original, dtype=np.float32)
        rec_np = np.array(recovered, dtype=np.float32)
        max_error = np.max(np.abs(orig_np - rec_np))
        mean_error = np.mean(np.abs(orig_np - rec_np))

        assert max_error < 0.3, f"Max error: {max_error}"
        assert mean_error < 0.08, f"Mean error: {mean_error}"

    def test_packed_shape(self):
        original = mx.array(
            np.random.randn(4, 64, 128).astype(np.float16),
            dtype=mx.float16,
        )
        packed, scales = quantize_kv_4bit(original)

        # Packed: 128 / 2 = 64
        assert packed.shape[-1] == 64
        assert packed.shape[-2] == 64  # num_tokens

    def test_compression_ratio(self):
        assert compute_compression_ratio(4) == 4.0
        assert compute_compression_ratio(2) == 8.0
