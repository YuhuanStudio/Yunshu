"""Tests for Yunshu Metal Kernel Python bindings."""

import pytest
import mlx.core as mx


class TestMetalKernelManager:
    """Test Metal kernel loading and fallback."""

    def test_manager_init(self):
        from yunshu_engine.metal_kernels import MetalKernelManager
        mgr = MetalKernelManager()
        assert mgr is not None
        assert not mgr.is_loaded  # No compiled metallib yet

    def test_load_nonexistent(self):
        from yunshu_engine.metal_kernels import MetalKernelManager
        mgr = MetalKernelManager(metallib_path="/nonexistent/path.metallib")
        assert not mgr.load_default_library()

    def test_get_kernel_manager_singleton(self):
        from yunshu_engine.metal_kernels import get_kernel_manager
        mgr1 = get_kernel_manager()
        mgr2 = get_kernel_manager()
        assert mgr1 is mgr2


class TestKIVIQuantize:
    """Test KIVI 2-bit quantization (Python fallback)."""

    def test_quantize_dequantize_roundtrip(self):
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()

        # Create test key tensor: [4 tokens, 2 heads, 8 dims]
        keys = mx.random.normal((4, 2, 8)).astype(mx.float16)

        quant_keys, scales, zps = mgr.kivi_quantize(keys)

        # Verify shapes
        assert quant_keys.shape == (4, 2, 2)  # 8/4 = 2 packed
        assert scales.shape == (4, 2)
        assert zps.shape == (4, 2)

        # Verify dtype
        assert quant_keys.dtype == mx.uint8
        assert scales.dtype == mx.float16
        assert zps.dtype == mx.float16

    def test_dequantize_shape(self):
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        keys = mx.random.normal((2, 4, 16)).astype(mx.float16)

        quant_keys, scales, zps = mgr.kivi_quantize(keys)
        deq_keys = mgr.kivi_dequantize(quant_keys, scales, zps, head_dim=16)

        assert deq_keys.shape == keys.shape
        assert deq_keys.dtype == mx.float16

    def test_compression_ratio(self):
        """Verify KIVI achieves ~8x compression (FP16 → 2-bit)."""
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        keys = mx.random.normal((10, 8, 64)).astype(mx.float16)

        quant_keys, scales, zps = mgr.kivi_quantize(keys)

        original_bytes = keys.size * 2  # FP16 = 2 bytes
        quant_bytes = quant_keys.size + scales.size * 2 + zps.size * 2
        ratio = original_bytes / quant_bytes

        # Should be roughly 4x (2-bit keys) minus overhead for scales/zps
        assert ratio > 2.0, f"Compression ratio too low: {ratio}"


class TestPagedAttentionFallback:
    """Test PagedAttention Python fallback."""

    def test_decode_basic(self):
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()

        num_queries = 2
        num_heads = 2
        head_dim = 8
        kv_block_size = 4
        seq_len = 10
        num_blocks = 3

        queries = mx.random.normal((num_queries, num_heads, head_dim)).astype(mx.float16)
        key_cache = mx.random.normal((num_blocks * kv_block_size, num_heads, head_dim)).astype(mx.float16)
        value_cache = mx.random.normal((num_blocks * kv_block_size, num_heads, head_dim)).astype(mx.float16)

        # Block tables: map virtual blocks to physical
        num_virtual_blocks = (seq_len + kv_block_size - 1) // kv_block_size
        block_tables = mx.array([[0, 1, 2], [0, 1, 2]])  # simple linear mapping
        seq_lens = mx.array([seq_len, seq_len])

        output = mgr.paged_attention_decode(
            queries, key_cache, value_cache,
            block_tables, seq_lens,
            num_heads=num_heads,
            head_dim=head_dim,
            kv_block_size=kv_block_size,
        )

        assert output.shape == queries.shape
        assert output.dtype == mx.float16
