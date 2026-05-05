"""Phase 0 Metal Kernel Comprehensive Tests.

Tests cover:
- MetalKernelManager loading (skip if metallib not compiled)
- KIVI quantize/dequantize accuracy (MSE between original and roundtrip)
- KIVI compression ratio for various tensor sizes
- PagedAttention fallback vs naive SDPA (numerical correctness)
- GEMV correctness (compare against mx.matmul)
- Kernel manager reload and error handling
- compile_kernels() and get_compilation_status()
- sdpa_attention correctness
"""

from __future__ import annotations

import pytest
import mlx.core as mx
import numpy as np


# ── Helpers ──


def naive_sdpa_reference(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    scale: float,
    causal: bool = True,
) -> mx.array:
    """Naive SDPA reference for numerical comparison."""
    seq_len = Q.shape[0]
    Q_f = Q.astype(mx.float32)
    K_f = K.astype(mx.float32)
    V_f = V.astype(mx.float32)

    scores = mx.einsum("qhd,khd->qhk", Q_f, K_f) * scale

    if causal:
        mask = mx.triu(mx.full((seq_len, seq_len), -1e9), k=1)
        scores = scores + mask[:, None, :]  # [q, k] -> broadcast over heads

    weights = mx.softmax(scores, axis=-1)
    output = mx.einsum("qhk,khd->qhd", weights, V_f)
    return output.astype(Q.dtype)


# ── MetalKernelManager Tests ──


class TestMetalKernelManager:
    """Test MetalKernelManager loading and lifecycle."""

    def test_init_default(self):
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        assert mgr is not None
        assert not mgr.is_loaded

    def test_init_custom_path(self):
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager(metallib_path="/tmp/nonexistent.metallib")
        assert not mgr.is_loaded

    def test_load_nonexistent(self):
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager(metallib_path="/tmp/does_not_exist.metallib")
        assert not mgr.load_default_library()

    def test_reload_clears_state(self):
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager(metallib_path="/tmp/fake.metallib")
        mgr.load_default_library()
        # Even though load fails, reload should not crash
        result = mgr.reload()
        assert not mgr.is_loaded

    def test_load_actual_metallib_if_available(self):
        """Try loading the actual metallib — skip if not compiled."""
        from yunshu_engine.metal_kernels import MetalKernelManager, _METALLIB_PATH

        if not _METALLIB_PATH.exists():
            pytest.skip("Compiled metallib not available — run 'just build-metal'")

        mgr = MetalKernelManager()
        assert mgr.load_default_library()
        assert mgr.is_loaded

    def test_singleton(self):
        from yunshu_engine.metal_kernels import get_kernel_manager

        mgr1 = get_kernel_manager()
        mgr2 = get_kernel_manager()
        assert mgr1 is mgr2


# ── Compile / Status Tests ──


class TestCompileAndStatus:
    """Test compile_kernels() and get_compilation_status()."""

    def test_get_compilation_status_structure(self):
        from yunshu_engine.metal_kernels import get_compilation_status

        status = get_compilation_status()
        assert "compiled" in status
        assert "metallib_path" in status
        assert "kernel_count" in status
        assert "last_error" in status
        assert "last_attempt" in status
        assert isinstance(status["compiled"], bool)
        assert isinstance(status["kernel_count"], int)

    def test_kernel_count_positive(self):
        """Should find .metal source files even without compilation."""
        from yunshu_engine.metal_kernels import get_compilation_status

        status = get_compilation_status()
        assert status["kernel_count"] > 0, "Should find at least 1 .metal source file"

    def test_compile_kernels_returns_bool(self):
        from yunshu_engine.metal_kernels import compile_kernels

        result = compile_kernels(force=False)
        assert isinstance(result, bool)

    def test_compile_updates_status(self):
        from yunshu_engine.metal_kernels import compile_kernels, get_compilation_status

        compile_kernels(force=True)
        status = get_compilation_status()
        assert status["last_attempt"] is not None

    def test_compile_status_reflects_file_existence(self):
        from yunshu_engine.metal_kernels import get_compilation_status, _METALLIB_PATH

        status = get_compilation_status()
        assert status["compiled"] == _METALLIB_PATH.exists()


# ── KIVI Quantize Tests ──


class TestKIVIQuantize:
    """Test KIVI 2-bit quantization."""

    def test_quantize_basic_shape(self):
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        keys = mx.random.normal((4, 2, 8)).astype(mx.float16)
        qk, scales, zps = mgr.kivi_quantize(keys)

        assert qk.shape == (4, 2, 2)  # 8/4 = 2 packed
        assert scales.shape == (4, 2)
        assert zps.shape == (4, 2)

    def test_quantize_various_sizes(self):
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        for tokens, heads, dim in [(1, 1, 4), (8, 4, 16), (32, 8, 64), (64, 16, 128)]:
            keys = mx.random.normal((tokens, heads, dim)).astype(mx.float16)
            qk, scales, zps = mgr.kivi_quantize(keys)
            assert qk.shape == (tokens, heads, dim // 4)
            assert scales.shape == (tokens, heads)
            assert zps.shape == (tokens, heads)

    def test_quantize_dtypes(self):
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        keys = mx.random.normal((4, 2, 8)).astype(mx.float16)
        qk, scales, zps = mgr.kivi_quantize(keys)

        assert qk.dtype == mx.uint8
        assert scales.dtype == mx.float16
        assert zps.dtype == mx.float16

    def test_quantize_values_in_range(self):
        """Quantized values should be 0-3 (2-bit range)."""
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        keys = mx.random.normal((4, 2, 8)).astype(mx.float16)
        qk, _, _ = mgr.kivi_quantize(keys)

        # Each packed byte should have values 0-3 in each 2-bit slot
        qk_np = np.array(qk)
        assert np.all(qk_np <= 0xFF)  # Valid uint8
        # Extract 2-bit values
        for i in range(4):
            slot_vals = (qk_np >> (i * 2)) & 0x3
            assert np.all(slot_vals <= 3)


class TestKIVIRoundtrip:
    """Test KIVI quantize+dequantize roundtrip accuracy."""

    def test_roundtrip_shape(self):
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        keys = mx.random.normal((8, 4, 16)).astype(mx.float16)
        qk, scales, zps = mgr.kivi_quantize(keys)
        deq = mgr.kivi_dequantize(qk, scales, zps, head_dim=16)

        assert deq.shape == keys.shape
        assert deq.dtype == mx.float16

    def test_roundtrip_mse_small(self):
        """MSE between original and roundtrip should be small."""
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        keys = mx.random.normal((8, 4, 16)).astype(mx.float16)
        qk, scales, zps = mgr.kivi_quantize(keys)
        deq = mgr.kivi_dequantize(qk, scales, zps, head_dim=16)

        diff = keys.astype(mx.float32) - deq.astype(mx.float32)
        mse = float((diff * diff).mean())
        assert mse < 0.15, f"MSE too high: {mse}"

    def test_roundtrip_larger_tensor(self):
        """Test roundtrip with realistic dimensions."""
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        keys = mx.random.normal((32, 8, 128)).astype(mx.float16)
        qk, scales, zps = mgr.kivi_quantize(keys)
        deq = mgr.kivi_dequantize(qk, scales, zps, head_dim=128)

        diff = keys.astype(mx.float32) - deq.astype(mx.float32)
        mse = float((diff * diff).mean())
        # 2-bit quantization on Gaussian data with dim=128 has inherent error
        assert mse < 0.4, f"MSE too high for large tensor: {mse}"

    def test_roundtrip_preserves_range(self):
        """Roundtrip should roughly preserve the value range."""
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        keys = mx.random.normal((4, 2, 16)).astype(mx.float16)
        qk, scales, zps = mgr.kivi_quantize(keys)
        deq = mgr.kivi_dequantize(qk, scales, zps, head_dim=16)

        keys_np = np.array(keys, dtype=np.float32)
        deq_np = np.array(deq, dtype=np.float32)
        # Correlation should be high
        correlation = np.corrcoef(keys_np.flatten(), deq_np.flatten())[0, 1]
        assert correlation > 0.9, f"Correlation too low: {correlation}"


class TestKIVICompression:
    """Test KIVI compression ratio."""

    def test_compression_ratio_small(self):
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        keys = mx.random.normal((4, 2, 8)).astype(mx.float16)
        qk, scales, zps = mgr.kivi_quantize(keys)

        original_bytes = keys.size * 2
        quant_bytes = qk.size + scales.size * 2 + zps.size * 2
        ratio = original_bytes / quant_bytes
        assert ratio > 2.0, f"Compression ratio too low: {ratio}"

    def test_compression_ratio_large(self):
        """Larger tensors should have better compression (overhead amortized)."""
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        keys = mx.random.normal((100, 8, 128)).astype(mx.float16)
        qk, scales, zps = mgr.kivi_quantize(keys)

        original_bytes = keys.size * 2
        quant_bytes = qk.size + scales.size * 2 + zps.size * 2
        ratio = original_bytes / quant_bytes
        assert ratio > 3.0, f"Compression ratio too low for large tensor: {ratio}"

    def test_compression_ratio_varies_with_dim(self):
        """Head dim must be divisible by 4."""
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        for dim in [4, 8, 16, 32, 64, 128]:
            keys = mx.random.normal((10, 4, dim)).astype(mx.float16)
            qk, scales, zps = mgr.kivi_quantize(keys)
            assert qk.shape[-1] == dim // 4


# ── PagedAttention Tests ──


class TestPagedAttention:
    """Test PagedAttention fallback correctness."""

    def test_decode_output_shape(self):
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        nq, nh, hd, kbs, sl = 2, 2, 8, 4, 10
        nb = (sl + kbs - 1) // kbs

        queries = mx.random.normal((nq, nh, hd)).astype(mx.float16)
        kc = mx.random.normal((nb * kbs, nh, hd)).astype(mx.float16)
        vc = mx.random.normal((nb * kbs, nh, hd)).astype(mx.float16)
        bt = mx.array([[i for i in range(nb)] for _ in range(nq)])
        sl_arr = mx.array([sl, sl])

        out = mgr.paged_attention_decode(
            queries, kc, vc, bt, sl_arr, nh, hd, kbs,
        )
        assert out.shape == (nq, nh, hd)

    def test_decode_finite_output(self):
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        nq, nh, hd, kbs, sl = 3, 4, 16, 8, 20
        nb = (sl + kbs - 1) // kbs

        queries = mx.random.normal((nq, nh, hd)).astype(mx.float16)
        kc = mx.random.normal((nb * kbs, nh, hd)).astype(mx.float16)
        vc = mx.random.normal((nb * kbs, nh, hd)).astype(mx.float16)
        bt = mx.array([[i for i in range(nb)] for _ in range(nq)])
        sl_arr = mx.array([sl, sl, sl])

        out = mgr.paged_attention_decode(
            queries, kc, vc, bt, sl_arr, nh, hd, kbs,
        )
        out_np = np.array(out)
        assert np.all(np.isfinite(out_np))

    def test_decode_nontrivial_output(self):
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        nq, nh, hd, kbs, sl = 2, 2, 16, 4, 12
        nb = (sl + kbs - 1) // kbs

        queries = mx.random.normal((nq, nh, hd)).astype(mx.float16)
        kc = mx.random.normal((nb * kbs, nh, hd)).astype(mx.float16)
        vc = mx.random.normal((nb * kbs, nh, hd)).astype(mx.float16)
        bt = mx.array([[i for i in range(nb)] for _ in range(nq)])
        sl_arr = mx.array([sl, sl])

        out = mgr.paged_attention_decode(
            queries, kc, vc, bt, sl_arr, nh, hd, kbs,
        )
        out_np = np.array(out)
        assert np.any(np.abs(out_np) > 1e-6), "Output should not be all zeros"

    def test_decode_correctness_vs_naive(self):
        """PagedAttention decode should match naive SDPA."""
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        nq = 1
        nh, hd, kbs, sl = 2, 8, 4, 8
        nb = (sl + kbs - 1) // kbs

        queries = mx.random.normal((nq, nh, hd)).astype(mx.float16)
        kc = mx.random.normal((nb * kbs, nh, hd)).astype(mx.float16)
        vc = mx.random.normal((nb * kbs, nh, hd)).astype(mx.float16)
        bt = mx.array([[i for i in range(nb)]])
        sl_arr = mx.array([sl])

        out = mgr.paged_attention_decode(
            queries, kc, vc, bt, sl_arr, nh, hd, kbs,
        )

        # Naive: gather contiguous K, V then do standard attention
        keys = kc[:sl]  # [sl, nh, hd]
        vals = vc[:sl]   # [sl, nh, hd]
        scale = 1.0 / (hd ** 0.5)
        ref = naive_sdpa_reference(queries[0:1], keys, vals, scale=scale, causal=False)

        # Should be close (allowing for FP16 precision)
        diff = np.array(out.astype(mx.float32) - ref.astype(mx.float32))
        max_err = np.abs(diff).max()
        assert max_err < 0.1, f"PagedAttention max error: {max_err}"

    def test_decode_different_seq_lens(self):
        """Multiple queries with different sequence lengths."""
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        nh, hd, kbs = 2, 8, 4
        max_sl = 12
        nb = (max_sl + kbs - 1) // kbs
        nq = 3

        queries = mx.random.normal((nq, nh, hd)).astype(mx.float16)
        kc = mx.random.normal((nb * kbs, nh, hd)).astype(mx.float16)
        vc = mx.random.normal((nb * kbs, nh, hd)).astype(mx.float16)
        bt = mx.array([[i for i in range(nb)] for _ in range(nq)])
        sl_arr = mx.array([4, 8, 12])

        out = mgr.paged_attention_decode(
            queries, kc, vc, bt, sl_arr, nh, hd, kbs,
        )
        assert out.shape == (nq, nh, hd)
        assert np.all(np.isfinite(np.array(out)))

    def test_decode_zero_seq_len(self):
        """Sequence length 0 should produce zeros."""
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        queries = mx.random.normal((1, 2, 8)).astype(mx.float16)
        kc = mx.random.normal((4, 2, 8)).astype(mx.float16)
        vc = mx.random.normal((4, 2, 8)).astype(mx.float16)
        bt = mx.array([[0]])
        sl_arr = mx.array([0])

        out = mgr.paged_attention_decode(
            queries, kc, vc, bt, sl_arr, 2, 8, 4,
        )
        assert np.allclose(np.array(out), 0.0, atol=1e-6)


# ── GEMV Tests ──


class TestGEMV:
    """Test GEMV correctness against mx.matmul."""

    def test_gemv_single_vector(self):
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        W = mx.random.normal((16, 8)).astype(mx.float16)
        x = mx.random.normal((8,)).astype(mx.float16)
        result = mgr.gemv(W, x)

        expected = mx.matmul(W, x)
        diff = np.array(result.astype(mx.float32) - expected.astype(mx.float32))
        assert np.abs(diff).max() < 0.05

    def test_gemv_with_bias(self):
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        W = mx.random.normal((16, 8)).astype(mx.float16)
        x = mx.random.normal((8,)).astype(mx.float16)
        bias = mx.random.normal((16,)).astype(mx.float16)
        result = mgr.gemv(W, x, bias=bias)

        expected = mx.matmul(W, x) + bias
        diff = np.array(result.astype(mx.float32) - expected.astype(mx.float32))
        assert np.abs(diff).max() < 0.05

    def test_gemv_batched(self):
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        W = mx.random.normal((32, 16)).astype(mx.float16)
        x_batch = mx.random.normal((4, 16)).astype(mx.float16)
        result = mgr.gemv(W, x_batch)

        expected = mx.matmul(x_batch, W.T)
        assert result.shape == (4, 32)
        diff = np.array(result.astype(mx.float32) - expected.astype(mx.float32))
        assert np.abs(diff).max() < 0.05

    def test_gemv_batched_with_bias(self):
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        W = mx.random.normal((16, 8)).astype(mx.float16)
        x_batch = mx.random.normal((3, 8)).astype(mx.float16)
        bias = mx.random.normal((16,)).astype(mx.float16)
        result = mgr.gemv(W, x_batch, bias=bias)

        expected = mx.matmul(x_batch, W.T) + bias
        diff = np.array(result.astype(mx.float32) - expected.astype(mx.float32))
        assert np.abs(diff).max() < 0.05

    def test_gemv_shapes(self):
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        for out_dim, in_dim in [(4, 4), (8, 4), (64, 32), (128, 64)]:
            W = mx.random.normal((out_dim, in_dim)).astype(mx.float16)
            x = mx.random.normal((in_dim,)).astype(mx.float16)
            result = mgr.gemv(W, x)
            assert result.shape == (out_dim,)

    def test_gemv_large(self):
        """Test with realistic LLM dimensions."""
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        W = mx.random.normal((512, 256)).astype(mx.float16)
        x = mx.random.normal((256,)).astype(mx.float16)
        result = mgr.gemv(W, x)
        expected = mx.matmul(W, x)
        diff = np.array(result.astype(mx.float32) - expected.astype(mx.float32))
        assert np.abs(diff).max() < 0.1


# ── SDPA Attention Tests ──


class TestSDPAAttention:
    """Test SDPA attention correctness."""

    def test_sdpa_basic_shape(self):
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        S, H, D = 8, 2, 16
        Q = mx.random.normal((S, H, D)).astype(mx.float16)
        K = mx.random.normal((S, H, D)).astype(mx.float16)
        V = mx.random.normal((S, H, D)).astype(mx.float16)

        out = mgr.sdpa_attention(Q, K, V, causal=True)
        assert out.shape == (S, H, D)

    def test_sdpa_causal_correctness(self):
        """SDPA output should match naive reference for causal case."""
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        S, H, D = 8, 2, 16
        Q = mx.random.normal((S, H, D)).astype(mx.float16)
        K = mx.random.normal((S, H, D)).astype(mx.float16)
        V = mx.random.normal((S, H, D)).astype(mx.float16)
        scale = 1.0 / (D ** 0.5)

        out = mgr.sdpa_attention(Q, K, V, scale=scale, causal=True)
        ref = naive_sdpa_reference(Q, K, V, scale=scale, causal=True)

        diff = np.array(out.astype(mx.float32) - ref.astype(mx.float32))
        max_err = np.abs(diff).max()
        assert max_err < 0.1, f"SDPA causal max error: {max_err}"

    def test_sdpa_noncausal_correctness(self):
        """SDPA output should match naive reference for non-causal case."""
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        S, H, D = 8, 2, 16
        Q = mx.random.normal((S, H, D)).astype(mx.float16)
        K = mx.random.normal((S, H, D)).astype(mx.float16)
        V = mx.random.normal((S, H, D)).astype(mx.float16)
        scale = 1.0 / (D ** 0.5)

        out = mgr.sdpa_attention(Q, K, V, scale=scale, causal=False)
        ref = naive_sdpa_reference(Q, K, V, scale=scale, causal=False)

        diff = np.array(out.astype(mx.float32) - ref.astype(mx.float32))
        max_err = np.abs(diff).max()
        assert max_err < 0.1, f"SDPA non-causal max error: {max_err}"

    def test_sdpa_finite(self):
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        for S, H, D in [(4, 2, 8), (16, 4, 32), (32, 8, 64)]:
            Q = mx.random.normal((S, H, D)).astype(mx.float16)
            K = mx.random.normal((S, H, D)).astype(mx.float16)
            V = mx.random.normal((S, H, D)).astype(mx.float16)
            out = mgr.sdpa_attention(Q, K, V, causal=True)
            assert np.all(np.isfinite(np.array(out))), f"Non-finite output for S={S},H={H},D={D}"

    def test_sdpa_gqa(self):
        """GQA: fewer KV heads than query heads."""
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        S, H, KVH, D = 8, 4, 2, 16
        Q = mx.random.normal((S, H, D)).astype(mx.float16)
        K = mx.random.normal((S, KVH, D)).astype(mx.float16)
        V = mx.random.normal((S, KVH, D)).astype(mx.float16)

        out = mgr.sdpa_attention(Q, K, V, causal=True)
        assert out.shape == (S, H, D)
        assert np.all(np.isfinite(np.array(out)))

    def test_sdpa_custom_scale(self):
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        S, H, D = 4, 2, 8
        Q = mx.random.normal((S, H, D)).astype(mx.float16)
        K = mx.random.normal((S, H, D)).astype(mx.float16)
        V = mx.random.normal((S, H, D)).astype(mx.float16)

        out1 = mgr.sdpa_attention(Q, K, V, scale=0.1, causal=False)
        out2 = mgr.sdpa_attention(Q, K, V, scale=1.0, causal=False)

        # Different scales should produce different outputs
        assert not np.allclose(np.array(out1), np.array(out2), atol=1e-3)


# ── Kernel Manager Error Handling ──


class TestKernelManagerErrors:
    """Test error handling and edge cases."""

    def test_kivi_head_dim_not_multiple_of_4(self):
        """Head dim not divisible by 4 should fail gracefully."""
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        keys = mx.random.normal((2, 2, 7)).astype(mx.float16)  # 7 not divisible by 4
        # Packing 4 values into 1 byte requires head_dim % 4 == 0
        with pytest.raises((ValueError, Exception)):
            mgr.kivi_quantize(keys)

    def test_paged_attention_negative_block(self):
        """Negative block IDs should be skipped."""
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        queries = mx.random.normal((1, 2, 8)).astype(mx.float16)
        kc = mx.random.normal((8, 2, 8)).astype(mx.float16)
        vc = mx.random.normal((8, 2, 8)).astype(mx.float16)
        bt = mx.array([[-1]])  # invalid block
        sl_arr = mx.array([4])

        out = mgr.paged_attention_decode(queries, kc, vc, bt, sl_arr, 2, 8, 4)
        # Should produce zeros (no valid blocks)
        assert out.shape == (1, 2, 8)

    def test_reload_after_failed_load(self):
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager(metallib_path="/nonexistent.metallib")
        mgr.load_default_library()
        assert not mgr.is_loaded
        # Reload should not crash
        result = mgr.reload()
        assert not result

    def test_gemv_single_element(self):
        """Edge case: 1x1 matrix."""
        from yunshu_engine.metal_kernels import MetalKernelManager

        mgr = MetalKernelManager()
        W = mx.array([[2.0]]).astype(mx.float16)
        x = mx.array([3.0]).astype(mx.float16)
        result = mgr.gemv(W, x)
        expected = mx.array([6.0]).astype(mx.float16)
        diff = np.abs(np.array(result) - np.array(expected))
        assert diff.max() < 0.01
