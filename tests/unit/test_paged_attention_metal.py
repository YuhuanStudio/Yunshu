"""Validate Metal PagedAttention decode kernel against MLX fallback.

Tests numerical correctness: Metal kernel output must match fallback
within FP16 tolerance for various configurations.
"""
import pytest
import mlx.core as mx

from yunshu_engine.metal_kernels import MetalKernelManager


@pytest.fixture
def mgr():
    return MetalKernelManager()


def _make_paged_kv(num_heads, head_dim, num_blocks, kv_block_size, dtype=mx.float16):
    """Create a paged KV cache with random data."""
    total_slots = num_blocks * kv_block_size
    keys = mx.random.normal(shape=(total_slots, num_heads, head_dim)).astype(dtype)
    vals = mx.random.normal(shape=(total_slots, num_heads, head_dim)).astype(dtype)
    return keys, vals


def _reference_attention(queries, keys, values, scale):
    """Standard scaled dot-product attention (no paged)."""
    # queries: [num_q, num_heads, head_dim]
    # keys/values: [seq_len, num_heads, head_dim]
    scores = (queries[:, None, :, :] * keys[None, :, :, :]).sum(axis=-1) * scale
    weights = mx.softmax(scores.astype(mx.float32), axis=1).astype(queries.dtype)
    out = (weights[:, :, :, None] * values[None, :, :, :]).sum(axis=1)
    return out


@pytest.mark.parametrize("num_heads", [1, 4, 8])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_paged_attention_single_query(num_heads, head_dim, mgr):
    """Single query against a short KV cache."""
    kv_block_size = 16
    seq_len = 47  # Not aligned to block boundary
    num_blocks = 4
    scale = 1.0 / (head_dim ** 0.5)

    # Create paged KV cache
    key_cache, val_cache = _make_paged_kv(num_heads, head_dim, num_blocks, kv_block_size)

    # Simple linear block table: block 0, 1, 2
    num_blocks_needed = (seq_len + kv_block_size - 1) // kv_block_size
    block_table = mx.array([[i for i in range(num_blocks_needed)]], dtype=mx.int32)
    seq_lens = mx.array([seq_len], dtype=mx.int32)

    # Query: single token
    queries = mx.random.normal(shape=(1, num_heads, head_dim)).astype(mx.float16)

    # Metal kernel
    metal_out = mgr.paged_attention_decode(
        queries, key_cache, val_cache, block_table, seq_lens,
        num_heads, head_dim, kv_block_size, scale,
    )

    # Fallback
    fallback_out = mgr._fallback_paged_attention(
        queries, key_cache, val_cache, block_table, seq_lens,
        num_heads, head_dim, kv_block_size, scale,
    )

    # Also compare against reference (non-paged) attention
    # Gather the actual KV sequence
    k_parts, v_parts = [], []
    for b in range(num_blocks_needed):
        start = b * kv_block_size
        end = start + min(kv_block_size, seq_len - b * kv_block_size)
        k_parts.append(key_cache[start:end])
        v_parts.append(val_cache[start:end])
    keys_seq = mx.concatenate(k_parts, axis=0)
    vals_seq = mx.concatenate(v_parts, axis=0)
    ref_out = _reference_attention(queries, keys_seq, vals_seq, scale)

    mx.eval(metal_out, fallback_out, ref_out)

    # Metal vs fallback
    assert mx.allclose(metal_out, fallback_out, atol=1e-2, rtol=1e-2), \
        f"Metal vs fallback mismatch for h={num_heads}, d={head_dim}: " \
        f"max_diff={float(mx.abs(metal_out - fallback_out).max()):.4f}"

    # Metal vs reference
    assert mx.allclose(metal_out, ref_out, atol=1e-2, rtol=1e-2), \
        f"Metal vs reference mismatch for h={num_heads}, d={head_dim}: " \
        f"max_diff={float(mx.abs(metal_out - ref_out).max()):.4f}"


@pytest.mark.parametrize("seq_len", [1, 16, 47, 64, 100])
def test_paged_attention_various_seq_lengths(seq_len, mgr):
    """Different sequence lengths, including edge cases."""
    num_heads = 4
    head_dim = 64
    kv_block_size = 16
    scale = 1.0 / (head_dim ** 0.5)

    num_blocks_needed = (seq_len + kv_block_size - 1) // kv_block_size
    num_blocks = num_blocks_needed + 1

    key_cache, val_cache = _make_paged_kv(num_heads, head_dim, num_blocks, kv_block_size)
    block_table = mx.array([[i for i in range(num_blocks_needed)]], dtype=mx.int32)
    seq_lens = mx.array([seq_len], dtype=mx.int32)
    queries = mx.random.normal(shape=(1, num_heads, head_dim)).astype(mx.float16)

    metal_out = mgr.paged_attention_decode(
        queries, key_cache, val_cache, block_table, seq_lens,
        num_heads, head_dim, kv_block_size, scale,
    )
    fallback_out = mgr._fallback_paged_attention(
        queries, key_cache, val_cache, block_table, seq_lens,
        num_heads, head_dim, kv_block_size, scale,
    )
    mx.eval(metal_out, fallback_out)

    if seq_len > 0:
        assert mx.allclose(metal_out, fallback_out, atol=1e-2, rtol=1e-2), \
            f"seq_len={seq_len}: max_diff={float(mx.abs(metal_out - fallback_out).max()):.4f}"


def test_paged_attention_multi_query(mgr):
    """Multiple queries in one call."""
    num_heads = 4
    head_dim = 64
    kv_block_size = 16
    num_queries = 3
    seq_lens_list = [31, 47, 16]
    scale = 1.0 / (head_dim ** 0.5)

    max_blocks = 4
    key_cache, val_cache = _make_paged_kv(num_heads, head_dim, max_blocks, kv_block_size)

    # Block tables: each query maps blocks linearly
    block_tables = []
    for sl in seq_lens_list:
        nb = (sl + kv_block_size - 1) // kv_block_size
        row = list(range(nb)) + [-1] * (max_blocks - nb)
        block_tables.append(row)
    block_tables = mx.array(block_tables, dtype=mx.int32)
    seq_lens = mx.array(seq_lens_list, dtype=mx.int32)

    queries = mx.random.normal(shape=(num_queries, num_heads, head_dim)).astype(mx.float16)

    metal_out = mgr.paged_attention_decode(
        queries, key_cache, val_cache, block_tables, seq_lens,
        num_heads, head_dim, kv_block_size, scale,
    )
    fallback_out = mgr._fallback_paged_attention(
        queries, key_cache, val_cache, block_tables, seq_lens,
        num_heads, head_dim, kv_block_size, scale,
    )
    mx.eval(metal_out, fallback_out)

    assert mx.allclose(metal_out, fallback_out, atol=1e-2, rtol=1e-2), \
        f"max_diff={float(mx.abs(metal_out - fallback_out).max()):.4f}"


def test_paged_attention_scattered_blocks(mgr):
    """Non-contiguous block mapping (scattered physical blocks)."""
    num_heads = 4
    head_dim = 64
    kv_block_size = 16
    seq_len = 48
    scale = 1.0 / (head_dim ** 0.5)

    num_blocks = 6
    key_cache, val_cache = _make_paged_kv(num_heads, head_dim, num_blocks, kv_block_size)

    # Scattered: logical blocks [3, 0, 5] instead of [0, 1, 2]
    block_table = mx.array([[3, 0, 5]], dtype=mx.int32)
    seq_lens = mx.array([seq_len], dtype=mx.int32)
    queries = mx.random.normal(shape=(1, num_heads, head_dim)).astype(mx.float16)

    metal_out = mgr.paged_attention_decode(
        queries, key_cache, val_cache, block_table, seq_lens,
        num_heads, head_dim, kv_block_size, scale,
    )
    fallback_out = mgr._fallback_paged_attention(
        queries, key_cache, val_cache, block_table, seq_lens,
        num_heads, head_dim, kv_block_size, scale,
    )
    mx.eval(metal_out, fallback_out)

    assert mx.allclose(metal_out, fallback_out, atol=1e-2, rtol=1e-2), \
        f"max_diff={float(mx.abs(metal_out - fallback_out).max()):.4f}"
