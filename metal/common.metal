/*
 * Yunshu Metal Kernels — Common types and utilities.
 *
 * Shared definitions for all Yunshu Metal compute kernels.
 * Metal 3.1+ targeting Apple GPU (G14/G15 architecture).
 *
 * Function constants are specialized at pipeline creation time,
 * allowing tile sizes to be tuned per-model without recompiling
 * the Metal source. Defaults match common LLM configurations.
 */

#ifndef YUNSHU_COMMON_METAL
#define YUNSHU_COMMON_METAL

#include <metal_stdlib>
using namespace metal;

// ── Compile-time configurable tile sizes ──
// Specialized via MTLFunctionConstantValues at pipeline creation.

// PagedAttention tile sizes
constant uint PA_BLOCK_Q  [[function_constant(0)]]  = 64;   // queries per block
constant uint PA_BLOCK_KV [[function_constant(1)]]  = 64;   // keys per block
constant uint PA_HEAD_DIM [[function_constant(2)]]  = 128;  // head dimension

// SDPA (Flash Attention) tile sizes
constant uint SDPA_TILE_Q  [[function_constant(3)]] = 64;   // query tile
constant uint SDPA_TILE_KV [[function_constant(4)]] = 64;   // KV tile

// Maximum head dimension (for threadgroup array sizing)
constant uint MAX_HEAD_DIM [[function_constant(5)]] = 128;

// KV block size for paged cache
constant uint KV_BLOCK_SIZE [[function_constant(10)]] = 64;  // tokens per KV block

// GEMV group size for quantized kernels
constant uint GEMV_GROUP_SIZE [[function_constant(11)]] = 32;

// SGMV max intermediate dimension (for threadgroup sizing)
constant uint SGMV_MAX_RANK [[function_constant(12)]] = 256;

// ── Common helper functions ──

/// Fast half-precision dot product for attention scores
template <typename T, int N>
inline T dot_product(
    device const T* a,
    device const T* b,
    uint offset_a,
    uint offset_b,
    uint stride_a,
    uint stride_b
) {
    T sum = 0;
    for (int i = 0; i < N; i++) {
        sum += a[offset_a + i * stride_a] * b[offset_b + i * stride_b];
    }
    return sum;
}

/// Warp-level sum reduction (Apple GPU 32-wide SIMD)
inline float simd_reduce_sum(float val, threadgroup float* shared) {
    for (uint16_t offset = 16; offset > 0; offset >>= 1) {
        val += simd_shuffle_down(val, offset);
    }
    return val;
}

/// Warp-level max reduction
inline float simd_reduce_max(float val, threadgroup float* shared) {
    for (uint16_t offset = 16; offset > 0; offset >>= 1) {
        val = max(val, simd_shuffle_down(val, offset));
    }
    return val;
}

/// Block index for paged KV: (block_table[block_idx] * KV_BLOCK_SIZE + offset_in_block)
inline uint kv_block_offset(
    device const int* block_table,
    uint seq_pos,
    uint head_idx,
    uint num_heads,
    uint head_dim,
    uint kv_block_size
) {
    uint block_idx = seq_pos / kv_block_size;
    uint offset_in_block = seq_pos % kv_block_size;
    uint physical_block = block_table[block_idx];
    return (physical_block * kv_block_size + offset_in_block) * num_heads * head_dim
           + head_idx * head_dim;
}

#endif // YUNSHU_COMMON_METAL
