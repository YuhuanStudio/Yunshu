/*
 * ⚠️  DEPRECATED: These .metal files are NOT used at runtime.
 *
 * The actual Metal kernels live as inline strings in:
 *   python/yunshu_engine/metal_kernels.py
 *
 * They are compiled via mx.fast.metal_kernel (JIT), not via
 * xcrun/make precompilation. Changes here have NO effect.
 *
 * If you need to edit a kernel, edit the Python inline source.
 * These files are kept for reference only.
 */

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
// Metal 3.1: function_constant cannot be used for array sizes.
// Use #define for threadgroup/register array dimensions, tunable via -D flags.

#ifndef PA_BLOCK_Q
#define PA_BLOCK_Q 64
#endif

#ifndef PA_BLOCK_KV
#define PA_BLOCK_KV 256
#endif

#ifndef SDPA_TILE_Q
#define SDPA_TILE_Q 64
#endif

#ifndef SDPA_TILE_KV
#define SDPA_TILE_KV 64
#endif

// Maximum head dimension (for register array sizing)
#ifndef MAX_HEAD_DIM
#define MAX_HEAD_DIM 128
#endif

// Number of SIMD groups per threadgroup (Apple GPU SIMD width = 32)
#ifndef NUM_SIMD_GROUPS
#define NUM_SIMD_GROUPS 2
#endif

// KV block size for paged cache
#ifndef KV_BLOCK_SIZE
#define KV_BLOCK_SIZE 16
#endif

// GEMV group size for quantized kernels
#ifndef GEMV_GROUP_SIZE
#define GEMV_GROUP_SIZE 64
#endif

// SGMV max intermediate dimension
#ifndef SGMV_MAX_RANK
#define SGMV_MAX_RANK 64
#endif

// Function constants for runtime-configurable values (NOT array sizes)
constant uint pa_block_q  [[function_constant(0)]];
constant uint pa_block_kv [[function_constant(1)]];
constant uint pa_head_dim [[function_constant(2)]];
constant uint sdpa_tile_q  [[function_constant(3)]];
constant uint sdpa_tile_kv [[function_constant(4)]];
constant uint max_head_dim [[function_constant(5)]];
constant uint kv_block_size [[function_constant(10)]];
constant uint gemv_group_size [[function_constant(11)]];
constant uint sgmv_max_rank [[function_constant(12)]];
constant uint num_simd_groups [[function_constant(20)]];

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
inline ulong kv_block_offset(
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
    return (ulong)(physical_block) * kv_block_size * num_heads * head_dim
           + (ulong)(offset_in_block) * num_heads * head_dim
           + (ulong)(head_idx) * head_dim;
}

#endif // YUNSHU_COMMON_METAL
