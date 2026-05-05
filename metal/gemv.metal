/*
 * Yunshu Metal Kernels — GEMV (General Matrix-Vector Multiply).
 *
 * Optimized GEMV for LLM inference workloads:
 * - FP16 weights with FP32 accumulation
 * - Quantized (4-bit) GEMV with on-the-fly dequantization
 * - Batched GEMV for multiple sequence processing
 *
 * Group size configurable via GEMV_GROUP_SIZE function constant.
 *
 * Reference: MLX gemv.metal, llama.cpp ggml-metal
 */

#include "common.metal"

// ── FP16 GEMV ──
// Compute y = W @ x + bias
// Grid: [out_dim / 32, 1, 1] — each threadgroup handles 32 output elements

kernel void gemv_fp16(
    device const half* W [[buffer(0)]],            // [out_dim, in_dim]
    device const half* x [[buffer(1)]],            // [in_dim]
    device const half* bias [[buffer(2)]],         // [out_dim] (optional, can be nullptr)
    device half* y [[buffer(3)]],                  // [out_dim]
    constant uint& in_dim [[buffer(4)]],
    constant uint& out_dim [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint row = gid.x * simd_groups_per_threadgroup + simd_group_id;
    if (row >= out_dim) return;

    device const half* w_row = W + row * in_dim;

    // Compute dot product using SIMD
    float sum = 0.0f;
    for (uint d = simd_lane_id; d < in_dim; d += 32) {
        sum += (float)w_row[d] * (float)x[d];
    }

    // SIMD reduction
    sum = simd_reduce_sum(sum, nullptr);

    // Add bias if present
    if (bias != nullptr) {
        sum += (float)bias[row];
    }

    y[row] = (half)sum;
}


// ── Batched FP16 GEMV ──
// Compute y[i] = W @ x[i] for batch of vectors
// Grid: [out_dim / 32, batch_size, 1]

kernel void gemv_fp16_batched(
    device const half* W [[buffer(0)]],            // [out_dim, in_dim]
    device const half* x [[buffer(1)]],            // [batch_size, in_dim]
    device half* y [[buffer(2)]],                  // [batch_size, out_dim]
    constant uint& in_dim [[buffer(3)]],
    constant uint& out_dim [[buffer(4)]],
    constant uint& batch_size [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint row = gid.x * simd_groups_per_threadgroup + simd_group_id;
    uint batch_idx = gid.y;

    if (row >= out_dim || batch_idx >= batch_size) return;

    device const half* w_row = W + row * in_dim;
    device const half* x_vec = x + batch_idx * in_dim;

    float sum = 0.0f;
    for (uint d = simd_lane_id; d < in_dim; d += 32) {
        sum += (float)w_row[d] * (float)x_vec[d];
    }

    sum = simd_reduce_sum(sum, nullptr);

    y[batch_idx * out_dim + row] = (half)sum;
}


// ── 4-bit Quantized GEMV ──
// Dequantize + GEMV fused kernel
// Weight layout: [out_dim, in_dim/2] (2 x 4-bit values per byte)
// With per-row scale and zero_point

kernel void gemv_q4(
    device const uchar* W_q [[buffer(0)]],         // [out_dim, in_dim/2] packed 4-bit
    device const half* scales [[buffer(1)]],        // [out_dim] per-row scale
    device const half* zero_points [[buffer(2)]],   // [out_dim] per-row zero_point
    device const half* x [[buffer(3)]],             // [in_dim]
    device half* y [[buffer(4)]],                   // [out_dim]
    constant uint& in_dim [[buffer(5)]],
    constant uint& out_dim [[buffer(6)]],
    constant uint& group_size [[buffer(7)]],       // quantization group size (default GEMV_GROUP_SIZE)
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint row = gid.x * simd_groups_per_threadgroup + simd_group_id;
    if (row >= out_dim) return;

    float scale = (float)scales[row];
    float zp = (float)zero_points[row];

    // Compute dot product with on-the-fly dequantization
    float sum = 0.0f;
    uint packed_dim = in_dim / 2;

    for (uint d = simd_lane_id; d < packed_dim; d += 32) {
        uchar packed = W_q[row * packed_dim + d];

        // Unpack two 4-bit values
        float val_lo = (float)(packed & 0xF) - zp;
        float val_hi = (float)((packed >> 4) & 0xF) - zp;

        sum += scale * val_lo * (float)x[d * 2];
        sum += scale * val_hi * (float)x[d * 2 + 1];
    }

    sum = simd_reduce_sum(sum, nullptr);
    y[row] = (half)sum;
}
