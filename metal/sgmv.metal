/*
 * Yunshu Metal Kernels — SGMV (Segmented Matrix-Vector Multiply).
 *
 * Batched GEMV for LoRA adapters: y = W @ x + Σ(A_i @ B_i @ x)
 * Each LoRA adapter has its own segment (low-rank matrices A, B).
 *
 * Max rank configurable via SGMV_MAX_RANK function constant.
 *
 * Optimized for Apple GPU's GEMV performance characteristics:
 * - 32-wide SIMD for dot products
 * - Threadgroup shared memory for intermediate results
 * - FP16 compute with FP32 accumulation
 *
 * Reference: SGMV kernel from Punica, vLLM, S-LoRA
 */

#include "common.metal"

// ── SGMV Forward Kernel ──
// Applies multiple LoRA adapters in a batched fashion.
// Each request can have a different adapter.
//
// y[seq_idx] = W @ x[seq_idx] + alpha * B[adapter_idx] @ A[adapter_idx] @ x[seg_slice]
//
// Grid: [total_tokens, 1, 1]

kernel void sgmv_forward(
    device const half* x [[buffer(0)]],            // [total_tokens, in_dim]
    device const half* lora_A [[buffer(1)]],       // [num_adapters, rank, in_dim]
    device const half* lora_B [[buffer(2)]],       // [num_adapters, out_dim, rank]
    device half* output [[buffer(3)]],             // [total_tokens, out_dim]
    device const int* segment_ids [[buffer(4)]],   // [total_tokens] → adapter index (-1 = none)
    device const int* segment_starts [[buffer(5)]],// [num_segments + 1] (unused, for future segmented dispatch)
    device const int* segment_ends [[buffer(6)]],  // [num_segments + 1] (unused)
    constant uint& in_dim [[buffer(7)]],
    constant uint& out_dim [[buffer(8)]],
    constant uint& rank [[buffer(9)]],
    constant float& alpha [[buffer(10)]],          // LoRA scaling factor
    uint3 gid [[thread_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    uint token_idx = gid.x;

    // Determine adapter for this token
    int adapter_idx = segment_ids[token_idx];
    if (adapter_idx < 0) return; // No adapter for this token

    device const half* x_ptr = x + token_idx * in_dim;

    // Step 1: Compute intermediate = A[adapter] @ x  → [rank]
    // A shape: [rank, in_dim], x shape: [in_dim]
    // Each SIMD lane processes a subset of rank elements
    threadgroup float shared_intermediate[SGMV_MAX_RANK];

    for (uint r = simd_lane_id; r < rank && r < SGMV_MAX_RANK; r += 32) {
        device const half* a_row = lora_A + adapter_idx * rank * in_dim + r * in_dim;

        float dot = 0.0f;
        for (uint d = 0; d < in_dim; d++) {
            dot += (float)a_row[d] * (float)x_ptr[d];
        }
        shared_intermediate[r] = dot;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Compute output = B[adapter] @ intermediate → [out_dim]
    // B shape: [out_dim, rank]
    // Each SIMD lane processes a subset of output dimensions
    for (uint o = simd_lane_id; o < out_dim; o += 32) {
        device const half* b_row = lora_B + adapter_idx * out_dim * rank + o * rank;

        float dot = 0.0f;
        for (uint r = 0; r < rank; r++) {
            dot += (float)b_row[r] * shared_intermediate[r];
        }

        output[token_idx * out_dim + o] += (half)(alpha * dot);
    }
}


// ── SGMV Batch GEMV (rank-1 optimization) ──
// For rank-1 LoRA adapters, fuse the A@x and B@intermediate into one pass.

kernel void sgmv_rank1(
    device const half* x [[buffer(0)]],
    device const half* lora_a [[buffer(1)]],       // [num_adapters, in_dim] (rank=1)
    device const half* lora_b [[buffer(2)]],       // [num_adapters, out_dim] (rank=1)
    device half* output [[buffer(3)]],
    device const int* adapter_ids [[buffer(4)]],
    constant uint& in_dim [[buffer(5)]],
    constant uint& out_dim [[buffer(6)]],
    constant float& alpha [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    uint token_idx = gid.x;
    int adapter_idx = adapter_ids[token_idx];
    if (adapter_idx < 0) return;

    device const half* x_ptr = x + token_idx * in_dim;

    // Compute scalar intermediate = A[adapter] · x
    device const half* a = lora_a + adapter_idx * in_dim;
    float intermediate = 0.0f;
    for (uint d = simd_lane_id; d < in_dim; d += 32) {
        intermediate += (float)a[d] * (float)x_ptr[d];
    }
    intermediate = simd_reduce_sum(intermediate, nullptr);

    // Compute output += alpha * intermediate * B[adapter]
    device const half* b = lora_b + adapter_idx * out_dim;
    device half* out = output + token_idx * out_dim;
    for (uint o = simd_lane_id; o < out_dim; o += 32) {
        out[o] += (half)(alpha * intermediate * (float)b[o]);
    }
}
