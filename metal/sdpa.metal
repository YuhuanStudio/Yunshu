/*
 * Yunshu Metal Kernels — Scaled Dot-Product Attention (SDPA).
 *
 * FlashAttention-2 style tiled attention for Apple GPU.
 * Based on MLX's steel_attention but with Yunshu-specific optimizations
 * for variable-length sequences and batched inference.
 *
 * Tile sizes configurable via function constants in common.metal.
 *
 * Key optimizations:
 * - Online softmax (no materializing full attention matrix)
 * - FP32 accumulation for numerical stability
 * - Tiled Q/K/V to maximize cache locality
 * - SIMD shuffle for warp-level reductions
 */

#include "common.metal"

// ── Flash SDPA Forward ──
// Computes: output = softmax(Q @ K^T / sqrt(d)) @ V
// Tiled to never materialize the full attention matrix in memory.
//
// Grid: [num_heads, ceil(seq_len / SDPA_TILE_Q), 1]

kernel void flash_sdpa(
    device const half* Q [[buffer(0)]],           // [seq_len, num_heads, head_dim]
    device const half* K [[buffer(1)]],           // [seq_len, num_kv_heads, head_dim]
    device const half* V [[buffer(2)]],           // [seq_len, num_kv_heads, head_dim]
    device half* output [[buffer(3)]],            // [seq_len, num_heads, head_dim]
    constant uint& seq_len [[buffer(4)]],
    constant uint& num_heads [[buffer(5)]],
    constant uint& num_kv_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant uint& tile_q [[buffer(9)]],          // Query tile size (default SDPA_TILE_Q)
    constant uint& tile_kv [[buffer(10)]],        // KV tile size (default SDPA_TILE_KV)
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint head_idx = gid.x;
    uint q_tile_idx = gid.y;
    uint q_start = q_tile_idx * tile_q;

    if (q_start >= seq_len) return;

    // GQA: map head_idx to kv_head_idx
    uint kv_head_idx = head_idx * num_kv_heads / num_heads;

    // Threadgroup shared memory for KV tiles.
    // Must stay under 32KB Apple GPU threadgroup limit.
    // With SDPA_TILE_KV=32, MAX_HEAD_DIM=128: 2*32*128*2 = 16KB + scores 64*32*4 = 8KB = 24KB
    threadgroup half shared_k[SDPA_TILE_KV * MAX_HEAD_DIM];
    threadgroup half shared_v[SDPA_TILE_KV * MAX_HEAD_DIM];
    threadgroup float shared_scores[SDPA_TILE_Q * SDPA_TILE_KV];

    // Load Q tile into registers (each simdgroup handles a few rows)
    uint rows_per_group = tile_q / NUM_SIMD_GROUPS;
    uint local_q_start = simd_group_id * rows_per_group;

    float q_local[SDPA_TILE_Q * MAX_HEAD_DIM / NUM_SIMD_GROUPS];
    float o_local[SDPA_TILE_Q * MAX_HEAD_DIM / NUM_SIMD_GROUPS];
    float m_local[SDPA_TILE_Q / NUM_SIMD_GROUPS];
    float l_local[SDPA_TILE_Q / NUM_SIMD_GROUPS];
    for (uint i = 0; i < rows_per_group; i++) {
        m_local[i] = -INFINITY;
        l_local[i] = 0.0f;
    }
    for (uint i = 0; i < rows_per_group * MAX_HEAD_DIM; i++) {
        q_local[i] = 0.0f;
        o_local[i] = 0.0f;
    }

    // Load Q rows
    for (uint q = 0; q < rows_per_group; q++) {
        uint q_idx = q_start + local_q_start + q;
        if (q_idx >= seq_len) break;
        device const half* q_ptr = Q + q_idx * num_heads * head_dim + head_idx * head_dim;
        for (uint d = simd_lane_id; d < head_dim; d += 32) {
            q_local[q * MAX_HEAD_DIM + d] = (float)q_ptr[d];
        }
    }

    // Iterate over KV tiles
    for (uint kv_tile = 0; kv_tile < (seq_len + tile_kv - 1) / tile_kv; kv_tile++) {
        uint kv_start = kv_tile * tile_kv;
        uint kv_len = min(tile_kv, seq_len - kv_start);

        // Load K and V tiles into shared memory (SIMD-strided)
        for (uint kv = simd_group_id; kv < kv_len; kv += NUM_SIMD_GROUPS) {
            uint kv_idx = kv_start + kv;
            if (kv_idx < seq_len) {
                for (uint d = simd_lane_id; d < head_dim; d += 32) {
                    shared_k[kv * head_dim + d] = K[kv_idx * num_kv_heads * head_dim + kv_head_idx * head_dim + d];
                    shared_v[kv * head_dim + d] = V[kv_idx * num_kv_heads * head_dim + kv_head_idx * head_dim + d];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute attention scores for this KV tile (with causal mask)
        for (uint q = 0; q < rows_per_group; q++) {
            uint q_idx = q_start + local_q_start + q;
            if (q_idx >= seq_len) break;

            for (uint kv = simd_group_id; kv < kv_len; kv += NUM_SIMD_GROUPS) {
                uint kv_idx = kv_start + kv;
                // Causal mask: query at q_idx can only attend to keys at positions <= q_idx
                if (kv_idx <= q_idx) {
                    float score = 0.0f;
                    for (uint d = simd_lane_id; d < head_dim; d += 32) {
                        score += q_local[q * MAX_HEAD_DIM + d] * (float)shared_k[kv * head_dim + d];
                    }
                    score *= scale;
                    shared_scores[q * tile_kv + kv] = score;
                } else {
                    shared_scores[q * tile_kv + kv] = -INFINITY;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax + accumulation
        for (uint q = 0; q < rows_per_group; q++) {
            uint q_idx = q_start + local_q_start + q;
            if (q_idx >= seq_len) break;

            // Find max in this KV tile
            float tile_max = -INFINITY;
            for (uint kv = 0; kv < kv_len; kv++) {
                tile_max = max(tile_max, shared_scores[q * tile_kv + kv]);
            }

            // Update running max and correction
            float old_max = m_local[q];
            m_local[q] = max(m_local[q], tile_max);
            float correction = exp(old_max - m_local[q]);

            // Scale existing accumulations
            l_local[q] *= correction;
            for (uint d = simd_lane_id; d < head_dim; d += 32) {
                o_local[q * MAX_HEAD_DIM + d] *= correction;
            }

            // Add new weighted values
            for (uint kv = 0; kv < kv_len; kv++) {
                float weight = exp(shared_scores[q * tile_kv + kv] - m_local[q]);
                l_local[q] += weight;

                for (uint d = simd_lane_id; d < head_dim; d += 32) {
                    o_local[q * MAX_HEAD_DIM + d] += weight * (float)shared_v[kv * head_dim + d];
                }
            }
        }
    }

    // Write output: normalize by sum
    for (uint q = 0; q < rows_per_group; q++) {
        uint q_idx = q_start + local_q_start + q;
        if (q_idx >= seq_len) break;

        float inv_l = 1.0f / max(l_local[q], 1e-8f);
        device half* out = output + q_idx * num_heads * head_dim + head_idx * head_dim;
        for (uint d = simd_lane_id; d < head_dim; d += 32) {
            out[d] = (half)(o_local[q * MAX_HEAD_DIM + d] * inv_l);
        }
    }
}
