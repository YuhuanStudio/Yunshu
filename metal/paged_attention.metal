/*
 * Yunshu Metal Kernels — PagedAttention (PA).
 *
 * Port of vLLM's PagedAttention to Apple GPU Metal.
 * Supports variable-length sequences with block-based KV cache.
 *
 * Tile sizes are configurable via function constants in common.metal,
 * specialized at pipeline creation time for each model architecture.
 *
 * Key differences from CUDA PagedAttention:
 * - Apple GPU has 32-wide SIMD (vs NVIDIA 32-wide warp)
 * - No shared memory bank conflicts (Apple GPU unified cache)
 * - Uses simd_shuffle_down for warp-level reductions
 * - FP16/BF16 compute with FP32 accumulation
 *
 * Reference: vLLM attention/ops/paged_attention/v2/
 * Reference: MLX steel_attention (SDPA pattern)
 */

#include "common.metal"

// ── PagedAttention Decode Kernel ──
// Single-query decode: one query token against paged KV cache.
// Grid: [num_heads, num_queries, 1]

kernel void paged_attention_decode(
    device const half* queries [[buffer(0)]],       // [num_queries, num_heads, head_dim]
    device const half* key_cache [[buffer(1)]],     // [num_blocks * block_size, num_heads, head_dim]
    device const half* value_cache [[buffer(2)]],   // [num_blocks * block_size, num_heads, head_dim]
    device const int* block_tables [[buffer(3)]],   // [num_queries, max_num_blocks_per_seq]
    device const int* seq_lens [[buffer(4)]],       // [num_queries]
    device half* output [[buffer(5)]],              // [num_queries, num_heads, head_dim]
    constant uint& num_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& kv_block_size [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint query_idx = gid.y;
    uint head_idx = gid.x;

    int seq_len = seq_lens[query_idx];
    if (seq_len <= 0) return;

    int num_kv_blocks = (seq_len + kv_block_size - 1) / kv_block_size;
    int block_table_stride = (seq_len + kv_block_size - 1) / kv_block_size;
    device const int* block_table = block_tables + query_idx * block_table_stride;

    // Threadgroup shared memory for reduction
    threadgroup float shared_logits[PA_BLOCK_KV];

    // Accumulate output in FP32
    float acc[MAX_HEAD_DIM];
    for (uint d = 0; d < head_dim && d < MAX_HEAD_DIM; d++) {
        acc[d] = 0.0f;
    }

    // Query vector pointer
    device const half* q = queries + query_idx * num_heads * head_dim + head_idx * head_dim;

    // Iterate over KV blocks using online softmax.
    // Track running max and running sum with correction factor when max changes.
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    for (int block = 0; block < num_kv_blocks; block++) {
        int physical_block = block_table[block];
        if (physical_block < 0) continue;

        int block_start = block * kv_block_size;
        int block_end = min((int)kv_block_size, seq_len - block_start);

        // Find local max for this block
        float local_max = -INFINITY;
        for (int kv_idx = simd_lane_id; kv_idx < block_end; kv_idx += 32) {
            device const half* k = key_cache +
                (physical_block * kv_block_size + kv_idx) * num_heads * head_dim +
                head_idx * head_dim;

            // Dot product: q · k
            float score = 0.0f;
            for (uint d = 0; d < head_dim; d++) {
                score += (float)q[d] * (float)k[d];
            }
            score *= scale;

            shared_logits[kv_idx] = score;
            local_max = max(local_max, score);
        }

        // SIMD-level max reduction for this block
        float block_max = simd_reduce_max(local_max, shared_logits);

        // Online softmax correction: update running max
        float old_running_max = running_max;
        running_max = max(running_max, block_max);
        float correction = exp(old_running_max - running_max);

        // Correct accumulated output and sum from previous blocks
        for (uint d = 0; d < head_dim && d < MAX_HEAD_DIM; d++) {
            acc[d] *= correction;
        }
        running_sum *= correction;

        // Compute exp(score - running_max) for this block
        float block_exp_sum = 0.0f;
        for (int kv_idx = simd_lane_id; kv_idx < block_end; kv_idx += 32) {
            float score = shared_logits[kv_idx];
            float weight = exp(score - running_max);
            shared_logits[kv_idx] = weight;
            block_exp_sum += weight;
        }

        // SIMD-level sum reduction for this block
        float block_sum = simd_reduce_sum(block_exp_sum, shared_logits);
        running_sum += block_sum;

        // Weighted accumulation: weight * v (unnormalized — divide by sum at end)
        for (int kv_idx = simd_lane_id; kv_idx < block_end; kv_idx += 32) {
            float weight = shared_logits[kv_idx];

            device const half* v = value_cache +
                (physical_block * kv_block_size + kv_idx) * num_heads * head_dim +
                head_idx * head_dim;

            for (uint d = 0; d < head_dim; d++) {
                acc[d] += weight * (float)v[d];
            }
        }
    }

    // Finalize: divide by global sum
    float inv_sum = 1.0f / max(running_sum, 1e-8f);

    // Write output
    device half* out = output + query_idx * num_heads * head_dim + head_idx * head_dim;
    for (uint d = simd_lane_id; d < head_dim; d += 32) {
        out[d] = (half)(acc[d] * inv_sum);
    }
}


// ── PagedAttention Prefill Kernel (FlashAttention-style) ──
// Processes query tokens in tiles of PA_BLOCK_Q.
// Grid: [num_heads, ceil(num_query_tokens / PA_BLOCK_Q), 1]

kernel void paged_attention_prefill(
    device const half* queries [[buffer(0)]],
    device const half* key_cache [[buffer(1)]],
    device const half* value_cache [[buffer(2)]],
    device const int* block_tables [[buffer(3)]],
    device const int* seq_lens [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& kv_block_size [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant uint& num_query_tokens [[buffer(10)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint head_idx = gid.x;
    uint query_block_idx = gid.y;
    uint block_q = PA_BLOCK_Q;
    uint q_start = query_block_idx * block_q;

    int seq_len = seq_lens[0]; // Prefill is single-sequence
    if (seq_len <= 0) return;

    int num_kv_blocks = (seq_len + kv_block_size - 1) / kv_block_size;
    device const int* block_table = block_tables;

    // Each simdgroup handles a slice of query rows
    uint rows_per_group = block_q / NUM_SIMD_GROUPS;
    uint q_local_start = simd_group_id * rows_per_group;

    // Query tile in registers
    float q_tile[PA_BLOCK_Q * MAX_HEAD_DIM / NUM_SIMD_GROUPS];
    // Initialize to zero
    for (uint i = 0; i < rows_per_group * MAX_HEAD_DIM; i++) {
        q_tile[i] = 0.0f;
    }

    // Load query tile
    for (uint q = 0; q < rows_per_group; q++) {
        uint q_idx = q_start + q_local_start + q;
        if (q_idx >= num_query_tokens) break;
        device const half* q_ptr = queries + q_idx * num_heads * head_dim + head_idx * head_dim;
        for (uint d = simd_lane_id; d < head_dim; d += 32) {
            q_tile[q * MAX_HEAD_DIM + d] = (float)q_ptr[d];
        }
    }

    // Accumulator for output (rows_per_group rows × head_dim)
    float out_acc[PA_BLOCK_Q * MAX_HEAD_DIM / NUM_SIMD_GROUPS];
    float max_scores[PA_BLOCK_Q / NUM_SIMD_GROUPS];
    float sum_weights[PA_BLOCK_Q / NUM_SIMD_GROUPS];
    for (uint i = 0; i < rows_per_group; i++) {
        max_scores[i] = -INFINITY;
        sum_weights[i] = 0.0f;
    }
    for (uint i = 0; i < rows_per_group * MAX_HEAD_DIM; i++) {
        out_acc[i] = 0.0f;
    }

    // Iterate over KV blocks
    for (int block = 0; block < num_kv_blocks; block++) {
        int physical_block = block_table[block];
        if (physical_block < 0) continue;

        int block_start = block * kv_block_size;
        int block_end = min((int)kv_block_size, seq_len - block_start);

        // Compute attention scores for this KV block
        for (int kv_idx = 0; kv_idx < block_end; kv_idx++) {
            device const half* k = key_cache +
                (physical_block * kv_block_size + kv_idx) * num_heads * head_dim +
                head_idx * head_dim;

            // Load key vector
            float k_vec[MAX_HEAD_DIM];
            for (uint d = simd_lane_id; d < head_dim; d += 32) {
                k_vec[d] = (float)k[d];
            }

            // Key position for causal mask
            int kv_pos = block * (int)kv_block_size + kv_idx;

            // Compute scores for our query rows
            for (uint q = 0; q < rows_per_group; q++) {
                uint q_idx = q_start + q_local_start + q;
                if (q_idx >= num_query_tokens) break;

                float score = 0.0f;
                // Causal mask: query can only attend to keys at positions <= q_idx
                if (kv_pos <= (int)q_idx) {
                    for (uint d = 0; d < head_dim; d++) {
                        score += q_tile[q * MAX_HEAD_DIM + d] * k_vec[d];
                    }
                    score *= scale;
                } else {
                    score = -INFINITY;
                }

                // Online softmax update
                float old_max = max_scores[q];
                max_scores[q] = max(max_scores[q], score);

                float correction = exp(old_max - max_scores[q]);
                for (uint d = 0; d < head_dim; d++) {
                    out_acc[q * MAX_HEAD_DIM + d] *= correction;
                }
                sum_weights[q] *= correction;

                float weight = exp(score - max_scores[q]);
                sum_weights[q] += weight;

                // Load value and accumulate
                device const half* v = value_cache +
                    (physical_block * kv_block_size + kv_idx) * num_heads * head_dim +
                    head_idx * head_dim;
                for (uint d = simd_lane_id; d < head_dim; d += 32) {
                    out_acc[q * MAX_HEAD_DIM + d] += weight * (float)v[d];
                }
            }
        }
    }

    // Finalize: divide by sum
    for (uint q = 0; q < rows_per_group; q++) {
        uint q_idx = q_start + q_local_start + q;
        if (q_idx >= num_query_tokens) break;

        device half* out = output + q_idx * num_heads * head_dim + head_idx * head_dim;
        float inv_sum = 1.0f / max(sum_weights[q], 1e-8f);
        for (uint d = simd_lane_id; d < head_dim; d += 32) {
            out[d] = (half)(out_acc[q * MAX_HEAD_DIM + d] * inv_sum);
        }
    }
}
