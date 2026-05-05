/*
 * Yunshu Metal Kernels — KIVI 2-bit KV Cache Quantization.
 *
 * KIVI (K-cache 2-bit, V-cache FP16) for Apple GPU.
 * Quantizes key cache to 2-bit while keeping values in FP16.
 * Based on KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.
 *
 * Quantization scheme: asymmetric per-channel 2-bit
 *   - 4 values per byte (2 bits each)
 *   - Scale + zero-point per channel (stored in FP16)
 *   - Dequantize: x = scale * (quant_val - zero_point)
 *
 * Metal 3.1+ for Apple GPU SIMD operations.
 */

#include "common.metal"

// ── KIVI Quantize Keys ──
// Quantize FP16 key vectors to 2-bit (4 values per byte)
// Grid: [num_tokens, num_heads, 1]

kernel void kivi_quantize_keys(
    device const half* keys [[buffer(0)]],        // [num_tokens, num_heads, head_dim]
    device uchar* quant_keys [[buffer(1)]],       // [num_tokens, num_heads, head_dim/4] (packed 2-bit)
    device half* scales [[buffer(2)]],            // [num_tokens, num_heads] per-channel scale
    device half* zero_points [[buffer(3)]],       // [num_tokens, num_heads] per-channel zero point
    constant uint& num_tokens [[buffer(4)]],
    constant uint& num_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint token_idx = gid.x;
    uint head_idx = gid.y;

    if (token_idx >= num_tokens || head_idx >= num_heads) return;

    uint base = token_idx * num_heads * head_dim + head_idx * head_dim;

    // Find min/max for this channel
    float channel_min = (float)keys[base];
    float channel_max = (float)keys[base];

    for (uint d = 1; d < head_dim; d++) {
        float val = (float)keys[base + d];
        channel_min = min(channel_min, val);
        channel_max = max(channel_max, val);
    }

    // Compute scale and zero_point for 2-bit (4 levels: 0-3)
    float range = channel_max - channel_min;
    float scale = range / 3.0f;  // 4 levels = range/3
    float zp = channel_min;      // zero_point maps to 0

    // Store scale and zero_point
    uint channel_idx = token_idx * num_heads + head_idx;
    scales[channel_idx] = (half)scale;
    zero_points[channel_idx] = (half)zp;

    // Quantize: pack 4 values into 1 byte
    float inv_scale = (scale > 1e-8f) ? (1.0f / scale) : 0.0f;
    uint out_offset = token_idx * num_heads * (head_dim / 4) + head_idx * (head_dim / 4);

    for (uint d = 0; d < head_dim; d += 4) {
        // Quantize 4 values
        uchar packed = 0;
        for (uint i = 0; i < 4; i++) {
            float val = (float)keys[base + d + i];
            float quantized = (val - zp) * inv_scale;
            // Clamp to [0, 3]
            int q = (int)clamp(quantized + 0.5f, 0.0f, 3.0f);
            packed |= (uchar)(q << (i * 2));
        }
        quant_keys[out_offset + d / 4] = packed;
    }
}


// ── KIVI Dequantize Keys ──
// Dequantize 2-bit packed keys back to FP16
// Grid: [num_tokens, num_heads, 1]

kernel void kivi_dequantize_keys(
    device const uchar* quant_keys [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const half* zero_points [[buffer(2)]],
    device half* keys [[buffer(3)]],              // Output: [num_tokens, num_heads, head_dim]
    constant uint& num_tokens [[buffer(4)]],
    constant uint& num_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint token_idx = gid.x;
    uint head_idx = gid.y;

    if (token_idx >= num_tokens || head_idx >= num_heads) return;

    uint channel_idx = token_idx * num_heads + head_idx;
    float scale = (float)scales[channel_idx];
    float zp = (float)zero_points[channel_idx];

    uint in_offset = token_idx * num_heads * (head_dim / 4) + head_idx * (head_dim / 4);
    uint out_offset = token_idx * num_heads * head_dim + head_idx * head_dim;

    for (uint d = 0; d < head_dim; d += 4) {
        uchar packed = quant_keys[in_offset + d / 4];

        for (uint i = 0; i < 4; i++) {
            int q = (packed >> (i * 2)) & 0x3;
            keys[out_offset + d + i] = (half)(scale * (float)q + zp);
        }
    }
}


// ── KIVI Batch Dequantize for Attention ──
// Dequantize + scale in one kernel to reduce memory bandwidth
// Grid: [num_heads, ceil(num_tokens / TILE), 1]

kernel void kivi_dequantize_for_attention(
    device const uchar* quant_keys [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const half* zero_points [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& num_tokens [[buffer(4)]],
    constant uint& num_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant float& attn_scale [[buffer(7)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    uint head_idx = gid.x;
    uint token_tile = gid.y;
    uint tile_size = 32;
    uint token_start = token_tile * tile_size;

    // Each SIMD group processes a tile of tokens for one head
    for (uint t = 0; t < tile_size; t++) {
        uint token_idx = token_start + t;
        if (token_idx >= num_tokens) break;

        uint channel_idx = token_idx * num_heads + head_idx;
        float scale = (float)scales[channel_idx];
        float zp = (float)zero_points[channel_idx];

        uint in_offset = token_idx * num_heads * (head_dim / 4) + head_idx * (head_dim / 4);
        uint out_offset = token_idx * num_heads * head_dim + head_idx * head_dim;

        // Dequantize and apply attention scale
        for (uint d = simd_lane_id; d < head_dim; d += 32) {
            uint byte_idx = d / 4;
            uint bit_offset = (d % 4) * 2;
            int q = (quant_keys[in_offset + byte_idx] >> bit_offset) & 0x3;
            output[out_offset + d] = (half)((scale * (float)q + zp) * attn_scale);
        }
    }
}
