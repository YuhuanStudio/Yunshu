# Derived from Splash 1.4.0 runtime/metal/kernels/prefill/ane_ffn.metal (Apache-2.0).
# Licensed under the Apache License, Version 2.0; see THIRD_PARTY_NOTICES.md.
# Modified for Yunshu: MLX custom-kernel form, MLX affine quantized weights instead of
# GGUF image tensors (one generic bit-stream decoder for 2..8 bits), per-tile status flags.
"""GPU side of the ANE FFN split as ``mx.fast.metal_kernel`` programs.

rotate: per token Hadamard rotation of the normalized hidden row (blocks of 128) and its int8
scale; pack: rotated rows to channel-major int8 IOSurface segments; weights: a layer's trailing
channels of a quantized projection rotated and requantized to int8 rows under one scale per row;
row_scale: that scale; join: adds the ANE's partial down projection to the GPU's output.
"""

from __future__ import annotations

HEADER = """
#define ROT_UNIT 128u
#define INT8_PEAK 127.0f
#define INT8_UNIT 128.0f
#define PEAK_FLOOR 6.103515625e-05f

// One block of K * 128 values held four per lane in each 128: two butterfly stages in
// registers and five across the simdgroup within each 128, then log2(K) across them.
template <uint K>
inline void ane_rotate_block(thread float (&value)[K][4], uint lane, device const float *sign) {
  for (uint k = 0; k < K; ++k) {
    thread float (&v)[4] = value[k];
    for (uint half_span = 1; half_span < 4; half_span <<= 1)
      for (uint e = 0; e < 4; ++e)
        if (!(e & half_span)) {
          const float a = v[e], b = v[e + half_span];
          v[e] = a + b;
          v[e + half_span] = a - b;
        }
    for (uint mask = 1; mask < 32; mask <<= 1)
      for (uint e = 0; e < 4; ++e) {
        const float other = simd_shuffle_xor(v[e], mask);
        v[e] = (lane & mask) ? other - v[e] : v[e] + other;
      }
  }
  for (uint half_span = 1; half_span < K; half_span <<= 1)
    for (uint k = 0; k < K; ++k)
      if (!(k & half_span))
        for (uint e = 0; e < 4; ++e) {
          const float a = value[k][e], b = value[k + half_span][e];
          value[k][e] = a + b;
          value[k + half_span][e] = a - b;
        }
  const float norm = rsqrt(float(K * ROT_UNIT));
  for (uint k = 0; k < K; ++k)
    for (uint e = 0; e < 4; ++e) value[k][e] *= sign[k * ROT_UNIT + lane * 4 + e] * norm;
}

// Value (row, col) of an MLX affine quantized matrix of K inputs per row.
template <typename S>
inline float ane_quant_value(device const uchar *w, device const S *scales, device const S *biases,
                             uint K, uint bits, uint gs, uint row, uint col) {
  const ulong bitpos = ulong(col) * bits;
  device const uchar *b = w + ulong(row) * (ulong(K) * bits / 8) + (bitpos >> 3);
  const uint shift = uint(bitpos & 7);
  uint v = uint(b[0]);
  if (shift + bits > 8) v |= uint(b[1]) << 8;
  const float code = float((v >> shift) & ((1u << bits) - 1u));
  const ulong g = ulong(row) * (K / gs) + col / gs;
  return code * float(scales[g]) + float(biases[g]);
}
"""

ROTATE = """
uint row = threadgroup_position_in_grid.x;
uint simd_group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint simd_groups = simdgroups_per_threadgroup;
threadgroup float peaks[32];
const uint blocks = H / ROT_UNIT / simd_groups;
float value[8][1][4];
float peak = 0.0f;
for (uint block = 0; block < blocks; ++block) {
  const uint origin = row * H + (simd_group * blocks + block) * ROT_UNIT + lane * 4;
  for (uint e = 0; e < 4; ++e) value[block][0][e] = float(inp[origin + e]);
  ane_rotate_block<1>(value[block], lane, sign);
  for (uint e = 0; e < 4; ++e) peak = max(peak, fabs(value[block][0][e]));
}
peak = simd_max(peak);
if (lane == 0) peaks[simd_group] = peak;
threadgroup_barrier(mem_flags::mem_threadgroup);
peak = 0.0f;
for (uint g = 0; g < simd_groups; ++g) peak = max(peak, peaks[g]);
const float scale = max(peak, PEAK_FLOOR) / INT8_PEAK, inverse = 1.0f / scale;
for (uint block = 0; block < blocks; ++block) {
  const uint origin = row * H + (simd_group * blocks + block) * ROT_UNIT + lane * 4;
  for (uint e = 0; e < 4; ++e) rotated[origin + e] = half(value[block][0][e] * inverse);
}
if (simd_group == 0 && lane == 0) ((device half *)(tx_surface))[row] = half(scale * INT8_UNIT);
"""

# grid (tiles*32, (SEG/32)*8, 1), threadgroup (32, 8, 1)
PACK = """
threadgroup half staged[32][33];
uint2 tile = uint2(threadgroup_position_in_grid.x, threadgroup_position_in_grid.y);
uint2 position = uint2(thread_position_in_threadgroup.x, thread_position_in_threadgroup.y);
const uint row = tile.x * 32, channel = tile.y * 32;
for (uint j = position.y; j < 32; j += 8)
  staged[j][position.x] = (row + j < R) ? rotated[(row + j) * H + CHANNEL0 + channel + position.x] : half(0);
threadgroup_barrier(mem_flags::mem_threadgroup);
device char *packed = (device char *)(surface);
for (uint j = position.y; j < 32; j += 8)
  packed[(channel + j) * STRIDE + row + position.x] =
      char(clamp(rint(float(staged[position.x][j])), -INT8_PEAK, INT8_PEAK));
if (tile.x == 0 && tile.y == 0 && position.x == 0 && position.y == 0) marker[0] = 1;
"""

# One simdgroup per weight row, 8 rows x one block of K*128 inputs per threadgroup.
# grid (rows/8*256, width/(K*128), 1), threadgroup (256, 1, 1)
WEIGHTS = """
uint2 tile = uint2(threadgroup_position_in_grid.x, threadgroup_position_in_grid.y);
uint simd_group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
const uint row = tile.x * 8 + simd_group, block = tile.y * K * ROT_UNIT;
float value[K][4];
for (uint k = 0; k < K; ++k)
  for (uint e = 0; e < 4; ++e)
    value[k][e] = ane_quant_value(reinterpret_cast<device const uchar *>(w), scales, biases, KIN, BITS, GS,
                                  ROW0 + row, IN0 + block + k * ROT_UNIT + lane * 4 + e);
ane_rotate_block<K>(value, lane, sign);
const float inverse = INT8_UNIT / float(row_scale[row]);
device char *out = (device char *)(surface);
for (uint k = 0; k < K; ++k) {
  char4 q;
  for (uint e = 0; e < 4; ++e) q[e] = char(clamp(rint(value[k][e] * inverse), -INT8_PEAK, INT8_PEAK));
  *(device char4 *)(out + ulong(row) * STRIDE + block + k * ROT_UNIT + lane * 4) = q;
}
if (tile.y == 0 && lane == 0) const_cast<device half *>(scale_surface)[row * SSTRIDE] = row_scale[row];
if (tile.x == 0 && tile.y == 0 && simd_group == 0 && lane == 0) marker[0] = 1;
"""

# Each row's largest rotated weight over inputs [IN0, IN0 + WIDTH) in blocks of K*128.
# grid (rows/8*256, 1, 1), threadgroup (256, 1, 1)
ROW_SCALE = """
uint tile = threadgroup_position_in_grid.x;
uint simd_group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
const uint row = tile * 8 + simd_group;
float peak = 0.0f;
for (uint block = 0; block < WIDTH; block += K * ROT_UNIT) {
  float value[K][4];
  for (uint k = 0; k < K; ++k)
    for (uint e = 0; e < 4; ++e)
      value[k][e] = ane_quant_value(reinterpret_cast<device const uchar *>(w), scales, biases, KIN, BITS, GS,
                                    ROW0 + row, IN0 + block + k * ROT_UNIT + lane * 4 + e);
  ane_rotate_block<K>(value, lane, sign);
  for (uint k = 0; k < K; ++k)
    for (uint e = 0; e < 4; ++e) peak = max(peak, fabs(value[k][e]));
}
peak = simd_max(peak);
if (lane == 0) row_scale[row] = half(max(peak, PEAK_FLOOR) / INT8_PEAK * INT8_UNIT);
"""

# Adds the partial down projection (scaled in fp32 by the token's scales) to the GPU's rows.
# grid (tiles*32, (H/32)*8, 1), threadgroup (32, 8, 1); status[tile] = 1 if a value was not finite.
JOIN = """
threadgroup float staged[32][33];
threadgroup float scale[32];
uint2 tile = uint2(threadgroup_position_in_grid.x, threadgroup_position_in_grid.y);
uint2 position = uint2(thread_position_in_threadgroup.x, thread_position_in_threadgroup.y);
const uint row = tile.x * 32, channel = tile.y * 32, token = row + position.x;
const bool chunk = token < R;
bool finite = true;
for (uint j = position.y; j < 32; j += 8) {
  const half value = partial[(channel + j) * PSTRIDE + token];
  finite &= !(chunk && ((as_type<ushort>(value) & 0x7c00) == 0x7c00));
  staged[j][position.x] = float(value);
}
if (position.y == 0) {
  const half intermediate = partial[H * PSTRIDE + token], input = token_scale[token];
  finite &= !(chunk && (((as_type<ushort>(intermediate) & 0x7c00) == 0x7c00) ||
                        ((as_type<ushort>(input) & 0x7c00) == 0x7c00)));
  scale[position.x] = float(intermediate) * float(input);
}
threadgroup uint bad;
if (position.x == 0 && position.y == 0) bad = 0u;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (!finite) bad = 1u;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (position.x == 0 && position.y == 0) status[tile.y * ((R + 31) / 32) + tile.x] = bad;
for (uint j = position.y; j < 32; j += 8) {
  if (row + j >= R) continue;
  const uint index = (row + j) * H + channel + position.x;
  out[index] = bfloat16_t(float(gpu[index]) + staged[position.x][j] * scale[j]);
}
"""

_cache: dict[str, object] = {}


def kernel(name: str):
    import mlx.core as mx

    if name in _cache:
        return _cache[name]
    table = {
        "rotate": (["inp", "sign", "tx_surface"], ["rotated"], ROTATE, False),
        "pack": (["rotated", "surface"], ["marker"], PACK, False),
        "weights": (
            ["w", "scales", "biases", "row_scale", "sign", "surface", "scale_surface"],
            ["marker"],
            WEIGHTS,
            False,
        ),
        "row_scale": (["w", "scales", "biases", "sign"], ["row_scale"], ROW_SCALE, True),
        "join": (["gpu", "partial", "token_scale"], ["out", "status"], JOIN, False),
    }
    inputs, outputs, source, contiguous = table[name]
    k = mx.fast.metal_kernel(
        name=f"yunshu_ane_ffn_{name}",
        input_names=inputs,
        output_names=outputs,
        source=source,
        header=HEADER,
        ensure_row_contiguous=contiguous,
    )
    _cache[name] = k
    return k
