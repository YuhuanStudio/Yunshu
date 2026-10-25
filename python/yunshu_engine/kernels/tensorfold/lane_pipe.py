# Upstream (derived): ashhart/TensorFold (MIT) src/tensorfold/kernels/qwen/dense/v1/lane_qmm.py @ 34bae79a
"""Software-pipelined M5 lane body: the next group's weight tile is in flight during this group's op.

``lane_m5._MAIN`` hands each 32-column x 64-K weight tile to the tensor op straight from device memory, so a
simdgroup alternates "wait for the tile, multiply, fma epilogue" with nothing else outstanding. This body loads
the tile of group g + 1 into registers while group g runs, parks it in a double-buffered threadgroup stage
and lets the same op read it from there. The op, its descriptor, the weight bytes it sees and the epilogue are
unchanged, so every output is bit-identical to ``lane_m5._MAIN``'s; only the memory schedule differs.

Applies to the tiled 4-bit layout (``tile_weight``): a tile group is NT columns x GS/2 bytes, column-major.
"""

_PIPE = r"""
  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;     // K slice
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);       // fragment row of this lane (and fm + 8)
  const short fn = ((qid & 2) | (lane & 1)) * 4;        // first of its four fragment columns
  const int M = mdims[0], MP = mdims[1];
  constexpr int KG = K / GS;
  constexpr int NF = NT / 16;
  constexpr int CB = GS / 2;                            // bytes of one column of one group
  constexpr int NV = CB / 16;                           // 16-byte loads a lane makes a group
  constexpr int TILE_V = NT * NV;                       // 16-byte units of one tile group
  static_assert(NT == 32 && CB % 16 == 0, "one column per lane, whole 16-byte loads");
  const int n0 = threadgroup_position_in_grid.x * NT;
  const int rb = threadgroup_position_in_grid.y * 16 * TMR;
  const int g_begin = (sg * KG) / SK;
  const int g_end = ((sg + 1) * KG) / SK;

  constexpr auto desc = matmul2d_descriptor(16 * TMR, NT, GS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + (int64_t)rb * K, dextents<int32_t, 2>(K, M - rb));

  threadgroup uint4 stage_all[SK * 2 * TILE_V];         // per K slice: two tile groups
  threadgroup uint4* stg = stage_all + (int)sg * 2 * TILE_V;
  const device uint4* Wt = (const device uint4*)Wq + (int64_t)threadgroup_position_in_grid.x * KG * TILE_V + (int)lane * NV;

  float C[TMR][NF * 8];
  for (int t = 0; t < TMR; t++) for (int i = 0; i < NF * 8; i++) C[t][i] = 0.0f;
  const device uint4* sbv = (const device uint4*)SBt;   // (s, b) bf16 pairs, [g][n]
  bool colok[NF];
  for (int f = 0; f < NF; f++) colok[f] = n0 + f * 16 + fn < N;

  uint4 wn[NV];
  if (g_begin < g_end) {
    for (int v = 0; v < NV; v++) stg[(int)lane * NV + v] = Wt[(int64_t)g_begin * TILE_V + v];
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (int g = g_begin, cur = 0; g < g_end; g++, cur ^= 1) {
    const bool more = g + 1 < g_end;
    if (more) for (int v = 0; v < NV; v++) wn[v] = Wt[(int64_t)(g + 1) * TILE_V + v];
    float s[NF][4], bb[NF][4];
    for (int f = 0; f < NF; f++) {
      const uint4 q = colok[f] ? sbv[(g * N + n0 + f * 16 + fn) / 4] : uint4(0);
      const vec<bfloat, 8> v = as_type<vec<bfloat, 8>>(q);
      for (int j = 0; j < 4; j++) { s[f][j] = float(v[2 * j]); bb[f][j] = float(v[2 * j + 1]); }
    }
    auto a = tA.slice(g * GS, 0);
    tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline> b(
        (threadgroup uchar*)(stg + cur * TILE_V), dextents<int32_t, 2>(GS, NT));
    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    op.run(a, b, P);
    if (more) for (int v = 0; v < NV; v++) stg[(cur ^ 1) * TILE_V + (int)lane * NV + v] = wn[v];
    for (int t = 0; t < TMR; t++) {
      const bool live = !EDGE || rb + t * 16 < MP;
      const float xs0 = live ? XS[g * MP + rb + t * 16 + fm] : 0.0f;
      const float xs1 = live ? XS[g * MP + rb + t * 16 + fm + 8] : 0.0f;
      for (int f = 0; f < NF; f++)
        for (int r = 0; r < 2; r++)
          for (int j = 0; j < 4; j++) {
            const int i = f * 8 + r * 4 + j;
            C[t][i] = fma(s[f][j], P[t * NF * 8 + i], fma(bb[f][j], r ? xs1 : xs0, C[t][i]));
          }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup float part[(SK > 1 ? SK - 1 : 1) * NF * 8 * 32];
  for (int t = 0; t < TMR; t++) {
    if (SK > 1) {
      if (sg > 0) for (int i = 0; i < NF * 8; i++) part[((sg - 1) * NF * 8 + i) * 32 + lane] = C[t][i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0)
        for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < NF * 8; i++) C[t][i] += part[((s2 - 1) * NF * 8 + i) * 32 + lane];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == 0)
      for (int f = 0; f < NF; f++)
        for (int r = 0; r < 2; r++) {
          const int m = rb + t * 16 + fm + 8 * r;
          const int n = n0 + f * 16 + fn;
          if (m < M && n < N)
            for (int j = 0; j < 4; j++) Y[m * N + n + j] = static_cast<bfloat>(C[t][f * 8 + r * 4 + j]);
        }
  }
"""
