# Upstream (derived): jundot/omlx (Apache-2.0) omlx/patches/qwen35_verify_sdpa_split.py @ a98d8c8c
# Upstream (inspired): ashhart/TensorFold (MIT) src/tensorfold/kernels/qwen/dense/v1/lane_attention.py @ 34bae79a
"""Decode/verify attention over per-row key lengths (ragged batch KV).

Keys and values live in one buffer per layer, ``[S, HKV, CAP, D]``; query row
``b`` reads buffer row ``slots[b]`` (default ``b``), whose ``lengths[b]`` valid
keys sit at positions ``0 .. lengths[b] - 1`` (right-aligned storage, no left
padding). Each query row reads only its own keys, so a short request in a batch
with a 16K-token neighbour no longer pays for the neighbour's length —
upstream's left-padded ``BatchKVCache`` pads every row to the longest one.

Split-K ("flash decoding"): keys are cut into fixed ``CK``-key chunks; the
partial kernels compute each chunk's softmax partials, then a merge kernel
combines a row's chunk partials in chunk order. Chunk boundaries depend only on
absolute key position and every in-chunk reduction order is fixed, so a row's
result does not depend on the other rows in the batch (the fixed-chunk +
ordered-merge idea follows TensorFold's lane_attention). ``T`` > 1 query tokens
(speculative verify) attend causally within the window: token ``t`` sees keys
``< lengths[b] - (T - 1 - t)``, and its bits equal a one-token call at that
position (every kernel below).

Three partial kernels:

- key-parallel (default for head_dim 256): one threadgroup per (row, query
  token, KV head, chunk); its 8 simdgroups read each K/V row once for all ``G``
  heads with 16-byte loads, score all keys of the chunk, take one softmax over
  the chunk, then accumulate P.V with the simdgroups split over dims — no
  per-key serial chain. Given host-side row lengths it launches only the rows'
  non-empty chunks (a work list, the T tokens of a chunk back to back), so
  short rows next to a long one do not pay for empty threadgroups.
- tile (bf16, head_dim 256, GPUs with MetalPerformancePrimitives tensor ops):
  one threadgroup per (row, KV head, chunk) serves all T tokens x G heads with
  tensor-op matmuls, so a verify step reads K/V once instead of once per token
  (16 layers, 27B heads, T=6: 6.6 ms at 32K keys vs 13.2 ms key-parallel, on
  par with oMLX's verify kernel). The speculative lane runs decode and verify
  on it (``ragged_kv.set_dense_lane``).
- per-head (fallback, any head_dim that is a multiple of 32): one simdgroup per
  query head walks the chunk key by key with an online softmax; the G
  simdgroups of a KV group each read the chunk.

At B=1, 131K keys the key-parallel kernel streams bf16 K/V at the attainable
read bandwidth of a plain 16-byte-load kernel (``scripts/research/
probe_bandwidth.py``); the per-head kernel reached ~83% there and ~60% at 8K
(``scripts/research/bench_ragged_decode_attention.py --ab``).

INT8 variant: keys/values stored as int8 codes with one symmetric fp16 scale per
(row, KV head, token, 32-dim group) (``scale = max|x| / 127``). Splash uses one
scale per (token, head); 32-dim groups keep a few outlier channels from
coarsening the whole head (synthetic 20x outliers on 7/256 dims: cosine to bf16
attention 0.9976 -> 0.9990) for 6% more bytes. Each lane owns 8 consecutive dims,
so it applies one group scale to its partial dot product. The kernel reads codes
and scales directly and dequantizes in registers — K/V traffic ~0.53x of bf16.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

CHUNK = 512
GROUP = 32  # int8 KV: dims per scale group
NS = 8  # key-parallel kernel: simdgroups per threadgroup

_PARTIAL = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint g = thread_position_in_threadgroup.y;             // query head within the KV group
  const uint hk = threadgroup_position_in_grid.y / NC;
  const uint c = threadgroup_position_in_grid.y % NC;          // key chunk
  const uint bt = threadgroup_position_in_grid.z;              // row * T + query token
  const uint b = bt / T, t = bt % T;
  const int nkeys = lengths[b] - int(T - 1 - t);
  const int kbeg = int(c) * CK;
  const int kend = min(kbeg + CK, nkeys);
  const uint h = hk * G + g;
  const int64_t base = ((int64_t)bt * NC + c) * H + h;       // partial slot (bt*NC + c, h)
  constexpr int DPL = D / 32;
  device float* op = PO + base * D + lane * DPL;
  if (kbeg >= kend) {
    for (int i = 0; i < DPL; i++) op[i] = 0.0f;
    if (lane == 0) { PM[base] = -INFINITY; PL[base] = 0.0f; }
    return;
  }
  const device bfloat16_t* qp = queries + (((int64_t)b * H + h) * T + t) * D + lane * DPL;
  const float s = scale[0];
  float q[DPL];
  for (int i = 0; i < DPL; i++) q[i] = s * float(qp[i]);
  const int64_t kv0 = (((int64_t)slots[b] * HKV + hk) * CAP) * D + lane * DPL;
  const device bfloat16_t* kp = keys + kv0 + (int64_t)kbeg * D;
  const device bfloat16_t* vp = values + kv0 + (int64_t)kbeg * D;
  float m = -INFINITY, l = 0.0f, o[DPL];
  for (int i = 0; i < DPL; i++) o[i] = 0.0f;
  for (int k = kbeg; k < kend; k++) {
    float sc = 0.0f;
    for (int i = 0; i < DPL; i++) sc += q[i] * float(kp[i]);
    sc = simd_sum(sc);
    const float nm = max(m, sc);
    const float f = fast::exp(m - nm);
    const float e = fast::exp(sc - nm);
    l = l * f + e;
    for (int i = 0; i < DPL; i++) o[i] = o[i] * f + e * float(vp[i]);
    m = nm;
    kp += D;
    vp += D;
  }
  for (int i = 0; i < DPL; i++) op[i] = o[i];
  if (lane == 0) { PM[base] = m; PL[base] = l; }
"""

# Same loop as _PARTIAL over int8 codes: q.k uses the key code dot product
# times the key's scale; the value update folds the value's scale into e.
_PARTIAL_Q8 = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint g = thread_position_in_threadgroup.y;
  const uint hk = threadgroup_position_in_grid.y / NC;
  const uint c = threadgroup_position_in_grid.y % NC;
  const uint bt = threadgroup_position_in_grid.z;
  const uint b = bt / T, t = bt % T;
  const int nkeys = lengths[b] - int(T - 1 - t);
  const int kbeg = int(c) * CK;
  const int kend = min(kbeg + CK, nkeys);
  const uint h = hk * G + g;
  const int64_t base = ((int64_t)bt * NC + c) * H + h;       // partial slot (bt*NC + c, h)
  constexpr int DPL = D / 32;
  device float* op = PO + base * D + lane * DPL;
  if (kbeg >= kend) {
    for (int i = 0; i < DPL; i++) op[i] = 0.0f;
    if (lane == 0) { PM[base] = -INFINITY; PL[base] = 0.0f; }
    return;
  }
  const device bfloat16_t* qp = queries + (((int64_t)b * H + h) * T + t) * D + lane * DPL;
  const float s = scale[0];
  float q[DPL];
  for (int i = 0; i < DPL; i++) q[i] = s * float(qp[i]);
  const int64_t row0 = ((int64_t)slots[b] * HKV + hk) * CAP;
  const device int8_t* kp = keys + (row0 + kbeg) * D + lane * DPL;
  const device int8_t* vp = values + (row0 + kbeg) * D + lane * DPL;
  constexpr int NG = D / GS;                                   // scale groups per token
  const int grp = int(lane) * DPL / GS;                        // this lane's group
  const device half* ks = kscales + (row0 + kbeg) * NG + grp;
  const device half* vs = vscales + (row0 + kbeg) * NG + grp;
  float m = -INFINITY, l = 0.0f, o[DPL];
  for (int i = 0; i < DPL; i++) o[i] = 0.0f;
  for (int k = kbeg; k < kend; k++) {
    float sc = 0.0f;
    for (int i = 0; i < DPL; i++) sc += q[i] * float(kp[i]);
    sc = simd_sum(sc * float(ks[0]));
    const float nm = max(m, sc);
    const float f = fast::exp(m - nm);
    const float e = fast::exp(sc - nm);
    l = l * f + e;
    const float ev = e * float(vs[0]);
    for (int i = 0; i < DPL; i++) o[i] = o[i] * f + ev * float(vp[i]);
    m = nm;
    kp += D;
    vp += D;
    ks += NG;
    vs += NG;
  }
  for (int i = 0; i < DPL; i++) op[i] = o[i];
  if (lane == 0) { PM[base] = m; PL[base] = l; }
"""

# Key-parallel partial (default for D == 256): one threadgroup of NS simdgroups
# per (row * token, KV head, chunk) serves all G query heads of the KV group, so
# the chunk's K and V are read once, not once per query head. Three phases with
# no per-key serial dependency:
#   1. scores: lanes own 8 dims (one 16-byte load per key row), simdgroups take
#      keys sg, sg + NS, ...; one simd_sum per (key, head) into S[G][CK];
#   2. softmax: simdgroup g takes the max and exp-sum of head g's CK scores in a
#      fixed lane-strided order (one exp per key and head);
#   3. P.V: simdgroup sg owns D/NS dims, LPK lanes x 8 dims cover them and the
#      simdgroup's KG = 32/LPK lane groups take keys kg, kg + KG, ...; the groups
#      are summed by a fixed xor-shuffle tree.
# Every reduction order depends only on CK, NS and the key position within the
# chunk, so a row's bits do not depend on the other rows of the batch.
_KEYPAR = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  uint hk, c, bt;
  if (WL) {                           // work list: one entry per non-empty chunk
    const uint item = uint(work[threadgroup_position_in_grid.y / HKV]);
    hk = threadgroup_position_in_grid.y % HKV;
    bt = item / NC;
    c = item % NC;
  } else {
    hk = threadgroup_position_in_grid.y / NC;
    c = threadgroup_position_in_grid.y % NC;
    bt = threadgroup_position_in_grid.z;
  }
  const uint w = uint(wstart[bt]) + c;  // partial slot of (bt, c)
  const uint b = bt / T, t = bt % T;
  const int kbeg = int(c) * CK;
  const int n = min(CK, lengths[b] - int(T - 1 - t) - kbeg);   // keys in this chunk
  if (n <= 0) return;                 // beyond the row: the merge never reads it
  threadgroup float S[G * CK];
  const int64_t row0 = ((int64_t)slots[b] * HKV + hk) * CAP + kbeg;   // chunk's first key
  {
    const float s = scale[0];
    float q[G][8];
    for (int g = 0; g < G; g++) {
      const device bfloat16_t* qp =
          queries + (((int64_t)b * H + hk * G + g) * T + t) * D + lane * 8;
      for (int i = 0; i < 8; i++) q[g][i] = s * float(qp[i]);
    }
    for (int k = int(sg); k < CK; k += NS) {
      if (k < n) {
        float kv[8];
        LOAD_K
        for (int g = 0; g < G; g++) {
          float a = 0.0f;
          for (int i = 0; i < 8; i++) a = fma(q[g][i], kv[i], a);
          a = simd_sum(a * KSCALE);
          if (lane == 0) S[g * CK + k] = a;
        }
      } else if (lane == 0) {
        for (int g = 0; g < G; g++) S[g * CK + k] = -INFINITY;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int g = int(sg); g < G; g += NS) {
    float mg = -INFINITY;
    for (int i = int(lane); i < CK; i += 32) mg = max(mg, S[g * CK + i]);
    mg = simd_max(mg);
    float lg = 0.0f;
    for (int i = int(lane); i < CK; i += 32) {
      const float p = fast::exp(S[g * CK + i] - mg);
      S[g * CK + i] = p;
      lg += p;
    }
    lg = simd_sum(lg);
    if (lane == 0) {
      const int64_t base = (int64_t)w * H + hk * G + g;
      PM[base] = mg; PL[base] = lg;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  constexpr int DS = D / NS, LPK = DS / 8, KG = 32 / LPK;
  const int d0 = int(sg) * DS + int(lane % LPK) * 8;
  const int kg = int(lane / LPK);
  float acc[G][8];
  for (int g = 0; g < G; g++) for (int i = 0; i < 8; i++) acc[g][i] = 0.0f;
  for (int k = kg; k < n; k += KG) {
    float vv[8];
    LOAD_V
    for (int g = 0; g < G; g++) {
      const float p = S[g * CK + k] * VSCALE;
      for (int i = 0; i < 8; i++) acc[g][i] = fma(p, vv[i], acc[g][i]);
    }
  }
  for (ushort off = LPK; off < 32; off <<= 1)
    for (int g = 0; g < G; g++) for (int i = 0; i < 8; i++)
      acc[g][i] += simd_shuffle_xor(acc[g][i], off);
  if (kg == 0) {
    for (int g = 0; g < G; g++) {
      const int64_t base = (int64_t)w * H + hk * G + g;
      for (int i = 0; i < 8; i++) PO[base * D + d0 + i] = acc[g][i];
    }
  }
"""

_UNPACK_BF16 = """
    {0}[0] = as_type<float>(r.x << 16); {0}[1] = as_type<float>(r.x & 0xffff0000u);
    {0}[2] = as_type<float>(r.y << 16); {0}[3] = as_type<float>(r.y & 0xffff0000u);
    {0}[4] = as_type<float>(r.z << 16); {0}[5] = as_type<float>(r.z & 0xffff0000u);
    {0}[6] = as_type<float>(r.w << 16); {0}[7] = as_type<float>(r.w & 0xffff0000u);"""
_UNPACK_Q8 = """
    const char4 c0 = as_type<char4>(r.x), c1 = as_type<char4>(r.y);
    {0}[0] = c0.x; {0}[1] = c0.y; {0}[2] = c0.z; {0}[3] = c0.w;
    {0}[4] = c1.x; {0}[5] = c1.y; {0}[6] = c1.z; {0}[7] = c1.w;"""

_KEYPAR_BF16 = (
    _KEYPAR.replace(
        "LOAD_K",
        "{ const uint4 r = *(const device uint4*)(keys + (row0 + k) * D + lane * 8);"
        + _UNPACK_BF16.format("kv")
        + " }",
    )
    .replace(
        "LOAD_V",
        "{ const uint4 r = *(const device uint4*)(values + (row0 + k) * D + d0);"
        + _UNPACK_BF16.format("vv")
        + " }",
    )
    .replace("KSCALE", "1.0f")
    .replace("VSCALE", "1.0f")
)
# int8: 8-byte code loads; the key's group scale multiplies the lane's partial
# dot product, the value's group scale folds into p.
_KEYPAR_Q8 = (
    _KEYPAR.replace(
        "LOAD_K",
        "{ const uint2 r = *(const device uint2*)(keys + (row0 + k) * D + lane * 8);"
        + _UNPACK_Q8.format("kv")
        + " }",
    )
    .replace(
        "LOAD_V",
        "{ const uint2 r = *(const device uint2*)(values + (row0 + k) * D + d0);"
        + _UNPACK_Q8.format("vv")
        + " }",
    )
    .replace("KSCALE", "float(kscales[(row0 + k) * (D / GS) + lane * 8 / GS])")
    .replace("VSCALE", "float(vscales[(row0 + k) * (D / GS) + d0 / GS])")
)

# Token-tile partial (bf16, D == 256, M5-class tensor ops): one threadgroup of
# 8 simdgroups per (row, KV head, chunk) serves every query token x head of the
# KV head at once, so a speculative verify step reads each K/V row once, not
# once per token. Queries arrive fused as [B, HKV, 8 * G, D] (row f = t * G +
# g, tokens padded to 8 with zeros), so the matmul shapes are the same for any
# T <= 8 — a verify row runs the same arithmetic as a one-token decode of the
# same position. The chunk is walked in N-key steps with an online softmax per
# row: Q.K^T and P.V run on MetalPerformancePrimitives matmul2d (fp32
# scores, bf16 probabilities, fp32 accumulators); four threads own a row's
# softmax. The structure follows oMLX's verify kernel
# (``kernels/omlx/qwen35_verify_sdpa_split._GQA_PARTIAL``, Apache-2.0) with
# fixed 512-key chunks, per-row lengths, slots and the causal limit per token.
_TILE_HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""

_TILE = r"""
  constexpr int N = 64;               // keys per step
  constexpr int KPL = N / 4;          // keys per thread in the softmax
  constexpr int M = 8 * G;            // fused rows: 8 tokens x G heads
  const uint tid = thread_position_in_threadgroup.x;
  uint hk, c, b;
  if (WL) {                           // work list: one entry per (row, chunk)
    const uint item = uint(work[threadgroup_position_in_grid.y / HKV]);
    hk = threadgroup_position_in_grid.y % HKV;
    b = item / NC;
    c = item % NC;
  } else {
    hk = threadgroup_position_in_grid.y / NC;
    c = threadgroup_position_in_grid.y % NC;
    b = threadgroup_position_in_grid.z;
  }
  const int kbeg = int(c) * CK;
  const int lim = lengths[b] - kbeg;  // keys of the last token in this chunk
  const int nmax = min(CK, lim);
  if (nmax <= 0) return;

  threadgroup float scores[M * N];
  threadgroup bfloat16_t probs[M * N];
  threadgroup float row_max[M];
  threadgroup float row_sum[M];
  threadgroup float row_scale[M];
  if (tid < uint(M)) { row_max[tid] = -INFINITY; row_sum[tid] = 0.0f; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto qt = tensor(
      const_cast<device bfloat16_t*>(queries) + ((int64_t)b * HKV + hk) * M * D,
      dextents<int, 2>{D, M}, array<int, 2>{1, D});
  auto st = tensor((threadgroup float*)scores, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto pt = tensor((threadgroup bfloat16_t*)probs, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto q0 = qt.template slice<D, M>(0, 0);
  auto p0 = pt.template slice<N, M>(0, 0);
  const int64_t row0 = ((int64_t)slots[b] * HKV + hk) * CAP;
  device bfloat16_t* kh = const_cast<device bfloat16_t*>(keys) + row0 * D;
  device bfloat16_t* vh = const_cast<device bfloat16_t*>(values) + row0 * D;
  constexpr auto qk_desc = matmul2d_descriptor(
      M, N, D, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto pv_desc = matmul2d_descriptor(
      M, D, N, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<8>> qk;
  matmul2d<pv_desc, execution_simdgroups<8>> pv;
  auto v_first = tensor(vh, dextents<int, 2>{D, N}, array<int, 2>{1, D}).template slice<D, N>(0, 0);
  auto running = pv.template get_destination_cooperative_tensor<
      decltype(p0), decltype(v_first), float>();
  for (ushort i = 0; i < running.get_capacity(); ++i)
    if (running.is_valid_element(i)) running[i] = 0.0f;

  const float s = scale[0];
  const int f = int(tid) / 4;         // four threads own fused row f
  const int col = (int(tid) % 4) * KPL;
  const int t_of_f = f / G;
  // keys this row may see in the chunk (causal limit of its token)
  const int nf = t_of_f < T ? lim - (T - 1 - t_of_f) : 0;
  const int kend = kbeg + nmax;
  for (int t0 = kbeg; t0 < kend; t0 += N) {
    const int ts = min(t0, CAP - N);  // stay inside the buffer; keys < t0 masked
    auto ks = tensor(kh + (int64_t)ts * D, dextents<int, 2>{D, N}, array<int, 2>{1, D})
        .template slice<D, N>(0, 0);
    auto vs = tensor(vh + (int64_t)ts * D, dextents<int, 2>{D, N}, array<int, 2>{1, D})
        .template slice<D, N>(0, 0);
    auto sc = qk.template get_destination_cooperative_tensor<
        decltype(q0), decltype(ks), float>();
    qk.run(q0, ks, sc);
    sc.store(st.template slice<N, M>(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (f < M) {
      float sv[KPL];
      float lmax = -INFINITY;
      for (int j = 0; j < KPL; ++j) {
        const int key = ts + col + j;
        const bool ok = key >= t0 && key - kbeg < nf;
        sv[j] = ok ? scores[f * N + col + j] * s : -INFINITY;
        lmax = max(lmax, sv[j]);
      }
      lmax = max(lmax, simd_shuffle_xor(lmax, ushort(1)));
      lmax = max(lmax, simd_shuffle_xor(lmax, ushort(2)));
      const float pmax = row_max[f];
      const float nm = max(pmax, lmax);
      float lsum = 0.0f;
      for (int j = 0; j < KPL; ++j) {
        const float p = sv[j] == -INFINITY ? 0.0f : fast::exp(sv[j] - nm);
        lsum += p;
        probs[f * N + col + j] = bfloat16_t(p);
      }
      lsum += simd_shuffle_xor(lsum, ushort(1));
      lsum += simd_shuffle_xor(lsum, ushort(2));
      if (col == 0) {
        const float a = nm == pmax ? 1.0f : fast::exp(pmax - nm);
        row_scale[f] = a;
        row_sum[f] = row_sum[f] * a + lsum;
        row_max[f] = nm;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort i = 0; i < running.get_capacity(); ++i) {
      if (!running.is_valid_element(i)) continue;
      auto ix = running.get_multidimensional_index(i);
      running[i] *= row_scale[ix[1]];
    }
    pv.run(p0, vs, running);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // partial slot of (token, chunk): wstart[b * T + t] + c; rows past T or
  // without keys in this chunk have no slot
  for (ushort i = 0; i < running.get_capacity(); ++i) {
    if (!running.is_valid_element(i)) continue;
    auto ix = running.get_multidimensional_index(i);
    const int fr = ix[1], t = fr / G;
    if (t >= T || lim - (T - 1 - t) <= 0) continue;
    const int64_t base = ((int64_t)wstart[b * T + t] + c) * H + hk * G + fr % G;
    PO[base * D + ix[0]] = running[i];
  }
  if (tid < uint(M)) {
    const int t = int(tid) / G;
    if (t < T && lim - (T - 1 - t) > 0) {
      const int64_t base = ((int64_t)wstart[b * T + t] + c) * H + hk * G + int(tid) % G;
      PM[base] = row_max[tid];
      PL[base] = row_sum[tid];
    }
  }
"""

_MERGE = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint r = threadgroup_position_in_grid.y;               // (row * T + token) * H + head
  constexpr int DPL = D / 32;
  const uint bt = r / H, h = r % H;
  const uint b = bt / T, t = bt % T;
  const int nkeys = lengths[b] - int(T - 1 - t);
  const int nc = min(NC, max(0, (nkeys + CK - 1) / CK));      // the row's own chunks
  float m = -INFINITY, l = 0.0f, o[DPL];
  for (int i = 0; i < DPL; i++) o[i] = 0.0f;
  for (int c = 0; c < nc; c++) {                               // chunk order: batch-independent bits
    const int64_t row = ((int64_t)wstart[bt] + c) * H + h;
    const float mc = PM[row];
    if (mc == -INFINITY) continue;
    const float nm = max(m, mc);
    const float f1 = fast::exp(m - nm), f2 = fast::exp(mc - nm);
    l = l * f1 + PL[row] * f2;
    for (int i = 0; i < DPL; i++) o[i] = o[i] * f1 + PO[row * D + lane * DPL + i] * f2;
    m = nm;
  }
  // output layout [B, H, T, D] like mx.fast.scaled_dot_product_attention
  device bfloat16_t* dst = out + (((int64_t)b * H + h) * T + t) * D + lane * DPL;
  const float inv = l > 0.0f ? 1.0f / l : 0.0f;
  for (int i = 0; i < DPL; i++) dst[i] = static_cast<bfloat16_t>(o[i] * inv);
"""

_KERNELS: dict = {}


def _kernel(name: str):
    if name not in _KERNELS:
        if name == "tile":
            _KERNELS[name] = mx.fast.metal_kernel(
                name="yunshu_ragged_attn_tile",
                input_names=[
                    "queries",
                    "keys",
                    "values",
                    "lengths",
                    "slots",
                    "scale",
                    "work",
                    "wstart",
                ],
                output_names=["PO", "PM", "PL"],
                source=_TILE,
                header=_TILE_HEADER,
                ensure_row_contiguous=False,
            )
        elif name in ("keypar", "keypar_q8"):
            q8 = name == "keypar_q8"
            _KERNELS[name] = mx.fast.metal_kernel(
                name=f"yunshu_ragged_attn_{name}",
                input_names=(
                    ["queries", "keys", "kscales", "values", "vscales", "lengths"]
                    if q8
                    else ["queries", "keys", "values", "lengths"]
                )
                + ["slots", "scale", "work", "wstart"],
                output_names=["PO", "PM", "PL"],
                source=_KEYPAR_Q8 if q8 else _KEYPAR_BF16,
                ensure_row_contiguous=False,
            )
        elif name == "partial_q8":
            _KERNELS[name] = mx.fast.metal_kernel(
                name="yunshu_ragged_attn_partial_q8",
                input_names=[
                    "queries",
                    "keys",
                    "kscales",
                    "values",
                    "vscales",
                    "lengths",
                    "slots",
                    "scale",
                ],
                output_names=["PO", "PM", "PL"],
                source=_PARTIAL_Q8,
                ensure_row_contiguous=False,
            )
        elif name == "partial":
            _KERNELS[name] = mx.fast.metal_kernel(
                name="yunshu_ragged_attn_partial",
                input_names=["queries", "keys", "values", "lengths", "slots", "scale"],
                output_names=["PO", "PM", "PL"],
                source=_PARTIAL,
                ensure_row_contiguous=False,
            )
        else:
            _KERNELS[name] = mx.fast.metal_kernel(
                name="yunshu_ragged_attn_merge",
                input_names=["PO", "PM", "PL", "lengths", "wstart"],
                output_names=["out"],
                source=_MERGE,
            )
    return _KERNELS[name]


_WORK: dict = {}
_TILE_READY: list = []


def tile_ready() -> bool:
    """Certify nonzero tensor-op attention and token-row invariance once.

    MPP can compile on older Apple GPUs using software tensor operations;
    successful compilation or an all-zero output alone proves no arithmetic.
    """
    if not _TILE_READY:
        try:
            q = ((mx.arange(1024).reshape(1, 2, 2, 256) % 19 - 9) / 16).astype(
                mx.bfloat16
            )
            k = ((mx.arange(16384).reshape(1, 1, 64, 256) % 23 - 11) / 16).astype(
                mx.bfloat16
            )
            v = ((mx.arange(16384).reshape(1, 1, 64, 256) % 29 - 14) / 16).astype(
                mx.bfloat16
            )
            lengths = mx.array([3])
            _TILE_READY.append(True)  # let the probe call through
            out = ragged_decode_attention(
                q, k, v, lengths, 0.0625, impl="tile", row_lengths=(3,)
            )
            one = ragged_decode_attention(
                q[:, :, -1:], k, v, lengths, 0.0625, impl="tile", row_lengths=(3,)
            )
            ref = mx.fast.scaled_dot_product_attention(
                q, k[:, :, :3], v[:, :, :3], scale=0.0625, mask="causal"
            )
            if not mx.array_equal(out[:, :, -1:], one).item():
                raise RuntimeError("tile attention differs by token-row count")
            if not (
                mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))) < 0.02
            ).item():
                raise RuntimeError("tile attention differs from reference")
        except Exception:  # noqa: BLE001 - any compile, launch or numerical failure
            _TILE_READY[:] = [False]
    return _TILE_READY[0]


_ARANGE: dict = {}


def _arange(n: int) -> mx.array:
    a = _ARANGE.get(n)
    if a is None:
        a = _ARANGE[n] = mx.arange(n, dtype=mx.int32)
    return a


def _work_list(
    row_lengths: tuple, T: int, nc: int, tile: bool = False
) -> tuple[mx.array, mx.array, int]:
    """The key-parallel kernel's work-list launch: ``work`` holds ``bt * NC +
    c`` for every non-empty chunk ``c`` of every (row, query token) ``bt``,
    and ``wstart[bt]`` is bt's first partial slot (slot of (bt, c) is
    ``wstart[bt] + c``, what the merge reads). Items run row, chunk, token:
    the T verify tokens of one chunk are launched back to back, so they read
    the chunk's K/V while it is still in cache. Cached per step (a decode
    step's layers share one list). ``tile``: one item ``b * NC + c`` per (row,
    chunk) for the token-tile kernel (same slots)."""
    key = (row_lengths, T, nc, CHUNK, tile)
    hit = _WORK.get(key)
    if hit is None:
        items, starts, total = [], [], 0
        for b, n in enumerate(row_lengths):
            counts = [min(nc, max(0, -(-(n - (T - 1 - t)) // CHUNK))) for t in range(T)]
            for t in range(T):
                starts.append(total)
                total += counts[t]
            for c in range(max(counts, default=0)):
                if tile:
                    items.append(b * nc + c)
                else:
                    items += [(b * T + t) * nc + c for t in range(T) if c < counts[t]]
        hit = (
            mx.array(items or [0], dtype=mx.int32),
            mx.array(starts, dtype=mx.int32),
            max(total, 1),
        )
        if len(_WORK) > 8:
            _WORK.clear()
        _WORK[key] = hit
    return hit


TILE_TOKENS = 8  # query tokens one tile launch serves (the kernel's fused rows)
MAX_WINDOW = 32  # widest speculative window (tokens) the tile kernel path takes


def ragged_decode_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    lengths: mx.array,
    scale: float,
    max_length: int | None = None,
    k_scales: mx.array | None = None,
    v_scales: mx.array | None = None,
    impl: str = "auto",
    row_lengths: Sequence[int] | None = None,
    slots: mx.array | None = None,
) -> mx.array:
    """``_attend`` for any window of up to ``MAX_WINDOW`` query tokens.

    Windows wider than the tile kernel's 8 fused tokens run as groups of 8
    consecutive tokens, each one launch of the same kernel over the same keys
    with the group's own causal limit. A token's arithmetic is that of the
    8-token launch whatever its window position, which equals a one-token
    decode of its position (tokens are padded to 8 there), so a wide verify
    row stays bit-identical to plain decode. Each group re-reads the chunk
    keys (the wide window's cost over keys is one read per 8 tokens).
    """
    T = int(queries.shape[2])
    if T <= TILE_TOKENS:
        return _attend(
            queries,
            keys,
            values,
            lengths,
            scale,
            max_length,
            k_scales,
            v_scales,
            impl,
            row_lengths,
            slots,
        )
    if impl != "tile" or T > MAX_WINDOW:
        raise ValueError(f"ragged_decode_attention: {T} tokens need impl='tile'")
    outs = []
    for g in range(0, T, TILE_TOKENS):
        tg = min(TILE_TOKENS, T - g)
        off = (
            T - g - tg
        )  # tokens after this group: its last token sees this many fewer keys
        outs.append(
            _attend(
                queries[:, :, g : g + tg],
                keys,
                values,
                lengths - off if off else lengths,
                scale,
                None if max_length is None else max_length - off,
                k_scales,
                v_scales,
                impl,
                None
                if row_lengths is None
                else tuple(int(n) - off for n in row_lengths),
                slots,
            )
        )
    return mx.concatenate(outs, axis=2)


def _attend(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    lengths: mx.array,
    scale: float,
    max_length: int | None = None,
    k_scales: mx.array | None = None,
    v_scales: mx.array | None = None,
    impl: str = "auto",
    row_lengths: Sequence[int] | None = None,
    slots: mx.array | None = None,
) -> mx.array:
    """Attention of ``queries`` [B, H, T, D] over each row's first ``lengths[b]``
    keys of ``keys``/``values`` [S, HKV, CAP, D]; returns [B, H, T, D] bf16.

    ``slots`` (int32 [B], default ``arange(B)``) maps query row ``b`` to its
    buffer row, so a cache can keep free rows and drop finished ones without
    moving the others.

    ``max_length`` (host int) sizes the chunk grid; defaults to the buffer
    capacity. The last ``T`` keys of each row are the query tokens' own keys.
    int8 ``keys``/``values`` need ``k_scales``/``v_scales`` [B, HKV, CAP, D/32]
    fp16 (see ``quantize_kv``).

    ``row_lengths`` (host ints equal to ``lengths``) lets the key-parallel
    kernel launch only each row's own chunks instead of ``max_length``'s worth
    per row; the bits are the same either way. ``impl``: ``"key_parallel"``,
    ``"per_head"`` or ``"auto"`` (key-parallel where it applies).
    """
    B, H, T, D = (int(v) for v in queries.shape)
    S, HKV, CAP, Dk = (int(v) for v in keys.shape)
    if slots is None:
        if S < B:
            raise ValueError("ragged_decode_attention: fewer buffer rows than queries")
        slots = _arange(B)
    if Dk != D or D % 32 or H % HKV or T > 8:
        raise ValueError(
            f"ragged_decode_attention: unsupported q={queries.shape} k={keys.shape}"
        )
    q8 = keys.dtype == mx.int8
    if queries.dtype != mx.bfloat16 or keys.dtype not in (mx.bfloat16, mx.int8):
        raise ValueError("ragged_decode_attention: bf16 queries, bf16 or int8 K/V")
    if q8 and (k_scales is None or v_scales is None or values.dtype != mx.int8):
        raise ValueError("ragged_decode_attention: int8 K/V need k_scales/v_scales")
    G = H // HKV
    nc = max(1, -(-int(max_length or CAP) // CHUNK))
    # The kernel indexes a dense [S, HKV, CAP, D] buffer; the ragged cache hands
    # over its whole buffer (not a slice), so these are no-ops there.
    keys, values, queries = (
        mx.contiguous(keys),
        mx.contiguous(values),
        mx.contiguous(queries),
    )
    tmpl = [
        ("D", D),
        ("G", G),
        ("T", T),
        ("H", H),
        ("HKV", HKV),
        ("CAP", CAP),
        ("CK", CHUNK),
        ("NC", nc),
    ]
    lengths = lengths.astype(mx.int32)
    slots = slots.astype(mx.int32)
    scale_arr = mx.array([float(scale)], dtype=mx.float32)
    # Kernels: "key_parallel" (default, D == 256: one query token per
    # threadgroup), "per_head" (any D % 32 == 0), and "tile" (bf16, D == 256,
    # tensor ops: every token x head of a KV head in one pass over the keys —
    # the speculative lane's decode and verify).
    if impl == "auto":
        impl = "key_parallel" if D == 256 else "per_head"
    tile = impl == "tile"
    if tile and (q8 or D != 256 or G > 8 or CAP < 64 or T > 8 or not tile_ready()):
        raise ValueError("ragged_decode_attention: tile needs bf16, D=256, G<=8")
    use_kp = impl == "key_parallel" and D == 256 and G * CHUNK * 4 <= 32 * 1024
    if impl == "key_parallel" and not use_kp:
        raise ValueError("ragged_decode_attention: key_parallel needs D=256")
    wl = (tile or use_kp) and row_lengths is not None
    if wl:
        work, wstart, nslot = _work_list(
            tuple(int(n) for n in row_lengths), T, nc, tile
        )
    else:
        # dense launch: every (bt, chunk < NC); slot bt * NC + c
        work = scale_arr
        wstart = mx.arange(B * T, dtype=mx.int32) * nc
        nslot = B * T * nc
    if tile:
        # fused rows f = t * G + g per (row, KV head), tokens padded to 8
        q = queries.reshape(B, HKV, G, T, D).transpose(0, 1, 3, 2, 4)
        if T < 8:
            q = mx.pad(q, [(0, 0), (0, 0), (0, 8 - T), (0, 0), (0, 0)])
        queries = mx.contiguous(q.reshape(B, HKV, 8 * G, D))
    if q8:
        name = "partial_q8"
        inputs = [
            queries,
            keys,
            mx.contiguous(k_scales.astype(mx.float16)),
            values,
            mx.contiguous(v_scales.astype(mx.float16)),
            lengths,
            slots,
            scale_arr,
        ]
    else:
        name = "partial"
        inputs = [queries, keys, values, lengths, slots, scale_arr]
    out_shapes = [(nslot * H * D,), (nslot * H,), (nslot * H,)]
    if tile:
        po, pm, pl = _kernel("tile")(
            inputs=inputs + [work, wstart],
            template=tmpl + [("WL", wl)],
            grid=(256, HKV * int(work.size), 1) if wl else (256, HKV * nc, B),
            threadgroup=(256, 1, 1),
            output_shapes=out_shapes,
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
    elif use_kp:
        po, pm, pl = _kernel(name.replace("partial", "keypar"))(
            inputs=inputs + [work, wstart],
            template=tmpl + [("NS", NS), ("WL", wl)] + ([("GS", GROUP)] if q8 else []),
            grid=(32 * NS, HKV * nslot, 1) if wl else (32 * NS, HKV * nc, B * T),
            threadgroup=(32 * NS, 1, 1),
            output_shapes=out_shapes,
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
    else:
        po, pm, pl = _kernel(name)(
            inputs=inputs,
            template=tmpl + [("GS", GROUP)] if q8 else tmpl,
            grid=(32, G * HKV * nc, B * T),
            threadgroup=(32, G, 1),
            output_shapes=out_shapes,
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
    return _kernel("merge")(
        inputs=[po, pm, pl, lengths, wstart],
        template=[("D", D), ("T", T), ("H", H), ("NC", nc), ("CK", CHUNK)],
        grid=(32, B * T * H, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, H, T, D)],
        output_dtypes=[mx.bfloat16],
    )[0]


# One threadgroup of 32 lanes per 32-dim group: the group's max |x| by one
# simd_max, then each lane writes its code. Same arithmetic as
# ``quantize_kv_reference``.
_QUANT = r"""
  const uint lane = thread_position_in_threadgroup.x;
  const uint gi = threadgroup_position_in_grid.y;              // (token row) * NG + group
  const bool isv = threadgroup_position_in_grid.z == 1;
  const float x = float((isv ? v : k)[gi * 32 + lane]);
  const half sh = half(simd_max(fabs(x)) / 127.0f);
  const float sf = float(sh);
  const float q = clamp(rint(x / (sf > 0.0f ? sf : 1.0f)), -127.0f, 127.0f);
  (isv ? vq : kq)[gi * 32 + lane] = int8_t(q);
  if (lane == 0) (isv ? vs : ks)[gi] = sh;
"""


def quantize_kv_pair(k: mx.array, v: mx.array) -> tuple[mx.array, ...]:
    """``quantize_kv`` of K and V (same shape, bf16) in one launch: returns
    ``(k_codes, k_scales, v_codes, v_scales)``."""
    if k.shape != v.shape or k.shape[-1] % GROUP:
        raise ValueError(f"quantize_kv_pair: k {k.shape} v {v.shape}")
    if "quant" not in _KERNELS:
        _KERNELS["quant"] = mx.fast.metal_kernel(
            name="yunshu_ragged_kv_quantize",
            input_names=["k", "v"],
            output_names=["kq", "ks", "vq", "vs"],
            source=_QUANT,
        )
    *lead, d = k.shape
    groups = k.size // GROUP
    kq, ks, vq, vs = _KERNELS["quant"](
        inputs=[k.astype(mx.bfloat16), v.astype(mx.bfloat16)],
        grid=(32, groups, 2),
        threadgroup=(32, 1, 1),
        output_shapes=[k.shape, (*lead, d // GROUP)] * 2,
        output_dtypes=[mx.int8, mx.float16] * 2,
    )
    return kq, ks, vq, vs


def quantize_kv(x: mx.array) -> tuple[mx.array, mx.array]:
    """Symmetric int8 over 32-dim groups of the last axis: ``x`` [..., D] ->
    int8 codes [..., D] and fp16 scales [..., D/32] (``max|x| / 127``; 0 for
    all-zero groups). Codes use the fp16-rounded scale, so dequantization in
    the kernel matches what was quantized."""
    kq, ks, _, _ = quantize_kv_pair(x, x)
    return kq, ks


def quantize_kv_reference(x: mx.array) -> tuple[mx.array, mx.array]:
    """``quantize_kv`` in MLX ops (the definition the kernel is tested
    against)."""
    *lead, d = x.shape
    xg = x.astype(mx.float32).reshape(*lead, d // GROUP, GROUP)
    s = (mx.max(mx.abs(xg), axis=-1) / 127.0).astype(mx.float16)
    sf = s.astype(mx.float32)
    safe = mx.where(sf > 0, sf, mx.ones_like(sf))
    q = mx.clip(mx.round(xg / safe[..., None]), -127, 127).astype(mx.int8)
    return q.reshape(*lead, d), s


__all__ = [
    "CHUNK",
    "quantize_kv",
    "quantize_kv_pair",
    "quantize_kv_reference",
    "ragged_decode_attention",
]
