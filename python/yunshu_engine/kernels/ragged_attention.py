"""Decode/verify attention over per-row key lengths (ragged batch KV).

Keys and values live in one buffer per layer, ``[B, HKV, CAP, D]``, with row
``b``'s ``lengths[b]`` valid keys at positions ``0 .. lengths[b] - 1`` (right-
aligned storage, no left padding). Each query row reads only its own keys, so a
short request in a batch with a 16K-token neighbour no longer pays for the
neighbour's length — upstream's left-padded ``BatchKVCache`` pads every row to
the longest one.

Split-K ("flash decoding"): keys are cut into fixed ``CK``-key chunks; one
threadgroup per (row, query token, KV head, chunk) computes the chunk's partial
softmax for the KV head's ``G`` query heads, then a merge kernel combines a
row's chunk partials in chunk order. Chunk boundaries depend only on absolute
key position and every in-chunk reduction order is fixed, so a row's result
does not depend on the other rows in the batch (the fixed-chunk + ordered-merge
idea follows TensorFold's lane_attention; these kernels are written
independently, plain SIMD). ``T`` > 1 query tokens (speculative verify) attend
causally within the window: token ``t`` sees keys ``< lengths[b] - (T - 1 - t)``.

Two partial kernels:

- key-parallel (default for head_dim 256): the threadgroup's 8 simdgroups read
  each K/V row once for all ``G`` heads with 16-byte loads, score all keys of
  the chunk, take one softmax over the chunk, then accumulate P.V with the
  simdgroups split over dims — no per-key serial chain. Given host-side row
  lengths it launches only the rows' non-empty chunks (a work list), so short
  rows next to a long one do not pay for empty threadgroups.
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
  const int64_t kv0 = (((int64_t)b * HKV + hk) * CAP) * D + lane * DPL;
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
  const int64_t row0 = ((int64_t)b * HKV + hk) * CAP;
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
  uint hk, c, bt, w;                  // w: partial slot of (bt, c)
  if (WL) {                           // work list: one entry per non-empty chunk
    w = threadgroup_position_in_grid.y / HKV;
    const uint item = uint(work[w]);
    hk = threadgroup_position_in_grid.y % HKV;
    bt = item / NC;
    c = item % NC;
  } else {
    hk = threadgroup_position_in_grid.y / NC;
    c = threadgroup_position_in_grid.y % NC;
    bt = threadgroup_position_in_grid.z;
    w = bt * NC + c;
  }
  const uint b = bt / T, t = bt % T;
  const int kbeg = int(c) * CK;
  const int n = min(CK, lengths[b] - int(T - 1 - t) - kbeg);   // keys in this chunk
  if (n <= 0) return;                 // beyond the row: the merge never reads it
  threadgroup float S[G * CK];
  const int64_t row0 = ((int64_t)b * HKV + hk) * CAP + kbeg;   // chunk's first key
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
        if name in ("keypar", "keypar_q8"):
            q8 = name == "keypar_q8"
            _KERNELS[name] = mx.fast.metal_kernel(
                name=f"yunshu_ragged_attn_{name}",
                input_names=(
                    ["queries", "keys", "kscales", "values", "vscales", "lengths"]
                    if q8
                    else ["queries", "keys", "values", "lengths"]
                )
                + ["scale", "work"],
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
                    "scale",
                ],
                output_names=["PO", "PM", "PL"],
                source=_PARTIAL_Q8,
                ensure_row_contiguous=False,
            )
        elif name == "partial":
            _KERNELS[name] = mx.fast.metal_kernel(
                name="yunshu_ragged_attn_partial",
                input_names=["queries", "keys", "values", "lengths", "scale"],
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


def _work_list(row_lengths: tuple, T: int, nc: int) -> tuple[mx.array, mx.array]:
    """Partial slots for the key-parallel kernel's work-list launch: ``work``
    holds ``bt * NC + c`` for every non-empty chunk ``c`` of every (row, query
    token) ``bt`` in (bt, c) order, and ``wstart[bt]`` is bt's first slot.
    Cached per step (a decode step's layers share one list)."""
    key = (row_lengths, T, nc, CHUNK)
    hit = _WORK.get(key)
    if hit is None:
        items, starts = [], []
        for b, n in enumerate(row_lengths):
            for t in range(T):
                starts.append(len(items))
                nk = n - (T - 1 - t)
                items += [
                    (b * T + t) * nc + c
                    for c in range(min(nc, max(0, -(-nk // CHUNK))))
                ]
        hit = (
            mx.array(items or [0], dtype=mx.int32),
            mx.array(starts, dtype=mx.int32),
        )
        _WORK.clear()
        _WORK[key] = hit
    return hit


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
) -> mx.array:
    """Attention of ``queries`` [B, H, T, D] over each row's first ``lengths[b]``
    keys of ``keys``/``values`` [B, HKV, CAP, D]; returns [B, H, T, D] bf16.

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
    _, HKV, CAP, Dk = (int(v) for v in keys.shape)
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
    # The kernel indexes a dense [B, HKV, CAP, D] buffer; the ragged cache hands
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
    scale_arr = mx.array([float(scale)], dtype=mx.float32)
    # Key-parallel kernel: 8 dims per lane in the score phase (D == 256) and a
    # G x CHUNK score tile in threadgroup memory; otherwise the per-head kernel.
    use_kp = impl != "per_head" and D == 256 and G * CHUNK * 4 <= 32 * 1024
    if impl == "key_parallel" and not use_kp:
        raise ValueError("ragged_decode_attention: key_parallel needs D=256")
    wl = use_kp and row_lengths is not None
    if wl:
        work, wstart = _work_list(tuple(int(n) for n in row_lengths), T, nc)
        slots = int(work.size)
    else:
        # dense launch: every (bt, chunk < NC); slot bt * NC + c
        work = scale_arr
        wstart = mx.arange(B * T, dtype=mx.int32) * nc
        slots = B * T * nc
    if q8:
        name = "partial_q8"
        inputs = [
            queries,
            keys,
            mx.contiguous(k_scales.astype(mx.float16)),
            values,
            mx.contiguous(v_scales.astype(mx.float16)),
            lengths,
            scale_arr,
        ]
    else:
        name = "partial"
        inputs = [queries, keys, values, lengths, scale_arr]
    out_shapes = [(slots * H * D,), (slots * H,), (slots * H,)]
    if use_kp:
        po, pm, pl = _kernel(name.replace("partial", "keypar"))(
            inputs=inputs + [work],
            template=tmpl + [("NS", NS), ("WL", wl)] + ([("GS", GROUP)] if q8 else []),
            grid=(32 * NS, HKV * slots, 1) if wl else (32 * NS, HKV * nc, B * T),
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


def quantize_kv(x: mx.array) -> tuple[mx.array, mx.array]:
    """Symmetric int8 over 32-dim groups of the last axis: ``x`` [..., D] ->
    int8 codes [..., D] and fp16 scales [..., D/32] (``max|x| / 127``; 0 for
    all-zero groups). Codes use the fp16-rounded scale, so dequantization in
    the kernel matches what was quantized."""
    *lead, d = x.shape
    xg = x.astype(mx.float32).reshape(*lead, d // GROUP, GROUP)
    s = (mx.max(mx.abs(xg), axis=-1) / 127.0).astype(mx.float16)
    sf = s.astype(mx.float32)
    safe = mx.where(sf > 0, sf, mx.ones_like(sf))
    q = mx.clip(mx.round(xg / safe[..., None]), -127, 127).astype(mx.int8)
    return q.reshape(*lead, d), s


__all__ = ["CHUNK", "quantize_kv", "ragged_decode_attention"]
