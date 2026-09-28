"""Decode/verify attention over per-row key lengths (ragged batch KV).

Keys and values live in one buffer per layer, ``[B, HKV, CAP, D]``, with row
``b``'s ``lengths[b]`` valid keys at positions ``0 .. lengths[b] - 1`` (right-
aligned storage, no left padding). Each query row reads only its own keys, so a
short request in a batch with a 16K-token neighbour no longer pays for the
neighbour's length — upstream's left-padded ``BatchKVCache`` pads every row to
the longest one.

Split-K ("flash decoding"): keys are cut into fixed ``CK``-key chunks; one
threadgroup per (row, query token, KV head, chunk) runs an online softmax for the
KV head's ``G`` query heads (one simdgroup each, 32 lanes x D/32 dims), then a
merge kernel combines a row's chunk partials in chunk order. Chunk boundaries
depend only on absolute key position, so a row's result does not depend on the
other rows in the batch (the fixed-chunk + ordered-merge idea follows
TensorFold's lane_attention; this kernel is written independently, plain SIMD).
``T`` > 1 query tokens (speculative verify) attend
causally within the window: token ``t`` sees keys ``< lengths[b] - (T - 1 - t)``.

INT8 variant: keys/values stored as int8 codes with one symmetric fp16 scale per
(row, KV head, token, 32-dim group) (``scale = max|x| / 127``). Splash uses one
scale per (token, head); 32-dim groups keep a few outlier channels from
coarsening the whole head (synthetic 20x outliers on 7/256 dims: cosine to bf16
attention 0.9976 -> 0.9990) for 6% more bytes. Each lane owns 8 consecutive dims,
so it applies one group scale to its partial dot product. The kernel reads codes
and scales directly and dequantizes in registers — K/V traffic ~0.53x of bf16.
"""

from __future__ import annotations

import mlx.core as mx

CHUNK = 512
GROUP = 32  # int8 KV: dims per scale group

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
  const int64_t base = ((int64_t)bt * H + h) * NC + c;
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
  const int64_t base = ((int64_t)bt * H + h) * NC + c;
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

_MERGE = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint r = threadgroup_position_in_grid.y;               // (row * T + token) * H + head
  constexpr int DPL = D / 32;
  float m = -INFINITY, l = 0.0f, o[DPL];
  for (int i = 0; i < DPL; i++) o[i] = 0.0f;
  for (int c = 0; c < NC; c++) {                               // chunk order: batch-independent bits
    const int64_t row = (int64_t)r * NC + c;
    const float mc = PM[row];
    if (mc == -INFINITY) continue;
    const float nm = max(m, mc);
    const float f1 = fast::exp(m - nm), f2 = fast::exp(mc - nm);
    l = l * f1 + PL[row] * f2;
    for (int i = 0; i < DPL; i++) o[i] = o[i] * f1 + PO[row * D + lane * DPL + i] * f2;
    m = nm;
  }
  // output layout [B, H, T, D] like mx.fast.scaled_dot_product_attention
  const uint bt = r / H, h = r % H;
  const uint b = bt / T, t = bt % T;
  device bfloat16_t* dst = out + (((int64_t)b * H + h) * T + t) * D + lane * DPL;
  const float inv = l > 0.0f ? 1.0f / l : 0.0f;
  for (int i = 0; i < DPL; i++) dst[i] = static_cast<bfloat16_t>(o[i] * inv);
"""

_KERNELS: dict = {}


def _kernel(name: str):
    if name not in _KERNELS:
        if name == "partial_q8":
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
                input_names=["PO", "PM", "PL"],
                output_names=["out"],
                source=_MERGE,
            )
    return _KERNELS[name]


def ragged_decode_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    lengths: mx.array,
    scale: float,
    max_length: int | None = None,
    k_scales: mx.array | None = None,
    v_scales: mx.array | None = None,
) -> mx.array:
    """Attention of ``queries`` [B, H, T, D] over each row's first ``lengths[b]``
    keys of ``keys``/``values`` [B, HKV, CAP, D]; returns [B, H, T, D] bf16.

    ``max_length`` (host int) sizes the chunk grid; defaults to the buffer
    capacity. The last ``T`` keys of each row are the query tokens' own keys.
    int8 ``keys``/``values`` need ``k_scales``/``v_scales`` [B, HKV, CAP, D/32]
    fp16 (see ``quantize_kv``).
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
    po, pm, pl = _kernel(name)(
        inputs=inputs,
        template=tmpl + [("GS", GROUP)] if q8 else tmpl,
        grid=(32, G * HKV * nc, B * T),
        threadgroup=(32, G, 1),
        output_shapes=[(B * T * H * nc * D,), (B * T * H * nc,), (B * T * H * nc,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return _kernel("merge")(
        inputs=[po, pm, pl],
        template=[("D", D), ("T", T), ("H", H), ("NC", nc)],
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
