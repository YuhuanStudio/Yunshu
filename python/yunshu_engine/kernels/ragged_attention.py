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
"""

from __future__ import annotations

import mlx.core as mx

CHUNK = 512

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
        if name == "partial":
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
) -> mx.array:
    """Attention of ``queries`` [B, H, T, D] over each row's first ``lengths[b]``
    keys of ``keys``/``values`` [B, HKV, CAP, D]; returns [B, H, T, D] bf16.

    ``max_length`` (host int) sizes the chunk grid; defaults to the buffer
    capacity. The last ``T`` keys of each row are the query tokens' own keys.
    """
    B, H, T, D = (int(v) for v in queries.shape)
    _, HKV, CAP, Dk = (int(v) for v in keys.shape)
    if Dk != D or D % 32 or H % HKV or T > 8:
        raise ValueError(
            f"ragged_decode_attention: unsupported q={queries.shape} k={keys.shape}"
        )
    if queries.dtype != mx.bfloat16 or keys.dtype != mx.bfloat16:
        raise ValueError("ragged_decode_attention: bf16 only")
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
    po, pm, pl = _kernel("partial")(
        inputs=[
            queries,
            keys,
            values,
            lengths.astype(mx.int32),
            mx.array([float(scale)], dtype=mx.float32),
        ],
        template=tmpl,
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


__all__ = ["CHUNK", "ragged_decode_attention"]
