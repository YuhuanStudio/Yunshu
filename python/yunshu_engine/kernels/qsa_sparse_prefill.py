# Upstream (derived): ashhart/TensorFold @ 7c22033 (Apache-2.0), zig/kernels/metal/prefill/fn_attn.metal (tf_sattn_nax) and zig/kernels/metal/nax.h; fragment helpers and the attention loop are kept, the key list is formed from the QSA block indices in the kernel and the fused output gate is dropped. See THIRD_PARTY_NOTICES.md.
"""Indexed-sparse QSA prefill attention on the M5 tensor units for Qwen3.8-Flash-Next (qwen4_exp).

Upstream's prefill (``qsa_kernel.qsa_sparse_attention``) runs one 1024-thread scalar kernel per
(query, head): every one of the 12 query heads that share a key head re-reads the same 2051 selected keys
and multiplies them on the SIMD ALUs. Here one threadgroup owns (query row, key head): the 12 heads are
one 16-row operand of the tensor-unit matmul, the selected keys are gathered once in blocks of 32 and the
softmax is an fp32 online softmax. Each row's result depends on that row alone, so the output does not
depend on how a prompt is split into chunks.

Selected keys of a row: the sorted ascending selected blocks (4 keys each) followed by the 0-3 tokens of the
incomplete tail block, the same set and order the upstream kernel visits. Short prefixes need no separate path:
every complete block is selected, so the set is every key up to the row's end.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import mlx.core as mx

logger = logging.getLogger(__name__)

D, H, KV, BLOCK, TOPK = 256, 24, 2, 4, 512

_HEADER = r"""
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

namespace tfp {
using namespace metal;
#define TF_UNROLL _Pragma("clang loop unroll(full)")
template <typename T>
using frag = vec<T, 8>;

// Lane l holds rows home.y and home.y + 8, columns home.x .. home.x + 3 of every fragment (the M5 operand layout).
inline short2 frag_home(ushort l) {
  return short2(short((l & 8) + ((l & 1) << 2)), short(((l & 16) >> 2) | ((l >> 1) & 3)));
}

template <typename T, typename S>
inline void frag_get_in(thread frag<T>& f, const device S* p, int ld, int r, int c, short2 home, int nr, int nc) {
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    const int rr = r + home.y + (e >> 2) * 8, cc = c + home.x + (e & 3);
    f[e] = (rr < nr && cc < nc) ? T(p[rr * ld + cc]) : T(0);
  }
}

// (lo | hi) += a * (b0 | b1): one 16x32x16 multiply-accumulate on the tensor unit, relaxed precision as MLX asks.
template <bool TA, bool TB, typename C, typename A, typename B>
inline void mma_16x32(thread frag<C>& lo, thread frag<C>& hi, thread const frag<A>& a, thread const frag<B>& b0,
                      thread const frag<B>& b1) {
  using namespace mpp::tensor_ops;
  constexpr auto shape = matmul2d_descriptor(16, 32, 16, TA, TB, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<shape, execution_simdgroup> op;
  auto left = op.template get_left_input_cooperative_tensor<A, B, C>();
  auto right = op.template get_right_input_cooperative_tensor<A, B, C>();
  auto acc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,
                                                            metal::remove_addrspace_t<decltype(right)>, C>();
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    left[e] = a[e];
    right[e] = b0[e];
    right[8 + e] = b1[e];
    acc[e] = lo[e];
    acc[8 + e] = hi[e];
  }
  op.run(left, right, acc);
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    lo[e] = acc[e];
    hi[e] = acc[8 + e];
  }
}
#define TF_COL(e) (e & 3)
}  // namespace tfp

namespace tfa {
using namespace tfp;
// A fragment row's max (MAX) or sum over the 32 columns of (f0 | f1), for the lane's rows home.y and home.y + 8.
template <bool MAX>
inline void row_fold(thread const frag<float>& f0, thread const frag<float>& f1, thread float (&r)[2]) {
  TF_UNROLL
  for (short h = 0; h < 2; h++) {
    const short b = 4 * h;
    float t0 = MAX ? max(max(f0[b], f0[b + 1]), max(f0[b + 2], f0[b + 3])) : (f0[b] + f0[b + 1]) + (f0[b + 2] + f0[b + 3]);
    float t1 = MAX ? max(max(f1[b], f1[b + 1]), max(f1[b + 2], f1[b + 3])) : (f1[b] + f1[b + 1]) + (f1[b + 2] + f1[b + 3]);
    float t = MAX ? max(t0, t1) : t0 + t1;
    const float u = simd_shuffle_xor(t, ushort(1));
    t = MAX ? max(t, u) : t + u;
    const float w = simd_shuffle_xor(t, ushort(8));
    t = MAX ? max(t, w) : t + w;
    r[h] = MAX ? max(r[h], t) : r[h] + t;
  }
}
// Key j of a row: the selected blocks (ascending, 4 keys each) then the incomplete tail block.
inline int qsa_key(const device int* bi, int j, int ncomplete, int tail_start) {
  return j < ncomplete ? bi[j >> 2] * 4 + (j & 3) : tail_start + (j - ncomplete);
}
}  // namespace tfa
"""

_SOURCE = r"""
  constexpr int D = 256, H = 24, GQA = 12, BK = 32, DS = 64, NB = 512;
  constexpr float masked = -3.402823466e+38f;
  const int T = int(Q_shape[2]);
  const long KROW = long(K_strides[2]), VROW = long(V_strides[2]);   // the cache's row stride; columns are contiguous
  const int r = int(threadgroup_position_in_grid.x), g = int(threadgroup_position_in_grid.y);
  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;
  const int qe = QE[r], nsel = NSEL[r];
  const int tail_start = (qe / 4) * 4;
  const int ncomplete = nsel * 4;
  const int n = ncomplete + (qe - tail_start);
  const device int* bi = BI + long(r) * NB + (NB - nsel);   // the selected blocks are the last nsel of the sorted row
  const device bfloat* Kh = K + long(g) * long(K_strides[1]) + DS * int(sg);
  const device bfloat* Vh = V + long(g) * long(V_strides[1]) + DS * int(sg);
  const tfp::short2 home = tfp::frag_home(lane);
  threadgroup float part[4][16 * BK];
  tfp::frag<bfloat> q[DS / 16];
  const device bfloat* Qr = Q + (long(GQA) * g * T + r) * D + DS * int(sg);
  TF_UNROLL
  for (short d = 0; d < DS / 16; d++) tfp::frag_get_in(q[d], Qr, T * D, 0, 16 * d, home, GQA, DS);
  tfp::frag<float> acc[DS / 16];
  TF_UNROLL
  for (short i = 0; i < DS / 16; i++) acc[i] = tfp::frag<float>(0);
  float top[2] = {masked, masked}, total[2] = {0.0f, 0.0f};
  const float scale2 = SC[0] * 1.44269504089f;
  const int blocks = (n + BK - 1) / BK;
  for (int kb = 0; kb < blocks; kb++) {
    const int j0 = kb * BK;
    int key[4];
    TF_UNROLL
    for (short i = 0; i < 4; i++) {
      const int j = j0 + 16 * (i >> 1) + home.y + 8 * (i & 1);
      key[i] = j < n ? tfa::qsa_key(bi, j, ncomplete, tail_start) : 0;
    }
    tfp::frag<float> s[2] = {tfp::frag<float>(0), tfp::frag<float>(0)};
    TF_UNROLL
    for (short d = 0; d < DS / 16; d++) {
      tfp::frag<bfloat> k0, k1;
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const int c = 16 * d + home.x + (e & 3);
        k0[e] = Kh[long(key[(e >> 2)]) * KROW + c];
        k1[e] = Kh[long(key[2 + (e >> 2)]) * KROW + c];
      }
      tfp::mma_16x32<false, true>(s[0], s[1], q[d], k0, k1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) part[sg][(home.y + (e >> 2) * 8) * BK + 16 * f + home.x + TF_COL(e)] = s[f][e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const int at = (home.y + (e >> 2) * 8) * BK + 16 * f + home.x + TF_COL(e);
        const float v = ((part[0][at] + part[1][at]) + part[2][at]) + part[3][at];
        const int j = j0 + 16 * f + home.x + TF_COL(e);
        s[f][e] = j < n ? v * scale2 : masked;
      }
    }
    float top_new[2] = {top[0], top[1]};
    tfa::row_fold<true>(s[0], s[1], top_new);
    TF_UNROLL
    for (short f = 0; f < 2; f++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) s[f][e] = fast::exp2(s[f][e] - top_new[e >> 2]);
    }
    float keep[2];
    TF_UNROLL
    for (short hh = 0; hh < 2; hh++) {
      keep[hh] = fast::exp2(top[hh] - top_new[hh]);
      top[hh] = top_new[hh];
      total[hh] = total[hh] * keep[hh];
    }
    tfa::row_fold<false>(s[0], s[1], total);
    TF_UNROLL
    for (short i = 0; i < DS / 16; i++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) acc[i][e] = acc[i][e] * keep[e >> 2];
    }
    TF_UNROLL
    for (short d = 0; d < DS / 16; d += 2) {
      TF_UNROLL
      for (short k = 0; k < 2; k++) {
        tfp::frag<bfloat> v0, v1;
        TF_UNROLL
        for (short e = 0; e < 8; e++) {
          const long row = long(key[2 * k + (e >> 2)]) * VROW;
          const int c = 16 * d + home.x + TF_COL(e);
          v0[e] = Vh[row + c];
          v1[e] = Vh[row + c + 16];
        }
        tfp::mma_16x32<false, false>(acc[d], acc[d + 1], s[k], v0, v1);
      }
    }
  }
  const float inv[2] = {1.0f / total[0], 1.0f / total[1]};
  TF_UNROLL
  for (short i = 0; i < DS / 16; i++) {
    TF_UNROLL
    for (short e = 0; e < 8; e++) {
      const int head = home.y + (e >> 2) * 8, col = DS * int(sg) + 16 * i + home.x + TF_COL(e);
      if (head < GQA) {
        const int h = GQA * g + head;
        O[(long(h) * T + r) * D + col] = bfloat(acc[i][e] * inv[e >> 2]);
      }
    }
  }
"""

COUNTS = {"engaged": 0, "fallback": 0}
_KERNEL: list[Any] = []
_READY: list[bool] = []


def _kernel() -> Any:
    if not _KERNEL:
        _KERNEL.append(
            mx.fast.metal_kernel(
                name="qsa_sparse_prefill_nax",
                input_names=["Q", "K", "V", "BI", "NSEL", "QE", "SC"],
                output_names=["O"],
                source=_SOURCE,
                header=_HEADER,
                ensure_row_contiguous=False,
            )
        )
    return _KERNEL[0]


def ready() -> bool:
    """M5-class GPU (applegpu_g17 / g18): the fragment layout in the header is the M5 one."""
    if not _READY:
        import re

        match = re.fullmatch(
            r"applegpu_g(\d+)[a-z]+", str(mx.device_info().get("architecture", ""))
        )
        _READY.append(bool(match and int(match[1]) in (17, 18)))
    return _READY[0]


def eligible(
    queries: Any, keys: Any, values: Any, block_indices: Any, block_size: int
) -> bool:
    return bool(
        isinstance(queries, mx.array)
        and queries.ndim == 4
        and queries.shape[0] == 1
        and queries.shape[1] == H
        and queries.shape[2] > 2
        and queries.shape[3] == D
        and queries.dtype == mx.bfloat16
        and isinstance(keys, mx.array)
        and isinstance(values, mx.array)
        and keys.dtype == mx.bfloat16
        and values.dtype == mx.bfloat16
        and keys.shape[:2] == (1, KV)
        and keys.shape[3] == D
        and keys.shape == values.shape
        and block_indices.shape[-1] == TOPK
        and block_size == BLOCK
        and ready()
    )


def attend(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    block_indices: mx.array,
    query_ends: mx.array,
    scale: float,
) -> mx.array:
    """Attention output ``[1, 24, T, 256]`` over each row's selected blocks and tail block."""
    rows = int(queries.shape[2])
    bi = mx.contiguous(mx.sort(block_indices.astype(mx.int32), axis=-1))
    nsel = mx.sum((bi >= 0).astype(mx.int32), axis=-1)
    qe = mx.contiguous(query_ends.astype(mx.int32))
    out = _kernel()(
        inputs=[
            mx.contiguous(queries),
            keys,
            values,
            bi.reshape(rows, TOPK),
            nsel.reshape(rows),
            qe.reshape(rows),
            mx.array([scale], dtype=mx.float32),
        ],
        grid=(rows * 128, KV, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )
    return cast("mx.array", out[0])


def install() -> bool:
    """Route eligible prefill calls of ``dispatch_qsa_attention`` to the tensor-unit kernel once."""
    try:
        from mlx_vlm.models.qwen4_exp import language as lang
    except ImportError:
        return False
    if getattr(lang.dispatch_qsa_attention, "_yunshu_sparse_prefill", False):
        return True
    original = lang.dispatch_qsa_attention

    def dispatch(queries, keys, values, block_indices, query_ends, **kw):
        if (
            kw.get("causal")
            and not kw.get("allow_sparse_decode")
            and eligible(queries, keys, values, block_indices, kw["block_size"])
        ):
            if not COUNTS["engaged"]:
                logger.info(
                    "QSA sparse prefill attention engaged (tensor-unit kernel, %d query rows)",
                    queries.shape[2],
                )
            COUNTS["engaged"] += 1
            return attend(
                queries, keys, values, block_indices, query_ends, float(kw["scale"])
            )
        if (
            kw.get("causal")
            and not kw.get("allow_sparse_decode")
            and queries.shape[2] > 2
        ):
            COUNTS["fallback"] += 1  # a prefill-shaped call that kept upstream's kernel
        return original(queries, keys, values, block_indices, query_ends, **kw)

    dispatch._yunshu_sparse_prefill = True  # type: ignore[attr-defined]
    lang.dispatch_qsa_attention = dispatch
    return True
