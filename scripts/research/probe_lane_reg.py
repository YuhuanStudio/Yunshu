# Upstream (inspired): ashhart/TensorFold (MIT) zig/kernels/metal/core/lane_projection.metal tf_lane_reg,
# zig/kernels/metal/core/lane_order.metal @ 76aafcf2
"""TensorFold 1.1.0's M5 register-order lane kernel (tf_lane_reg) vs our lane_matmul.

tf_lane_reg keeps tf_lane's per-group arithmetic (fma(s, P, fma(b, xs, C)), split-K slices
added in slice order) but takes its 4-bit codes pre-permuted into the tensor op's right-input
register order, so a lane builds its operand with four integer ops a word instead of the op
loading uint4b_format codes. Rows go in 16-row blocks on the grid's y. Question: same bits as
our lane_matmul at every row count, and how much faster with weights streamed cold.

    python scripts/research/probe_lane_reg.py   (gpuq only)
"""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from yunshu_engine.kernels.tensorfold import lane_qmm  # noqa: E402

SHAPES = {  # name: (K in, N out, copies)
    "mlp_up": (5120, 17408, 24),
    "mlp_down": (17408, 5120, 24),
    "gdn_qkv": (5120, 10240, 40),
    "attn_o": (6144, 5120, 64),
}
ROWS = (1, 8, 16, 17, 24, 32, 48)

_REG = r"""
  const int r0 = int(threadgroup_position_in_grid.y) * 16, rows = mdims[0] - r0;
  if (rows <= 0) return;
  constexpr int KG = K / 64;
  const int lane = int(thread_index_in_simdgroup), ks = int(simdgroup_index_in_threadgroup);
  const int M = min(rows, 16), stride = mdims[1], tile = int(threadgroup_position_in_grid.x), n0 = tile * 32;
  const int nb = 4 * (lane & 1) + 8 * ((lane >> 3) & 1), mb = ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1);
  const int first = (ks * KG) / SK, last = ((ks + 1) * KG) / SK;
  constexpr auto desc = matmul2d_descriptor(16, 32, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> a((device bfloat*)X + (int64_t)r0 * K, dextents<int32_t, 2>(K, M));
  auto a0 = a.slice(0, 0);
  auto cb = op.template get_right_input_cooperative_tensor<bfloat, half, float>();
  auto P = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a0)>, metal::remove_addrspace_t<decltype(cb)>, float>();
  thread half2* b2 = (thread half2*)&cb[0];
  const device uint4* wl = (const device uint4*)Wr + (size_t)tile * KG * 64 + lane;
  const device uint4* sbv = (const device uint4*)SBt;
  const device float* xsr = XS + r0;
  const half2 bias = half2(1024.0h);
  float C[16];
  for (int i = 0; i < 16; i++) C[i] = 0.0f;
  uint4 w0n = 0, w1n = 0, q0n = 0, q1n = 0;
  float x0n = 0.0f, x1n = 0.0f;
  if (first < last) {
    w0n = wl[(size_t)first * 64]; w1n = wl[(size_t)first * 64 + 32];
    q0n = sbv[((size_t)first * N + n0 + nb) / 4]; q1n = sbv[((size_t)first * N + n0 + nb + 16) / 4];
    x0n = xsr[first * stride + mb]; x1n = xsr[first * stride + mb + 8];
  }
  for (int g = first; g < last; g++) {
    const uint4 w0 = w0n, w1 = w1n, q0 = q0n, q1 = q1n;
    const float x0 = x0n, x1 = x1n;
    for (int w = 0; w < 8; w++) {
      const uint word = w < 4 ? w0[w] : w1[w - 4], lo = word & 0x0F0F0F0Fu, hi = (word >> 4) & 0x0F0F0F0Fu;
      const int e = 16 * (w & 1) + 2 * (w >> 1);
      b2[e] = as_type<half2>((lo & 0x00FF00FFu) | 0x64006400u) - bias;
      b2[e + 1] = as_type<half2>((hi & 0x00FF00FFu) | 0x64006400u) - bias;
      b2[e + 8] = as_type<half2>(((lo >> 8) & 0x00FF00FFu) | 0x64006400u) - bias;
      b2[e + 9] = as_type<half2>(((hi >> 8) & 0x00FF00FFu) | 0x64006400u) - bias;
    }
    if (g + 1 < last) {
      w0n = wl[(size_t)(g + 1) * 64]; w1n = wl[(size_t)(g + 1) * 64 + 32];
      q0n = sbv[((size_t)(g + 1) * N + n0 + nb) / 4]; q1n = sbv[((size_t)(g + 1) * N + n0 + nb + 16) / 4];
      x0n = xsr[(g + 1) * stride + mb]; x1n = xsr[(g + 1) * stride + mb + 8];
    }
    auto x = a.slice(g * 64, 0);
    op.run(x, cb, P);
    for (int j = 0; j < 4; j++) {
      const float s0 = as_type<float>(q0[j] << 16), b0 = as_type<float>(q0[j] & 0xFFFF0000u);
      const float s1 = as_type<float>(q1[j] << 16), b1 = as_type<float>(q1[j] & 0xFFFF0000u);
      C[j] = fma(s0, P[j], fma(b0, x0, C[j]));
      C[4 + j] = fma(s0, P[4 + j], fma(b0, x1, C[4 + j]));
      C[8 + j] = fma(s1, P[8 + j], fma(b1, x0, C[8 + j]));
      C[12 + j] = fma(s1, P[12 + j], fma(b1, x1, C[12 + j]));
    }
  }
  threadgroup float part[(SK > 1 ? SK - 1 : 1) * 16 * 32];
  if (SK > 1) {
    if (ks > 0) for (int i = 0; i < 16; i++) part[((ks - 1) * 16 + i) * 32 + lane] = C[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks == 0) for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < 16; i++) C[i] += part[((s2 - 1) * 16 + i) * 32 + lane];
  }
  if (ks == 0)
    for (int i = 0; i < 16; i++) {
      const int m = mb + 8 * ((i >> 2) & 1), n = n0 + nb + 16 * (i >> 3) + (i & 3);
      if (m < M && n < N) Y[(size_t)(r0 + m) * N + n] = static_cast<bfloat>(C[i]);
    }
"""

_kernel = lane_qmm._Baked(
    "probe_lane_reg", _REG, ["X", "XS", "Wr", "SBt", "mdims"], ["Y"]
)


def reg_order(wt: mx.array) -> mx.array:
    """lane_order.metal's permutation of tile-order 4-bit g64 words, as array ops."""
    blocks = wt.reshape(-1, 256)
    t = mx.arange(256, dtype=mx.uint32)
    lane = (t >> 2) & 31
    w = 4 * (t >> 7) + (t & 3)
    at = (
        (((lane >> 1) & 3) + 4 * ((lane >> 4) & 1) + 8 * (w >> 1)) * 8
        + ((lane >> 3) & 1)
        + 4 * (w & 1)
    )
    shift = 16 * (lane & 1)
    a = (mx.take(blocks, at.astype(mx.int32), axis=1) >> shift) & 0xFFFF
    c = (mx.take(blocks, (at + 2).astype(mx.int32), axis=1) >> shift) & 0xFFFF
    out = (
        (a & 0xF)
        | ((a >> 4) & 0xF0)
        | ((c & 0xF) << 8)
        | ((c << 4) & 0xF000)
        | (((a >> 4) & 0xF) << 16)
        | (((a >> 8) & 0xF0) << 16)
        | (((c >> 4) & 0xF) << 24)
        | (((c >> 8) & 0xF0) << 24)
    )
    return out.astype(mx.uint32).reshape(wt.shape)


def lane_reg(x, wr, sbt, n, k):
    m = int(x.shape[0])
    mp = 16 * ((m + 15) // 16)
    mdims = lane_qmm._mdims(m, mp)
    xs = lane_qmm._kernel("xsum")(
        inputs=[x, mdims],
        template=[("K", k), ("GS", 64)],
        grid=(k // 64, mp, 1),
        threadgroup=(min(k // 64, 256), 1, 1),
        output_shapes=[(k // 64, mp)],
        output_dtypes=[mx.float32],
    )[0]
    sk = lane_qmm.split_k(n, k)
    return lane_qmm_call(x, xs, wr, sbt, mdims, n, k, sk, mp)


def lane_qmm_call(x, xs, wr, sbt, mdims, n, k, sk, mp):
    return _kernel(
        inputs=[x, xs, wr, sbt, mdims],
        template=[("N", n), ("K", k), ("SK", sk)],
        grid=((n // 32) * 32 * sk, mp // 16, 1),
        threadgroup=(32 * sk, 1, 1),
        output_shapes=[(int(x.shape[0]), n)],
        output_dtypes=[mx.bfloat16],
    )[0]


def timed(fn, reps=5):
    mx.eval(fn())
    mx.synchronize()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    out.sort()
    return round(out[len(out) // 2], 3)


def main():
    mx.random.seed(0)
    ok = True
    for name, (k, n, copies) in SHAPES.items():
        w = mx.random.normal((n, k)).astype(mx.bfloat16)
        wq, sc, bi = mx.quantize(w, group_size=64, bits=4)
        wt = lane_qmm.tile_weight(wq, 32, 64, bits=4)
        sbt = lane_qmm.pack_scales(sc, bi)
        wr = reg_order(wt)
        mx.eval(wt, sbt, wr)
        copies_t = [(wt + 0, sbt + 0) for _ in range(copies)]
        copies_r = [(wr + 0, sbt + 0) for _ in range(copies)]
        mx.eval(copies_t, copies_r)
        gb = copies * (wt.nbytes + sbt.nbytes) / 1e9
        row = {"shape": name, "gb": round(gb, 2), "sk": lane_qmm.split_k(n, k)}
        for m in ROWS:
            x = (mx.random.normal((m, k)) * 0.5).astype(mx.bfloat16)
            mx.eval(x)
            ref = lane_qmm.lane_matmul(x, wt, sbt, tiled=True, group=64)
            new = lane_reg(x, wr, sbt, n, k)
            same = bool(mx.array_equal(ref, new).item())
            ok &= same
            lane_ms = timed(
                lambda x=x: [
                    lane_qmm.lane_matmul(x, a, b, tiled=True, group=64)
                    for a, b in copies_t
                ]
            )
            reg_ms = timed(lambda x=x: [lane_reg(x, a, b, n, k) for a, b in copies_r])
            row[f"M{m}"] = {
                "lane_ms": lane_ms,
                "reg_ms": reg_ms,
                "lane_GBps": round(gb / lane_ms * 1e3),
                "reg_GBps": round(gb / reg_ms * 1e3),
                "same": same,
            }
        print(json.dumps(row), flush=True)
        copies_t = copies_r = []
        mx.clear_cache()
    print(json.dumps({"complete": True, "bit_equal": ok}), flush=True)
    sys.exit(0 if ok else 1)


main()
