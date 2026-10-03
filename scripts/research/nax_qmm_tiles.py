"""Research-only MLX 0.32.3 NAX QMM tile sweep (upstream Apple MIT kernels).

Use the installed MLX headers rather than a newer reference checkout. No
serving dispatch is changed. Changing BM/BN is accepted only after bit parity
and an end-to-end win; BK stays 64 to preserve the accumulation order.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import re
from functools import cache
from pathlib import Path


@cache
def header() -> str:
    root = Path(importlib.metadata.distribution("mlx").locate_file("mlx/include"))
    seen = set()

    def expand(name):
        if name in seen:
            return ""
        seen.add(name)
        text = (root / name).read_text()
        text = re.sub(
            r'^\s*#include "([^"]+)"', lambda m: expand(m[1]), text, flags=re.M
        )
        return text.replace("#pragma once", "")

    # mx.fast.metal_kernel prepends metal::utils(); avoid defining it twice.
    expand("mlx/backend/metal/kernels/utils.h")
    result = "\n".join(
        expand("mlx/backend/metal/kernels/" + name)
        for name in (
            "steel/gemm/gemm.h",
            "steel/gemm/nax.h",
            "steel/gemm/loader.h",
            "quantized_nax.h",
        )
    )
    # Custom-kernel inputs use device buffers. Scalar arguments here need no
    # address-space reference; passing values leaves the arithmetic unchanged.
    result = re.sub(
        r"const constant int& ([KNM])\b(?!\s*\[\[buffer)", r"const int \1", result
    )
    return result


@cache
def lane_header() -> str:
    """Same native QMM arithmetic; replace only its input layout loader."""
    h = header()
    point = h.index("METAL_FUNC void qmm_t_nax_tgp_impl")
    start = h.rfind("template <", 0, point)
    opening = h.index("{", point)
    depth = 1
    end = opening + 1
    while depth:
        depth += (h[end] == "{") - (h[end] == "}")
        end += 1
    body = h[start:end].replace("qmm_t_nax_tgp_impl", "nax_prefill_lane_tgp_impl")
    begin = body.index("  using loader_w_t = QuantizedBlockLoader<")
    finish = body.index(";", begin) + 1
    body = (
        body[:begin]
        + "  using loader_w_t = NaxLaneLoader<T, BN, BK_padded, WM * WN * SIMD_SIZE, bits>;"
        + body[finish:]
    )
    for shift in (
        "  wl += y_col * K_w;",
        "  scales += y_col * K_g;",
        "  biases += y_col * K_g;",
    ):
        if body.count(shift) != 1:
            raise RuntimeError("unsupported installed MLX QMM helper")
        body = body.replace(shift, "")
    old = "loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);"
    if body.count(old) != 1:
        raise RuntimeError("unsupported installed MLX loader signature")
    body = body.replace(
        old, "loader_w_t loader_w(wl, scales, K, N, y_col, Ws, simd_gid, simd_lid);"
    )
    loader = r"""
template<typename T, short BN, short LD, short TG, short bits>
struct NaxLaneLoader {
  MLX_MTL_CONST short PF = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short BP = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short CP = 64 / PF;
  MLX_MTL_CONST short READS = BN * CP / TG;
  const device uint8_t* W;
  const device T* SB;
  threadgroup T* dst;
  const int KG, N, n, bi, bj;
  int g;
  NaxLaneLoader(const device uint8_t* w, const device T* sb,
      int k, int ncols, int n0, threadgroup T* ws, int sg, int lane) thread
    : W(w), SB(sb),
      dst(ws + (READS * (sg*32+lane) / CP)*LD + (READS*(sg*32+lane)%CP)*PF),
      KG(k/64), N(ncols), n(n0+READS*(sg*32+lane)/CP),
      bi(READS*(sg*32+lane)/CP), bj(READS*(sg*32+lane)%CP), g(0) {}
  void load_unsafe() const thread {
    const device uint8_t* src = W + (((n/32)*KG+g)*32+n%32)*(64*bits/8) + bj*BP;
    const T scale = SB[2*(g*N+n)], bias = SB[2*(g*N+n)+1];
    for(int i=0;i<READS;i++) dequantize<T,PF,bits>(src+i*BP,scale,bias,dst+i*PF);
  }
  void load_safe(short2 dims) const thread {
    if(bi < dims.y) load_unsafe();
    else for(int i=0;i<READS*PF;i++) dst[i] = T(0);
  }
  void next() thread { g++; }
};
"""
    return h + loader + body


@cache
def compiled(bits, bm, bn, wm, wn, lane):
    import mlx.core as mx

    h = lane_header() if lane else header()
    fn = "nax_prefill_lane_tgp_impl" if lane else "qmm_t_nax_tgp_impl"
    scale_ptr, bias_ptr = ("SB", "SB+1") if lane else ("S", "B")
    source = f"""    threadgroup bfloat Ws[{bn} * (64 + 8)];
    {fn}<bfloat, 64, {bits}, true, {bm}, 64, {bn}, {wm}, {wn}>(
        W, {scale_ptr}, {bias_ptr}, X, Y, Ws, dims[0], dims[1], dims[2],
        threadgroup_position_in_grid, thread_index_in_threadgroup,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
    """
    name = "nax_prefill_tile_" + hashlib.sha256((h + source).encode()).hexdigest()[:16]
    return mx.fast.metal_kernel(
        name=name,
        input_names=["X", "W", "S", "B", "dims"]
        if not lane
        else ["X", "W", "SB", "dims"],
        output_names=["Y"],
        header=h,
        source=source,
    )


@cache
def dimensions(k, n, m):
    import mlx.core as mx

    dims = mx.array([k, n, m], dtype=mx.int32)
    mx.eval(dims)
    return dims


def make(mx, x, q, scales, biases, *, bits, bm, bn, wm=2, wn=2, sbt=None):
    m, k = x.shape
    n = scales.shape[0] if sbt is None else sbt.shape[1]
    if k % 64 or n % bn or m % bm:
        raise ValueError("tile sweep requires fully aligned inputs")
    kernel = compiled(bits, bm, bn, wm, wn, sbt is not None)
    dims = dimensions(k, n, m)

    def run():
        return kernel(
            inputs=[x, q, scales, biases, dims] if sbt is None else [x, q, sbt, dims],
            grid=((n // bn) * 32, (m // bm) * wn, wm),
            threadgroup=(32, wn, wm),
            output_shapes=[(m, n)],
            output_dtypes=[mx.bfloat16],
        )[0]

    return run
