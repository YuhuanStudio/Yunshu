# Upstream (derived): ml-explore/mlx (MIT), quantized_nax.h / steel/gemm/nax.h @ v0.32.3
"""Native NAX QMM arithmetic over lane-tiled weights and paired scales/biases.

Only the input loader and M tile change. K tiles and float accumulation order
remain MLX 0.32.3's. Unsupported devices/versions/shapes retain stock QMM.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import re
from functools import cache
from pathlib import Path
from typing import Any, cast

import mlx.core as mx


@cache
def header() -> str:
    root = Path(str(importlib.metadata.distribution("mlx").locate_file("mlx/include")))
    seen = set()

    def expand(name: str) -> str:
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
def compiled(bits: int, bm: int, bn: int, wm: int, wn: int, lane: bool) -> Any:
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
def dimensions(k: int, n: int, m: int) -> mx.array:
    import mlx.core as mx

    dims = mx.array([k, n, m], dtype=mx.int32)
    mx.eval(dims)
    return dims


def make(
    mx: Any,
    x: mx.array,
    q: mx.array,
    scales: Any,
    biases: Any,
    *,
    bits: int,
    bm: int,
    bn: int,
    wm: int = 2,
    wn: int = 2,
    sbt: Any = None,
) -> Any:
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


_enabled = False
_dispatches = 0
logger = logging.getLogger(__name__)


def enable() -> bool:
    """Enable only the measured MLX/device pair; failed ABI checks keep stock."""
    global _enabled, _dispatches
    _enabled = False
    _dispatches = 0
    if importlib.metadata.version("mlx") != "0.32.3":
        return False
    if mx.device_info().get("device_name") != "Apple M5 Max":
        return False
    try:
        lane_header()
    except (OSError, ValueError, RuntimeError):
        return False
    _enabled = True
    return True


def enabled() -> bool:
    return _enabled


def disable() -> None:
    global _enabled, _dispatches
    _enabled = False
    _dispatches = 0


def record_dispatch(m: int, k: int, n: int, bits: int) -> None:
    global _dispatches
    if not _dispatches:
        logger.info("NAX prefill engaged: rows=%d K=%d N=%d bits=%d", m, k, n, bits)
    _dispatches += 1


@cache
def arithmetic_id() -> str:
    source = (
        lane_header()
        + "bm128-bn64-bk64;bm64-if-M>4096-K>8192;narrow32-gt512-aligned128-v1"
    )
    return "nax-lane-" + hashlib.sha256(source.encode()).hexdigest()[:16]


def eligible(m: int, k: int, n: int, bits: int, group: int, tiled: bool) -> bool:
    return bool(
        _enabled
        and 512 < m <= 8192
        and m % 128 == 0
        and k % 64 == 0
        and 256 <= n < 100_000
        and n % 64 == 0
        and bits in (4, 5, 8)
        and group == 64
        and tiled
    )


def narrow_eligible(m: int, k: int, n: int, bits: int, group: int) -> bool:
    return bool(
        _enabled
        and 512 < m <= 8192
        and m % 128 == 0
        and k % 64 == 0
        and 0 < n < 256
        and bits in (4, 5, 8)
        and group == 64
    )


def matmul(x: mx.array, weight: mx.array, sbt: mx.array, *, bits: int) -> mx.array:
    m, k = x.shape
    record_dispatch(m, k, sbt.shape[1], bits)
    bm = 64 if m > 4096 and k > 8192 else 128
    return cast(
        mx.array, make(mx, x, weight, None, None, bits=bits, bm=bm, bn=64, sbt=sbt)()
    )


def warmup(bits_used: set[int]) -> None:
    """Compile measured tile variants during model setup, before request TTFT."""
    if not _enabled:
        return
    x = mx.zeros((128, 64), dtype=mx.bfloat16)
    sbt = mx.zeros((1, 64, 2), dtype=mx.bfloat16)
    for bits in sorted(bits_used & {4, 5, 8}):
        weight = mx.zeros((64, 2 * bits), dtype=mx.uint32)
        for bm in (64, 128):
            mx.eval(make(mx, x, weight, None, None, bits=bits, bm=bm, bn=64, sbt=sbt)())
