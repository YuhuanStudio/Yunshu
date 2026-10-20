# Upstream (derived): ml-explore/mlx (MIT), steel_attention_nax.h @ v0.32.3
"""Head-dim-split NAX attention with a free query tile (BQ) for 27B prefill chunks.

MLX 0.32.3 runs D=256 prefill through ``attention_nax_dsplit`` with BQ=64, BK=32, WM=4, WN=2.
Per-row arithmetic (key-block order, D-split reduction, online softmax) does not depend on BQ,
so a larger query tile re-reads K/V half as often with bit-identical results. This module
harvests the installed MLX kernel and compiles it for a chosen (BQ, WM, WN).

    prefillspeed_attn.py --out F.jsonl [--keys ...] [--bq 64 128]   (GPU: through gpuq only)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
import time
from functools import cache
from pathlib import Path

H, HKV, D, BK = 24, 4, 256, 32


@cache
def header(align_q: bool, align_k: bool, opts: str = "") -> str:
    root = Path(str(importlib.metadata.distribution("mlx").locate_file("mlx/include")))
    seen = {"mlx/backend/metal/kernels/utils.h"}  # custom kernel supplies utils

    def expand(name: str) -> str:
        if name in seen:
            return ""
        seen.add(name)
        text = (root / name).read_text()
        return re.sub(
            r'^\s*#include "([^"]+)"', lambda m: expand(m[1]), text, flags=re.M
        ).replace("#pragma once", "")

    path = "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h"
    text = (root / path).read_text()
    prefix = text[: text.index("// clang-format off")]
    at = text.index("void attention_nax_dsplit(")
    begin = text.rfind("template <", 0, at)
    body = text[begin:]
    opening = body.index("{ // clang-format on")
    signature, impl = body[:opening], body[opening:]
    signature = re.sub(r"\[\[.*?\]\]", "", signature, flags=re.S)
    signature = signature.replace(
        " void attention_nax_dsplit(", " METAL_FUNC void prefillspeed_dsplit("
    )
    signature = signature.replace("const constant Attn", "const thread Attn")
    signature = signature.replace(
        "uint3 lid )", "uint3 lid, threadgroup AccumType (&s_xchg)[WM][WN][BK * 16])"
    )
    declaration = "threadgroup AccumType s_xchg[WM][WN][TK * kEPF * 32];"
    if declaration not in impl or "(&s_xchg)" not in signature:
        raise ValueError("native D-split scratch signature changed")
    impl = impl.replace(
        declaration, 'static_assert(TK * kEPF * 32 == BK * 16, "scratch shape");'
    )
    if "swz" in opts:
        # Lane-minor exchange layout: the native per-lane-linear layout strides lanes by
        # TK*kEPF floats (bank conflicts). Same values, same reduction order.
        old = "base + ik * kEPF + i]"
        if impl.count(old) != 3:
            raise ValueError("native exchange indexing changed")
        impl = impl.replace(old, "(ik * kEPF + i) * 32 + short(simd_lane_id)]")
    if "lazy" in opts:
        # x * 1.0f == x exactly: skip the accumulator rescale when no row's max moved.
        old = "    Otile.template row_bin_op<MulOp>(factor);\n"
        if impl.count(old) != 1:
            raise ValueError("native rescale statement changed")
        impl = impl.replace(
            old,
            "    bool all_one = true;\n"
            "    for (short i = 0; i < kRowsPT; ++i) all_one = all_one && (factor[i] == 1.0f);\n"
            "    if (!simd_all(all_one)) { Otile.template row_bin_op<MulOp>(factor); }\n",
        )
    values = {
        "align_Q": align_q,
        "align_K": align_k,
        "has_mask": False,
        "do_causal": True,
        "has_sinks": False,
    }
    prefix = re.sub(
        r"constant bool (\w+) \[\[function_constant\(\d+\)\]\];",
        lambda m: f"constant bool {m[1]} = {'true' if values[m[1]] else 'false'};",
        prefix,
    )
    prefix = re.sub(
        r'^\s*#include "([^"]+)"', lambda m: expand(m[1]), prefix, flags=re.M
    )
    pipe = ""
    if "pipe" in opts:
        pipe = (Path(__file__).parent / "prefillspeed_pipe.metal.txt").read_text()
    if "tq" in opts:
        pipe = (Path(__file__).parent / "prefillspeed_tq.metal.txt").read_text()
    if "kb2" in opts:
        pipe = (Path(__file__).parent / "prefillspeed_kb2.metal.txt").read_text()
    return prefix + signature + impl + pipe


@cache
def compiled(
    mx, bq: int, wm: int, wn: int, align_q: bool, align_k: bool, opts: str = ""
):
    h = header(align_q, align_k, opts)
    tq = 2 if "tq2" in opts else 1
    if "tq" in opts:
        if not (align_q and align_k):
            raise ValueError("tq variant needs aligned rows and keys")
        call = (
            f"prefillspeed_tq<bfloat, {bq}, {BK}, {D}, {wm}, {wn}, {tq}, float>(Q, K, V, O, &params, "
            "thread_index_in_simdgroup, simdgroup_index_in_threadgroup, "
            "threadgroup_position_in_grid, s_xchg);"
        )
    elif "kb2" in opts:
        if not (align_q and align_k):
            raise ValueError("kb2 variant needs aligned rows and keys")
        call = (
            f"prefillspeed_kb2<bfloat, {bq}, {BK}, {D}, {wm}, {wn}, float>(Q, K, V, O, &params, "
            "thread_index_in_simdgroup, simdgroup_index_in_threadgroup, "
            "threadgroup_position_in_grid, s_xchg);"
        )
        tq = 2
    elif "pipe" in opts:
        if not (align_q and align_k):
            raise ValueError("pipe variant needs aligned rows and keys")
        call = (
            f"prefillspeed_pipe<bfloat, {bq}, {BK}, {D}, {wm}, {wn}, float>(Q, K, V, O, &params, "
            "thread_index_in_simdgroup, simdgroup_index_in_threadgroup, "
            "threadgroup_position_in_grid, s_xchg);"
        )
    else:
        call = (
            f"prefillspeed_dsplit<bfloat, {bq}, {BK}, {D}, {wm}, {wn}, float, float>(Q, K, V, O, &params, "
            "nullptr, nullptr, nullptr, thread_index_in_simdgroup, "
            "simdgroup_index_in_threadgroup, threadgroup_position_in_grid, "
            "thread_position_in_threadgroup, s_xchg);"
        )
    if "rev" in opts:
        # Heaviest causal query blocks first: shortens the launch tail.
        call = (
            "uint3 tgrev = threadgroup_position_in_grid; tgrev.x = params.NQ - 1 - tgrev.x; "
            + call.replace("threadgroup_position_in_grid", "tgrev")
        )
    source = f"""
    threadgroup float s_xchg[{wm}][{wn}][{tq} * {BK} * 16];
    AttnParams params;
    params.B = 1; params.H = {H}; params.D = {D};
    params.qL = Q_shape[2]; params.kL = K_shape[2]; params.gqa_factor = {H // HKV};
    params.scale = 0.0625f;
    params.NQ = (params.qL + {bq} - 1) / {bq};
    params.NK = (params.kL + {BK} - 1) / {BK};
    params.NQ_aligned = params.qL / {bq}; params.NK_aligned = params.kL / {BK};
    params.qL_rem = params.qL % {bq}; params.kL_rem = params.kL % {BK};
    params.qL_off = params.kL - params.qL;
    for (int i = 0; i < 3; i++) {{
      params.Q_strides[i] = Q_strides[i];
      params.K_strides[i] = K_strides[i];
      params.V_strides[i] = V_strides[i];
    }}
    params.O_strides[0] = {H} * params.qL * {D};
    params.O_strides[1] = params.qL * {D}; params.O_strides[2] = {D};
    {call}
    """
    name = "prefillspeed_attn_" + hashlib.sha256((h + source).encode()).hexdigest()[:16]
    return mx.fast.metal_kernel(
        name=name,
        input_names=["Q", "K", "V"],
        output_names=["O"],
        header=h,
        source=source,
        ensure_row_contiguous=False,
    )


def make(mx, q, k, v, bq: int = 128, opts: str = ""):
    wn = 2
    wm = bq // (32 if "tq2" in opts else 16)
    rows, keys = q.shape[-2], k.shape[-2]
    if (
        q.shape != (1, H, rows, D)
        or k.shape != (1, HKV, keys, D)
        or v.shape != k.shape
        or keys < rows
        or q.dtype != mx.bfloat16
    ):
        raise ValueError("Q24/KV4 D256 bf16 shapes required")
    kernel = compiled(mx, bq, wm, wn, rows % bq == 0, keys % BK == 0, opts)
    return lambda: kernel(
        inputs=[q, k, v],
        grid=((rows + bq - 1) // bq * 32, H * wm * wn, 1),
        threadgroup=(32, wm * wn, 1),
        output_shapes=[q.shape],
        output_dtypes=[mx.bfloat16],
    )[0]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rows", type=int, default=2048)
    p.add_argument("--keys", type=int, nargs="+", default=[2048, 8192, 32768, 131072])
    p.add_argument("--bq", type=int, nargs="+", default=[64, 128])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument(
        "--opts", nargs="*", default=[], help="kernel variants, e.g. swz lazy swz+lazy"
    )
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    for bq in a.bq:
        if bq % 16 or not 16 <= bq <= 256 or a.rows % bq:
            print(f"bad bq {bq}", file=sys.stderr)
            return 2
    if a.dry_run:
        print(json.dumps(dict(bq=a.bq, keys=a.keys)))
        return 0
    import mlx.core as mx

    sys.path.insert(0, str(Path(__file__).parent))
    from prefillspeed_micro import sdpa_flops

    a.out.parent.mkdir(parents=True, exist_ok=True)
    mx.random.seed(0)
    n = 0
    with a.out.open("w") as f:

        def emit(rec):
            nonlocal n
            n += 1
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(json.dumps(rec), flush=True)

        emit(
            dict(
                type="engagement",
                device=mx.device_info().get("device_name"),
                mlx=mx.__version__,
            )
        )
        for lk in a.keys:
            q = (mx.random.normal((1, H, a.rows, D)) * 0.5).astype(mx.bfloat16)
            k = (mx.random.normal((1, HKV, lk, D)) * 0.5).astype(mx.bfloat16)
            v = (mx.random.normal((1, HKV, lk, D)) * 0.5).astype(mx.bfloat16)
            mx.eval(q, k, v)
            fns = {
                "stock": lambda: mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=D**-0.5, mask="causal"
                )
            }
            for bq in a.bq:
                fns[f"bq{bq}"] = make(mx, q, k, v, bq)
                for o in a.opts:
                    fns[f"bq{bq}{o}"] = make(mx, q, k, v, bq, o)
            ref = fns["stock"]()
            mx.eval(ref)
            nrep = max(2, min(a.reps, int(2e12 // max(sdpa_flops(a.rows, lk), 1))))
            res = {}
            for name, fn in fns.items():
                out = fn()
                mx.eval(out)
                equal = bool(mx.array_equal(out, ref).item())
                if not equal and name != "stock":
                    emit(dict(type="mismatch", keys=lk, arm=name))
                    return 1
                res[name] = dict(equal=equal, ts=[])
            for _ in range(3):  # interleaved reps
                for name, fn in fns.items():
                    mx.eval(fn())
                    t0 = time.perf_counter()
                    mx.eval([fn() for _ in range(nrep)])
                    res[name]["ts"].append((time.perf_counter() - t0) / nrep)
            fl = sdpa_flops(a.rows, lk)
            for name, r in res.items():
                med = sorted(r["ts"])[len(r["ts"]) // 2]
                emit(
                    dict(
                        type="attn",
                        keys=lk,
                        arm=name,
                        equal=r["equal"],
                        ms=round(med * 1e3, 3),
                        tflops=round(fl / med / 1e12, 2),
                    )
                )
        f.write(json.dumps(dict(type="complete", success=True, records=n)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
