# Upstream (derived): ml-explore/mlx (MIT), sdpa_vector.h and steel_attention.h @ 36969c094354086e15d2a327a9c50feb20b9cf11
# Copyright © 2024-26 Apple Inc. (harvested Metal kernel bodies).
# SPDX-License-Identifier: MIT
"""Pinned prior-art screening; GPU execution belongs exclusively inside gpuq.

This is a component screen, never a serving/token-identity qualification.
External oMLX modules are loaded by exact file hash without importing its server.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = Path("/Volumes/Micron/yunshu-refsync/omlx/omlx/patches")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--candidate", choices=("attention", "prefill", "gdn", "moe"), required=True
    )
    p.add_argument("--contexts", default="32768,65536,131072")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    args.contexts = [int(v) for v in args.contexts.split(",")]
    if not args.contexts or any(v not in (32768, 65536, 131072) for v in args.contexts):
        p.error("long-context screen requires 32K/64K/128K cells")
    if args.reps < 1 or args.iterations < 1:
        p.error("reps and iterations must be positive")
    return args


def validate(rows, candidate, contexts, reps):
    expected = {(c, r) for c in contexts for r in range(reps)}
    actual = {(row["context"], row["rep"]) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("missing, duplicate or short rows")
    for row in rows:
        if row["candidate"] != candidate:
            raise ValueError("wrong engaged candidate")
        for key in ("ours_ms", "external_ms", "max_abs"):
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError("invalid measurement")
        if not isinstance(row["bit_equal"], bool):
            raise ValueError("missing parity")
    return {
        "complete": True,
        "candidate": candidate,
        "rows": len(rows),
        "component_parity": all(row["bit_equal"] for row in rows),
        "serving_qualified": False,
    }


def external_module(name):
    path = EXTERNAL / (name + ".py")
    alias = "yunshu_engine.kernels.omlx.kernelstudy_external_" + name
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module, {
        "external_file": str(path),
        "external_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def metal_body(text, name):
    start = text.index("void " + name + "(")
    opening = text.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
        end += 1
    return text[opening + 1 : end - 1]


def latest_vector(mx, q, k, v, context):
    """Execute release-0.32.4 vector bodies on the installed Metal runtime."""
    path = Path(
        "/Volumes/Micron/yunshu-refsync/mlx/mlx/backend/metal/kernels/sdpa_vector.h"
    )
    text = path.read_text()
    blocks = (
        512 if context < 65536 else 1024
    )  # screen-selected splits; not upstream g17s defaults
    body1 = metal_body(text, "sdpa_vector_2pass_1")
    body2 = metal_body(text, "sdpa_vector_2pass_2")
    prefix1 = f"""
    constexpr int D=256,V=256,TK=4,blocks={blocks};
    constexpr bool query_transposed=false,bool_mask=false,float_mask=false,has_sinks=false,do_causal=false;
    const int N=keys_shape[2];
    const size_t k_head_stride=keys_strides[1],k_seq_stride=keys_strides[2];
    const size_t v_head_stride=values_strides[1],v_seq_stride=values_strides[2];
    const float scale=0.0625f;
    const int mask_head_stride=0,mask_kv_seq_stride=0,mask_q_seq_stride=0;
    const device bool* bmask=nullptr;const device T* fmask=nullptr;const device T* sinks=nullptr;
    const uint3 tptg=threads_per_threadgroup,tidtg=thread_position_in_threadgroup;
    const uint3 tid=threadgroup_position_in_grid,tpg=threadgroups_per_grid;
    const uint simd_lid=thread_index_in_simdgroup;
    """
    prefix2 = f"""
    constexpr int D=256,blocks={blocks};
    const uint3 tid=threadgroup_position_in_grid,tpg=threadgroups_per_grid;
    const uint simd_gid=simdgroup_index_in_threadgroup,simd_lid=thread_index_in_simdgroup;
    """
    first = mx.fast.metal_kernel(
        name="kernelstudy_vector0324_first",
        input_names=["queries", "keys", "values"],
        output_names=["out", "sums", "maxs"],
        source=prefix1 + body1,
    )
    second = mx.fast.metal_kernel(
        name="kernelstudy_vector0324_merge",
        input_names=["partials", "sums", "maxs"],
        output_names=["out"],
        source=prefix2 + body2,
    )

    def call():
        parts, sums, maxs = first(
            inputs=[q, k, v],
            template=[("T", mx.bfloat16)],
            grid=(4 * 32, 6, blocks),
            threadgroup=(32, 6, 1),
            output_shapes=[
                (1, 24, 1, blocks, 256),
                (1, 24, 1, blocks),
                (1, 24, 1, blocks),
            ],
            output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
        )
        return second(
            inputs=[parts, sums, maxs],
            template=[("T", mx.bfloat16)],
            grid=(24 * 32, 32, 1),
            threadgroup=(32, 32, 1),
            output_shapes=[q.shape],
            output_dtypes=[mx.bfloat16],
        )[0]

    return call


def attention(mx, context):
    from yunshu_engine.kernels.ragged_attention import ragged_decode_attention

    q = mx.random.normal((1, 24, 1, 256)).astype(mx.bfloat16)
    k = mx.random.normal((1, 4, context, 256)).astype(mx.bfloat16)
    v = mx.random.normal(k.shape).astype(mx.bfloat16)
    lengths = mx.array([context], mx.int32)
    mx.eval(q, k, v, lengths)

    def ours():
        return ragged_decode_attention(
            q,
            k,
            v,
            lengths,
            256**-0.5,
            max_length=context,
            impl="tile",
            row_lengths=[context],
        )

    theirs = latest_vector(mx, q, k, v, context)

    return (
        ours,
        theirs,
        {
            "shape": [1, 24, 1, 256],
            "kv_shape": list(k.shape),
            "engaged": "ours=ragged tile CK512; external=harvested MLX0.32.4 vector TK4 2pass (q_len=1,D256)",
            "external_file": "mlx/backend/metal/kernels/sdpa_vector.h:181",
            "external_sha256": hashlib.sha256(
                Path(
                    "/Volumes/Micron/yunshu-refsync/mlx/mlx/backend/metal/kernels/sdpa_vector.h"
                ).read_bytes()
            ).hexdigest(),
        },
    )


def steel_header(root):
    """Harvest upstream Steel FA2 without altering loop/reduction expressions."""
    path = root / "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"
    text = path.read_text()
    start = text.index("template <", text.index("struct DivOp"))
    # DivOp has an inner template; the kernel starts at the subsequent template.
    start = text.index("template <", text.index("};", start) + 2)
    prefix, body = text[:start], text[start:]
    opening = body.index("{ // clang-format on")
    signature, implementation = body[:opening], body[opening:]
    signature = re.sub(r"\[\[.*?\]\]", "", signature, flags=re.S)
    signature = signature.replace(
        " void attention(", " METAL_FUNC void kernelstudy_steel("
    )
    signature = signature.replace("const constant Attn", "const thread Attn")
    signature = signature.replace(
        "uint3 lid )", "uint3 lid, threadgroup T* Q_smem, threadgroup T* KV_smem)"
    )
    for declaration in (
        "threadgroup T Q_smem[q_tgp_mem_s];",
        "threadgroup T KV_smem[tgp_mem_s];",
    ):
        if implementation.count(declaration) != 1:
            raise ValueError("Steel threadgroup scratch contract changed")
        implementation = implementation.replace(declaration, "")
    if "threadgroup T* Q_smem" not in signature:
        raise ValueError("Steel signature contract changed")
    values = dict(
        align_Q=True,
        align_K=True,
        has_mask=False,
        do_causal=True,
        has_sinks=False,
        save_lse=False,
    )
    prefix = re.sub(
        r"constant bool (\w+) \[\[function_constant\(\d+\)\]\];",
        lambda m: f"constant bool {m[1]} = {'true' if values[m[1]] else 'false'};",
        prefix,
    )
    seen = {"mlx/backend/metal/kernels/utils.h"}

    def expand(match):
        name = match[1]
        if name in seen:
            return ""
        seen.add(name)
        source = (root / name).read_text().replace("#pragma once", "")
        return re.sub(r'^\s*#include "([^"]+)"', expand, source, flags=re.M)

    prefix = re.sub(r'^\s*#include "([^"]+)"', expand, prefix, flags=re.M)
    return prefix + signature + implementation


def prefill(mx, context):
    root = Path("/Volumes/Micron/yunshu-refsync/mlx")
    header = steel_header(root)
    source = """
    threadgroup bfloat Qscratch[1];
    threadgroup bfloat KVscratch[10240];
    AttnParams params;
    params.B=1; params.H=24; params.D=256;
    params.qL=Q_shape[2]; params.kL=K_shape[2]; params.gqa_factor=6;
    params.scale=0.0625f; params.NQ=params.qL/32; params.NK=params.kL/32;
    params.NQ_aligned=params.NQ; params.NK_aligned=params.NK;
    params.qL_rem=0; params.kL_rem=0; params.qL_off=params.kL-params.qL;
    for(int i=0;i<3;++i){params.Q_strides[i]=Q_strides[i];params.K_strides[i]=K_strides[i];params.V_strides[i]=V_strides[i];}
    params.O_strides[0]=24*params.qL*256;params.O_strides[1]=params.qL*256;params.O_strides[2]=256;
    kernelstudy_steel<bfloat,32,32,256,256,4,2,float,float>(Q,K,V,O,&params,nullptr,nullptr,nullptr,nullptr,
        thread_index_in_simdgroup,simdgroup_index_in_threadgroup,threadgroup_position_in_grid,
        thread_position_in_threadgroup,Qscratch,KVscratch);
    """
    kernel = mx.fast.metal_kernel(
        name="kernelstudy_steel_fa2",
        input_names=["Q", "K", "V"],
        output_names=["O"],
        header=header,
        source=source,
    )
    q = mx.random.normal((1, 24, 2048, 256)).astype(mx.bfloat16)
    k = mx.random.normal((1, 4, context, 256)).astype(mx.bfloat16)
    v = mx.random.normal(k.shape).astype(mx.bfloat16)
    mx.eval(q, k, v)

    def ours():
        return mx.fast.scaled_dot_product_attention(
            q, k, v, scale=0.0625, mask="causal"
        )

    def theirs():
        return kernel(
            inputs=[q, k, v],
            grid=(64 * 256, 24, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[q.shape],
            output_dtypes=[mx.bfloat16],
        )[0]

    return (
        ours,
        theirs,
        dict(
            shape=list(q.shape),
            kv_shape=list(k.shape),
            engaged="ours=stock MLX NAX Dsplit; external=Steel simdgroup FA2 BQ32 BK32 WM4 WN2 fp32 accumulator",
            external_file=str(
                root / "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"
            ),
            external_sha256=hashlib.sha256(header.encode()).hexdigest(),
        ),
    )


def gdn(mx, context):
    from yunshu_engine.kernels.omlx import qwen35_gdn_verify_fused as local

    other, info = external_module("qwen35_gdn_verify_fused")
    # Flash-Next config: 16 key heads,48 value heads,D128,2 verify rows.
    state = mx.random.normal((1, 48, 128, 128)).astype(mx.float32)
    q = mx.random.normal((1, 2, 16, 128)).astype(mx.bfloat16)
    k = mx.random.normal(q.shape).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, 48, 128)).astype(mx.bfloat16)
    a = mx.random.normal((1, 2, 48)).astype(mx.bfloat16)
    b = mx.random.normal(a.shape).astype(mx.bfloat16)
    alog = mx.zeros((48,), mx.bfloat16)
    dt = mx.zeros((48,), mx.bfloat16)
    mx.eval(state, q, k, v, a, b, alog, dt)

    def call(module):
        return module._kernel(True, False)(
            inputs=[state, alog, dt, q, k, v, a, b],
            template=[
                ("InT", mx.bfloat16),
                ("Hk", 16),
                ("Hv", 48),
                ("Dk", 128),
                ("Dv", 128),
                ("T", 2),
            ],
            grid=(32, 128, 48),
            threadgroup=(32, 4, 1),
            output_shapes=[v.shape],
            output_dtypes=[mx.bfloat16],
        )[0]

    info.update(
        shape=list(v.shape),
        state_shape=list(state.shape),
        engaged="fused recurrence only, no replay/norm/cache; recurrent shape independent of context",
    )
    return lambda: call(local), lambda: call(other), info


def moe(mx, context):
    from yunshu_engine.kernels.omlx import moe_verify_gather as local

    other, info = external_module("moe_verify_gather")
    # Flash-Next real gate/up geometry: E512,N640,K2560,top10,verify2.
    w = mx.random.normal((512, 640, 2560)).astype(mx.bfloat16)
    weight, scales, biases = mx.quantize(w, group_size=64, bits=4)
    mx.eval(weight, scales, biases)
    del w

    class Linear(dict):
        bits = 4
        group_size = 64

    linear = Linear(weight=weight, scales=scales, biases=biases)
    x = mx.random.normal((2, 2560)).astype(mx.bfloat16)
    # Same expert picks across rows to expose reuse; synthetic, no token claim.
    picks = mx.tile(mx.arange(10, dtype=mx.uint32), 2)
    mx.eval(x, picks)
    info.update(
        shape=[2, 2560],
        weight_shape=[512, 640, 2560],
        top_k=10,
        engaged="grouped affine q4g64 gate projection; synthetic repeated experts, context independent",
    )
    return (
        lambda: local.gather_qmv(linear, x, picks, 10),
        lambda: other.gather_qmv(linear, x, picks, 10),
        info,
    )


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "candidate": args.candidate,
                    "contexts": args.contexts,
                    "reps": args.reps,
                    "gpu_executed": False,
                }
            )
        )
        return 0
    import mlx.core as mx

    if not mx.metal.is_available():
        raise RuntimeError("M5 Metal required")
    mx.random.seed(181026)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.out.open("w") as out:

        def emit(row):
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(json.dumps(row), flush=True)

        emit(
            {
                "kind": "metadata",
                "candidate": args.candidate,
                "device": mx.device_info(),
                "base_sha": "5ac5452c3d82f43a2208d34730cd35136bd73fa0",
            }
        )
        for context in args.contexts:
            ours, theirs, info = globals()[args.candidate](mx, context)
            a, b = ours(), theirs()
            mx.eval(a, b)
            equal = bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item())
            error = float(
                mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()
            )
            # Fail fast for invalid numerics; unequal arithmetic is a recorded rejection.
            if not math.isfinite(error):
                raise ValueError("non-finite output")
            for rep in range(args.reps):
                elapsed = {}
                for name, fn in (
                    [("ours", ours), ("external", theirs)]
                    if rep % 2 == 0
                    else [("external", theirs), ("ours", ours)]
                ):
                    samples = []
                    for _ in range(args.iterations):
                        mx.synchronize()
                        start = time.perf_counter()
                        result = fn()
                        mx.eval(result)
                        samples.append((time.perf_counter() - start) * 1000)
                    elapsed[name + "_ms"] = statistics.median(samples)
                row = dict(
                    candidate=args.candidate,
                    context=context,
                    rep=rep,
                    bit_equal=equal,
                    max_abs=error,
                    **elapsed,
                    **info,
                )
                rows.append(row)
                emit(row)
            del ours, theirs, a, b
            mx.synchronize()
            mx.clear_cache()
        emit(validate(rows, args.candidate, args.contexts, args.reps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
