"""Host cost of TensorFold kernel dispatch, and whether mx.compile removes it.

Builds (no eval) N chained HC-norm + HC-project calls with the real oQ4e weights of one layer, eagerly and
under mx.compile, and compares build time and result equality.  Run through gpuq only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path


async def run(args):
    import mlx.core as mx

    from yunshu_engine import settings
    from yunshu_engine.kernels import tf_hc_forward as tfh
    from yunshu_engine.kernels.tf_flash_next import hc
    from yunshu_engine.types import EngineConfig
    from yunshu_engine.vlm_engine import VLMEngine

    settings.set_override("YUNSHU_VLM_APC_MEMORY_GB", 1.0)
    engine = VLMEngine(
        str(args.model), EngineConfig(prefill_step_size=512, completion_batch_size=1)
    )
    await engine.start()
    lm = engine._model.language_model
    layer = lm.model.layers[1]
    conn = layer.attn_hyper_connection
    plan = tfh._plan(conn)
    streams = plan["streams"]
    out = args.out.open("w")

    def emit(row):
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps(row), flush=True)

    h = (mx.random.normal((args.rows, streams * 2560)) * 0.5).astype(mx.bfloat16)
    br = (mx.random.normal((args.rows, 2560)) * 0.2).astype(mx.bfloat16)
    inj = (mx.random.uniform(shape=(args.rows, streams)) * 2).astype(mx.bfloat16)

    def chain(h, br, inj):
        for _ in range(args.calls):
            hn, ssp = hc.hc_norm(
                h, streams=streams, write_back="plain", branch=(br,), inject=inj
            )
            mixed, inj = hc.hc_project(
                hn,
                ssp,
                plan["down"],
                plan["up"],
                plan["scale"],
                eps=plan["eps"],
                streams=streams,
                low=plan["low"],
            )
            inj = inj[: args.rows]
            h, br = hn, mixed
        return h, br

    compiled = mx.compile(chain)
    res = {}
    for name, fn in (("eager", chain), ("compiled", compiled)):
        try:
            mx.eval(fn(h, br, inj))  # warm / trace
            builds, evals = [], []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                o = fn(h, br, inj)
                t1 = time.perf_counter()
                mx.eval(o)
                t2 = time.perf_counter()
                builds.append((t1 - t0) * 1e3)
                evals.append((t2 - t1) * 1e3)
            res[name] = {
                "build_ms": round(statistics.median(builds), 3),
                "eval_ms": round(statistics.median(evals), 3),
            }
            res[name + "_out"] = fn(h, br, inj)
            mx.eval(res[name + "_out"])
        except Exception as exc:  # noqa: BLE001
            res[name] = {"error": repr(exc)[:300]}
    same = None
    if "eager_out" in res and "compiled_out" in res:
        same = all(
            bool(mx.array_equal(a, b).item())
            for a, b in zip(res["eager_out"], res["compiled_out"], strict=True)
        )
    emit(
        {
            "kind": "dispatch",
            "calls": args.calls,
            "rows": args.rows,
            "eager": res["eager"],
            "compiled": res["compiled"],
            "bit_equal": same,
        }
    )
    emit({"complete": True})
    out.close()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--calls", type=int, default=97)
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    return asyncio.run(run(ap.parse_args(argv)))


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--calls", type=int, default=97)
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    return ap


if __name__ == "__main__":
    sys.exit(main())
