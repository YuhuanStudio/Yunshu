"""Where does a Flash-Next (qwen4_exp) decode token go? Host graph build vs GPU vs PLE sync.

Loads the pack through the Yunshu VLM engine (same loader, PLE manifest hook), prefills ``--ctx`` tokens,
then times ``--steps`` single-token decode steps in modes:

  serial        build (python graph construction, includes any mid-graph sync) + mx.eval, per step
  pipelined     mlx-lm style async_eval of the next token
  noPLE-*       same two, with the PLE lookup replaced by zeros (DIAGNOSTIC ONLY, not lossless):
                the gap to the real modes is the cost of the synchronous NumPy row read
  layers        per-layer-type / per-block GPU time with an eval after every block (sync overhead
                included; use the ratios, not the totals)

Run through gpuq only.  Exit code 1 and no ``complete`` unless every mode produced timings.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path


def summarize(samples_ms):
    """Median / mean / min of a list of per-step milliseconds."""
    if not samples_ms:
        raise ValueError("no samples")
    return {
        "n": len(samples_ms),
        "median_ms": round(statistics.median(samples_ms), 3),
        "mean_ms": round(statistics.fmean(samples_ms), 3),
        "min_ms": round(min(samples_ms), 3),
    }


def tok_s(ms):
    return round(1000.0 / ms, 2)


async def run(args):
    import mlx.core as mx
    from mlx_vlm.models.qwen4_exp import language as ql
    from mlx_vlm.models.qwen4_exp import ple_storage as ps

    from yunshu_engine import settings
    from yunshu_engine.types import EngineConfig
    from yunshu_engine.vlm_engine import VLMEngine

    settings.set_override("YUNSHU_VLM_APC_MEMORY_GB", 1.0)
    engine = VLMEngine(
        str(args.model), EngineConfig(prefill_step_size=512, completion_batch_size=1)
    )
    await engine.start()
    lm = engine._model.language_model
    out = args.out.open("w")

    def emit(row):
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps(row), flush=True)

    ids = [(i * 7919) % 200000 + 1000 for i in range(args.ctx)]
    cache = lm.make_cache()
    for i in range(0, args.ctx, 512):
        o = lm(mx.array([ids[i : i + 512]]), cache=cache)
        mx.eval(o.logits)
        mx.eval([c.state for c in cache if hasattr(c, "state")])
    emit(
        {
            "kind": "prefilled",
            "ctx": args.ctx,
            "active_gb": mx.get_active_memory() / 1e9,
        }
    )

    def step(y):
        return mx.argmax(lm(y, cache=cache).logits[:, -1, :], axis=-1)[:, None]

    y0 = mx.array([[ids[-1]]])

    def serial():
        y = y0
        build, ev = [], []
        for _ in range(args.steps + 4):
            t0 = time.perf_counter()
            y = step(y)
            t1 = time.perf_counter()
            mx.eval(y)
            t2 = time.perf_counter()
            build.append((t1 - t0) * 1e3)
            ev.append((t2 - t1) * 1e3)
        return build[4:], ev[4:]

    def pipelined():
        y = step(y0)
        mx.async_eval(y)
        t_start = None
        for i in range(args.steps + 4):
            if i == 4:
                t_start = time.perf_counter()
            nxt = step(y)
            mx.async_eval(nxt)
            y.item()
            y = nxt
        y.item()
        return (time.perf_counter() - t_start) * 1e3 / args.steps

    results = {}
    for tag in ("real", "noPLE"):
        orig = ps.QuantizedMMapNGramEmbedding.__call__
        if tag == "noPLE":

            def zero_call(self, ids_arr):
                shape = ids_arr.shape
                return mx.zeros((*shape, self.row_width), dtype=mx.bfloat16)

            ps.QuantizedMMapNGramEmbedding.__call__ = zero_call
        try:
            from yunshu_engine.weight_residency import ple_lookup_stats

            before = ple_lookup_stats()
            b, e = serial()
            p = pipelined()
            after = ple_lookup_stats()
        finally:
            ps.QuantizedMMapNGramEmbedding.__call__ = orig
        row = {
            "kind": "decode",
            "ple": tag,
            "serial_build": summarize(b),
            "serial_eval": summarize(e),
            "serial_total_ms": round(
                statistics.median(x + y for x, y in zip(b, e, strict=True)), 3
            ),
            "pipelined_ms_per_tok": round(p, 3),
            "ple_lookups": after["lookups"] - before["lookups"],
            "ple_ms_per_lookup": round(
                (after["elapsed_seconds"] - before["elapsed_seconds"])
                * 1e3
                / max(after["lookups"] - before["lookups"], 1),
                3,
            ),
            "ple_bytes_per_lookup": (after["bytes_read"] - before["bytes_read"])
            // max(after["lookups"] - before["lookups"], 1),
        }
        row["serial_tok_s"] = tok_s(row["serial_total_ms"])
        row["pipelined_tok_s"] = tok_s(p)
        results[tag] = row
        emit(row)

    # No per-layer async_eval: build = pure Python/C++ graph tracing, eval = encode + GPU execution.
    original_async = mx.async_eval
    mx.async_eval = lambda *a, **k: None
    try:
        b, e = serial()
    finally:
        mx.async_eval = original_async
    split = {
        "kind": "trace_vs_gpu",
        "trace_ms": summarize(b),
        "eval_ms": summarize(e),
        "note": "async_eval per layer disabled; trace_ms is host-only tracing, eval_ms is encode + GPU",
    }
    emit(split)

    # cProfile over a few serial steps: where does host time go, and how often does a step sync?
    import cProfile
    import pstats

    prof = cProfile.Profile()
    y = y0
    n_prof = 8
    prof.enable()
    for _ in range(n_prof):
        y = step(y)
        mx.eval(y)
    prof.disable()
    st = pstats.Stats(prof)
    rows = []
    for (fn, line, name), (_cc, nc, tt, ct, _callers) in st.stats.items():
        rows.append((tt, ct, nc, f"{Path(fn).name}:{line}:{name}"))
    rows.sort(reverse=True)
    emit(
        {
            "kind": "cprofile",
            "steps": n_prof,
            "top_by_tottime": [
                {
                    "fn": f,
                    "tottime_ms_per_step": round(t * 1e3 / n_prof, 3),
                    "calls_per_step": round(n / n_prof, 1),
                }
                for t, _ct, n, f in rows[:30]
            ],
            "sync_calls_per_step": {
                f: round(n / n_prof, 1)
                for _t, _ct, n, f in rows
                if any(
                    k in f
                    for k in ("eval", "tolist", "item", "synchronize", "__array__")
                )
            },
        }
    )

    # per-block timing, eval after every block
    acc = defaultdict(list)

    def wrap(cls, name):
        inner = cls.__call__

        def timed(self, *a, **k):
            t0 = time.perf_counter()
            r = inner(self, *a, **k)
            mx.eval(r)
            acc[name].append((time.perf_counter() - t0) * 1e3)
            return r

        cls.__call__ = timed
        return inner

    saved = []
    for cls, name in (
        (ql.Qwen4ExpGatedDeltaNet, "gdn"),
        (ql.Qwen4ExpAttention, "full_attn"),
        (ql.Qwen4ExpSparseMoeBlock, "moe"),
        (ql.Qwen4ExpPLELayer, "ple_layer"),
        (ql.Qwen4ExpGatedResidual, "hyper_conn"),
    ):
        saved.append((cls, wrap(cls, name)))
    try:
        y = y0
        for _ in range(args.steps // 2 + 3):
            y = step(y)
            mx.eval(y)
    finally:
        for cls, inner in saved:
            cls.__call__ = inner
    per_token = {}
    n_tok = args.steps // 2 + 3
    for name, vals in acc.items():
        per_token[name] = {
            "calls_per_token": round(len(vals) / n_tok, 1),
            "ms_per_token": round(sum(vals) / n_tok, 3),
        }
    emit(
        {
            "kind": "blocks",
            "per_token": per_token,
            "note": "eval after each block; sync overhead inflates totals",
        }
    )

    real, nople = results["real"], results["noPLE"]
    ok = all(r["serial_total_ms"] > 0 for r in results.values())
    emit(
        {
            "kind": "verdict",
            "ple_sync_cost_ms_serial": round(
                real["serial_total_ms"] - nople["serial_total_ms"], 3
            ),
            "ple_sync_cost_ms_pipelined": round(
                real["pipelined_ms_per_tok"] - nople["pipelined_ms_per_tok"], 3
            ),
            "host_bound_noPLE": nople["serial_build"]["median_ms"]
            > nople["serial_eval"]["median_ms"],
        }
    )
    emit({"complete": ok})
    out.close()
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    return asyncio.run(run(a))


if __name__ == "__main__":
    sys.exit(main())
