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

    passage = (
        "The lighthouse stood at the edge of the cliff, its white tower weathered by a hundred winters. "
        "Every evening the keeper climbed the spiral stairs, trimmed the wick, and watched the ships pass "
        "far below. In the morning he wrote the weather in a worn ledger, noting the wind, the tide, and "
        "the color of the sky. "
    )
    tokenizer = getattr(engine, "_tokenizer", None)
    if tokenizer is not None:
        base = tokenizer.encode(passage)
        ids = (base * (args.ctx // len(base) + 1))[: args.ctx]
    else:
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

    def verify_ms(k):
        """Median ms of the real speculative verify forward of k rows (batch-invariant path, rolled back)."""
        yk = mx.array([[ids[-1]] * k])
        times = []
        for i in range(args.steps // 2 + 3):
            t0 = time.perf_counter()
            hidden, _, txn = lm.speculative_verify_hidden(yk, cache)
            mx.eval(hidden)
            if i >= 3:
                times.append((time.perf_counter() - t0) * 1e3)
            txn.abort()
        return round(statistics.median(times), 3)

    results = {}
    plan = [
        ("real", "off"),
        ("real", "hc+moe"),
        ("real", "hc+moe+gdn"),
        ("real", "hc+moe"),
        ("real", "hc+moe+gdn"),
        ("noPLE", "hc+moe"),
        ("noPLE", "hc+moe+gdn"),
    ]
    from yunshu_engine.kernels import tf_hc_forward

    for tag, mode in plan:
        if mode == "off":
            tf_hc_forward.uninstall()
        else:
            tf_hc_forward.install()
            tf_hc_forward.set_moe(mode.startswith("hc+moe"))
            tf_hc_forward.set_gdn(mode.endswith("+gdn"))
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
            "verify_ms": {k: verify_ms(k) for k in (2, 3, 4)},
            "tf_hc": mode,
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
        if mode == "hc+moe+gdn":
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

    # Graph census of one decode step: number of primitives by type (async_eval off, nothing evaluated).
    mx.async_eval = lambda *a, **k: None
    try:
        dot_path = args.out.with_suffix(".dot")
        y = step(y0)
        mx.export_to_dot(str(dot_path), y)
        mx.eval(y)
    finally:
        mx.async_eval = original_async
    import re

    text = dot_path.read_text()
    # nodes look like `1234 [label="Add"];` or `... [label = "Add"]`; edges contain "->"
    labels = re.findall(r'\[\s*label\s*=\s*"([^"]+)"', text)
    hist = defaultdict(int)
    for label in labels:
        hist[label.split()[0] if label.split() else label] += 1
    emit(
        {
            "kind": "graph",
            "nodes": len(labels),
            "top_primitives": sorted(hist.items(), key=lambda kv: -kv[1])[:25],
        }
    )

    # Who builds the small kernels?  Count mx.arange / mx.zeros / mx.full / mx.concatenate callers in one step.
    import traceback
    from collections import Counter

    callers = Counter()
    wrapped = {}
    for fname in (
        "arange",
        "zeros",
        "full",
        "ones",
        "concatenate",
        "stack",
        "where",
        "broadcast_to",
    ):
        fn = getattr(mx, fname)
        wrapped[fname] = fn

        def make(fn=fn, fname=fname):
            def inner(*a, **k):
                fr = traceback.extract_stack(limit=3)[0]
                callers[
                    f"{fname}  {Path(fr.filename).parent.name}/{Path(fr.filename).name}:{fr.lineno}"
                ] += 1
                return fn(*a, **k)

            return inner

        setattr(mx, fname, make())
    try:
        mx.async_eval = lambda *a, **k: None
        yy = step(y0)
        mx.eval(yy)
    finally:
        mx.async_eval = original_async
        for fname, fn in wrapped.items():
            setattr(mx, fname, fn)
    emit({"kind": "callers", "top": callers.most_common(40)})

    # Per-component graph census with the real layers: which block owns the Sum/Square/AsType nodes?
    import re as _re

    from mlx_vlm.models.qwen4_exp import language as _ql

    fwd = _ql._QWEN4_BATCH_INVARIANT_FORWARD
    mx.async_eval = lambda *a, **k: None

    def census(label, fn, calls):
        try:
            _census(label, fn, calls)
        except Exception as exc:  # noqa: BLE001 - keep the other components
            emit({"kind": "component", "label": label, "error": repr(exc)[:300]})

    def _census(label, fn, calls):
        out = fn()
        path = args.out.with_suffix(f".{label}.dot")
        mx.export_to_dot(str(path), out)
        text = path.read_text()
        path.unlink()
        hist = defaultdict(int)
        for lab in _re.findall(r'\[\s*label\s*=\s*"([^"]+)"', text):
            hist[lab.split()[0] if lab.split() else lab] += 1
        mx.eval(out)
        emit(
            {
                "kind": "component",
                "label": label,
                "calls_per_step": calls,
                "nodes": sum(hist.values()),
                "top": sorted(hist.items(), key=lambda kv: -kv[1])[:14],
            }
        )

    try:
        layers = lm.model.layers
        h4 = mx.random.normal((1, 1, 10240)).astype(mx.bfloat16)
        h1 = mx.random.normal((1, 1, 2560)).astype(mx.bfloat16)
        gdn_i = next(i for i, ly in enumerate(layers) if ly.is_linear)
        att_i = next(i for i, ly in enumerate(layers) if not ly.is_linear)
        census(
            "hyper_connection",
            lambda: fwd._hyper_connection(layers[1].attn_hyper_connection, h4)[0],
            97,
        )
        census("feed_forward", lambda: fwd._feed_forward(layers[1].mlp, h1), 48)
        census(
            "inject",
            lambda: fwd._inject(h1, h4, mx.ones((1, 1, 4), dtype=mx.bfloat16)),
            96,
        )
        census(
            "gated_delta",
            lambda: fwd._gated_delta(layers[gdn_i].linear_attn, h1, None, cache[gdn_i]),
            36,
        )
        pos = mx.array([[args.ctx]], dtype=mx.int32)
        census(
            "qsa_attention",
            lambda: fwd._qsa_attention(
                layers[att_i].self_attn, h1, cache[att_i], pos, None
            ),
            12,
        )
    finally:
        mx.async_eval = original_async

    # Where does a k-row verify spend its time?  Component timers (eval after each call, sync overhead included).
    from collections import defaultdict as _dd

    from yunshu_engine.kernels import tf_hc_forward as _tfh
    from yunshu_engine.kernels.tf_flash_next import hc as _hc

    _tfh.install()
    _tfh.set_moe(True)
    acc2 = _dd(float)

    def timed(owner, name, label):
        inner = getattr(owner, name)

        def wrapper(*a, **k):
            t0 = time.perf_counter()
            out = inner(*a, **k)
            if isinstance(out, tuple) and out and isinstance(out[0], str):
                mx.eval(out[1])
            elif out is not None:
                mx.eval(out)
            acc2[label] += (time.perf_counter() - t0) * 1e3
            return out

        setattr(owner, name, wrapper)
        return owner, name, inner

    patches = [
        timed(_tfh, "_moe", "moe_fused"),
        timed(_hc, "hc_project", "hc_project"),
        timed(_hc, "hc_norm", "hc_norm"),
        timed(_tfh, "_gated_delta", "gated_delta"),
        timed(_ql.Qwen4ExpBatchInvariantForward, "_qsa_attention", "qsa_attention"),
    ]
    try:
        for k in (1, 2, 4):
            yk = mx.array([[ids[-1]] * k])
            acc2.clear()
            n = 6
            for _ in range(n):
                hidden, _, txn = lm.speculative_verify_hidden(yk, cache)
                mx.eval(hidden)
                txn.abort()
            emit(
                {
                    "kind": "verify_components",
                    "rows": k,
                    "ms_per_forward": {
                        c: round(v / n, 2) for c, v in sorted(acc2.items())
                    },
                }
            )
    finally:
        for owner, name, inner in patches:
            setattr(owner, name, inner)

    # Expert-union census on real text: greedy-decode a continuation, record each token's routed experts per
    # layer, then count distinct experts per layer for windows of k consecutive tokens (a k-row verify reads that many).
    import numpy as np

    from yunshu_engine.kernels import tf_hc_forward as _tfh2
    from yunshu_engine.kernels.tf_flash_next import experts as _experts

    _tfh2.install()
    _tfh2.set_moe(True)
    _tfh2.set_gdn(True)
    seen = []
    orig_gateup = _experts.expert_gateup

    def spy(x, logits, top_k, n_experts, *a, **k):
        out = orig_gateup(x, logits, top_k, n_experts, *a, **k)
        seen.append(out[1])
        return out

    _experts.expert_gateup = spy
    try:
        picks_by_token = []
        yy = mx.array([[ids[-1]]])
        for _ in range(48):
            seen.clear()
            yy = step(yy)
            mx.eval(yy, seen)
            picks_by_token.append(
                np.stack([np.array(p)[0] for p in seen])
            )  # [layers, top_k]
    finally:
        _experts.expert_gateup = orig_gateup
    arr = np.stack(picks_by_token)  # [T, layers, top_k]
    union = {}
    for k in (1, 2, 3, 4):
        sizes = []
        for t0 in range(0, arr.shape[0] - k + 1):
            win = arr[t0 : t0 + k]  # [k, layers, top_k]
            sizes.append(
                np.mean(
                    [
                        len(np.unique(win[:, layer_i].reshape(-1)))
                        for layer_i in range(win.shape[1])
                    ]
                )
            )
        union[k] = round(float(np.mean(sizes)), 2)
    emit(
        {
            "kind": "expert_union",
            "tokens": int(arr.shape[0]),
            "layers": int(arr.shape[1]),
            "distinct_per_layer_by_window": union,
        }
    )

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
