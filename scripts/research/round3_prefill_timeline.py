"""Per-step timeline of the round driver's prefill with N concurrent cold prompts.

Prints one record per prefill / decode step (start offset, duration, tokens per
row) and each request's first-token time. Run under gpuq with
YUNSHU_ROUND_DRIVER=1.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
MODEL = "/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
PROMPTS = Path("/Volumes/P5Plus/yunshu-build/tfnew/prompts")


def summarize(steps):
    """Collapse the step list into (kind, count, seconds, tokens)."""
    out = {}
    for s in steps:
        k = out.setdefault(s["kind"], [0, 0.0, 0])
        k[0] += 1
        k[1] += s["dur"]
        k[2] += s.get("tokens", 0)
    return {k: dict(n=v[0], s=round(v[1], 2), tokens=v[2]) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument(
        "--frac", type=float, default=1.0, help="use this fraction of the prompt file"
    )
    ap.add_argument(
        "--gdn-chunked",
        action="store_true",
        help="ablation: chunked GDN in driver prefill",
    )
    ap.add_argument("--off", action="store_true", help="expect no driver (arm off)")
    ap.add_argument("--min-conc", type=int, default=2, help="driver routing threshold")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from yunshu_engine import vlm_batch_runner
    from yunshu_engine.vlm_batch_runner import RunStats
    from yunshu_engine.vlm_engine import VLMEngine

    vlm_batch_runner.DRIVER_MIN_CONCURRENCY = a.min_conc
    vlm_batch_runner.DRIVER_MAX_UNCACHED_TOKENS = 10**9  # measure the driver itself
    if a.gdn_chunked:
        import contextlib

        from yunshu_engine.kernels import gdn_prefill

        gdn_prefill.step_kernel = contextlib.nullcontext
    engine = VLMEngine(MODEL)
    asyncio.run(engine.start())
    runner = engine._batch_runner
    drv = runner.driver
    if (drv is None) != a.off:
        print(f"driver present={drv is not None} does not match --off={a.off}")
        return 1
    if drv is None:  # off arm: the upstream path, no step hooks
        from types import SimpleNamespace

        drv = SimpleNamespace(_prefill_step=None, _decode_step=None)
    steps = []
    t0 = [0.0]
    orig_p, orig_d = drv._prefill_step, drv._decode_step

    def wrap_p(waiting, decoding):
        s = time.perf_counter()
        r = orig_p(waiting, decoding)
        steps.append(
            dict(
                kind="prefill_dec" if decoding else "prefill_idle",
                t=round(s - t0[0], 3),
                dur=time.perf_counter() - s,
                tokens=sum(x.done for x in waiting),
                rows=[(x.done, len(x.req.ids), x.pending is not None) for x in waiting],
                events=[round(time.perf_counter() - t0[0], 3) for _ in r[:1]],
                n_events=len(r),
            )  # fmt: skip
        )
        return r

    def wrap_d():
        s = time.perf_counter()
        r = orig_d()
        steps.append(
            dict(kind="decode", t=round(s - t0[0], 3), dur=time.perf_counter() - s)
        )
        return r

    drv._prefill_step, drv._decode_step = wrap_p, wrap_d
    import cProfile
    import io
    import pstats

    prof = cProfile.Profile()
    slice_body = runner._drive_slice_body

    def profiled(*args, **kw):
        prof.enable()
        try:
            return slice_body(*args, **kw)
        finally:
            prof.disable()

    runner._drive_slice_body = profiled
    prompt_text = (PROMPTS / f"code-{a.ctx}.txt").read_text()
    prompt_text = prompt_text[: int(len(prompt_text) * a.frac)]
    texts = [f"UNIQUE-{i}-{time.time_ns()}\n{prompt_text}" for i in range(a.n)]

    def ids_for(text):
        return engine._executor.submit(
            engine._runner_input,
            [{"role": "user", "content": text}], [], [], False,
            {"enable_thinking": False},
        ).result()  # fmt: skip

    reqs = [ids_for(t) for t in texts[: a.n]]
    ttft = [None] * a.n
    errs = []

    def go(i):
        ids, kw, salt = reqs[i]
        try:
            for _ in runner.iter_tokens(
                ids, max_tokens=a.tokens, temperature=0.0, seed=1,
                allow_draft=True, prompt_kwargs=kw, apc_semantic_hash=salt,
                stats=RunStats(),
            ):  # fmt: skip
                if ttft[i] is None:
                    ttft[i] = time.perf_counter() - t0[0]
                    print(f"first token {i}: {ttft[i]:.1f}s", flush=True)
        except Exception as e:
            errs.append(repr(e))

    # warm-up (compile), not timed
    w = ids_for("Say hi.")
    list(runner.iter_tokens(w[0], max_tokens=8, temperature=0.0, seed=1,
                            allow_draft=True, prompt_kwargs=w[1],
                            apc_semantic_hash=w[2], stats=RunStats()))  # fmt: skip
    import mlx.core as mx

    mx.reset_peak_memory()
    gib = 2**30
    print("warm-up done", flush=True)
    steps.clear()
    t0[0] = time.perf_counter()
    th = [threading.Thread(target=go, args=(i,)) for i in range(a.n)]
    [t.start() for t in th]
    [t.join() for t in th]
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(28)
    print(buf.getvalue(), flush=True)
    mem = dict(
        peak_gib=round(mx.get_peak_memory() / gib, 2),
        active_gib=round(mx.get_active_memory() / gib, 2),
        cache_gib=round(mx.get_cache_memory() / gib, 2),
    )
    import gc

    gc.collect()
    mx.clear_cache()
    mem["active_after_gc_gib"] = round(mx.get_active_memory() / gib, 2)
    mgr = runner.apc_manager
    entries = getattr(mgr, "_exact_cache", None)
    if isinstance(entries, dict):
        mem["apc_entries"] = sorted(len(e.token_ids) for e in entries.values())
    for name in ("clear", "clear_all", "reset"):
        if mgr is not None and hasattr(mgr, name):
            getattr(mgr, name)()
            gc.collect()
            mx.clear_cache()
            mem["active_after_apc_clear_gib"] = round(mx.get_active_memory() / gib, 2)
            break
    asyncio.run(engine.stop())
    rec = dict(mem=mem, n=a.n, ctx=a.ctx, prompt_tokens=[len(r[0]) for r in reqs],
               ttft=ttft, errors=errs, summary=summarize(steps), steps=steps)  # fmt: skip
    a.out.write_text(json.dumps(rec))
    print(json.dumps({k: v for k, v in rec.items() if k != "steps"}), flush=True)
    return 1 if errs or None in ttft else 0


if __name__ == "__main__":
    sys.exit(main())
