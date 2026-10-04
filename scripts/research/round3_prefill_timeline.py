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
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from yunshu_engine.vlm_batch_runner import RunStats
    from yunshu_engine.vlm_engine import VLMEngine

    engine = VLMEngine(MODEL)
    asyncio.run(engine.start())
    runner = engine._batch_runner
    drv = runner.driver
    if drv is None:
        print("no driver", flush=True)
        return 1
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
    texts = [(PROMPTS / f"{k}-{a.ctx}.txt").read_text() for k in ("prose", "code")]
    texts += [(PROMPTS / f"conc-code-{i}.txt").read_text() for i in range(4)]

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
        except Exception as e:
            errs.append(repr(e))

    # warm-up (compile), not timed
    w = ids_for("Say hi.")
    list(runner.iter_tokens(w[0], max_tokens=8, temperature=0.0, seed=1,
                            allow_draft=True, prompt_kwargs=w[1],
                            apc_semantic_hash=w[2], stats=RunStats()))  # fmt: skip
    steps.clear()
    t0[0] = time.perf_counter()
    th = [threading.Thread(target=go, args=(i,)) for i in range(a.n)]
    [t.start() for t in th]
    [t.join() for t in th]
    asyncio.run(engine.stop())
    rec = dict(n=a.n, ctx=a.ctx, prompt_tokens=[len(r[0]) for r in reqs],
               ttft=ttft, errors=errs, summary=summarize(steps), steps=steps)  # fmt: skip
    a.out.write_text(json.dumps(rec))
    print(json.dumps({k: v for k, v in rec.items() if k != "steps"}), flush=True)
    return 1 if errs or None in ttft else 0


if __name__ == "__main__":
    sys.exit(main())
