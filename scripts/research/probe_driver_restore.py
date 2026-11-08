"""Where a repeated prompt's time to first token goes in the round driver.

One document of N tokens, cold, then repeated (APC hit): wall to the first
token event, split by phase (APC lookup, prefill forward, MTP absorb, decode
batch join, head join, draft, ...), each timed with a synchronize.

    python scripts/research/probe_driver_restore.py <ckpt> --tokens 32768
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep_round_driver import load  # noqa: E402

TIMES: dict = {}


def wrap(obj, name, label):
    fn = getattr(obj, name)

    def timed(*a, **k):
        mx.synchronize()
        t0 = time.perf_counter()
        out = fn(*a, **k)
        mx.synchronize()
        TIMES[label] = TIMES.get(label, 0.0) + (time.perf_counter() - t0) * 1e3
        return out

    setattr(obj, name, timed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    from mlx_vlm.apc import APCManager

    from yunshu_engine.round_driver import driver as drv
    from yunshu_engine.round_driver import forward as fwd

    model, _, drafter, _ = load(a.ckpt, False)
    apc = APCManager(num_blocks=512, block_size=16, overrides={"memory_max_gb": 16})
    d = drv.RoundDriver(
        model, drafter=None if a.no_mtp else drafter, stop_tokens=set(), apc=apc
    )
    wrap(d, "_restore", "restore(lookup)")
    wrap(d.apc, "lookup", "  apc.lookup")
    wrap(d.batch, "join", "batch.join")
    if d.head is not None:
        wrap(d.head, "absorb_prompt", "head.absorb_prompt")
        wrap(d.head, "join", "head.join")
        wrap(d.head, "draft", "head.draft")
    wrap(d, "_store_checkpoint", "store_checkpoint")
    real_forward = drv.forward

    def timed_forward(lm, segs):
        mx.synchronize()
        t0 = time.perf_counter()
        h = real_forward(lm, segs)
        mx.eval(h)
        TIMES["forward"] = TIMES.get("forward", 0.0) + (time.perf_counter() - t0) * 1e3
        return h

    drv.forward = timed_forward
    rng = random.Random(0)
    ids = [rng.randrange(1000, 20000) for _ in range(a.tokens)]
    rows = []
    for rep in range(a.repeats + 1):
        TIMES.clear()
        mx.synchronize()
        t0 = time.perf_counter()
        hit = d.add(drv.Request(ids, 8, handle=rep, extra_hash=1))
        first = None
        toks = []
        steps = []
        while d.busy():
            s0 = time.perf_counter()
            evs = d.step()
            steps.append(round((time.perf_counter() - s0) * 1e3, 1))
            for e in evs:
                toks.append(e.token)
                if first is None:
                    first = time.perf_counter() - t0
        rec = {
            "rep": rep,
            "hit": hit,
            "ttft_s": round(first, 3),
            "steps_ms": steps[:6],
            "phases_ms": {k: round(v, 1) for k, v in TIMES.items()},
            "tokens": toks,
        }
        print(json.dumps(rec), flush=True)
        rows.append(rec)
    if a.output:
        with a.output.open("a") as f:
            for r in rows:
                f.write(json.dumps({**r, "tokens": len(r["tokens"])}) + "\n")
    assert all(r["tokens"] == rows[0]["tokens"] for r in rows), "repeat differs"
    _ = fwd


if __name__ == "__main__":
    main()
