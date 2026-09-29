"""Round driver decode step anatomy at N rows: where the milliseconds go.

Adds ``--rows`` prompts, runs them past prefill, then times ``--steps`` decode
steps: the whole step, the forward's graph construction (host time before
anything runs), sampling / evaluation, and the MTP head's absorb and draft.

    python scripts/research/probe_driver_decode.py <ckpt> --rows 8 --steps 40
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep_round_driver import PROMPTS, encode, load  # noqa: E402


def timed(obj, name, sink):
    fn = getattr(obj, name)

    def wrap(*a, **k):
        t0 = time.perf_counter()
        out = fn(*a, **k)
        sink.setdefault(name, []).append((time.perf_counter() - t0) * 1e3)
        return out

    setattr(obj, name, wrap)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--context", type=int, default=0)
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    model, processor, drafter, _ = load(a.ckpt, a.quantize)
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    tok = processor.tokenizer
    prompts = [encode(tok, p, a.context) for p in PROMPTS]
    d = RoundDriver(model, drafter=None if a.no_mtp else drafter, stop_tokens=set())
    for i in range(a.rows):
        d.add(Request(prompts[i % len(prompts)], 10_000, handle=i))
    while any(r.pending is None for r in d.rows):
        d.step()
    sink: dict = {}
    timed(d.batch, "forward", sink)
    timed(d.batch, "commit", sink)
    timed(d, "_draw", sink)
    if d.head is not None:
        timed(d.head, "absorb", sink)
        timed(d.head, "draft", sink)
    for _ in range(4):  # warm-up: kernel compiles, cost curve
        d.step()
    sink.clear()
    steps, tokens = [], 0
    for _ in range(a.steps):
        mx.synchronize()
        t0 = time.perf_counter()
        tokens += len(d.step())
        mx.synchronize()
        steps.append((time.perf_counter() - t0) * 1e3)
    rec = {
        "rows": a.rows,
        "mtp": not a.no_mtp,
        "context": a.context,
        "step_ms": round(statistics.median(steps), 1),
        "tok_per_step": round(tokens / a.steps, 2),
        "agg_tps": round(tokens / (sum(steps) / 1e3), 1),
        **{f"{k}_ms": round(sum(v) / a.steps, 1) for k, v in sink.items()},
        "cost_points": {k: round(v, 1) for k, v in sorted(d.cost.points.items())},
        "chain_ms": round(d.chain_ms, 2),
    }
    print(json.dumps(rec), flush=True)
    if a.output:
        with a.output.open("a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
