"""Round driver with several long prompts at once: time per step kind.

Adds ``--rows`` prompts of ``--context`` tokens together and runs them to
``--tokens`` outputs, every step synchronized and timed. Prints when each
row's first token arrived and, per step kind (prefill / decode), the count and
total and median milliseconds — where the wall time of N long prompts goes.

    python scripts/research/probe_driver_multi.py <ckpt> --rows 4 --context 32768
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--steps", action="store_true", help="log every step")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    model, processor, drafter, _ = load(a.ckpt, a.quantize)
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    tok = processor.tokenizer
    prompts = [encode(tok, PROMPTS[i % len(PROMPTS)], a.context) for i in range(a.rows)]
    d = RoundDriver(model, drafter=None if a.no_mtp else drafter, stop_tokens=set())
    for i, p in enumerate(prompts):
        d.add(Request(p, a.tokens, handle=i))
    first, kinds = {}, {"prefill": [], "decode": []}
    t0 = time.perf_counter()
    while d.busy():
        kind = "decode" if d._will_decode() else "prefill"
        mx.synchronize()
        s = time.perf_counter()
        events = d.step()
        mx.synchronize()
        ms = (time.perf_counter() - s) * 1e3
        kinds[kind].append(ms)
        for e in events:
            first.setdefault(e.handle, time.perf_counter() - t0)
        if a.steps:
            print(
                json.dumps(
                    {
                        "kind": kind,
                        "ms": round(ms, 1),
                        "events": len(events),
                        "decoding": len(d.batch.rows),
                        "waiting": sum(r.pending is None for r in d.rows),
                        "active_gib": round(mx.get_active_memory() / 2**30, 1),
                        "cache_gib": round(mx.get_cache_memory() / 2**30, 1),
                    }
                )
            )
    wall = time.perf_counter() - t0
    rec = {
        "rows": a.rows,
        "context": a.context,
        "mtp": not a.no_mtp,
        "wall_s": round(wall, 1),
        "first_token_s": [round(first[i], 1) for i in sorted(first)],
        **{
            k: {
                "steps": len(v),
                "total_s": round(sum(v) / 1e3, 1),
                "median_ms": round(statistics.median(v), 1) if v else None,
            }
            for k, v in kinds.items()
        },
        "peak_gib": round(mx.get_peak_memory() / 2**30, 1),
    }
    print(json.dumps(rec), flush=True)
    if a.output:
        with a.output.open("a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
