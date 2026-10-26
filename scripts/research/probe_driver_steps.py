"""Round driver per-step cost: wall, packed tokens and kind of every step.

One row of N prompt tokens through ``RoundDriver`` to ``--tokens`` outputs,
with and without the MTP head; every step is synchronized and logged
(prefill / decode, tokens, ms, process footprint). Prints a per-context
summary: prefill tok/s, median decode step, MTP absorb / draft share.

    python scripts/research/probe_driver_steps.py <ckpt> --lengths 4096 8192 16384
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep_round_driver import load  # noqa: E402


def footprint_gib() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**30
    except ImportError:  # pragma: no cover
        return -1.0


def run(model, drafter, ids, tokens, log):
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    d = RoundDriver(model, drafter=drafter)
    d.add(Request(ids, tokens, handle=0))
    row = d.rows[0]
    steps = []
    while d.busy():
        kind = "prefill" if row.pending is None else "decode"
        mx.synchronize()
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        d.step()
        mx.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        rec = {
            "kind": kind,
            "ms": round(ms, 1),
            "active_gib": round(mx.get_active_memory() / 2**30, 2),
            "peak_gib": round(mx.get_peak_memory() / 2**30, 2),
            "rss_gib": round(footprint_gib(), 2),
        }
        steps.append(rec)
        log(rec)
    return steps


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("ckpt")
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--lengths", type=int, nargs="*", default=[4096, 8192, 16384])
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--steps", action="store_true", help="log every step")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    model, _, drafter, lanes = load(a.ckpt, a.quantize)
    out = a.output.open("a") if a.output else None

    def put(obj):
        line = json.dumps(obj)
        print(line, flush=True)
        if out:
            out.write(line + "\n")
            out.flush()

    put({"kind": "meta", "ckpt": a.ckpt, "lanes": lanes["converted"]})
    rng = random.Random(0)
    for n in a.lengths:
        ids = [rng.randrange(1000, 20000) for _ in range(n)]
        for dr in [None] if a.no_mtp else [None, drafter]:
            steps = run(
                model,
                dr,
                ids,
                a.tokens,
                (lambda r, n=n, m=dr is not None: put({"step": n, "mtp": m, **r}))
                if a.steps
                else (lambda r: None),
            )
            pre = [s["ms"] for s in steps if s["kind"] == "prefill"]
            dec = [s["ms"] for s in steps if s["kind"] == "decode"]
            put(
                {
                    "kind": "summary",
                    "tokens": n,
                    "mtp": dr is not None,
                    "prefill_steps": len(pre),
                    "prefill_s": round(sum(pre) / 1e3, 2),
                    "prefill_tok_s": round(n / (sum(pre) / 1e3), 1),
                    "prefill_step_ms": [round(x) for x in pre],
                    "decode_steps": len(dec),
                    "decode_median_ms": round(statistics.median(dec), 1)
                    if dec
                    else None,
                    "decode_max_ms": max(dec) if dec else None,
                    "peak_active_gib": max(s["active_gib"] for s in steps),
                    "peak_step_gib": max(s["peak_gib"] for s in steps),
                    "peak_rss_gib": max(s["rss_gib"] for s in steps),
                }
            )


if __name__ == "__main__":
    main()
