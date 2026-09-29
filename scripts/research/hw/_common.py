"""Shared helpers for the M5 Max hardware probes (JSONL output, timing)."""

import json
import statistics
import time
from pathlib import Path

RUNS = Path(__file__).resolve().parents[3] / "docs/research/runs/2026-09-29-m5max-hw"


class Out:
    def __init__(self, name):
        RUNS.mkdir(parents=True, exist_ok=True)
        self.f = open(RUNS / f"{name}.jsonl", "a")  # noqa: SIM115

    def __call__(self, **row):
        line = json.dumps(row)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def timeit(fn, iters=20, warmup=3):
    """Median and min wall time (s) of fn(); fn must block until done."""
    for _ in range(warmup):
        fn()
    s = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        s.append(time.perf_counter() - t0)
    return statistics.median(s), min(s)
