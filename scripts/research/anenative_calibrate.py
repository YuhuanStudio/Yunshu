"""Calibrate (or recall) the Splash-style ANE FFN split on the real 27B layers, like server start.

Writes the remembered calibration into --cache (the same file the engine recalls) and a JSON
receipt with the timings Splash's fit uses.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from anenative_splash_micro import load_layers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp")
    p.add_argument("--layers", type=int, default=64)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--share", type=float, default=None)
    p.add_argument("--min-rows", type=int, default=None)
    a = p.parse_args()

    from yunshu_engine.ane import ffn_startup
    from yunshu_engine.kernels import nax_prefill

    if not nax_prefill.enable():
        raise SystemExit("NAX prefill unavailable")
    t0 = time.perf_counter()
    layers = load_layers(a.model, 0, a.layers, True)
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    outcome = ffn_startup.start(
        layers, a.cache, a.model, given_share=a.share, given_minimum_rows=a.min_rows
    )
    record = {
        "kind": outcome.kind,
        "reason": outcome.reason,
        "load_s": load_s,
        "start_s": time.perf_counter() - t0,
        "detail": outcome.detail,
    }
    if outcome.split is not None:
        record["share"] = outcome.split.share
        record["minimum_rows"] = outcome.split.minimum_rows
        record["resident_extra_mib"] = None
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record)[:1500], flush=True)
    if outcome.kind not in ("split", "no_gain"):
        raise SystemExit(f"split unavailable: {outcome.reason}")
    print(json.dumps({"complete": True}))


if __name__ == "__main__":
    main()
