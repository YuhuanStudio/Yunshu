"""CPU-only summary of completed RAM-hot ANE interference receipts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from anenative_tfbench import rate_in_window


def summarize(directory: Path):
    if (
        json.loads((directory / "complete.json").read_text()).get("complete")
        is not True
    ):
        raise ValueError("incomplete interference run")
    rows = [
        json.loads(line)
        for line in (directory / "decode.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not rows or rows[-1].get("complete") is not True:
        raise ValueError("incomplete decode receipt")
    workers = {
        backend: json.loads((directory / f"{backend}.json").read_text())
        for backend in ("native", "coreml")
    }
    if any(w.get("complete") is not True for w in workers.values()):
        raise ValueError("incomplete background worker")
    output = []
    for kind in ("prose", "code"):
        cells = [
            r for r in rows if r.get("stage") == "decode" and r.get("kind") == kind
        ]
        if len(cells) != 10 or len({r.get("arm") for r in cells}) != 10:
            raise ValueError("expected ten distinct interleaved arms per kind")
        digests = {r.get("sha") for r in cells}
        cached = {r.get("cached") for r in cells}
        if (
            len(digests) != 1
            or not next(iter(digests))
            or len(cached) != 1
            or not isinstance(next(iter(cached)), int)
            or next(iter(cached)) <= 0
        ):
            raise ValueError("digest or RAM cache differs across arms")
        for r in cells:
            if (
                r.get("ct") != 512
                or r.get("finish") != "length"
                or r.get("xy", {}).get("speculative", {}).get("mode") != "dflash"
            ):
                raise ValueError("incomplete or unengaged cell")
            if (
                not isinstance(r.get("dec_tps"), (float, int))
                or not math.isfinite(r["dec_tps"])
                or r["dec_tps"] <= 0
            ):
                raise ValueError("invalid decode rate")
        base = statistics.median(r["dec_tps"] for r in cells if r["backend"] == "off")
        for backend in ("native", "coreml"):
            arms = [r for r in cells if r["backend"] == backend]
            if len(arms) != 3:
                raise ValueError("expected three background repetitions")
            worker = workers[backend]
            rates = []
            for r in arms:
                start, end = r["_first_at_s"], r["_end_at_s"]
                if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                    raise ValueError("invalid decode window")
                samples = worker["rate_samples"]
                coverage = sum(
                    max(0.0, min(end, s["end_s"]) - max(start, s["start_s"]))
                    for s in samples
                    if s.get("active") is True
                )
                if coverage < (end - start) * 0.99:
                    raise ValueError("background load missing in decode window")
                rate = rate_in_window(samples, start, end)
                if rate is None or rate <= 0:
                    raise ValueError("no background progress during decode")
                rates.append(rate * worker["rows"])
            median = statistics.median(r["dec_tps"] for r in arms)
            item = {
                "kind": kind,
                "backend": backend,
                "cached": next(iter(cached)),
                "base_dec_tps": base,
                "background_dec_tps": median,
                "decode_slowdown_percent": (1 - median / base) * 100,
                "background_rows_per_second": statistics.median(rates),
                "whole_mlp": worker["whole_mlp"],
            }
            if worker["whole_mlp"]:
                item["mlp_only_model64_tps_upper_bound"] = (
                    item["background_rows_per_second"] / 64
                )
            output.append(item)
    return {
        "complete": True,
        "device": "m5",
        "scope": "RAM-hot decode; no TTFT claim; whole-layer prefill is not measured. Check gpuq contention before accepting timings.",
        "cells": output,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.write_text(json.dumps(summarize(a.directory), indent=2) + "\n")


if __name__ == "__main__":
    main()
