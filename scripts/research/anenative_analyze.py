"""CPU-only summary of complete native/CoreML receipts; no unmeasured speed claims."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from anenative_micro import economics


def summarize(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or rows[-1].get("complete") is not True:
        raise ValueError("incomplete native ANE receipt")
    gpu = {r["rows"]: r["median_ms"] for r in rows if r.get("kind") == "gpu_ffn"}
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("kind") != "slice":
            continue
        if r.get("cached") is not False:
            raise ValueError("unexpected cached timing row")
        key = (r["rows"], r["mode"], r["channels"])
        backend = "native" if r["arm"].startswith("cand") else "coreml"
        cells[key][backend].append(r)
    output = []
    for (count, mode, channels), arms in sorted(cells.items()):
        out = {"rows": count, "mode": mode, "channels": channels}
        for backend, records in arms.items():
            out[backend + "_ms"] = statistics.median(
                v for r in records for v in r["ms"]
            )
            if backend == "native":
                out["copy_ms"] = statistics.median(
                    v for r in records for v in r["copy_ms"]
                )
            out[backend + "_rel_l2"] = max(r["rel_l2_fp32"] for r in records)
        output.append(out)
    fits = []
    for count, mode in sorted({(r["rows"], r["mode"]) for r in output}):
        series = [r for r in output if r["rows"] == count and r["mode"] == mode]
        if len(series) < 2:
            continue
        x = [r["channels"] for r in series]
        for backend in ("native", "coreml"):
            if not all(backend + "_ms" in r for r in series):
                continue
            y = [r[backend + "_ms"] for r in series]
            xm, ym = statistics.mean(x), statistics.mean(y)
            slope = sum((a - xm) * (b - ym) for a, b in zip(x, y, strict=True)) / sum(
                (a - xm) ** 2 for a in x
            )
            intercept = ym - slope * xm
            estimate = economics(gpu[count], max(0.0, intercept), max(0.0, slope))
            fits.append(
                {
                    "rows": count,
                    "mode": mode,
                    "backend": backend,
                    "floor_ms": intercept,
                    "ms_per_1000_channels": slope * 1000,
                    "gpu_ffn_ms": gpu[count],
                    "optimistic_no_copy_contention": estimate,
                    "fit_max_residual_ms": max(
                        abs(b - intercept - slope * a)
                        for a, b in zip(x, y, strict=True)
                    ),
                }
            )
    return {"cells": output, "fits": fits, "complete": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.write_text(json.dumps(summarize(a.receipt), indent=2) + "\n")


if __name__ == "__main__":
    main()
