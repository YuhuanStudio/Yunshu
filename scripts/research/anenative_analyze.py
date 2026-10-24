"""CPU-only summary of complete native/CoreML receipts; no unmeasured speed claims."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from anenative_micro import economics


def measured_curve(gpu_ms, points, total_channels=17408, serialize_copies=False):
    """Interpolate measured costs only; GPU remainder is an optimistic linear model."""
    points = sorted(points)
    if (
        gpu_ms <= 0
        or not points
        or any(
            n <= 0 or n > total_channels or ms < 0 or copies < 0
            for n, ms, copies in points
        )
    ):
        raise ValueError("invalid measured cost curve")
    if len({p[0] for p in points}) != len(points):
        raise ValueError("duplicate channel cost")
    best = {
        "channels": 0,
        "share": 0.0,
        "split_ms": gpu_ms,
        "saving_ms": 0.0,
        "offload_ms": 0.0,
        "host_copy_ms": 0.0,
    }
    for count in range(64, points[-1][0] + 1, 64):
        if count < points[0][0]:
            continue
        for index, (n, ms, copies) in enumerate(points):
            if n >= count:
                if n == count or index == 0:
                    ane, host = ms, copies
                else:
                    lo, low_ms, low_copy = points[index - 1]
                    weight = (count - lo) / (n - lo)
                    ane = low_ms + (ms - low_ms) * weight
                    host = low_copy + (copies - low_copy) * weight
                break
        split = max(gpu_ms * (1 - count / total_channels), ane) + (
            host if serialize_copies else 0.0
        )
        if split < best["split_ms"]:
            best = {
                "channels": count,
                "share": count / total_channels,
                "split_ms": split,
                "saving_ms": gpu_ms - split,
                "offload_ms": ane,
                "host_copy_ms": host,
            }
    return best


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
                    "descriptive_linear_extrapolation_not_decision": estimate,
                    "fit_max_residual_ms": max(
                        abs(b - intercept - slope * a)
                        for a, b in zip(x, y, strict=True)
                    ),
                }
            )
    curves = []
    for count, mode in sorted({(r["rows"], r["mode"]) for r in output}):
        series = [r for r in output if r["rows"] == count and r["mode"] == mode]
        for backend in ("native", "coreml"):
            if not all(backend + "_ms" in r for r in series):
                continue
            points = [
                (
                    r["channels"],
                    r[backend + "_ms"],
                    r.get("copy_ms", 0.0) if backend == "native" else 0.0,
                )
                for r in series
            ]
            item = {
                "rows": count,
                "mode": mode,
                "backend": backend,
                "gpu_ffn_ms": gpu[count],
                "scope": "Measured-span interpolation; assumes GPU remainder scales linearly. Excludes handoff/join and measured hardware contention.",
                "no_extra_copies": measured_curve(gpu[count], points),
                "serialized_host_copies": measured_curve(
                    gpu[count], points, serialize_copies=True
                ),
            }
            full = next((r for r in series if r["channels"] == 17408), None)
            if full:
                item["model64_mlp_only_tps_if_layer40_representative"] = count / (
                    64 * full[backend + "_ms"] * 0.001
                )
            curves.append(item)
    return {
        "cells": output,
        "fits": fits,
        "measured_curve_economics": curves,
        "complete": True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.write_text(json.dumps(summarize(a.receipt), indent=2) + "\n")


if __name__ == "__main__":
    main()
