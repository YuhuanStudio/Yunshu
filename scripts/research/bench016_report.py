"""Build private long-context tables from independently revalidated gpuq receipts."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from types import SimpleNamespace

from scripts.research import bench016 as b

DEPTHS = (32768, 65536, 131072)
METRICS = (
    "cold_ttft_s",
    "decode_tps",
    "warm_ttft_s",
    "followup_ttft_s",
    "cached_tokens",
    "peak_gib",
)


def metrics(rows: list[dict]) -> dict:
    phases = {r["phase"]: r for r in rows if r.get("part") == "decode"}
    memory = [r for r in rows if r.get("part") == "memory"]
    values = dict(
        zip(
            METRICS,
            (
                phases["cold"]["ttft_s"],
                phases["cold"]["dec_tps"],
                phases["warm"]["ttft_s"],
                phases["turn2"]["ttft_s"],
                phases["warm"]["cached"],
                max(r["peak_gib"] for r in memory),
            ),
            strict=True,
        )
    )
    if any(
        not isinstance(v, int | float) or not math.isfinite(v) or v < 0
        for key, v in values.items()
        if key != "cached_tokens" or v is not None
    ):
        raise ValueError("missing or invalid metric")
    return values


def summarize(samples: dict[int, dict]) -> dict:
    """Repetition IDs, rather than attempts, determine sample size."""
    if set(samples) != {0, 1, 2}:
        return {"status": "incomplete", "reps": sorted(samples)}
    return {
        "status": "qualified",
        "reps": sorted(samples),
        "metrics": {
            key: None
            if any(s[key] is None for s in samples.values())
            else {
                "median": statistics.median(s[key] for s in samples.values()),
                "min": min(s[key] for s in samples.values()),
                "max": max(s[key] for s in samples.values()),
            }
            for key in METRICS
        },
    }


def collect(manifest: dict, state: dict, queue: Path) -> dict:
    samples = {}
    rejected = {}
    recovered = {}
    for name, entry in state.items():
        match = re.fullmatch(r"(.+)-L(32768|65536|131072)-r([0-2])-try\d+", name)
        if not match:
            continue
        engine, depth, rep = match[1], int(match[2]), int(match[3])
        try:
            receipt = json.loads((queue / "jobs" / f"{entry['id']}.json").read_text())
            rows = b.bs.read_rows(Path(entry["out"]))
            job = SimpleNamespace(
                engine=engine,
                ctxs=(depth,),
                kinds=("code",),
                stage="cell",
                part="decode",
                env={"TFB_EXPECT_YUNSHU_SHA": manifest["yunshu_sha"]},
            )
            log_path = queue / "logs" / f"{entry['id']}.log"
            receipt, recovery = b.revalidate_memory_receipt(
                receipt,
                log_path.read_text(errors="replace") if log_path.exists() else "",
            )
            errors = b.verdict(job, rows, receipt)
            if receipt.get("state") != "done":
                errors.append("receipt not done")
            if errors:
                rejected[name] = errors
                continue
            value = metrics(rows)
            if recovery:
                recovered[name] = {"id": entry["id"], **recovery}
            bucket = samples.setdefault((engine, depth), {})
            # First valid attempt is selected; never select the fastest rerun.
            bucket.setdefault(rep, value)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            rejected[name] = [str(exc)]
    cells = [
        {
            "engine": engine,
            "depth": depth,
            **summarize(samples.get((engine, depth), {})),
        }
        for depth in DEPTHS
        for engine in manifest["order"]
    ]
    return {
        "yunshu_sha": manifest["yunshu_sha"],
        "cells": cells,
        "rejected": rejected,
        "recovered": recovered,
        "complete": all(c["status"] == "qualified" for c in cells),
    }


def markdown(report: dict) -> str:
    lines = [
        "# bench016 provisional evidence",
        "",
        "Incomplete cells are not standings. Code prompt; cold decode at depth; median [min,max].",
        "Different checkpoints remain different-weight observations; no overall winner is inferred.",
        "",
        "| Depth | Engine | Reps | Cold TTFT s | Decode tok/s | Warm TTFT s | Follow-up TTFT s | Cached tokens | Peak GiB |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in report["cells"]:
        vals = (
            [
                f"{m['median']} [{m['min']},{m['max']}]"
                if m is not None
                else "not reported"
                for m in cell["metrics"].values()
            ]
            if cell["status"] == "qualified"
            else ["incomplete"] * len(METRICS)
        )
        lines.append(
            f"| {cell['depth']} | {cell['engine']} | {len(cell['reps'])}/3 | "
            + " | ".join(vals)
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    b.configure(manifest)
    state = json.loads(args.manifest.with_name("state.json").read_text())
    report = collect(manifest, state, b.QUEUE)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(markdown(report))


if __name__ == "__main__":
    main()
