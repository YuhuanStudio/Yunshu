"""Summarize only complete measured cells; retain all raw repetitions."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in args.input.read_text().splitlines() if l.strip()]
    groups = defaultdict(list)
    for row in rows:
        if "profile" in row and "case" in row:
            groups[row["profile"], row["size"], row["case"]].append(row)
    table = []
    for (profile, size, case), items in sorted(groups.items()):
        entry = {
            "profile": profile,
            "context_target": size,
            "case": case,
            "reps": len(items),
            "eligible_reps": 0,
            "medians": {},
            "raw": items,
        }
        for field in [
            "ttft_s",
            "decode_tps",
            "wall_s",
            "aggregate_decode_tps",
            "aggregate_e2e_tps",
            "peak_gib",
            "peak_process_tree_rss_bytes",
            "ready_s",
        ]:
            values = [r[field] for r in items if isinstance(r.get(field), (int, float))]
            if len(values) >= 3:
                entry["medians"][field] = statistics.median(values)
        if case in ("cold", "warm", "turn2"):
            entry["eligible_reps"] = sum(
                bool(r.get("done"))
                and r.get("usage", {}).get("completion_tokens", 0) >= 64
                for r in items
            )
            if entry["eligible_reps"] < 3:
                entry["medians"].pop("decode_tps", None)
        table.append(entry)
    result = {
        "complete": any(r.get("complete") and not r.get("dry_run") for r in rows),
        "failures": sum("error" in r for r in rows),
        "cells": table,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "profile | context | case | reps | median TTFT ms | decode tok/s | aggregate decode | aggregate e2e | peak GiB"
    )
    for e in table:
        m = e["medians"]
        if e["case"] not in (
            "cold",
            "warm",
            "turn2",
            "concurrent8",
            "physical_memory",
            "startup",
        ):
            continue

        def num(key, scale=1):
            return f"{m[key] * scale:.3f}" if key in m else "UNMEASURED"

        print(
            " | ".join(
                [
                    e["profile"],
                    str(e["context_target"]),
                    e["case"],
                    str(e["reps"]),
                    num("ttft_s", 1000),
                    num("decode_tps"),
                    num("aggregate_decode_tps"),
                    num("aggregate_e2e_tps"),
                    num("peak_gib"),
                ]
            )
        )


if __name__ == "__main__":
    main()
