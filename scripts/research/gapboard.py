#!/usr/bin/env python3
"""Aggregate cross-engine cell jsonl files into a ratio board (CPU only)."""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load(out):
    g = defaultdict(dict)
    for path in sorted(Path(out).glob("*.jsonl")):
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        if any(r.get("part") == "failure" for r in rows):
            continue
        if not any(r.get("part") == "part_done" and r.get("complete") for r in rows):
            continue
        for r in rows:
            if r.get("part") != "decode":
                continue
            cell = (r["ctx"], r["kind"], r["phase"])
            rep = r["snapshot_rep"]
            g[(r["engine"], cell)][rep] = (
                r.get("dec_tps"),
                r["ttft_s"],
                r.get("cached"),
            )
    return g


def med(xs):
    xs = [float(x) for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def board(out):
    g = load(out)
    engines = sorted({k[0] for k in g})
    cells = sorted({k[1] for k in g}, key=lambda c: (c[2], c[0], c[1]))
    lines = []
    for metric, idx, higher in (("decode tok/s", 0, True), ("TTFT s", 1, False)):
        lines += [f"\n### {metric} (median of reps; n in brackets)\n"]
        lines.append(
            "| cell | " + " | ".join(engines) + " | best other | Yunshu/best |"
        )
        lines.append("|---|" + "---|" * (len(engines) + 2))
        for c in cells:
            if idx == 0 and c[2] != "cold":
                continue
            vals, ns = {}, {}
            for e in engines:
                reps = g.get((e, c))
                if reps:
                    v = med(x[idx] for x in reps.values())
                    if v is not None:
                        vals[e], ns[e] = v, len(reps)
            others = {e: v for e, v in vals.items() if not e.startswith("yunshu")}
            ys = {e: v for e, v in vals.items() if e.startswith("yunshu")}
            if not others or not ys:
                continue
            pick = max if higher else min
            be = pick(others, key=others.get)
            yv = ys.get("yunshu-main", next(iter(ys.values())))
            ratio = yv / others[be] if higher else others[be] / yv
            flag = "RED" if ratio < 0.97 else ("ok" if ratio < 1.03 else "LEAD")
            row = [f"{c[0]} {c[1]} {c[2]}"]
            for e in engines:
                row.append(f"{vals[e]:.3f} [{ns[e]}]" if e in vals else "-")
            row += [be, f"{ratio:.2f} {flag}"]
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(board(sys.argv[1]))
