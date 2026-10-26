"""Summarise COLDTTFT_PROFILE lines of a turn2gap log segment: spans grouped by kind."""

import json
import sys
from pathlib import Path
from collections import defaultdict


def kind(key):
    if key.startswith("gemm:"):
        return "gemm"
    return key.split(":")[0]


def summarise(text):
    out = []
    for line in text.splitlines():
        if "COLDTTFT_PROFILE " not in line:
            continue
        d = json.loads(line.split("COLDTTFT_PROFILE ", 1)[1])
        g = defaultdict(float)
        n = defaultdict(int)
        shapes = {}
        for k, v in d["spans"].items():
            g[kind(k)] += v["exclusive_s"]
            n[kind(k)] += v["count"]
            shapes[k] = v
        out.append(
            dict(
                start=d["start"],
                rows=d["rows"],
                cached=d["cached"],
                wall_s=d["wall_s"],
                attributed_s=d["attributed_s"],
                groups=dict(g),
                counts=dict(n),
                method=d["method"],
                spans=shapes,
                timeline=d.get("timeline"),
            )
        )
    return out


if __name__ == "__main__":
    for path in sys.argv[1:]:
        for s in summarise(Path(path).read_text(errors="replace")):
            print(
                path.split("/")[-1],
                f"start={s['start']} rows={s['rows']} wall={s['wall_s'] * 1e3:.0f}ms "
                f"attr={s['attributed_s'] * 1e3:.0f}ms",
                {k: round(v * 1e3) for k, v in s["groups"].items()},
            )
