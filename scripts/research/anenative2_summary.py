"""Median cold TTFT per arm and context of the anenative2 matrix, with the first tokens' identity."""

import glob
import json
import statistics
import sys
from collections import defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/P5Plus/yunshu-build/anenative2/gate"
v = defaultdict(list)
heads = defaultdict(set)
cont = 0
for f in sorted(glob.glob(d + "/*.jsonl")):
    arm, ctx, _ = f.rsplit("/", 1)[1][:-6].split("-")
    for line in open(f):
        r = json.loads(line)
        if r.get("part") == "decode" and r.get("phase") == "cold":
            v[(int(ctx), arm)].append(r["ttft_s"])
            heads[(int(ctx), arm)].add(r["sha"])
            cont += bool(r.get("contended"))
for ctx in sorted({k[0] for k in v}):
    base = statistics.median(v[(ctx, "off")]) if v.get((ctx, "off")) else None
    row = f"{ctx:>6}"
    for arm in ("off", "cal", "fix"):
        x = v.get((ctx, arm))
        if x:
            m = statistics.median(x)
            pct = f" ({100 * (m / base - 1):+.1f}%)" if base and arm != "off" else ""
            row += f" | {arm} {m:.3f}s{pct} n={len(x)} same={len(heads[(ctx, arm)]) == 1}"
    print(row)
print("contended records:", cont)
