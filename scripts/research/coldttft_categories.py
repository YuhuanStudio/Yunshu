"""CPU-only: group analyzed spans into categories; print seconds, floor and gap."""

import collections
import json
import sys
from pathlib import Path

a = json.loads(Path(sys.argv[1]).read_text())["profile"]
cat = collections.defaultdict(lambda: [0.0, 0.0, 0])
for s in a["spans"]:
    n = s["name"]
    if n.startswith("gemm:"):
        k = "gemm:" + (
            "lm_head"
            if ":N248320" in n
            else "N<=64 tiny"
            if int(n.split(":N")[1].split(":")[0]) <= 64
            else "projection"
        )
    elif n.startswith("attention"):
        k = "attention"
    elif n.startswith("gdn.core"):
        k = "gdn.core"
    elif n.startswith(("norm", "elementwise", "copy", "gdn.conv", "gdn.gated")):
        k = n.split(".")[0] + "." + n.split(".")[1] if n.startswith("gdn") else n
    else:
        k = "module remainder (" + n.split(".")[0] + ")"
    cat[k][0] += s["seconds"]
    cat[k][1] += s.get("arithmetic_floor_s", 0)
    cat[k][2] += s["count"]
tot = sum(v[0] for v in cat.values())
for k, v in sorted(cat.items(), key=lambda p: -p[1][0]):
    print(
        f"{k:40} {v[0]:7.3f}s {100 * v[0] / tot:5.1f}% floor65TF={v[1]:7.3f}s n={v[2]}"
    )
print("sum", round(tot, 3))
