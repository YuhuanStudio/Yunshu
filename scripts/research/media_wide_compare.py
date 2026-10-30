"""Compare two media_identity.py results on their ``wide`` prompts (main stock path vs the driver).

For every prompt: identical text, or the first diverging token and the two arms' top-1 vs top-2 logprob
gap at that step. A flip with a small gap in BOTH arms is a near-tie (the two arithmetics rank two
almost equal tokens differently); a flip with a large gap in either arm is not and is reported as a
suspect. Exit 1 when any suspect exists or more than ``--max-flip-fraction`` of the prompts diverge.

    python scripts/research/media_wide_compare.py base.json cand.json
"""

from __future__ import annotations

import argparse
import json
import sys


def first_divergence(a: list, b: list) -> int | None:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x["token"] != y["token"]:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(base: list, cand: list, tie_gap: float = 0.5) -> dict:
    rows, suspects = [], []
    for n, (x, y) in enumerate(zip(base, cand, strict=True)):
        i = first_divergence(x["steps"], y["steps"])
        if i is None:
            rows.append({"prompt": n, "equal": True})
            continue
        gx = x["steps"][i]["gap"] if i < len(x["steps"]) else None
        gy = y["steps"][i]["gap"] if i < len(y["steps"]) else None
        near = gx is not None and gy is not None and gx < tie_gap and gy < tie_gap
        row = {
            "prompt": n,
            "equal": False,
            "at": i,
            "gap_base": gx,
            "gap_cand": gy,
            "near_tie": near,
        }
        rows.append(row)
        if not near:
            suspects.append(row)
    flips = [r for r in rows if not r["equal"]]
    return {
        "prompts": len(rows),
        "flips": len(flips),
        "suspects": suspects,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("cand")
    ap.add_argument("--tie-gap", type=float, default=0.5)
    ap.add_argument("--max-flip-fraction", type=float, default=0.25)
    a = ap.parse_args()
    base = json.load(open(a.base))["wide"]
    cand = json.load(open(a.cand))["wide"]
    out = compare(base, cand, a.tie_gap)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    for r in out["rows"]:
        if not r["equal"]:
            print(json.dumps(r))
    if out["suspects"] or out["flips"] > a.max_flip_fraction * out["prompts"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
