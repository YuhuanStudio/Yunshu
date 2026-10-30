"""Compare two media_identity.py results on their ``wide`` prompts (main stock path vs the driver).

Two views of the same data:

* flips: for every prompt, identical text or the first diverging token and the two arms' top-1 vs top-2
  logprob gap at that step. A flip with a small gap in BOTH arms is a near-tie; a flip with a large
  gap in either arm is a suspect. Exit 1 on any suspect or when more than ``--max-flip-fraction`` of the
  prompts diverge.
* paired teacher-forced measure: up to the first divergence both arms were fed the SAME tokens, so each
  such position is a paired observation of the forced token's logprob. Reported: mean / median / p90 /
  p99 / max |dlogprob|, how many are exactly zero, top-1 agreement, and the first generated token
  (prefill only) apart from the rest.

    python media_wide_compare.py base.json cand.json [--key wide|wide_text]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


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


def paired(base: list, cand: list) -> dict:
    """|dlogprob| of the forced token at every position where both arms held the same context."""
    deltas: list[float] = []
    first: list[float] = []
    agree = total = 0
    for b, c in zip(base, cand, strict=True):
        for i, (x, y) in enumerate(zip(b["steps"], c["steps"], strict=False)):
            total += 1
            if x["token"] != y["token"]:
                break
            agree += 1
            d = abs(x["logprob"] - y["logprob"])
            deltas.append(d)
            if i == 0:
                first.append(d)
    deltas.sort()

    def q(p: float) -> float:
        return deltas[min(len(deltas) - 1, int(p * len(deltas)))] if deltas else 0.0

    return {
        "positions": len(deltas),
        "compared": total,
        "top1_agreement": agree / total if total else 1.0,
        "mean_abs_dlogprob": statistics.mean(deltas) if deltas else 0.0,
        "median": statistics.median(deltas) if deltas else 0.0,
        "p90": q(0.9),
        "p99": q(0.99),
        "max": deltas[-1] if deltas else 0.0,
        "exactly_zero": sum(1 for d in deltas if d == 0),
        "first_token_n": len(first),
        "first_token_nonzero": sum(1 for d in first if d != 0),
        "first_token_mean": statistics.mean(first) if first else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("cand")
    ap.add_argument("--key", default="wide")
    ap.add_argument("--tie-gap", type=float, default=0.5)
    ap.add_argument("--max-flip-fraction", type=float, default=0.25)
    a = ap.parse_args()
    base = json.loads(Path(a.base).read_text())[a.key]
    cand = json.loads(Path(a.cand).read_text())[a.key]
    out = compare(base, cand, a.tie_gap)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    print(json.dumps(paired(base, cand), indent=1))
    for r in out["rows"]:
        if not r["equal"]:
            print(json.dumps(r))
    if out["suspects"] or out["flips"] > a.max_flip_fraction * out["prompts"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
