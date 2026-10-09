"""Compare decode JSONL of a spec-on and a spec-off run: same text per cell, and spec engaged.

spec_identity_compare.py ON.jsonl OFF.jsonl [...pairs]   exits nonzero on any mismatch
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def cells(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        d = json.loads(line)
        if d.get("part") != "decode":
            continue
        spec = ((d.get("xy") or {}).get("speculative")) or {}
        out[(d["ctx"], d["kind"], d["phase"])] = {
            "text": d["text"],
            "drafted": int(spec.get("drafted") or 0),
            "rounds": int(spec.get("rounds") or 0),
        }
    return out


def compare(on: dict, off: dict) -> list[str]:
    problems = []
    if not on or set(on) != set(off):
        return [f"cell sets differ or empty: {sorted(on)} vs {sorted(off)}"]
    for key in sorted(on):
        if on[key]["text"] != off[key]["text"]:
            a, b = on[key]["text"], off[key]["text"]
            i = next(
                (k for k, (x, y) in enumerate(zip(a, b, strict=False)) if x != y),
                min(len(a), len(b)),
            )
            problems.append(f"{key}: texts differ at char {i}")
        if on[key]["drafted"] <= 0:
            problems.append(f"{key}: speculative lane never engaged (drafted=0)")
    for ctx, kind in sorted({(c, k) for c, k, _ in on}):
        cold, warm = on.get((ctx, kind, "cold")), on.get((ctx, kind, "warm"))
        if cold and warm and cold["text"] != warm["text"]:
            problems.append(f"{(ctx, kind)}: APC hit (warm) differs from miss (cold)")
    return problems


def main(argv: list[str]) -> int:
    bad = 0
    for on_path, off_path in zip(argv[0::2], argv[1::2], strict=True):
        problems = compare(cells(Path(on_path)), cells(Path(off_path)))
        print(on_path, "OK" if not problems else problems)
        bad += bool(problems)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
