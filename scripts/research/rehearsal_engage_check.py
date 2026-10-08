"""Engagement proof for next-turn rehearsal (CPU analysis of two tfbench decode outputs).

Usage: rehearsal_engage_check.py OFF.jsonl ON.jsonl [--out RESULT.json]
Passes only when, for every (ctx, kind) cell of the ON run:
  rehearsals_completed > 0, turn2 cached_tokens > cold prompt_tokens (the reply was cached),
  and every phase's text equals the OFF run's (greedy, identical output).
Exit 1 otherwise (fail closed: an empty or incomplete file is a failure).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path):
    rows = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    if not any(
        r.get("part") == "part_done" and r.get("complete") is True for r in rows
    ):
        raise ValueError(f"{path}: no final complete record")
    return [r for r in rows if r.get("part") == "decode"]


def check(off_rows, on_rows):
    problems, cells = [], {}
    off = {(r["ctx"], r["kind"], r["phase"]): r for r in off_rows}
    on = {(r["ctx"], r["kind"], r["phase"]): r for r in on_rows}
    if not on:
        return ["no decode rows in the ON run"], cells
    for ctx, kind in sorted({(c, k) for c, k, _ in on}):
        cold, t2 = on.get((ctx, kind, "cold")), on.get((ctx, kind, "turn2"))
        if not cold or not t2:
            problems.append(f"{ctx}/{kind}: missing cold or turn2 row")
            continue
        info = {
            "completed": t2.get("rehearsals_completed"),
            "cold_prompt": cold.get("pt"),
            "turn2_cached": t2.get("cached"),
            "turn2_ttft_off": (off.get((ctx, kind, "turn2")) or {}).get("ttft_s"),
            "turn2_ttft_on": t2.get("ttft_s"),
        }
        cells[f"{ctx}/{kind}"] = info
        if not (info["completed"] or 0) > 0:
            problems.append(f"{ctx}/{kind}: completed rehearsals = {info['completed']}")
        if not (info["turn2_cached"] or 0) > (info["cold_prompt"] or 0):
            problems.append(
                f"{ctx}/{kind}: turn2 cached {info['turn2_cached']} <= cold prompt {info['cold_prompt']}"
            )
        for phase in ("cold", "warm", "turn2"):
            a, b = off.get((ctx, kind, phase)), on.get((ctx, kind, phase))
            if a is None or b is None or a.get("text") != b.get("text"):
                problems.append(f"{ctx}/{kind}/{phase}: output differs or missing")
    return problems, cells


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("off")
    ap.add_argument("on")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    try:
        problems, cells = check(load(a.off), load(a.on))
    except (OSError, ValueError) as exc:
        problems, cells = [str(exc)], {}
    result = {"pass": not problems, "problems": problems, "cells": cells}
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
