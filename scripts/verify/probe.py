"""probe: deterministic engine counters from one short run per arm (no timing reps).

Parses the server log of a tfbench decode run into per-request counters (tokens/round, nodes/round,
rounds, acceptance histogram) plus an aggregate ms/round, joins them with the tfbench rows (decode
tok/s, TTFT, ct) and compares base with candidate. Greedy counters are deterministic, so any move
is flagged with its numbers; ms/round and tok/s move within a tolerance. Pure functions only (the
stage in stages.py runs the servers); no counter line = no evidence = error, never a pass."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROW_DRIVER = re.compile(
    r"Round driver (\w+) row: (\d+) rounds, ([\d.]+) tokens/round, ([\d.]+) nodes/round"
    r"(?:, (\d+) copy rounds)?(?:.*?accepted-length histogram (\[[\d, ]*\]))?"
)
WINDOW_DRIVER = re.compile(
    r"Round driver (\w+) window of (\d+) rounds \(total \d+\): .*?([\d.]+) ms/round"
)
PROFILE_OLD = re.compile(
    r"(\w+) round profile: (\d+) rounds, ([\d.]+) nodes/round, draft ([\d.]+) ms, "
    r"verify ([\d.]+) ms, commit\+host ([\d.]+) ms"
)

# defaults; every one overridable through the suite config
TOL = {
    "tpr_pct": 0.1,  # same log format: tokens/round (greedy-deterministic)
    "tpr_pct_cross": 3.0,  # base and cand print different formats (old lane vs driver)
    "nodes_abs": 0.1,  # nodes/round is printed with one decimal
    "ms_pct": 3.0,
    "tps_pct": 5.0,  # one rep: a loose timing flag only
}


def parse_log(text: str) -> dict:
    """{"fmt", "requests": [{rounds, tpr|None, nodes, hist|None, ms|None}], "windows": [(rounds, ms)]}.
    Raises ValueError when no counter line exists."""
    reqs, windows, fmts = [], [], set()
    for line in text.splitlines():
        m = ROW_DRIVER.search(line)
        if m:
            fmts.add("driver")
            reqs.append(
                {
                    "rounds": int(m[2]),
                    "tpr": float(m[3]),
                    "nodes": float(m[4]),
                    "hist": json.loads(m[6]) if m[6] else None,
                    "ms": None,
                }
            )
            continue
        m = WINDOW_DRIVER.search(line)
        if m:
            windows.append((int(m[2]), float(m[3])))
            continue
        m = PROFILE_OLD.search(line)
        if m:
            fmts.add("lane")
            reqs.append(
                {
                    "rounds": int(m[2]),
                    "tpr": None,  # filled from the tfbench row's token count
                    "nodes": float(m[3]),
                    "hist": None,
                    "ms": float(m[4]) + float(m[5]) + float(m[6]),
                    "verify_ms": float(m[5]),
                }
            )
    if not reqs:
        raise ValueError(
            "no round counter line in the server log (driver row / round profile)"
        )
    if len(fmts) > 1:
        raise ValueError(f"mixed counter formats in one log: {sorted(fmts)}")
    return {"fmt": fmts.pop(), "requests": reqs, "windows": windows}


def join_rows(parsed: dict, rows: list) -> list:
    """Per decode row (request order), the counters of its request: the last N counter records
    (earlier ones are warmup). Fails closed when there are fewer records than rows."""
    dec = [r for r in rows if r.get("part") == "decode" and r.get("phase")]
    reqs = parsed["requests"]
    if not dec:
        raise ValueError("no tfbench decode rows")
    if len(reqs) < len(dec):
        raise ValueError(f"{len(reqs)} counter records for {len(dec)} decode requests")
    out = []
    for row, q in zip(dec, reqs[len(reqs) - len(dec) :]):  # noqa: B905
        tpr = q["tpr"]
        if tpr is None:
            tpr = (row.get("ct") or 0) / q["rounds"] if q["rounds"] else None
        out.append(
            {
                "cell": f"{row.get('kind')}@{row.get('ctx')}:{row.get('phase')}",
                "rounds": q["rounds"],
                "tpr": tpr,
                "nodes": q["nodes"],
                "hist": q["hist"],
                "dec_tps": row.get("dec_tps"),
                "ttft_s": row.get("ttft_s"),
            }
        )
    return out


def arm_ms(parsed: dict) -> float | None:
    """Mean ms/round over the run, weighted by rounds."""
    if parsed["fmt"] == "driver":
        ws = parsed["windows"]
        tot = sum(r for r, _ in ws)
        return sum(r * m for r, m in ws) / tot if tot else None
    qs = [q for q in parsed["requests"] if q["ms"] is not None]
    tot = sum(q["rounds"] for q in qs)
    return sum(q["rounds"] * q["ms"] for q in qs) / tot if tot else None


def summarize(text: str, rows: list) -> dict:
    parsed = parse_log(text)
    return {
        "fmt": parsed["fmt"],
        "ms_per_round": arm_ms(parsed),
        "cells": join_rows(parsed, rows),
    }


def _pct(a: float, b: float) -> float:
    return 100.0 * (b - a) / a if a else float("inf")


def compare(base: dict, cand: dict, tol: dict | None = None) -> dict:
    """{"moved": [str], "ok": bool, "numbers": {...}}. `moved` lists every counter outside its
    tolerance with both numbers; empty = nothing moved (PASS without timing reps)."""
    t = {**TOL, **(tol or {})}
    cross = base["fmt"] != cand["fmt"]
    moved, cells = [], []
    bc, cc = base["cells"], cand["cells"]
    if [c["cell"] for c in bc] != [c["cell"] for c in cc]:
        return {
            "ok": False,
            "moved": [
                f"cell lists differ: {[c['cell'] for c in bc]} vs {[c['cell'] for c in cc]}"
            ],
            "numbers": {},
        }
    tpr_tol = t["tpr_pct_cross"] if cross else t["tpr_pct"]
    for b, c in zip(bc, cc):  # noqa: B905
        d = {"cell": b["cell"]}
        for name in ("tpr", "nodes", "rounds"):
            d[name] = [b[name], c[name]]
        if b["tpr"] is None or c["tpr"] is None:
            moved.append(f"{b['cell']}: tokens/round missing")
        elif abs(_pct(b["tpr"], c["tpr"])) > tpr_tol:
            moved.append(f"{b['cell']}: tokens/round {b['tpr']:.2f} -> {c['tpr']:.2f}")
        if abs(c["nodes"] - b["nodes"]) > t["nodes_abs"]:
            moved.append(
                f"{b['cell']}: nodes/round {b['nodes']:.1f} -> {c['nodes']:.1f}"
            )
        if not cross and b["rounds"] != c["rounds"]:
            moved.append(f"{b['cell']}: rounds {b['rounds']} -> {c['rounds']}")
        if b["hist"] and c["hist"] and b["hist"] != c["hist"]:
            moved.append(
                f"{b['cell']}: acceptance histogram {b['hist']} -> {c['hist']}"
            )
        if b["dec_tps"] and c["dec_tps"]:
            dp = _pct(b["dec_tps"], c["dec_tps"])
            d["dec_tps"] = [b["dec_tps"], c["dec_tps"], round(dp, 1)]
            if b["cell"].endswith(":cold") and dp < -t["tps_pct"]:
                moved.append(
                    f"{b['cell']}: decode tok/s {b['dec_tps']:.1f} -> {c['dec_tps']:.1f} ({dp:+.1f}%)"
                )
        if b["ttft_s"] and c["ttft_s"]:
            d["ttft_s"] = [
                b["ttft_s"],
                c["ttft_s"],
                round(_pct(b["ttft_s"], c["ttft_s"]), 1),
            ]
        cells.append(d)
    bm, cm = base["ms_per_round"], cand["ms_per_round"]
    if bm and cm:
        if _pct(bm, cm) > t["ms_pct"]:
            moved.append(
                f"ms/round {bm:.1f} -> {cm:.1f} ({_pct(bm, cm):+.1f}%, limit {t['ms_pct']}%)"
            )
    else:
        moved.append("ms/round missing in an arm (no window/profile line)")
    return {
        "ok": not moved,
        "moved": moved,
        "numbers": {
            "formats": [base["fmt"], cand["fmt"]],
            "ms_per_round": [bm, cm],
            "cells": cells,
            "tol": t,
        },
    }


def verdict_text(res: dict) -> str:
    if res["ok"]:
        return "probe PASS: counters and ms/round unchanged; no timing reps needed"
    return "probe flagged, run the speed stage: " + "; ".join(res["moved"])


def cache_path(root: Path, key: str) -> Path:
    return root / "probe" / f"{key}.json"


def cache_load(root: Path, key: str) -> dict | None:
    p = cache_path(root, key)
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return d if d.get("cells") else None


def cache_store(root: Path, key: str, summary: dict) -> None:
    p = cache_path(root, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary))
    tmp.replace(p)
