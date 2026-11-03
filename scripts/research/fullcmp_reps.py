#!/usr/bin/env python3
"""Rep policy for the full comparison: 2 reps per cell, automatic 3rd rep when the reps disagree.

Policy (user decision, 2026-11-03):
  * every cell runs 2 reps (r0, r1); the spread between them is reported as the two values;
  * per metric, relative gap = (max - min) / min * 100 over the reps of a cell;
  * if any metric's gap exceeds its threshold after 2 reps, queue rep 3 for that cell, in the
    rotated order slot (rep index 2), and report the median of the 3 reps flagged '3 reps (spread)';
  * if after rep 3 any metric's gap is still > 2 x its threshold, the cell is 'unstable' and every
    value of every rep is reported instead of one number.

Thresholds come from the A/A control runs (rep_thresholds.json, see derive_thresholds):
  tok/s   = max(3 x A/A max gap, 3 %)
  TTFT    = max(3 x A/A max gap, 5 %)
  memory  = 5 % fixed

Cell layout read by the CLI is the audit layout (fullcmp_audit.py): <results>/<engine>/<cell>-r<rep>.jsonl,
one session header row (part "session") followed by request rows (part "decode"), and optionally a
"memory" row carrying peak_gib. METRICS mirrors fullcmp_audit.METRICS (kept local so this module does
not import a file another change is editing).

CLI:
    fullcmp_reps.py --results-dir DIR --thresholds rep_thresholds.json [--manifest MANIFEST.json]
        prints one JSON line per cell with status 'need_rep3' (queue rep 3) or 'unstable' (all values
        in the record); counts go to stderr. Exit 1 if a cell is unstable without all of its values
        recorded, 2 on unreadable input.
    fullcmp_reps.py --derive-thresholds CELLS_DIR [CELLS_DIR ...] --out rep_thresholds.json
        A/A control runs: reads speed.<arm>-r<k>.jsonl files (non-.a1) and writes the thresholds file.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# metric -> (phase, field, class). Mirrors fullcmp_audit.METRICS plus the class used for thresholds.
METRICS = {
    "decode_tps": ("cold", "dec_tps", "tok_s"),
    "turn2_decode_tps": ("turn2", "dec_tps", "tok_s"),
    "cold_ttft": ("cold", "ttft_s", "ttft"),
    "warm_ttft": ("warm", "ttft_s", "ttft"),
    "turn2_ttft": ("turn2", "ttft_s", "ttft"),
}
MEMORY_METRIC = "peak_gib"
CLASS_FLOOR_PCT = {"tok_s": 3.0, "ttft": 5.0, "memory": 5.0}
FACTOR = 3.0  # threshold = max(FACTOR x A/A max gap, floor)
MIN_SAMPLES = 3  # fewer A/A samples than this = low confidence, floor only
REP3_INDEX = 2  # rep index of the automatic third rep

REP_FILE = re.compile(r"^(?P<cell>.+)-r(?P<rep>\d+)\.jsonl$")
AA_FILE = re.compile(r"^speed\.(?P<arm>base|cand)-r(?P<rep>\d+)\.jsonl$")


def is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def rotate(order, rep):
    """Rotated interleave order for rep index `rep` (same rule as fullcmp_audit.rotate)."""
    k = rep % len(order)
    return list(order[k:]) + list(order[:k])


def rep3_order(order, rep: int = REP3_INDEX) -> list:
    """Order of the engines in the rep-3 slot group. Engine e moves from slot p (rep 0) to slot
    (p + 2) mod n, so it never reuses the slot of rep 0 or rep 1 when the group has >= 3 engines."""
    return rotate(order, rep)


def rel_gap_pct(values) -> float:
    """(max - min) / min x 100; for two values this is |a - b| / min(a, b)."""
    lo, hi = min(values), max(values)
    if lo <= 0:
        raise ValueError(f"non-positive value in gap: {values}")
    return (hi - lo) / lo * 100.0


def per_rep_metrics(rows) -> dict:
    """Metric values of one rep (one cell, one rep): median over the measured request rows."""
    out = {}
    for name, (phase, fld, _cls) in METRICS.items():
        vals = [
            r[fld]
            for r in rows
            if r.get("part") == "decode"
            and r.get("phase") == phase
            and not r.get("error")
            and is_num(r.get(fld))
        ]
        if vals:
            out[name] = statistics.median(vals)
    mem = [r[MEMORY_METRIC] for r in rows if r.get("part") == "memory" and is_num(r.get(MEMORY_METRIC))]
    if mem:
        out[MEMORY_METRIC] = mem[-1]
    return out


def metric_class(metric: str) -> str:
    if metric == MEMORY_METRIC:
        return "memory"
    return METRICS[metric][2]


def decide(cell_rows, thresholds) -> dict:
    """Decide one cell from its reps.

    cell_rows: list of 2 or 3 dicts, one per rep, {metric: value}.
    thresholds: {metric: threshold_pct}. A metric is judged only if every rep has it; a metric
    present in some reps but not all raises ValueError.

    Returns {'status': 'ok' | 'need_rep3' | 'spread_flag' | 'unstable', 'per_metric': {...},
    'value': {metric: median} | {metric: [values]} | None, 'exceeding': [...], 'flags': [...]}.
    """
    n = len(cell_rows)
    if n not in (2, 3):
        raise ValueError(f"a cell has 2 or 3 reps, got {n}")
    names = set(thresholds)
    present = [set(r) & names for r in cell_rows]
    judged = set.intersection(*present)
    partial = set.union(*present) - judged
    if partial:
        raise ValueError(f"metrics missing in some reps: {sorted(partial)}")

    per_metric = {}
    for m in sorted(judged):
        vals = [float(r[m]) for r in cell_rows]
        gap = rel_gap_pct(vals)
        thr = float(thresholds[m])
        per_metric[m] = {
            "values": vals,
            "gap_pct": gap,
            "threshold_pct": thr,
            "exceeds": gap > thr,
            "exceeds_2x": gap > 2.0 * thr,
        }
    exceeding = [m for m, v in per_metric.items() if v["exceeds"]]

    if n == 2:
        if not exceeding:
            return {
                "status": "ok",
                "per_metric": per_metric,
                "value": {m: statistics.median(v["values"]) for m, v in per_metric.items()},
                "exceeding": [],
                "flags": [],
            }
        return {
            "status": "need_rep3",
            "per_metric": per_metric,
            "value": None,
            "exceeding": exceeding,
            "flags": [],
        }

    if any(v["exceeds_2x"] for v in per_metric.values()):
        return {
            "status": "unstable",
            "per_metric": per_metric,
            "value": {m: list(v["values"]) for m, v in per_metric.items()},
            "exceeding": exceeding,
            "flags": ["3 reps (spread)", "unstable"],
        }
    return {
        "status": "spread_flag",
        "per_metric": per_metric,
        "value": {m: statistics.median(v["values"]) for m, v in per_metric.items()},
        "exceeding": exceeding,
        "flags": ["3 reps (spread)"],
    }


def thresholds_from_file(path) -> dict:
    data = json.loads(Path(path).read_text())
    return {m: float(v) for m, v in data["thresholds_pct"].items()}


# ---------------------------------------------------------------------------
# A/A thresholds
# ---------------------------------------------------------------------------
def aa_gap_samples(cells_dir) -> dict:
    """Per metric, the relative gaps (%) between identical-config runs in one A/A cells dir.

    Every pair of speed.<arm>-r<k>.jsonl files (base and cand are the same tree in an A/A run)
    gives one gap per (ctx, kind) group and metric. A/A runs have no real difference, so the
    gaps are pure noise.
    """
    files = []
    for p in sorted(Path(cells_dir).glob("speed.*.jsonl")):
        m = AA_FILE.match(p.name)
        if m:
            files.append((m.group("arm"), int(m.group("rep")), p))
    samples = defaultdict(list)
    per_file = {}
    for arm, rep, p in files:
        groups = defaultdict(list)
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("part") == "decode":
                groups[(row.get("ctx"), row.get("kind"))].append(row)
        per_file[(arm, rep)] = {g: per_rep_metrics(rows) for g, rows in groups.items()}
    for (a_key, b_key) in itertools.combinations(sorted(per_file), 2):
        for g in set(per_file[a_key]) & set(per_file[b_key]):
            for m in METRICS:
                va, vb = per_file[a_key][g].get(m), per_file[b_key][g].get(m)
                if va is not None and vb is not None:
                    samples[m].append(rel_gap_pct([va, vb]))
    return dict(samples)


def derive_thresholds(samples: dict, sources: list) -> dict:
    """Thresholds file body. threshold = max(3 x A/A max gap, class floor); fewer than
    MIN_SAMPLES samples = low confidence, floor only. Memory is fixed at its floor."""
    metrics = {}
    thresholds = {}
    for m in METRICS:
        cls = metric_class(m)
        vals = samples.get(m, [])
        n = len(vals)
        floor = CLASS_FLOOR_PCT[cls]
        if n < MIN_SAMPLES:
            thr, low = floor, True
        else:
            thr, low = max(FACTOR * max(vals), floor), False
        thresholds[m] = round(thr, 2)
        metrics[m] = {
            "class": cls,
            "aa_samples": n,
            "aa_max_gap_pct": round(max(vals), 3) if vals else None,
            "aa_median_gap_pct": round(statistics.median(vals), 3) if vals else None,
            "floor_pct": floor,
            "threshold_pct": thresholds[m],
            "low_confidence": low,
        }
    thresholds[MEMORY_METRIC] = CLASS_FLOOR_PCT["memory"]
    metrics[MEMORY_METRIC] = {
        "class": "memory",
        "aa_samples": 0,
        "floor_pct": CLASS_FLOOR_PCT["memory"],
        "threshold_pct": CLASS_FLOOR_PCT["memory"],
        "low_confidence": False,
        "note": "fixed 5% (user decision), not derived from A/A",
    }
    return {
        "formula": {
            "tok_s": "max(3 x A/A max gap %, 3 %)",
            "ttft": "max(3 x A/A max gap %, 5 %)",
            "memory": "5 % fixed",
            "gap": "(max - min) / min x 100 over the reps of a cell",
            "low_confidence": f"fewer than {MIN_SAMPLES} A/A samples: floor only",
            "unstable": "after rep 3, any gap > 2 x threshold",
        },
        "sources": sources,
        "thresholds_pct": thresholds,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Results scan (CLI)
# ---------------------------------------------------------------------------
def load_results(results_dir) -> dict:
    """{(engine, cell): {'group': str|None, 'reps': {rep: rows}}} from <results>/<engine>/<cell>-rN.jsonl."""
    cells = defaultdict(lambda: {"group": None, "reps": {}})
    for p in sorted(Path(results_dir).glob("*/*.jsonl")):
        m = REP_FILE.match(p.name)
        if not m or p.name.endswith(".telemetry.jsonl"):
            continue
        rows = []
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
        engine, cell = p.parent.name, m.group("cell")
        entry = cells[(engine, cell)]
        for r in rows:
            if r.get("part") == "session" and entry["group"] is None:
                entry["group"] = r.get("group")
        entry["reps"][int(m.group("rep"))] = rows
    return cells


def group_order(manifest: dict | None, group: str | None):
    if not manifest or group is None:
        return None
    return (manifest.get("groups", {}).get(group) or {}).get("order")


def scan(results_dir, thresholds: dict, manifest: dict | None = None):
    """Yields (engine, cell, group, record) for every cell with at least 2 reps."""
    for (engine, cell), entry in sorted(load_results(results_dir).items()):
        reps = sorted(entry["reps"])
        if len(reps) < 2:
            continue
        rows = [per_rep_metrics(entry["reps"][k]) for k in reps[:3]]
        verdict = decide(rows, thresholds)
        record = {
            "engine": engine,
            "cell": cell,
            "group": entry["group"],
            "reps_present": reps,
            "status": verdict["status"],
            "exceeding": verdict["exceeding"],
            "value": verdict["value"],
            "flags": verdict["flags"],
        }
        if verdict["status"] == "need_rep3":
            order = group_order(manifest, entry["group"])
            record["rep"] = REP3_INDEX
            record["rep3_order"] = rep3_order(order) if order else None
        yield engine, cell, entry["group"], record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", help="result dir: <dir>/<engine>/<cell>-rN.jsonl")
    ap.add_argument("--thresholds", help="rep_thresholds.json")
    ap.add_argument("--manifest", help="optional audit manifest JSON (for rep-3 order slots)")
    ap.add_argument("--derive-thresholds", nargs="+", metavar="CELLS_DIR")
    ap.add_argument("--out", help="output path for --derive-thresholds")
    args = ap.parse_args(argv)

    if args.derive_thresholds:
        if not args.out:
            ap.error("--derive-thresholds needs --out")
        samples = defaultdict(list)
        for d in args.derive_thresholds:
            for m, vals in aa_gap_samples(d).items():
                samples[m].extend(vals)
        body = derive_thresholds(dict(samples), sources=[str(Path(d)) for d in args.derive_thresholds])
        Path(args.out).write_text(json.dumps(body, indent=2) + "\n")
        print(json.dumps(body["thresholds_pct"]), file=sys.stderr)
        return 0

    if not (args.results_dir and args.thresholds):
        ap.error("--results-dir and --thresholds are required")
    try:
        thresholds = thresholds_from_file(args.thresholds)
        manifest = json.loads(Path(args.manifest).read_text()) if args.manifest else None
        results = list(scan(args.results_dir, thresholds, manifest))
    except (OSError, ValueError, KeyError) as exc:
        print(f"fullcmp_reps: unreadable input: {exc}", file=sys.stderr)
        return 2

    rc = 0
    counts = defaultdict(int)
    for _engine, _cell, _group, record in results:
        counts[record["status"]] += 1
        if record["status"] in ("need_rep3", "unstable"):
            print(json.dumps(record, sort_keys=True))
        if record["status"] == "unstable":
            values = record["value"] or {}
            if not values or not all(len(v) == 3 for v in values.values()):
                rc = 1
    print(json.dumps({"summary": dict(counts)}), file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
