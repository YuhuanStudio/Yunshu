"""Infra report from gpuq job history (stdlib only, python 3.9 safe).

    gpuq report [--since 7d] [--label-prefix X] [--json]

Numbers (all from job records; nothing is estimated):

* queue busy %   union of [started, ended] minus pauses, over the window. This is queue occupancy, NOT hardware GPU
                 utilization (a job can hold the slot while idle).
* wasted minutes run time (minus pauses) of jobs that ended failed / lost / stalled / timeout / cancelled-after-start,
                 plus run time of jobs flagged contended (their timing is untrustworthy). Reported by cause.
* time-to-verdict per candidate: label ends in a 7-12 hex sha; first submit -> last end of the jobs sharing that sha.
* queue wait     started - submitted, by priority (median / p90 / max).
"""

# ruff: noqa: UP031, UP045
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

BAD = ("failed", "lost", "stalled", "timeout", "cancelled")
SHA = re.compile(r"(?:^|-)([0-9a-f]{7,12})(?:-|$)")


def parse_since(text: str, now: float) -> float:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([mhd])", text)
    if not m:
        return float(text)
    return now - float(m.group(1)) * {"m": 60, "h": 3600, "d": 86400}[m.group(2)]


def load_jobs(directory: Path) -> list:
    jobs = []
    for f in sorted(directory.glob("*.json")):
        try:
            j = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(j, dict) and j.get("id"):
            jobs.append(j)
    return jobs


def _pause_s(j: dict, lo: float, hi: float) -> float:
    total = 0.0
    for p in j.get("pauses") or []:
        try:
            a, b = float(p[0]), float(p[1] if len(p) > 1 and p[1] else hi)
        except (TypeError, ValueError, IndexError):
            continue
        total += max(0.0, min(b, hi) - max(a, lo))
    return total


def _union(intervals: list) -> float:
    total, end = 0.0, None
    for a, b in sorted(intervals):
        if end is None or a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def _pct(vals: list, q: float) -> float:
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))] if s else 0.0


def build(jobs: list, since: float, now: float, prefix: str = "") -> dict:
    jobs = [j for j in jobs if not prefix or str(j.get("label", "")).startswith(prefix)]
    ran = [j for j in jobs if j.get("started") and j.get("ended") and j["ended"] > since]
    intervals, waste, by_state = [], {}, {}
    for j in ran:
        a, b = max(float(j["started"]), since), min(float(j["ended"]), now)
        if b <= a:
            continue
        intervals.append((a, b))
        run = (b - a) - _pause_s(j, a, b)
        by_state[j.get("state")] = by_state.get(j.get("state"), 0) + 1
        cause = j.get("state") if j.get("state") in BAD else ("contended" if j.get("contended") else None)
        if cause:
            waste[cause] = waste.get(cause, 0.0) + run / 60
    first = min([a for a, _ in intervals] + [since]) if intervals else since
    span = max(now - max(since, first if first > since else since), 1.0)
    busy_s = _union(intervals)
    pausing = sum(_pause_s(j, float(j["started"]), float(j["ended"])) for j in ran)
    waits = {}
    for j in jobs:
        if j.get("started") and j.get("submitted") and j["started"] > since:
            waits.setdefault(int(j.get("priority", 0)), []).append((j["started"] - j["submitted"]) / 60)
    wait = {p: {"n": len(v), "median_min": _pct(v, .5), "p90_min": _pct(v, .9), "max_min": max(v)}
            for p, v in sorted(waits.items())}
    cands = {}
    for j in jobs:
        m = SHA.search(str(j.get("label", "")))
        if m and j.get("ended") and j.get("submitted") and j["ended"] > since:
            c = cands.setdefault(m.group(1), {"first": j["submitted"], "last": j["ended"], "jobs": 0, "bad": 0})
            c["first"], c["last"] = min(c["first"], j["submitted"]), max(c["last"], j["ended"])
            c["jobs"] += 1
            c["bad"] += j.get("state") in BAD
    verdict = {s: {"hours": round((c["last"] - c["first"]) / 3600, 2), "jobs": c["jobs"], "bad": c["bad"]}
               for s, c in cands.items()}
    hrs = [v["hours"] for v in verdict.values()]
    return {
        "window_h": round(span / 3600, 1), "jobs_ran": len(ran), "states": by_state,
        "queue_busy_pct": round(100 * (busy_s - pausing) / span, 1),
        "paused_min": round(pausing / 60, 1),
        "run_min": round((busy_s) / 60, 1),
        "wasted_min": {k: round(v, 1) for k, v in sorted(waste.items())},
        "wasted_total_min": round(sum(waste.values()), 1),
        "queue_wait_by_priority": wait,
        "time_to_verdict": {"candidates": len(verdict), "median_h": round(_pct(hrs, .5), 2),
                            "p90_h": round(_pct(hrs, .9), 2), "per_candidate": verdict},
    }


def render(r: dict) -> str:
    out = ["gpuq report: window %.1f h, %d jobs ran" % (r["window_h"], r["jobs_ran"]),
           "queue busy (occupancy, not hardware GPU util): %.1f %%  (run %.0f min, paused %.0f min)"
           % (r["queue_busy_pct"], r["run_min"], r["paused_min"]),
           "wasted GPU minutes: %.0f  %s" % (r["wasted_total_min"], r["wasted_min"]), "",
           "queue wait by priority (min)", "  prio     n  median    p90    max"]
    for p, w in r["queue_wait_by_priority"].items():
        out.append("  %4d %5d %7.1f %6.1f %6.1f" % (p, w["n"], w["median_min"], w["p90_min"], w["max_min"]))
    t = r["time_to_verdict"]
    out += ["", "time to verdict: %d candidates, median %.2f h, p90 %.2f h" % (t["candidates"], t["median_h"], t["p90_h"])]
    return "\n".join(out)


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="gpuq report")
    ap.add_argument("--since", default="7d")
    ap.add_argument("--label-prefix", default="")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--dir", default=os.environ.get("GPUQ_DIR", ""))
    a = ap.parse_args(argv)
    if not a.dir:
        print("GPUQ_DIR not set", file=sys.stderr)
        return 2
    now = time.time()
    jobs = load_jobs(Path(a.dir) / "jobs")
    if not jobs:
        print("no job records: refusing to report", file=sys.stderr)
        return 1
    r = build(jobs, parse_since(a.since, now), now, a.label_prefix)
    print(json.dumps(r, indent=1) if a.json else render(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
