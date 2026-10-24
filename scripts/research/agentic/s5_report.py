"""Aggregate agent-bench jsonl files per client (CPU only).

    s5_report.py <dir with *.jsonl>

Per client: runs, passes with Wilson 95% interval, token-weighted cache hit, median TTFT over all
requests, and the failed runs with the client-visible tail of the run (last request's finish reason and
tool-call count, so "the turn ended without a tool call" is told apart from a timeout or a wrong answer).
"""

from __future__ import annotations

import glob
import json
import math
import statistics
import sys
from collections import defaultdict


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def load(paths: list[str]) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    for p in paths:
        for line in open(p):
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("type") == "run":
                by[d["agent"]].append(d)
    return by


def stalled(run: dict) -> bool:
    """The last request ended normally with no tool call (the agent stops there)."""
    log = run.get("request_log") or []
    if not log:
        return False
    last = log[-1]
    return not last.get("tool_calls") and last.get("finish_reason") not in (
        "length",
        "max_tokens",
    )


def report(by: dict[str, list[dict]]) -> str:
    out = [
        "| client | runs | pass | rate (Wilson 95%) | cache hit | TTFT med s | wall med s | api err | malformed | leaked |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    fails = []
    for agent in sorted(by):
        runs = by[agent]
        n = len(runs)
        k = sum(bool(r.get("passed")) for r in runs)
        lo, hi = wilson(k, n)
        pt = sum(r.get("prompt_tokens") or 0 for r in runs)
        ct = sum(r.get("cached_tokens") or 0 for r in runs)
        ttft = [
            q["ttft_s"]
            for r in runs
            for q in (r.get("request_log") or [])
            if q.get("ttft_s") is not None
        ]
        wall = [r["wall_s"] for r in runs if r.get("wall_s") is not None]
        out.append(
            f"| {agent} | {n} | {k} | {100 * k / n:.0f}% ({100 * lo:.0f}-{100 * hi:.0f}) | "
            f"{100 * ct / pt:.1f}% | {statistics.median(ttft):.2f} | "
            f"{statistics.median(wall):.0f} | {sum(r.get('api_errors') or 0 for r in runs)} | "
            f"{sum(r.get('malformed_tool_calls') or 0 for r in runs)} | "
            f"{sum(r.get('leaked_tool_markup') or 0 for r in runs)} |"
        )
        for r in runs:
            if not r.get("passed"):
                last = (r.get("request_log") or [{}])[-1]
                fails.append(
                    f"| {agent} | {r['task']} | r{r['repeat']} | {r.get('wall_s')} | "
                    f"{'timeout' if r.get('timed_out') else 'stalled-turn' if stalled(r) else 'other'} | "
                    f"{last.get('finish_reason')}/{last.get('tool_calls')} | {r.get('artifacts')} |"
                )
    out += [
        "",
        "| client | task | rep | wall s | tail | last finish/tool_calls | artifacts |",
        "|---|---|---|---|---|---|---|",
        *fails,
    ]
    return "\n".join(out)


if __name__ == "__main__":
    print(report(load(sorted(glob.glob(sys.argv[1] + "/*.jsonl")))))
