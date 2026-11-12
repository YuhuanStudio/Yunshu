"""Per-span PLE wait of cold Flash-Next prefills, from server logs' ``fn_ple span=a:b ple_wait=X ms rows=N prefetched=M`` lines.

Usage: ple_wait_report.py LOG [LOG ...]   (a request starts where a span starts at 0)
"""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

LINE = re.compile(
    r"fn_ple span=(\d+):(\d+) ple_wait=([\d.]+) ms rows=(\d+) prefetched=(\d+)"
)


def parse(text: str) -> list[list[dict]]:
    """Requests, each a list of span dicts (a new request begins at span start 0)."""
    requests: list[list[dict]] = []
    for m in LINE.finditer(text):
        start, end, wait, rows, served = m.groups()
        span = dict(
            start=int(start),
            end=int(end),
            wait_ms=float(wait),
            rows=int(rows),
            prefetched=int(served),
        )
        if span["start"] == 0 or not requests:
            requests.append([])
        requests[-1].append(span)
    return requests


def summarize(requests: list[list[dict]], buckets=(8192, 32768)) -> dict:
    """Per context bucket (prompt end nearest to a bucket, within 5%): wait per span median/max,
    prefetched row share, first span (never prefetched) apart from the rest."""
    out: dict = {}
    for req in requests:
        n = req[-1]["end"]
        bucket = next((b for b in buckets if abs(n - b) <= b * 0.05), None)
        if bucket is None:
            continue
        agg = out.setdefault(bucket, {"first": [], "rest": [], "rows": 0, "served": 0})
        agg["first"].append(req[0]["wait_ms"])
        agg["rest"].extend(s["wait_ms"] for s in req[1:])
        agg["rows"] += sum(s["rows"] for s in req)
        agg["served"] += sum(s["prefetched"] for s in req)
    table = {}
    for bucket, agg in sorted(out.items()):
        rest = agg["rest"] or [0.0]
        table[bucket] = dict(
            requests=len(agg["first"]),
            first_span_ms=statistics.median(agg["first"]),
            later_median_ms=statistics.median(rest),
            later_max_ms=max(rest),
            prefetched_share=agg["served"] / agg["rows"] if agg["rows"] else 0.0,
        )
    return table


def main(argv: list[str]) -> int:
    requests: list[list[dict]] = []
    for path in argv:
        requests.extend(parse(Path(path).read_text(errors="ignore")))
    table = summarize(requests)
    if not table:
        print("no fn_ple lines for a bucketed prompt", file=sys.stderr)
        return 1
    for bucket, row in table.items():
        print(bucket, row)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
