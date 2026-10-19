#!/usr/bin/env python3
"""Read-only retrospective policy review. Never edits/resubmits live jobs.

Legacy host-only records cannot establish job-local swap or minimum free memory.
Report which host rejections clear conditionally, but never certify missing evidence.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from gpuq_contention import DEFAULTS
from gpuq_swap import memory_verdict


def review(job):
    host = job.get("host_swap_mb", job.get("swap_mb"))
    failures = memory_verdict(
        host, job.get("job_swap_mb"), job.get("available_min_mb"), DEFAULTS
    )
    # Recompute only the legacy host threshold. Never erase an unreadable
    # interval merely because a later sample supplied a number.
    others = [r for r in job.get("contention_reasons", []) if r != "swap"]
    if job.get("contended") and not job.get("contention_reasons"):
        others.append("unexplained_contention_latch")
    reasons = list(dict.fromkeys(others + failures))
    missing = any(r.endswith("_unreadable") for r in reasons)
    rejected = any(not r.endswith("_unreadable") for r in reasons)
    return dict(
        id=job["id"],
        state=job.get("state"),
        rc=job.get("rc"),
        host_swap_mb=host,
        old_reasons=job.get("contention_reasons", []),
        host_rejection_clears=(
            "swap" in job.get("contention_reasons", [])
            and host is not None
            and host <= DEFAULTS["host_swap_mb"]
        ),
        verdict="REJECT"
        if rejected
        else "UNVERIFIABLE"
        if missing
        else "CONTENTION_CLEAR",
        reasons=reasons,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--prefix", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        review(json.loads(p.read_text()))
        for p in sorted(args.jobs.glob("*.json"))
        if any(p.name.startswith(x) for x in args.prefix)
    ]
    args.output.write_text(
        json.dumps(
            dict(
                complete=True,
                policy=DEFAULTS,
                measurement_success_evaluated=False,
                counts=dict(Counter(r["verdict"] for r in rows)),
                host_rejection_clears=sum(r["host_rejection_clears"] for r in rows),
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
