"""Count skipped tests in a ci-local pytest log and fail when they exceed the committed baseline.

    python scripts/dev/ci_skips.py LABEL LOG [--baseline scripts/dev/ci_skip_baseline.json] [--update]

A skip is a test that did not run: under ci-local's sandbox the path-gated tests skip instead of failing, so a
rising count means a test silently stopped running. The baseline is per pass (plain, paravirtual).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_SUMMARY = re.compile(r"=*\s*(?:[\d,]+ \w+, )*?([\d,]+) skipped\b")
_REASON = re.compile(r"^SKIPPED \[(\d+)\] (\S+?):\d+: (.*)$")


def parse(text: str) -> tuple[int, Counter]:
    """(total skipped over every pytest summary line in the log, reason -> count)."""
    total = 0
    reasons: Counter = Counter()
    for line in text.splitlines():
        m = _REASON.match(line)
        if m:
            reasons[m.group(3).strip()[:160]] += int(m.group(1))
            continue
        if (
            re.search(r"\d+ (passed|failed|deselected|skipped)", line)
            and " in " in line
        ):
            s = _SUMMARY.search(line)
            if s:
                total += int(s.group(1).replace(",", ""))
    return total, reasons


def judge(label: str, total: int, baseline: dict) -> tuple[bool, str]:
    limit = baseline.get(label)
    if limit is None:
        return False, f"no committed baseline for pass '{label}' (run with --update)"
    if total > limit:
        return False, f"{total} skipped > baseline {limit}: a test stopped running"
    return True, f"{total} skipped <= baseline {limit}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("label")
    ap.add_argument("log", type=Path)
    ap.add_argument(
        "--baseline",
        type=Path,
        default=Path(__file__).with_name("ci_skip_baseline.json"),
    )
    ap.add_argument(
        "--update", action="store_true", help="write the count as the new baseline"
    )
    a = ap.parse_args()
    total, reasons = parse(a.log.read_text(errors="replace"))
    print(f"ci-local: {a.label}: {total} skipped tests; reasons:")
    for reason, n in reasons.most_common():
        print(f"  {n:5d}  {reason}")
    baseline = json.loads(a.baseline.read_text()) if a.baseline.exists() else {}
    if a.update:
        baseline[a.label] = total
        a.baseline.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
        print(f"ci-local: baseline {a.label} = {total}")
        return 0
    ok, msg = judge(a.label, total, baseline)
    print(f"ci-local: skip check {a.label}: {'PASS' if ok else 'FAIL'} - {msg}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
