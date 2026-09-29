"""Print `path tokens` for the fastest ANE layer packages (tokens per second) from ane_probe rows.

python scripts/research/hw/pick_best_ane.py 2 [layer27_x1]
"""

import json
import os
import sys
from pathlib import Path

RUNS = (
    Path(__file__).resolve().parents[3]
    / "docs/research/runs/2026-09-29-m5max-hw/ane_probe.jsonl"
)
WORK = Path(os.environ.get("YUNSHU_ANE_WORK", "~/.cache/yunshu/ane")).expanduser()


def main():
    n = int(sys.argv[1])
    case = sys.argv[2] if len(sys.argv) > 2 else "layer27_x1"
    rows = []
    for line in RUNS.read_text().splitlines():
        d = json.loads(line)
        if d.get("case") == case and "ne" in d:
            rows.append((d["M"] / d["ne"]["ms_median"], d))
    rows.sort(key=lambda r: -r[0])
    for _, d in rows[:n]:
        print(f"{WORK}/{d['case']}_M{d['M']}_{d['weights']}.mlpackage {d['M']}")


if __name__ == "__main__":
    main()
