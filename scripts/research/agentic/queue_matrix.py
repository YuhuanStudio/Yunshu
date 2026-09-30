"""Queue the full agentic matrix as resumable gpuq jobs (one server per job).

    queue_matrix.py --snapshot /Volumes/P5Plus/yunshu-build/agentic/snapshot --out DIR \
        --checkpoint $M [--shards 12] [--passes 2] [--priority -3] [--dry-run]

Combos: Yunshu (default settings) x {opencode, claude, codex}, TensorFold x {opencode} (TensorFold
serves only /v1/chat/completions, so Claude Code (Messages) and Codex (Responses) cannot use it).
Each job runs one shard of a combo's (task, repeat) list and stops starting new runs after
--budget-min, so it ends inside its queue timeout; later passes finish what earlier ones left
(a finished shard exits before starting a server).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

GPUQ = "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/scripts/dev/gpuq"
PY = "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python"
COMBOS = [
    ("yunshu-default", "yunshu", "opencode"),
    ("yunshu-default", "yunshu", "claude"),
    ("yunshu-default", "yunshu", "codex"),
    ("tensorfold-default", "tensorfold", "opencode"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shards", type=int, default=12)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--priority", type=int, default=-3)
    ap.add_argument("--budget-min", type=float, default=33)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    snap = Path(a.snapshot)
    ids = []
    for p in range(a.passes):
        for label, serve, agent in COMBOS:
            for i in range(a.shards):
                cmd = [
                    GPUQ, "submit", "--label",
                    f"agentic-{label}-{agent}-p{p + 1}s{i}",
                    "--timeout", "60", "--stall", "10", "--priority", str(a.priority),
                    "--", "/usr/bin/env", f"AGENTIC_YUNSHU_SRC={snap}/python",
                    PY, f"{snap}/research/agentic/run_agentic.py", "run",
                    "--serve", serve, "--checkpoint", a.checkpoint,
                    "--engine-label", label, "--agent", agent, "--tasks", "all",
                    "--repeat", str(a.repeat), "--shard-i", str(i),
                    "--shard-n", str(a.shards), "--budget-min", str(a.budget_min),
                    "--timeout-min", "20",
                    "--output", f"{a.out}/{label}-{agent}.jsonl",
                ]  # fmt: skip
                if a.dry_run:
                    print(" ".join(cmd))
                    continue
                jid = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
                ids.append(jid)
    print("\n".join(ids))
    print(f"{len(ids)} jobs", file=sys.stderr)


if __name__ == "__main__":
    main()
