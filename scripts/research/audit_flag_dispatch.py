#!/usr/bin/env python3
"""CPU-only dispatcher: a real successful tiny smoke precedes 27B submission.

Poll at most once per ten minutes. Preserve an audit status/digest journal while
waiting. Never change serving code, queue order, or defaults from this watcher.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

MAIN = Path("/Users/yuhuan/Documents/YuhuanStudio/Yunshu")
GPUQ = MAIN / "scripts/dev/gpuq"
PYTHON = MAIN / ".venv/bin/python"


def eligible_smoke(job, receipt, matrix):
    return (
        job.get("state") == "done"
        and job.get("rc") == 0
        and matrix.validate_smoke(receipt, matrix.source_sha(), list(matrix.AREAS))
    )


def timing_command(snapshot, output, label):
    return [
        str(GPUQ),
        "submit",
        "--label",
        label,
        "--priority",
        "-1",
        "--quiet",
        "--timeout",
        "1440",
        "--stall",
        "30",
        "--mem-gb",
        "85",
        "--out",
        str(output / "timing.json"),
        "--expect-complete",
        "--",
        "env",
        "GPUQ_OWNER=codex-audit",
        "PYTHONPATH=" + str(snapshot / "python"),
        str(PYTHON),
        str(snapshot / "scripts/research/audit_flag_matrix.py"),
        "--out",
        str(output / "timing.json"),
        "--require-smoke",
        str(output / "smoke.json"),
    ]


def load_matrix(snapshot):
    spec = importlib.util.spec_from_file_location(
        "pinned_audit_matrix", snapshot / "scripts/research/audit_flag_matrix.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def monitor(job_dir):
    rows = []
    for path in sorted(job_dir.glob("*audit-*.json")):
        try:
            job = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        rows.append(
            {
                k: job.get(k)
                for k in ("id", "label", "state", "rc", "contended", "pause_s")
            }
        )
    digest = subprocess.run(
        [
            str(GPUQ),
            "digest",
            "--label-prefix",
            "audit-",
            "--since",
            "12h",
            "--peek",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    return {
        "polled_at": time.time(),
        "jobs": rows,
        "digest_rc": digest.returncode,
        "digest": json.loads(digest.stdout)
        if digest.stdout.strip().startswith("{")
        else {"error": digest.stderr[-400:]},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--smoke-id", required=True)
    ap.add_argument("--timing-label", required=True)
    ap.add_argument(
        "--job-dir",
        type=Path,
        default=Path(
            os.environ.get("GPUQ_DIR", str(Path.home() / ".cache/yunshu/gpuq"))
        )
        / "jobs",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    matrix = load_matrix(args.snapshot)
    command = timing_command(args.snapshot, args.output, args.timing_label)
    args.output.mkdir(parents=True, exist_ok=True)
    planned = {
        "smoke_id": args.smoke_id,
        "timing_label": args.timing_label,
        "timing_argv": command,
        "source_sha": matrix.source_sha(),
        "poll_interval_s": 600,
    }
    (args.output / "dispatch-plan.json").write_text(
        json.dumps(planned, indent=2) + "\n"
    )
    if args.dry_run:
        print(json.dumps(planned))
        return
    while True:
        journal = monitor(args.job_dir)
        (args.output / "queue-journal.json").write_text(
            json.dumps(journal, indent=2) + "\n"
        )
        job = next((row for row in journal["jobs"] if row["id"] == args.smoke_id), {})
        if job.get("state") in ("failed", "cancelled", "timeout") or (
            job.get("state") == "done" and job.get("rc") != 0
        ):
            (args.output / "dispatch-result.json").write_text(
                json.dumps(
                    {
                        "timing_submitted": False,
                        "reason": "tiny smoke failed; no 27B job submitted",
                        "job": job,
                    },
                    indent=2,
                )
                + "\n"
            )
            raise SystemExit(1)
        if job.get("state") == "done":
            try:
                receipt = json.loads((args.output / "smoke.json").read_text())
            except (OSError, ValueError):
                receipt = {}
            if not eligible_smoke(job, receipt, matrix):
                (args.output / "dispatch-result.json").write_text(
                    json.dumps(
                        {
                            "timing_submitted": False,
                            "reason": "missing/mismatched/incomplete/parity-failed smoke",
                        },
                        indent=2,
                    )
                    + "\n"
                )
                raise SystemExit(1)
            submitted = subprocess.run(
                command, cwd=args.snapshot, capture_output=True, text=True
            )
            result = {
                "timing_submitted": submitted.returncode == 0,
                "submit_rc": submitted.returncode,
                "stdout": submitted.stdout.strip(),
                "stderr": submitted.stderr.strip(),
                "complete": submitted.returncode == 0,
            }
            (args.output / "dispatch-result.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(json.dumps(result), flush=True)
            if submitted.returncode:
                raise SystemExit(submitted.returncode)
            return
        print(
            json.dumps(
                {
                    "waiting_for_smoke": args.smoke_id,
                    "state": job.get("state"),
                    "next_poll_s": 600,
                }
            ),
            flush=True,
        )
        time.sleep(600)


if __name__ == "__main__":
    main()
