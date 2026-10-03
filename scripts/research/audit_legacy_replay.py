#!/usr/bin/env python3
"""Replay pinned legacy audit commands, with per-arm receipts and smoke gates.

CPU --dry-run invokes every command's preflight. GPU --smoke exercises the tiny
checkpoint first. --dispatch submits at most four grouped replacements after
smoke success and journals terminal jobs every ten minutes. Use gpuq for runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = Path("/Users/yuhuan/Documents/YuhuanStudio/Yunshu")
GPUQ = MAIN / "scripts/dev/gpuq"


def source_sha():
    digest = hashlib.sha256()
    for path in sorted(
        list((ROOT / "python").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py"))
    ):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def final_record(path):
    try:
        text = path.read_text()
        if path.suffix == ".json":
            return json.loads(text).get("complete") is True
        records = [
            json.loads(line) for line in text.splitlines() if line.startswith("{")
        ]
        return bool(records) and records[-1].get("complete") is True
    except (OSError, ValueError):
        return False


def run_arms(arms, *, dry_run=False):
    rows = []
    for arm in arms:
        log = Path(arm["log"])
        log.parent.mkdir(parents=True, exist_ok=True)
        argv = arm["argv"] + (["--dry-run"] if dry_run else [])
        Path(arm["out"]).parent.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith("YUNSHU_")}
        config = log.parent / "settings.toml"
        config.write_text("# isolated audit request settings\n")
        env.update(
            PYTHONPATH=str(ROOT / "python"),
            YUNSHU_CONFIG=str(config),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            YUNSHU_AUTH_DISABLED="1",
        )
        with log.open("w") as stream:
            rc = subprocess.run(
                argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
            ).returncode
        # Sweeps use stdout as the declared structured output; other scripts write
        # their own result file. The wrapper never fabricates a child completion.
        output = Path(arm["out"])
        if arm.get("stdout_result") and not dry_run:
            output.write_text(log.read_text())
        completed = rc == 0 if dry_run else final_record(output)
        row = {
            "name": arm["name"],
            "rc": rc,
            "output": str(output),
            "log": str(log),
            "complete": completed,
            "log_tail": log.read_text(errors="replace")[-1800:],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def eligible_smoke(job, receipt, digest):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from device_evidence import require_same_device

    require_same_device([job, receipt], performance=True)
    return (
        job.get("state") == "done"
        and job.get("rc") == 0
        and receipt.get("smoke") is True
        and receipt.get("dry_run") is False
        and receipt.get("source_sha") == digest
        and receipt.get("complete") is True
        and bool(receipt.get("arms"))
        and all(
            row.get("rc") == 0 and row.get("complete") is True
            for row in receipt["arms"]
        )
    )


def dispatch(plan, smoke_id, plan_path):
    base = Path(plan["base"])
    job_dir = Path(plan["job_dir"])
    state = {"smoke_id": smoke_id, "submitted": [], "harvested": {}, "complete": False}
    state_path = base / "dispatch.json"
    while True:
        job = json.loads((job_dir / (smoke_id + ".json")).read_text())
        if job["state"] in ("done", "failed", "cancelled", "timeout"):
            receipt_path = base / "smoke.json"
            receipt = (
                json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
            )
            state["smoke_job"] = {
                k: job.get(k) for k in ("id", "state", "rc", "contended")
            }
            if not eligible_smoke(job, receipt, source_sha()):
                state["blocked"] = (
                    "Real tiny smoke failed/incomplete/source mismatch; replacements NOT submitted"
                )
                write(state_path, state)
                return 1
            break
        state["waiting_for_smoke"] = job["state"]
        write(state_path, state)
        print(json.dumps(state), flush=True)
        time.sleep(600)
    for group, config in plan["groups"].items():
        output = base / (group + ".json")
        argv = [
            str(GPUQ),
            "submit",
            "--label",
            plan["label"] + "-" + group,
            "--priority",
            str(config["priority"]),
            "--timeout",
            str(config["timeout_min"]),
            "--stall",
            "30",
            "--mem-gb",
            str(config.get("mem_gb", 85)),
            "--out",
            str(output),
            "--expect-complete",
        ]
        if config["quiet"]:
            argv.append("--quiet")
        argv += [
            "--",
            sys.executable,
            __file__,
            "--plan",
            str(plan_path),
            "--group",
            group,
            "--out",
            str(output),
            "--require-smoke",
            str(base / "smoke.json"),
        ]
        proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
        row = {
            "group": group,
            "submit_rc": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "argv": argv,
        }
        if proc.returncode == 0:
            row["id"] = proc.stdout.strip().splitlines()[-1]
        state["submitted"].append(row)
        write(state_path, state)
        if proc.returncode:
            return proc.returncode
    while True:
        pending = []
        for submitted in state["submitted"]:
            job_id = submitted["id"]
            job = json.loads((job_dir / (job_id + ".json")).read_text())
            if job["state"] not in ("done", "failed", "cancelled", "timeout"):
                pending.append(job_id)
                continue
            if job_id not in state["harvested"]:
                path = base / (submitted["group"] + ".json")
                log = Path(plan["queue_log_dir"]) / (job_id + ".log")
                state["harvested"][job_id] = {
                    **{k: job.get(k) for k in ("state", "rc", "contended", "pause_s")},
                    "valid_output": final_record(path),
                    "out": str(path),
                    "log_tail": log.read_text(errors="replace")[-3000:]
                    if log.exists()
                    else "missing log",
                    "verdict": "eligible evidence"
                    if job.get("rc") == 0
                    and final_record(path)
                    and not job.get("contended")
                    else "review failure/incomplete/contention",
                }
        state["pending"] = pending
        state["complete"] = not pending
        write(state_path, state)
        digest = subprocess.run(
            [
                str(GPUQ),
                "digest",
                "--label-prefix",
                "audit-",
                "--since",
                "12h",
                "--peek",
            ],
            capture_output=True,
            text=True,
        )
        (base / "digest.txt").write_text(digest.stdout + digest.stderr)
        print(
            json.dumps({"pending": pending, "harvested": len(state["harvested"])}),
            flush=True,
        )
        if not pending:
            return 0
        time.sleep(600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--group")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--require-smoke", type=Path)
    parser.add_argument("--dispatch", metavar="SMOKE_ID")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.dispatch:
        raise SystemExit(dispatch(plan, args.dispatch, args.plan))
    if args.out is None:
        parser.error("--out required")
    if not args.dry_run and not args.smoke:
        receipt = (
            json.loads(args.require_smoke.read_text()) if args.require_smoke else {}
        )
        if not eligible_smoke({"state": "done", "rc": 0}, receipt, source_sha()):
            raise ValueError("matching real tiny smoke required")
    if args.smoke:
        arms = plan["smoke_arms"]
    elif args.group:
        arms = plan["groups"][args.group]["arms"]
    else:
        arms = [arm for group in plan["groups"].values() for arm in group["arms"]]
    result = {
        "source_sha": source_sha(),
        "smoke": args.smoke,
        "dry_run": args.dry_run,
        "complete": False,
        "arms": [],
    }
    write(args.out, result)
    for arm in arms:
        result["arms"].extend(run_arms([arm], dry_run=args.dry_run))
        write(args.out, result)
    result["complete"] = all(
        row["rc"] == 0 and row["complete"] for row in result["arms"]
    )
    write(args.out, result)
    raise SystemExit(0 if result["complete"] else 1)


if __name__ == "__main__":
    main()
