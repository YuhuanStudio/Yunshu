"""Candidate-only generic-serving pilot with a sampled whole-machine reserve guard.

This is an absolute admission/correctness probe, not a cross-model speed verdict.
It imports no MLX; the child server runs only inside its enclosing gpuq job.
"""

import argparse
import contextlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from bigmoe_census import GIB, census
from process_memory import process_tree_memory


def available_bytes(text):
    """Reclaimable pages; purgeable is a subset and must not be counted twice."""
    page = int(text.split("page size of ")[1].split()[0])
    values = {}
    for line in text.splitlines():
        if line.startswith(("Pages free:", "Pages inactive:", "Pages speculative:")):
            key, value = line.split(":", 1)
            values[key] = int(value.strip().rstrip("."))
    return page * sum(
        values[k] for k in ("Pages free", "Pages inactive", "Pages speculative")
    )


def sample_available():
    return available_bytes(subprocess.check_output(["vm_stat"], text=True, timeout=5))


def validate(rows, log):
    decode = [r for r in rows if r.get("part") == "decode"]
    if not any(r.get("complete") is True for r in rows):
        return "missing complete record"
    if not decode or any(
        not r.get("ct") or not str(r.get("text", "")).strip() for r in decode
    ):
        return "empty generation"
    if "VLM batch runner:" not in log or "draft=off" not in log:
        return "generic VLM runner/draft-off not confirmed"
    return ""


def stop_owned(proc):
    """Only this child and descendants; never signal an inherited process group."""
    rows = [
        tuple(map(int, line.split()))
        for line in subprocess.check_output(
            ["ps", "-axo", "pid=,ppid="], text=True
        ).splitlines()
    ]
    owned = {proc.pid}
    while True:
        expanded = owned | {pid for pid, parent in rows if parent in owned}
        if expanded == owned:
            break
        owned = expanded
    for pid in owned:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        for pid in owned:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        proc.wait(timeout=5)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reserve-gib", type=float, default=30)
    parser.add_argument("--guard-gib", type=float, default=2)
    parser.add_argument("--workspace-gib", type=float, default=8)
    return parser


def run(args):
    inventory = census(args.model)
    available = sample_available()
    result = dict(
        schema=1,
        device="m5",
        complete=False,
        inventory=inventory,
        reserve_gib=args.reserve_gib,
        guard_gib=args.guard_gib,
        min_available_bytes=available,
        samples=0,
        physical_footprint_peak_bytes=0,
    )
    threshold = (args.reserve_gib + args.guard_gib) * GIB
    need = inventory["base_bytes"] + args.workspace_gib * GIB + threshold
    if available < need:
        result["failure"] = (
            "admission: insufficient reclaimable memory for weights, workspace and reserve"
        )
        return result
    raw = args.out.with_suffix(".raw.jsonl")
    work = args.out.parent / (args.out.stem + ".tfbench")
    env = dict(
        os.environ,
        TFB_YUNSHU_SRC=str(args.src),
        TFB_OUT=str(work),
        TFB_PORT_LAST="18996",
    )
    cmd = [
        sys.executable,
        str(args.src.parent / "scripts/research/tfbench.py"),
        "--engine",
        "yunshu",
        "--model",
        str(args.model),
        "--out",
        str(raw),
        "--part",
        "decode",
        "--smoke",
        "--rep",
        "0",
    ]
    for setting in [
        "YUNSHU_VLM_DRAFT=off",
        "YUNSHU_VLM_APC_MEMORY_GB=0",
        "YUNSHU_PREFILL_BUFFER_CACHE_GB=0",
        "YUNSHU_FOOTPRINT_SAMPLE_MS=20",
    ]:
        cmd += ["--env", setting]
    last_progress = -math.inf
    with args.out.with_suffix(".harness.log").open("w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                current = sample_available()
                result["samples"] += 1
                footprint = process_tree_memory(proc.pid)[
                    "physical_footprint_sum_bytes"
                ]
                result["physical_footprint_peak_bytes"] = max(
                    result["physical_footprint_peak_bytes"], footprint
                )
                result["min_available_bytes"] = min(
                    current, result["min_available_bytes"]
                )
                now = time.monotonic()
                if now - last_progress >= 30:
                    print(
                        json.dumps(
                            dict(
                                event="memory_sample",
                                available_bytes=current,
                                footprint_bytes=footprint,
                            )
                        ),
                        flush=True,
                    )
                    last_progress = now
                if current < threshold:
                    result["failure"] = "reserve guard crossed"
                    stop_owned(proc)
                    break
                time.sleep(0.25)
            result["rc"] = proc.wait()
        except BaseException:
            stop_owned(proc)
            raise
    result["raw"] = str(raw)
    logs = list(work.rglob("server-*.log"))
    log_text = "\n".join(p.read_text(errors="replace") for p in logs)
    result["server_logs"] = [str(p) for p in logs]
    if result.get("failure"):
        return result
    if result["rc"] != 0:
        result["failure"] = "harness failed"
        return result
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    error = validate(rows, log_text)
    if error:
        result["failure"] = error
    else:
        result.update(
            complete=True,
            engaged="generic VLM runner; draft=off; APC disabled",
            cells=[r for r in rows if r.get("part") == "decode"],
        )
    return result


def main():
    parser = build_parser()
    args = parser.parse_args()
    budgets = (args.reserve_gib, args.guard_gib, args.workspace_gib)
    if (
        not all(math.isfinite(v) and v >= 0 for v in budgets)
        or args.reserve_gib < 30
        or args.guard_gib < 2
    ):
        parser.error("finite budgets required: reserve >= 30 GiB, guard >= 2 GiB")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args)
    except Exception as exc:
        result = dict(schema=1, device="m5", complete=False, failure=str(exc))
    args.out.write_text(json.dumps(result) + "\n")
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
