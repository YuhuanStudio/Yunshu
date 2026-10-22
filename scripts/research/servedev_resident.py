#!/usr/bin/env python3
"""New measurement: launchd resident plus synthetic research allocation, inside gpuq.

Does not edit live serving.json or live jobs. Own launchd label, external state,
dev port, research process group and isolated fake queue records only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "python"))


def in_gpuq() -> bool:
    """gpuq sets GPUQ_RC / GPUQ_DEVICE for the job (it never sets a job id variable)."""
    return bool(os.environ.get("GPUQ_RC") and os.environ.get("GPUQ_DEVICE"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model")
    p.add_argument("--out", type=Path)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--ctx", default="1024,32768")
    p.add_argument("--research-gb", type=float, default=18)
    p.add_argument("--research-child", action="store_true")
    p.add_argument("--resident", choices=("process", "launchd"), default="process")
    p.add_argument("--dry-run", action="store_true")
    return p


def judge(value):
    rows = value.get("rows", [])
    if value.get("complete") is not True or not rows:
        return False, "missing complete evidence"
    for r in rows:
        if not r.get("identity") or r.get("pause_latency_ms", 1e9) > 2000:
            return False, "identity / bounded pause failed"
        if r.get("headroom_gb", -1) < 16:
            return False, "headroom below 16 GB"
    return True, ""


def researcher(gb):
    import mlx.core as mx

    allocation = mx.ones((int(gb * (1 << 30) / 4),), dtype=mx.float32)
    mx.eval(allocation)
    a = mx.ones((1024, 1024))
    print(
        json.dumps({"ready": True, "active_bytes": mx.get_active_memory()}), flush=True
    )
    while True:
        mx.eval(a @ a)
        # Retain the unified-memory allocation for the entire job.
        if allocation.size == 0:
            raise RuntimeError("empty research allocation")


def request(url, body):
    import httpx

    t0 = time.perf_counter()
    first = None
    ids = []
    done = False
    finish = None
    stats = {}
    with httpx.stream(
        "POST",
        url + "/v1/chat/completions",
        json={
            **body,
            "stream": True,
            "stream_options": {"include_usage": True},
            "logprobs": True,
        },
        timeout=240,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                done = True
                continue
            obj = json.loads(line[6:])
            stats = obj.get("x_yunshu", stats)
            if obj.get("usage"):
                stats = obj["usage"].get("x_yunshu", stats)
            for choice in obj.get("choices", []):
                delta = choice.get("delta", {})
                if first is None and any(
                    delta.get(k) for k in ("content", "reasoning_content", "tool_calls")
                ):
                    first = time.perf_counter()
                for lp in (choice.get("logprobs") or {}).get("content", []):
                    if "token_id" not in lp:
                        raise RuntimeError("exact token IDs unavailable")
                    ids.append(lp["token_id"])
                finish = choice.get("finish_reason") or finish
    if not done or not finish or first is None or not ids:
        raise RuntimeError("incomplete response or no token IDs")
    return {
        "ttft_ms": (first - t0) * 1000,
        "ids": ids,
        "finish": finish,
        "stats": stats,
    }


def run(a):
    if not in_gpuq():
        raise RuntimeError("GPU work requires gpuq")
    from research.agentic.servers import free_ports

    name = "dev-servedev-" + uuid.uuid4().hex[:12]
    directory = a.out.parent / name
    directory.mkdir(parents=True)
    port = free_ports(1, wait_s=60)[0]
    url = f"http://127.0.0.1:{port}"
    activity = directory / "activity.json"
    config = directory / "config.toml"
    config.write_text("")
    cli = [
        sys.executable,
        "-m",
        "yunshu_cli",
        "service",
        "--instance",
        name,
        "--state-dir",
        str(directory),
    ]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "python"))
    # A local gate exercises the candidate scheduler without touching the live daemon.
    spec = importlib.util.spec_from_file_location(
        "servedev_gpuq", ROOT / "scripts/dev/gpuq.py"
    )
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    g.LOGS = directory
    gate = g.ServingGate()
    gate.configure(
        {
            "activity_files": [str(activity)],
            "poll_s": 0.02,
            "idle_start_s": 0.2,
            "idle_resume_s": 0.2,
            "reserve_gb": 16,
        }
    )
    from yunshu_gateway.resident_activity import atomic_json

    atomic_json(activity, {"schema": 1, "ticket": "initial", "active": 0})
    child = None
    pauser = None
    stop = threading.Event()
    failures = []
    pause_latencies = []

    def scheduler():
        last = ""
        while not stop.wait(0.02):
            try:
                now = time.time()
                gate.poll(now)
                if pauser:
                    pauser.step(gate, False, now)
                else:
                    gate.acknowledge(True)
                state = json.loads(activity.read_text())
                ack_path = Path(str(activity) + ".ack")
                ack = json.loads(ack_path.read_text()) if ack_path.exists() else {}
                if (
                    state["active"]
                    and ack.get("ticket") == state["ticket"]
                    and last != state["ticket"]
                ):
                    last = state["ticket"]
                    pause_latencies.append((ack["updated"] - state["updated"]) * 1000)
            except Exception as e:
                failures.append(str(e))
                stop.set()

    worker = threading.Thread(target=scheduler)
    installed = False
    resident = None
    rows = []
    try:
        args = [
            *cli,
            "install",
            "--model",
            a.model,
            "--port",
            str(port),
            "--config",
            str(config),
        ]
        for key, value in {
            "YUNSHU_RESEARCH_ACTIVITY_FILE": str(activity),
            "YUNSHU_REPLAY_TOKEN_IDS": "1",
            "YUNSHU_VLM_APC_DISK_DIR": str(directory / "apc"),
            "YUNSHU_SERVE_LOG": "1",
            "YUNSHU_SERVE_LOG_DIR": str(directory),
            "YUNSHU_MEDIA_DIR": str(directory / "media"),
        }.items():
            args += ["--set", f"{key}={value}"]
        if a.resident == "launchd":
            r = subprocess.run(args, env=env, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(
                    "launchd install failed: " + (r.stdout + r.stderr)[-1500:]
                )
            installed = True
        else:  # plain child process in this gpuq job: same server, no launchd dependency
            senv = dict(env)
            for i, tok in enumerate(args):
                if tok == "--set":
                    k, v = args[i + 1].split("=", 1)
                    senv[k] = v
            resident_log = (directory / "yunshu.log").open("w")
            resident = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "yunshu_cli",
                    "serve",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--model",
                    a.model,
                ],
                env=senv,
                stdout=resident_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        worker.start()
        import httpx

        deadline = time.monotonic() + 180
        while True:
            try:
                r = httpx.get(url + "/health", timeout=2)
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if resident is not None and resident.poll() is not None:
                raise RuntimeError(
                    "resident exited: " + (directory / "yunshu.log").read_text()[-3000:]
                )
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "resident not ready; "
                    + (directory / "yunshu.log").read_text()[-3000:]
                )
            time.sleep(0.5)
        bodies = [
            {
                "model": a.model,
                "messages": [
                    {"role": "user", "content": "hello " * n + "Reply briefly."}
                ],
                "max_tokens": 16,
                "temperature": 0,
            }
            for n in map(int, a.ctx.split(","))
        ]
        for (
            body
        ) in bodies:  # warm each shape so the baseline is not a cold first request
            request(url, body)
        for rep in range(a.reps):
            for body in bodies:
                baseline = request(url, body)
                free = g.free_memory_gb()
                if free is None or free - a.research_gb < 16:
                    raise RuntimeError("research memory admission rejected")
                child_log = (directory / f"research-{rep}-{len(rows)}.log").open("w")
                child = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--research-child",
                        "--research-gb",
                        str(a.research_gb),
                    ],
                    stdout=child_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                deadline = time.monotonic() + 60
                while "ready" not in Path(child_log.name).read_text():
                    if child.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError(
                            "research allocation failed: "
                            + Path(child_log.name).read_text()
                        )
                    time.sleep(0.1)
                headroom = g.free_memory_gb()
                job = {
                    "id": "synthetic-research",
                    "pid": child.pid,
                    "priority": -1,
                    "state": "running",
                }
                pauser = g.Pauser(job, directory / "synthetic-job.json")
                result = request(url, body)
                if not pauser.pauses:
                    raise RuntimeError("research was never paused")
                # Observe resumption after the configured idle window.
                deadline = time.monotonic() + 3
                while pauser.paused and time.monotonic() < deadline:
                    time.sleep(0.02)
                if pauser.paused:
                    raise RuntimeError("research did not resume after idle")
                row = {
                    "rep": rep,
                    "prompt_tokens": result["stats"].get("prompt_tokens"),
                    "baseline_ttft_ms": baseline["ttft_ms"],
                    "research_ttft_ms": result["ttft_ms"],
                    "ttft_penalty_ms": result["ttft_ms"] - baseline["ttft_ms"],
                    "identity": baseline["ids"] == result["ids"]
                    and baseline["finish"] == result["finish"],
                    "pause_latency_ms": pause_latencies[-1],
                    "headroom_gb": headroom,
                    "research_allocation_gb": a.research_gb,
                    "resumed": True,
                    "spec_mode": (result["stats"].get("speculative") or {}).get("mode"),
                }
                print(json.dumps(row), flush=True)
                rows.append(row)
                if not row["identity"]:
                    raise RuntimeError("identity failed on first paired request")
                old_child = child
                pauser = None
                os.killpg(old_child.pid, signal.SIGCONT)
                os.killpg(old_child.pid, signal.SIGTERM)
                old_child.wait(timeout=10)
                child = None
                child_log.close()
        if failures:
            raise RuntimeError(str(failures))
        value = {
            "complete": True,
            "device": "m5",
            "rows": rows,
            "resident": a.resident,
            "research_kind": "synthetic allocation + matmul",
            "log": str(directory / "yunshu.log"),
        }
        ok, reason = judge(value)
        value.update(ok=ok, reason=reason)
        a.out.write_text(json.dumps(value) + "\n")
        return 0 if ok else 1
    finally:
        stop.set()
        if worker.is_alive():
            worker.join(timeout=3)
        if child and child.poll() is None:
            os.killpg(child.pid, signal.SIGCONT)
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=10)
        if installed:
            subprocess.run([*cli, "uninstall"], env=env, check=True)
        if resident and resident.poll() is None:
            os.killpg(resident.pid, signal.SIGTERM)
            try:
                resident.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(resident.pid, signal.SIGKILL)


def main(argv=None):
    a = parser().parse_args(argv)
    if a.dry_run:
        print(
            json.dumps(
                {
                    "model": a.model,
                    "ctx": a.ctx,
                    "reps": a.reps,
                    "research_gb": a.research_gb,
                }
            )
        )
        return 0
    if a.research_child:
        if not in_gpuq():
            raise RuntimeError("research allocation requires gpuq")
        researcher(a.research_gb)
        return 0
    if not a.model or not a.out or a.reps < 1 or a.research_gb <= 0:
        raise ValueError("model, output, positive reps and allocation required")
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
