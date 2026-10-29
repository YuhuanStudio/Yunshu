"""Does the console jump under a REAL engine's bursty updates? (run through gpuq with a small model)

    GPUQ_DIR=/Volumes/P5Plus/yunshu-gpuq GPUQ_OWNER=brand scripts/dev/gpuq submit --label brand-tiny-live-jump \
        --priority 0 -- python scripts/research/brand_live_jump.py RUN_NAME --model /Volumes/P5Plus/models/Qwen3.5-0.8B-MLX-bf16

Starts `yunshu serve -m MODEL` (the console process comes with it), builds the console, drives a bursty
load of short streamed chats (bursts of 2-8 concurrent requests, one every 100-400 ms inside a burst, a
few longer ones, a cancel, short idle gaps) and, while it runs, records the boxes of every visible
element, every animation started and every changing number (frontend/scripts/live-jump-real.mjs, the
same rules as browser/live-jump.spec.ts) on the overview and requests pages.

Output: /Volumes/P5Plus/yunshu-build/claude-alt/brand/live/RUN_NAME/{jump.json, strip/*.jpg, summary.json}.
Exit 0 only when the load ran clean AND every jump rule passed.
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = Path("/Volumes/P5Plus/yunshu-build/claude-alt/brand/live")
MODEL = "/Volumes/P5Plus/models/Qwen3.5-0.8B-MLX-bf16"


def build_plan(seed: int, seconds: float) -> list[tuple[float, str]]:
    """(seconds after start, kind) of every request: bursts of 2-8 requests 100-400 ms apart, 8-20 s apart."""
    rng = random.Random(seed)
    plan: list[tuple[float, str]] = []
    t = 2.0
    while t < seconds - 8:
        n = rng.choice([2, 3, 4, 6, 8])
        at = t
        for i in range(n):
            kind = (
                "long"
                if i == 0 and n >= 4
                else "cancel"
                if i == n - 1 and n >= 6
                else "short"
            )
            plan.append((round(at, 2), kind))
            at += rng.uniform(0.1, 0.4)
        t += rng.uniform(8, 20)
    return plan


def _get(url: str, timeout: float = 5.0):
    import urllib.request  # noqa: PLC0415

    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def request(base: str, model: str, kind: str, results: list[dict]) -> None:
    host, port = base.split("//")[1].split(":")
    prompt = {
        "short": "Say hello in one short sentence.",
        "long": "Write a detailed essay about tides.",
        "cancel": "Count slowly from 1 to 500.",
    }[kind]
    body = {
        "model": model,
        "stream": True,
        "max_tokens": {"short": 48, "long": 400, "cancel": 400}[kind],
        "messages": [{"role": "user", "content": prompt}],
    }
    events = 0
    try:
        c = http.client.HTTPConnection(host, int(port), timeout=300)
        c.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(body),
            {"content-type": "application/json"},
        )
        r = c.getresponse()
        status = r.status
        for raw in r:
            if raw.startswith(b"data:"):
                events += 1
                if kind == "cancel" and events >= 20:
                    break
        c.close()
        results.append({"kind": kind, "status": status, "events": events})
    except Exception as exc:  # noqa: BLE001
        results.append({"kind": kind, "error": repr(exc)[:120]})


def judge(results: list[dict], jump_rc: int) -> list[str]:
    """Problems with the run itself (the jump rules are judged by live-jump-real.mjs): pure, unit-tested."""
    problems = []
    if len(results) < 6:
        problems.append(f"only {len(results)} requests ran")
    bad = [r for r in results if r.get("error") or r.get("status") != 200]
    if bad:
        problems.append(f"{len(bad)} requests failed: {bad[:2]}")
    if jump_rc:
        problems.append(f"jump rules failed (rc {jump_rc}), see jump.json")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--engine-port", type=int, default=18990)
    ap.add_argument("--console-port", type=int, default=18992)
    ap.add_argument("--seconds", type=float, default=90)
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--startup-timeout", type=float, default=300)
    a = ap.parse_args(argv)
    out = OUT_ROOT / a.run
    out.mkdir(parents=True, exist_ok=True)
    home = out / "tmp"
    home.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "python"),
        "YUNSHU_CONSOLE_DB": str(home / "history.sqlite"),
        "YUNSHU_CONSOLE_PORT": str(a.console_port),
    }
    engine = f"http://127.0.0.1:{a.engine_port}"
    console = f"http://127.0.0.1:{a.console_port}"
    procs: list[subprocess.Popen] = []
    try:
        build = subprocess.run(
            ["node", "./node_modules/vite/bin/vite.js", "build"],
            cwd=ROOT / "frontend",
            capture_output=True,
            text=True,
        )
        if build.returncode:
            print("FAIL frontend build\n" + build.stderr[-800:])
            return 1
        log = open(home / "engine.log", "wb")  # noqa: SIM115
        eng = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "yunshu_cli",
                "serve",
                "-m",
                a.model,
                "--port",
                str(a.engine_port),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        procs.append(eng)
        t0 = time.time()
        while True:
            if eng.poll() is not None:
                print("FAIL engine exited early")
                return 1
            if time.time() - t0 > a.startup_timeout:
                print("FAIL engine did not come up")
                return 1
            try:
                if _get(f"{engine}/health", 3).get("status") and _get(
                    f"{console}/v1/yunshu/console", 3
                ).get("up"):
                    break
            except Exception:  # noqa: BLE001
                time.sleep(1.0)
        model_id = _get(f"{engine}/v1/models")["data"][0]["id"]
        print(f"engine up in {time.time() - t0:.0f}s, model {model_id}", flush=True)
        warm: list[dict] = []
        request(engine, model_id, "short", warm)
        if not warm or warm[0].get("status") != 200:
            print(f"FAIL warm-up request: {warm}")
            return 1
        jump = subprocess.Popen(
            [
                "node",
                "scripts/live-jump-real.mjs",
                "--base",
                console,
                "--out",
                str(out),
                "--seconds",
                str(int(a.seconds / 2) - 8),
            ],
            cwd=ROOT / "frontend",
            stdout=open(out / "jump.log", "wb"),  # noqa: SIM115
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        procs.append(jump)
        results: list[dict] = []
        threads: list[threading.Thread] = []
        start = time.time()
        for at, kind in build_plan(a.seed, a.seconds):
            while time.time() - start < at:
                time.sleep(0.05)
            th = threading.Thread(
                target=request, args=(engine, model_id, kind, results), daemon=True
            )
            th.start()
            threads.append(th)
        for th in threads:
            th.join(timeout=300)
        jump_rc = jump.wait(timeout=1200)
        problems = judge(results, jump_rc)
        summary = {
            "requests": len(results),
            "model": model_id,
            "jump_rc": jump_rc,
            "problems": problems,
            "engine_last": (_get(f"{engine}/v1/yunshu/status").get("last") or {}).get(
                "request_id"
            ),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))
        print("FAIL" if problems else f"PASS: {out}")
        return 1 if problems else 0
    finally:
        for p in reversed(procs):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(p.pid, 15)
        time.sleep(3)
        for p in procs:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(p.pid, 9)


if __name__ == "__main__":
    raise SystemExit(main())
