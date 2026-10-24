"""Real check of observe mode: start the serve command with --observe and prove every observability feature is live.

Usage (through gpuq):  observe_check.py --model PATH --port 18991 --home DIR --out FILE.json

Starts the server with an isolated HOME, sends one streamed and one logprobs chat request while polling
/debug/requests, then asserts: every /debug route answers 200 without a token (loopback), /v1/yunshu/status
says observe, the effective config shows source "observe" for the six settings, host telemetry has a
reading, the serve log and the generation capture hold lines, the audit file exists, and logprobs carry
exact token ids. Exits nonzero (and writes the failures) when anything is missing. The server is killed
on exit.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEBUG_ROUTES = (
    "engine",
    "system",
    "models",
    "requests",
    "kv-cache",
    "spec-decode",
    "memory-guard",
    "memory-census",
    "weight-residency",
    "ssd-cache",
    "per-model",
    "all",
)
OBSERVED = (
    "YUNSHU_DEBUG_ROUTES",
    "YUNSHU_TELEMETRY",
    "YUNSHU_SERVE_LOG",
    "YUNSHU_AUDIT_LOG_FILE",
    "YUNSHU_REPLAY_TOKEN_IDS",
    "YUNSHU_DEBUG_STREAM_CAPTURE",
)
YUNSHU = "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/yunshu"


def get(url: str, timeout: float = 30.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, None


def post(url: str, body: dict, timeout: float = 300.0):
    req = urllib.request.Request(
        url,
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode()


def count_lines(path: Path) -> int:
    return sum(1 for _ in path.open()) if path.exists() else 0


def verify(base: str, home: Path, in_flight_seen: list, logprobs_body: dict) -> list[str]:
    """Every failed expectation as a sentence; empty when observe mode is fully live."""
    bad: list[str] = []
    for route in DEBUG_ROUTES:
        code, _ = get(f"{base}/debug/{route}")
        if code != 200:
            bad.append(f"/debug/{route} -> {code}")
    code, st = get(f"{base}/v1/yunshu/status")
    if code != 200 or (st or {}).get("observe") is not True:
        bad.append(f"/v1/yunshu/status observe={(st or {}).get('observe')}")
    code, cfg = get(f"{base}/v1/yunshu/config?include=all")
    rows = {r["name"]: r for r in (cfg or {}).get("settings", (cfg or {}).get("data", []))}
    for name in OBSERVED:
        src = rows.get(name, {}).get("source")
        if src != "observe":
            bad.append(f"config {name} source={src}")
    code, host = get(f"{base}/v1/yunshu/host")
    tel = (host or {}).get("telemetry") or {}
    if tel.get("state") not in ("ok", "partial"):
        bad.append(f"telemetry state={tel.get('state')} reason={tel.get('reason')}")
    logs = home / ".yunshu" / "logs"
    if not list(logs.glob("*.jsonl*")) or not any(count_lines(p) for p in logs.glob("serve*")):
        bad.append(f"serve log empty under {logs}: {sorted(p.name for p in logs.glob('*'))}")
    if not (logs / "audit.log").exists():
        bad.append("audit.log was not created")
    if count_lines(logs / "stream_capture.jsonl") < 1:
        bad.append("stream_capture.jsonl has no line")
    ids = [
        t.get("token_id")
        for c in logprobs_body.get("choices", [])
        for t in (c.get("logprobs") or {}).get("content") or []
    ]
    if not ids or any(i is None for i in ids):
        bad.append(f"logprobs without exact token_id: {ids[:3]}")
    if not in_flight_seen:
        bad.append("/debug/requests never listed an in-flight request")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--home", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", action="store_true", help="leave the server running (console screenshots)")
    ap.add_argument("--hold", type=float, default=0.0, help="seconds to keep it up after the check")
    a = ap.parse_args()
    home = Path(a.home)
    home.mkdir(parents=True, exist_ok=True)
    # the tree this script lives in, not the checkout the binary was installed from
    tree = str(Path(__file__).resolve().parents[2] / "python")
    env = dict(
        os.environ,
        HOME=str(home),
        HF_HUB_OFFLINE="1",
        NO_PROXY="127.0.0.1",
        PYTHONPATH=tree,
    )
    for k in [k for k in env if k.startswith("YUNSHU_")]:
        del env[k]
    base = f"http://127.0.0.1:{a.port}"
    log = open(home / "server.log", "wb")
    proc = subprocess.Popen(
        [YUNSHU, "serve", "--observe", "--no-console", "-m", a.model, "--port", str(a.port)],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    result: dict = {"failures": ["did not run"]}
    try:
        t0 = time.time()
        while time.time() - t0 < 300:
            if proc.poll() is not None:
                raise SystemExit(f"server exited rc={proc.returncode}; see {home / 'server.log'}")
            try:
                if get(base + "/v1/models", 3)[0] == 200:
                    break
            except Exception:
                time.sleep(1)
        else:
            raise SystemExit("server not ready in 300 s")
        import threading

        seen: list = []
        stop = threading.Event()

        def poll() -> None:
            while not stop.is_set():
                code, body = get(base + "/debug/requests", 5)
                if code == 200 and (body or {}).get("in_flight"):
                    seen.append(body["in_flight"])
                time.sleep(0.2)

        th = threading.Thread(target=poll)
        th.start()
        msgs = [{"role": "user", "content": "Count from 1 to 40, one number per line."}]
        post(base + "/v1/chat/completions", {"model": "m", "messages": msgs, "max_tokens": 200, "stream": True})
        _, lp = post(
            base + "/v1/chat/completions",
            {"model": "m", "messages": msgs, "max_tokens": 16, "logprobs": True, "top_logprobs": 2},
        )
        stop.set()
        th.join()
        time.sleep(2)  # serve log / capture are written when a request finishes
        failures = verify(base, home, seen, json.loads(lp))
        result = {"failures": failures, "in_flight_samples": len(seen), "first_sample": seen[:1]}
        print(json.dumps(result, indent=2))
        if a.hold and not failures:
            # something in flight for the console screenshots
            long = {"model": "m", "messages": msgs, "max_tokens": 1500, "stream": True}
            threading.Thread(target=lambda: post(base + "/v1/chat/completions", long), daemon=True).start()
            for _ in range(int(a.hold // 20)):
                print("holding for screenshots", flush=True)
                time.sleep(20)
    finally:
        if not a.keep or result["failures"]:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        Path(a.out).write_text(json.dumps(result, indent=2))
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
