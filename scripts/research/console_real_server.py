"""Real-server check of the console process (run through gpuq with a small model).

    gpuq run --label console-real -- python scripts/research/console_real_server.py MODEL \
        --engine-port 18996 --console-port 18997 --requests 6

Starts `yunshu serve -m MODEL` (which starts the console process next to it), sends requests to the
ENGINE with no browser anywhere, then asks the CONSOLE process for what it recorded:

  1. every request is in the recorded request log with sane metadata (ids, tokens, TTFT, no content);
  2. the metrics history has a row about every second for the whole run, with real memory numbers;
  3. a streamed request through the console's proxy delivers its first event before the stream ends;
  4. the engine is killed (SIGKILL): the console process stays up, reports it, still answers its
     history, and records the outage as an event.

Everything is judged by :func:`evaluate` (pure, unit-tested on CPU); the script exits non-zero on the
first failed check and kills every process it started, whatever happens.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def get(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def post(url: str, body: dict, timeout: float = 300.0):
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def evaluate(
    n_requests: int,
    history: dict,
    metrics: dict,
    run_seconds: float,
    stream: dict,
    after_kill: dict,
) -> list[str]:
    """Problems found (empty = pass). Pure: takes what the console process answered."""
    bad: list[str] = []
    rows = history.get("data") or []
    if len(rows) < n_requests:
        bad.append(f"request log has {len(rows)} rows, expected at least {n_requests}")
    for r in rows:
        if not r.get("request_id") or r.get("t") is None:
            bad.append(f"request row without id or time: {r}")
        if (r.get("completion_tokens") or 0) <= 0 or (r.get("prompt_tokens") or 0) <= 0:
            bad.append(f"request {r.get('request_id')} has no token counts")
        if r.get("ttft_ms") is None or r["ttft_ms"] <= 0:
            bad.append(f"request {r.get('request_id')} has no TTFT")
        blob = json.dumps(r).lower()
        if any(k in blob for k in ('"messages"', '"content"', '"output"', '"prompt":')):
            bad.append(f"request row carries content: {sorted(r)}")
    t = (metrics.get("series") or {}).get("t") or []
    expected = max(5, int(run_seconds * 0.6))
    if len(t) < expected:
        bad.append(
            f"metrics history has {len(t)} rows over {run_seconds:.0f} s, expected >= {expected}"
        )
    active = [v for v in (metrics.get("series") or {}).get("active_gb") or [] if v]
    if not active or max(active) < 0.05:
        bad.append("no Metal memory recorded")
    if metrics.get("gaps"):
        bad.append(f"unexpected gaps while the engine was up: {metrics['gaps']}")
    if not stream.get("first_event_before_end"):
        bad.append(f"proxy did not stream: {stream}")
    if after_kill.get("console_up") is not True:
        bad.append("console process did not survive the engine being killed")
    if after_kill.get("state_up") is not False:
        bad.append(f"console did not notice the engine was gone: {after_kill}")
    if "engine_unreachable" not in (after_kill.get("event_kinds") or []):
        bad.append(f"no engine_unreachable event: {after_kill.get('event_kinds')}")
    if (after_kill.get("history_rows") or 0) < n_requests:
        bad.append("history stopped answering after the engine died")
    return bad


def rss_mib(pid: int) -> float | None:
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True
        )
        return round(int(out.stdout.strip()) / 1024, 1)
    except (ValueError, OSError):
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("model")
    ap.add_argument("--engine-port", type=int, default=18996)
    ap.add_argument("--console-port", type=int, default=18997)
    ap.add_argument("--requests", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--startup-timeout", type=float, default=600.0)
    ap.add_argument("--out", help="write the evidence JSON here")
    args = ap.parse_args(argv)

    home = Path(
        tempfile.mkdtemp(prefix="console-real-", dir=os.environ.get("CONSOLE_REAL_TMP"))
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "python"),
        "YUNSHU_CONSOLE_DB": str(home / "history.sqlite"),
        "YUNSHU_CONSOLE_PORT": str(args.console_port),
    }
    engine_url = f"http://127.0.0.1:{args.engine_port}"
    console_url = f"http://127.0.0.1:{args.console_port}"
    log = open(home / "engine.log", "wb")  # noqa: SIM115
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "yunshu_cli",
            "serve",
            "-m",
            args.model,
            "--port",
            str(args.engine_port),
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    evidence: dict = {"model": args.model, "home": str(home)}
    try:
        t0 = time.time()
        while time.time() - t0 < args.startup_timeout:
            if proc.poll() is not None:
                print(f"FAIL engine exited early; see {home / 'engine.log'}")
                return 1
            try:
                if get(f"{engine_url}/health", 3).get("status") and get(
                    f"{console_url}/v1/yunshu/console", 3
                ).get("up"):
                    break
            except Exception:
                time.sleep(1.0)
        else:
            print("FAIL engine or console did not come up")
            return 1
        model_id = get(f"{engine_url}/v1/models")["data"][0]["id"]
        started = time.time()

        # 1. requests with no browser anywhere
        for i in range(args.requests):
            out = post(
                f"{engine_url}/v1/chat/completions",
                {
                    "model": model_id,
                    "max_tokens": args.max_tokens,
                    "messages": [
                        {"role": "user", "content": f"Count from 1 to 20, run {i}."}
                    ],
                },
            )
            assert out["choices"][0]["message"], out
        time.sleep(3.0)

        # 3. a streamed request through the console's proxy
        stream = {"first_event_before_end": False}
        req = urllib.request.Request(
            f"{console_url}/v1/chat/completions",
            json.dumps(
                {
                    "model": model_id,
                    "max_tokens": 64,
                    "stream": True,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Write a long sentence about the sea.",
                        }
                    ],
                }
            ).encode(),
            {"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            first_at = None
            events = 0
            for raw in r:
                if raw.startswith(b"data:"):
                    events += 1
                    first_at = first_at or time.time()
            end_at = time.time()
        stream.update(
            events=events, first_to_end_s=round(end_at - (first_at or end_at), 3)
        )
        stream["first_event_before_end"] = (
            events > 2 and (end_at - (first_at or end_at)) > 0.02
        )
        time.sleep(2.0)

        history = get(f"{console_url}/v1/yunshu/requests/history?limit=100")
        run_seconds = time.time() - started
        metrics = get(f"{console_url}/v1/yunshu/metrics/history?since={started - 2}")
        console_pid = get(f"{console_url}/v1/yunshu/console")
        evidence.update(
            requests_recorded=history.get("count"),
            metrics_rows=len((metrics.get("series") or {}).get("t") or []),
            tier=metrics.get("tier"),
            stream=stream,
            run_seconds=round(run_seconds, 1),
            console=console_pid.get("store"),
        )
        children = subprocess.run(
            ["pgrep", "-P", str(get(f"{engine_url}/v1/yunshu/status")["pid"])],
            capture_output=True,
            text=True,
        )
        pids = [int(x) for x in children.stdout.split() if x.strip()]
        evidence["console_rss_mib"] = rss_mib(pids[0]) if pids else None

        # 4. kill the engine
        engine_pid = get(f"{engine_url}/v1/yunshu/status").get("pid")
        os.kill(engine_pid, signal.SIGKILL)
        time.sleep(4.0)
        after: dict = {}
        try:
            state = get(f"{console_url}/v1/yunshu/console")
            after["console_up"] = True
            after["state_up"] = state.get("up")
            m = get(f"{console_url}/v1/yunshu/metrics/history?since={started - 2}")
            after["event_kinds"] = [e["kind"] for e in m.get("events", [])]
            after["history_rows"] = get(
                f"{console_url}/v1/yunshu/requests/history?limit=100"
            ).get("count")
        except Exception as exc:  # noqa: BLE001
            after["console_up"] = False
            after["error"] = repr(exc)
        evidence["after_kill"] = after

        problems = evaluate(args.requests, history, metrics, run_seconds, stream, after)
        evidence["problems"] = problems
        evidence["complete"] = True
        if args.out:
            Path(args.out).write_text(json.dumps(evidence, indent=1))
        print(json.dumps(evidence, indent=1))
        if problems:
            print("FAIL " + "; ".join(problems))
            return 1
        print("PASS console process real-server check")
        return 0
    finally:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                break
            time.sleep(1.5)
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
