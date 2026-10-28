"""Server start-up and idle-memory probe: ready time, import cost, and the RSS / MLX / ledger figures at
ready, after one request, and after an idle wait. One run = one server; the server is killed on exit.

  python startup_probe.py --model PATH --port 18995 --out DIR [--idle 20] [--env K=V ...]
Writes DIR/startup.json and DIR/server.log (-X importtime goes to the log's start).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

_IMPORT = re.compile(r"^import time:\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\S.*)$")


def top_imports(text: str, n: int = 25) -> list[dict]:
    """Cumulative import cost (ms) of the slowest modules in a ``-X importtime`` log."""
    rows = []
    for line in text.splitlines():
        m = _IMPORT.match(line)
        if m:
            rows.append(
                {"module": m.group(3).strip(), "cum_ms": int(m.group(2)) / 1000.0}
            )
    rows.sort(key=lambda r: -r["cum_ms"])
    return rows[:n]


def rss_mb(pid: int) -> float:
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True
    ).stdout.strip()
    return int(out) / 1024.0 if out else 0.0


def get_json(url: str, timeout: float = 10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def snapshot(proc: subprocess.Popen, url: str) -> dict:
    return {
        "rss_mb": round(rss_mb(proc.pid), 1),
        "ledger": get_json(url + "/v1/yunshu/memory"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=18995)
    ap.add_argument("--out", required=True)
    ap.add_argument("--idle", type=float, default=20.0)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--ready-timeout", type=float, default=900.0)
    a = ap.parse_args()
    if not 18990 <= a.port <= 18999:
        print("port must be 18990-18999", file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    tree = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(
        os.environ,
        PYTHONPATH=os.path.join(tree, "python"),
        YUNSHU_AUTH_DISABLED="true",
        HF_HUB_OFFLINE="1",
    )
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    log_path = os.path.join(a.out, "server.log")
    log = open(log_path, "w")  # noqa: SIM115
    t0 = time.time()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-X",
            "importtime",
            "-m",
            "yunshu_cli",
            "serve",
            "--model",
            a.model,
            "--port",
            str(a.port),
        ],
        cwd=tree,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    url = f"http://127.0.0.1:{a.port}"
    res: dict = {"model": a.model, "env": a.env}
    rc = 1
    try:
        http_s = None
        while time.time() - t0 < a.ready_timeout:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited rc={proc.returncode}")
            try:
                urllib.request.urlopen(url + "/health", timeout=2)
                http_s = http_s or time.time() - t0
            except Exception:  # noqa: BLE001
                pass
            try:
                urllib.request.urlopen(url + "/health/ready", timeout=2)
                res["ready_s"] = round(time.time() - t0, 2)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.25)
        else:
            raise RuntimeError("server not ready")
        res["http_up_s"] = round(http_s, 2) if http_s else None
        res["at_ready"] = snapshot(proc, url)
        body = json.dumps(
            {
                "model": "x",
                "messages": [{"role": "user", "content": "Say hi."}],
                "max_tokens": 16,
                "temperature": 0,
            }
        ).encode()
        t1 = time.time()
        req = urllib.request.Request(
            url + "/v1/chat/completions",
            body,
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            ok = json.loads(r.read())
        res["first_request_s"] = round(time.time() - t1, 2)
        if not ok.get("choices"):
            raise RuntimeError(f"first request had no choices: {ok}")
        res["after_request"] = snapshot(proc, url)
        time.sleep(a.idle)
        res["after_idle"] = snapshot(proc, url)
        res["idle_s"] = a.idle
        rc = 0
    except BaseException as exc:  # noqa: BLE001
        res["error"] = repr(exc)
        print("FAIL", repr(exc), file=sys.stderr)
    finally:
        with __import__("contextlib").suppress(Exception):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        log.close()
    res["top_imports"] = top_imports(Path(log_path).read_text(errors="replace"))
    with open(os.path.join(a.out, "startup.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "top_imports"}, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
