"""Log line to stream-reader latency (a gpuq job; no MLX): how long after a server log record is
emitted does a `GET /v1/yunshu/logs/stream` reader get it?

    PYTHONPATH=<tree>/python python scripts/research/front_log_latency.py [--n 40]

Runs a tiny app with the real admin_logs router and ring in a child uvicorn process (port 18996),
emits one record per 0.37 s through a route, and prints the latency distribution in ms.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

PORT = 18996


def serve() -> None:
    import uvicorn
    from fastapi import FastAPI

    from yunshu_gateway import log_ring
    from yunshu_gateway.routers import admin_logs, models

    models._check_permission = lambda *a, **k: None  # type: ignore[assignment]
    admin_logs._check_permission = lambda *a, **k: None  # type: ignore[assignment]
    log_ring.install()
    app = FastAPI()
    app.include_router(admin_logs.router, prefix="/v1")
    log = logging.getLogger("lat")
    log.setLevel(logging.INFO)

    @app.get("/emit")
    def emit() -> dict:
        t = time.time()
        log.info("mark %.6f", t)
        return {"t": t}

    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


def percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--serve", action="store_true")
    a = ap.parse_args()
    if a.serve:
        serve()
        return 0
    proc = subprocess.Popen(
        [sys.executable, __file__, "--serve"], start_new_session=True
    )
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{PORT}/emit", timeout=1
                ).read()
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.2)
        else:
            print("FAIL server did not start")
            return 1
        lat: list[float] = []
        resp = urllib.request.urlopen(
            f"http://127.0.0.1:{PORT}/v1/yunshu/logs/stream", timeout=30
        )

        def emitter() -> None:
            for _ in range(a.n):
                time.sleep(0.37)
                urllib.request.urlopen(
                    f"http://127.0.0.1:{PORT}/emit", timeout=5
                ).read()

        threading.Thread(target=emitter, daemon=True).start()
        deadline = time.time() + a.n * 0.37 + 5
        while len(lat) < a.n and time.time() < deadline:
            line = resp.readline().decode()
            if line.startswith("data:"):
                rec = json.loads(line[5:])
                if rec["msg"].startswith("mark "):
                    lat.append((time.time() - float(rec["msg"].split()[1])) * 1000)
        if len(lat) < a.n * 0.9:
            print(f"FAIL only {len(lat)} of {a.n} records arrived")
            return 1
        print(
            json.dumps(
                {
                    "n": len(lat),
                    "p50_ms": round(statistics.median(lat), 1),
                    "p95_ms": round(percentile(lat, 0.95), 1),
                    "max_ms": round(max(lat), 1),
                }
            )
        )
        return 0
    finally:
        os.killpg(proc.pid, signal.SIGKILL)


if __name__ == "__main__":
    raise SystemExit(main())
