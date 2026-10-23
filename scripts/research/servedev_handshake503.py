#!/usr/bin/env python3
"""Real-server check of the research-activity handshake timeout path (inside gpuq, small model).

1. No scheduler acknowledges: a chat request must return 503 research_pause_timeout within the timeout and
   must not start inference.
2. gpuq's ServingGate acknowledges (nothing of ours running): the same request returns 200 with tokens.
Complements servedev_resident.py (pause before inference, resume after).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "python"))


def judge(r503: dict, r200: dict) -> tuple[bool, str]:
    if r503.get("status") != 503 or r503.get("code") != "research_pause_timeout":
        return False, f"expected 503 research_pause_timeout, got {r503}"
    if r503.get("elapsed_s", 99) > 5:
        return False, "503 took too long"
    if r200.get("status") != 200 or not r200.get("tokens"):
        return False, f"expected 200 with tokens after ack, got {r200}"
    return True, ""


def post(url: str, model: str) -> dict:
    import httpx

    t0 = time.perf_counter()
    r = httpx.post(
        url + "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 8,
            "temperature": 0,
        },
        timeout=120,
    )
    out = {"status": r.status_code, "elapsed_s": time.perf_counter() - t0}
    try:
        body = r.json()
    except ValueError:
        body = {}
    if r.status_code == 503:
        out["code"] = (body.get("error") or {}).get("code")
    else:
        out["tokens"] = (body.get("usage") or {}).get("completion_tokens")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.dry_run:
        print(json.dumps({"model": a.model}))
        return 0
    if not (os.environ.get("GPUQ_RC") and os.environ.get("GPUQ_DEVICE")):
        raise RuntimeError("GPU work requires gpuq")
    from research.agentic.servers import Server, start_server

    work = Path(tempfile.mkdtemp(prefix="handshake-", dir=str(a.out.parent)))
    activity = work / "activity.json"
    os.environ.update(
        YUNSHU_RESEARCH_ACTIVITY_FILE=str(activity),
        YUNSHU_RESEARCH_PAUSE_TIMEOUT_S="0.5",
        AGENTIC_YUNSHU_SRC=str(ROOT / "python"),
        YUNSHU_VLM_APC_DISK="0",
    )
    spec = importlib.util.spec_from_file_location("gq", ROOT / "scripts/dev/gpuq.py")
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    gate = g.ServingGate()
    gate.configure({"activity_files": [str(activity)], "poll_s": 0.02})
    from yunshu_gateway.resident_activity import atomic_json

    atomic_json(activity, {"schema": 1, "ticket": "initial", "active": 0})
    server = None
    stop = threading.Event()
    ack_on = threading.Event()

    def scheduler():
        while not stop.wait(0.02):
            if ack_on.is_set():
                gate.poll(time.time())
                gate.acknowledge(True)

    t = threading.Thread(target=scheduler, daemon=True)
    try:
        server, _ = start_server(
            1, lambda p: Server("yunshu", a.model, p, work / "server.log"), wait_s=120.0
        )
        t.start()
        r503 = post(server.url, server.model_id)
        ack_on.set()
        r200 = post(server.url, server.model_id)
        ok, reason = judge(r503, r200)
        a.out.write_text(
            json.dumps({"ok": ok, "reason": reason, "r503": r503, "r200": r200}) + "\n"
        )
        print(
            json.dumps({"ok": ok, "reason": reason, "r503": r503, "r200": r200}),
            flush=True,
        )
        return 0 if ok else 1
    finally:
        stop.set()
        if server is not None:
            server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
