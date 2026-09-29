"""Diagnose Responses 'cancelled' seen through the WS bridge: same request over plain HTTP SSE vs the bridge."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, "python")
from yunshu_client import YunshuStream  # noqa: E402

PORT = 18992
BASE = f"http://127.0.0.1:{PORT}"
BODY = {
    "model": "m",
    "input": "Say hi.",
    "max_output_tokens": 1500,
    "stream": True,
    "reasoning": {"effort": "none"},
}


def terminal(lines):
    out = []
    for ln in lines:
        if (
            ln.startswith("data:")
            and "response." in ln
            and ("incomplete" in ln or "completed" in ln)
        ):
            d = json.loads(ln[5:])
            out.append((d["type"], (d.get("response") or {}).get("incomplete_details")))
    return out


p = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "yunshu_cli",
        "serve",
        "--model",
        sys.argv[1],
        "--port",
        str(PORT),
    ],
    env=dict(os.environ, PYTHONPATH="python"),
    start_new_session=True,
)
try:
    for _ in range(200):
        try:
            if urllib.request.urlopen(BASE + "/health/ready", timeout=3).status == 200:
                break
        except Exception:  # noqa: BLE001
            time.sleep(2)
    req = urllib.request.Request(
        BASE + "/v1/responses",
        json.dumps(BODY).encode(),
        {"content-type": "application/json", "x-request-id": "http-1"},
    )
    lines = urllib.request.urlopen(req).read().decode().splitlines()
    print("HTTP  terminal:", terminal(lines), flush=True)
    lines = urllib.request.urlopen(req).read().decode().splitlines()
    print("HTTP2 terminal:", terminal(lines), flush=True)

    async def ws():
        async with YunshuStream(f"ws://127.0.0.1:{PORT}/v1/stream") as c:
            for rid in ("a1", "a2", "a3", "has space"):
                got = []
                async for m in c.responses(
                    {k: v for k, v in BODY.items()}, id=rid, with_done=True
                ):
                    if m["type"] == "event":
                        got.append("data: " + json.dumps(m["data"]))
                    elif m["type"] == "done":
                        print(
                            "WS", rid, "done:", m["reason"], m["request_id"], flush=True
                        )
                print("WS   ", rid, "terminal:", terminal(got), flush=True)

    asyncio.run(ws())
finally:
    os.killpg(p.pid, signal.SIGKILL)
    p.wait()
