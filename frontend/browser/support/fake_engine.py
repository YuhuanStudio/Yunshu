"""A stand-in engine for the console-process end-to-end tests: just the endpoints the console reads
(status, finished requests with a cursor, host), plus /__ controls. No mlx, no yunshu imports.

    python fake_engine.py <port>
"""

from __future__ import annotations

import os
import sys
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

STARTED = time.monotonic()
BOOT = uuid.uuid4().hex[:12]
state = {"seq": 0, "ring": [], "busy": False, "load_error": None, "models": [{"id": "org/e2e-model", "type": "VLMEngine", "loaded": True, "loading": False, "pinned": False}]}

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/yunshu/status")
async def status():
    busy = state["busy"]
    items = (
        [{"request_id": "live_1", "elapsed_s": 2.0, "phase": "decode", "completion_tokens": 120, "tokens_per_second": 42.0}]
        if busy
        else []
    )
    return {
        "object": "yunshu.status",
        "version": "e2e",
        "state": "running",
        "uptime_s": round(time.monotonic() - STARTED, 1),
        "pid": os.getpid(),
        "load_error": state["load_error"],
        "models": state["models"],
        "memory": {"active_gb": 12.0, "active_bytes": 12 * 1024**3, "cache_gb": 1.0, "cache_bytes": 1024**3, "peak_gb": 14.0, "peak_bytes": 14 * 1024**3, "total_gb": 64.0, "total_bytes": 64 * 1024**3},
        "requests": {"active": len(items), "queued": 0, "prefill": 0, "decode": len(items), "items": items},
        "last": state["ring"][-1] if state["ring"] else None,
        "throughput": {"window_s": 60, "requests": len(state["ring"]), "prompt_tokens": 0, "completion_tokens": 0, "live_decode_tps": 42.0 if busy else None, "mean_prefill_tps": None, "mean_decode_tps": None},
    }


@app.get("/v1/yunshu/requests/recent")
async def recent(after_seq: int | None = None, limit: int = 100):
    rows = [r for r in state["ring"] if after_seq is None or r["seq"] > after_seq]
    return {"object": "list", "data": list(reversed(rows))[:limit], "count": len(rows), "capacity": 512, "boot_id": BOOT, "latest_seq": state["seq"]}


@app.get("/v1/yunshu/host")
async def host():
    return {"object": "yunshu.host", "telemetry": {"state": "ok", "watts": {"gpu": 9.0, "package": 15.0}, "gpu": {"frequency_mhz": 1100, "active_ratio": 0.4}, "temperature": {"die_max_c": 55}}}


@app.post("/__finish")
async def finish(request: Request):
    body = await request.json() if request.headers.get("content-length") not in (None, "0") else {}
    for i in range(int(body.get("n", 1))):
        state["seq"] += 1
        state["ring"].append(
            {
                "seq": state["seq"],
                "request_id": f"req_e2e_{BOOT}_{state['seq']}",
                "t": time.time(),
                "model": "org/e2e-model",
                "path": "/v1/chat/completions",
                "status": 200,
                "stream": True,
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "cached_tokens": 50,
                "ttft_ms": 140.0 + i,
                "decode_tps": 41.5,
                "prefill_tps": 900.0,
                "finish_reason": "stop",
            }
        )
    return {"seq": state["seq"]}


@app.post("/__busy/{on}")
async def busy(on: int):
    state["busy"] = bool(on)
    return {"busy": state["busy"]}


@app.post("/__load_error")
async def load_error(request: Request):
    body = await request.json()
    state["load_error"] = body.get("message")
    if state["load_error"]:
        state["models"] = [{**m, "loaded": False} for m in state["models"]]
    return {"ok": True}


@app.exception_handler(404)
async def not_found(request, exc):
    return JSONResponse({"error": {"message": "n/a", "type": "not_found"}}, status_code=404)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="error")
