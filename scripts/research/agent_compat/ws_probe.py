"""Record the frames Codex sends over the Responses WebSocket (`supports_websockets = true`).

A tiny asyncio WebSocket server answers every `response.create` with a canned completed response and
logs each inbound frame to <out>/ws_frames.jsonl; Codex runs under the same isolation as census.py.

    python ws_probe.py [--out DIR]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import threading
import time
from pathlib import Path

import census
import websockets


def resp_obj(i: int, status: str, output: list, model: str) -> dict:
    return {
        "id": f"resp_probe{i}",
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


async def serve(port_box: list, log: Path, ready: threading.Event):
    n = 0

    async def handler(ws):
        nonlocal n
        with log.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "event": "connect",
                        "path": ws.request.path,
                        "headers": dict(ws.request.headers),
                    }
                )
                + "\n"
            )
        async for raw in ws:
            msg = json.loads(raw)
            with log.open("a") as f:
                f.write(json.dumps({"event": "frame", "msg": msg}) + "\n")
            if msg.get("type") != "response.create":
                continue
            n += 1
            model = msg.get("model", "m")
            gen = msg.get("generate", True)
            item = {
                "type": "message",
                "id": f"msg_{n}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "pong", "annotations": []}],
            }
            out = [item] if gen is not False else []
            base = resp_obj(n, "in_progress", [], model)
            await ws.send(json.dumps({"type": "response.created", "response": base}))
            if out:
                await ws.send(
                    json.dumps(
                        {
                            "type": "response.output_item.added",
                            "output_index": 0,
                            "item": {**item, "content": [], "status": "in_progress"},
                        }
                    )
                )
                await ws.send(
                    json.dumps(
                        {
                            "type": "response.output_text.delta",
                            "item_id": item["id"],
                            "output_index": 0,
                            "content_index": 0,
                            "delta": "pong",
                        }
                    )
                )
                await ws.send(
                    json.dumps(
                        {
                            "type": "response.output_item.done",
                            "output_index": 0,
                            "item": item,
                        }
                    )
                )
            await ws.send(
                json.dumps(
                    {
                        "type": "response.completed",
                        "response": resp_obj(n, "completed", out, model),
                    }
                )
            )

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port_box.append(server.sockets[0].getsockname()[1])
        ready.set()
        await asyncio.Future()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(census.OUT_ROOT / "cx_ws_frames"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "ws_frames.jsonl"
    log.unlink(missing_ok=True)
    port_box: list = []
    ready = threading.Event()
    threading.Thread(
        target=lambda: asyncio.run(serve(port_box, log, ready)), daemon=True
    ).start()
    ready.wait(10)
    run = census.BUILD / "runs" / "ws_probe"
    work = census.BUILD / "work" / "ws_probe"
    for d in (run, work):
        subprocess.run(["rm", "-rf", str(d)])
        d.mkdir(parents=True)
    launch = census.agents.prepare(
        "codex",
        run,
        work,
        f"http://127.0.0.1:{port_box[0]}",
        "probe-model",
        "Reply with pong",
    )
    census.HOOKS["cx_ws"](launch.home, launch, {})
    p = subprocess.run(
        launch.cmd,
        cwd=launch.cwd,
        env=launch.env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
    )
    (out / "stdout.txt").write_text(p.stdout)
    (out / "stderr.txt").write_text(p.stderr)
    print("rc", p.returncode)
    print(log.read_text()[:6000])


if __name__ == "__main__":
    main()
