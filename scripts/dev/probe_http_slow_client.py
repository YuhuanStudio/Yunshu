"""HTTP-level slow-client coupling probe through the real server's /v1/chat/completions SSE.

Starts uvicorn on a free port with the model from --model, then runs two SSE clients at once:
"both_normal" (two normal readers) and "one_slow" (one reader sleeps --sleep s after every chunk).
Interleaved reps. Reports the normal client's chunks/s in each scenario and the ratio.
"""

import argparse
import asyncio
import contextlib
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def wait_ready(client, base, proc, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited rc={proc.returncode}")
        with contextlib.suppress(Exception):
            r = await client.get(f"{base}/v1/models")
            if r.status_code == 200:
                return r.json()["data"][0]["id"]
        await asyncio.sleep(1.0)
    raise TimeoutError("server not ready")


async def read_sse(client, base, model, max_tokens, sleep_s):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Tell a long story about a river."}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    chunks = 0
    t0 = time.monotonic()
    async with client.stream("POST", f"{base}/v1/chat/completions", json=body) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            choices = event.get("choices") or []
            delta = (choices[0].get("delta") or {}) if choices else {}
            if any(delta.get(k) for k in ("content", "reasoning_content", "reasoning")):
                chunks += 1
            if sleep_s:
                await asyncio.sleep(sleep_s)
    wall = time.monotonic() - t0
    return {
        "chunks": chunks,
        "wall_s": round(wall, 3),
        "chunks_s": round(chunks / wall, 2),
    }


async def scenario(client, base, model, label, specs, max_tokens):
    t0 = time.monotonic()
    results = await asyncio.gather(
        *(read_sse(client, base, model, max_tokens, s) for s in specs)
    )
    row = {
        "label": label,
        "streams": results,
        "scenario_wall_s": round(time.monotonic() - t0, 3),
    }
    print(json.dumps(row), flush=True)
    return row


async def main_async(args):
    import httpx

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    env = dict(os.environ, YUNSHU_MODEL=args.model, YUNSHU_AUTH_DISABLED="true")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "yunshu_gateway.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        start_new_session=True,
    )
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            model = await wait_ready(client, base, proc)
            await scenario(client, base, model, "warmup", [0.0], 8)
            rows = []
            for rep in range(args.reps):
                rows.append(
                    await scenario(
                        client,
                        base,
                        model,
                        f"both_normal.r{rep}",
                        [0.0, 0.0],
                        args.tokens,
                    )
                )
                rows.append(
                    await scenario(
                        client,
                        base,
                        model,
                        f"one_slow.r{rep}",
                        [args.sleep, 0.0],
                        args.tokens,
                    )
                )
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        with contextlib.suppress(Exception):
            proc.wait(timeout=30)
    base_rates = [
        r["streams"][0]["chunks_s"] for r in rows if r["label"].startswith("both")
    ]
    slow_rates = [
        r["streams"][1]["chunks_s"] for r in rows if r["label"].startswith("one_slow")
    ]
    summary = {
        "reps": args.reps,
        "normal_chunks_s_both_normal": base_rates,
        "normal_chunks_s_with_slow_peer": slow_rates,
        "median_both_normal": statistics.median(base_rates),
        "median_with_slow_peer": statistics.median(slow_rates),
    }
    summary["ratio"] = round(
        summary["median_with_slow_peer"] / summary["median_both_normal"], 3
    )
    print("SUMMARY", json.dumps(summary), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=600)
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
