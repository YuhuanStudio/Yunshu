#!/usr/bin/env python3
"""Isolate single-request TTFT (time-to-first-token) overhead in the SERVED path.

Measures, for ONE small model, the median streaming TTFT over real localhost HTTP and
compares it to the in-process fast-path TTFT for the identical prompt. The delta is the
gateway/serving overhead (middleware stack, validation, routing) that in-process testing
hides — the thing the regression report flags (yunshu serve TTFT ≫ mlx-lm).

Safe on a 36GB Mac: loads a SINGLE small model (no co-loading). Default Qwen3.5-0.8B.

Usage:
  PYTHONPATH=python .venv/bin/python scripts/bench/bench_ttft_probe.py \
      --model models/Qwen3.5-0.8B-MLX-bf16 --port 8013 --n 12
"""
from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROMPT = "Explain in one sentence why the sky is blue."


def _median_ms(xs: list[float]) -> float:
    return round(statistics.median(xs) * 1000, 1) if xs else 0.0


async def _http_ttft(port: int, model: str, n: int) -> list[float]:
    import httpx
    out: list[float] = []
    async with httpx.AsyncClient(timeout=120) as c:
        # warmup (load/JIT/settle + prime any caches the steady state would have)
        for _ in range(2):
            try:
                async with c.stream("POST", f"http://127.0.0.1:{port}/v1/chat/completions",
                                    json={"model": model, "stream": True, "max_tokens": 8,
                                          "messages": [{"role": "user", "content": PROMPT}]}) as r:
                    async for _ in r.aiter_lines():
                        break
            except Exception as e:
                print(f"  warmup err: {e}")
        for _ in range(n):
            t0 = time.perf_counter()
            first = None
            async with c.stream("POST", f"http://127.0.0.1:{port}/v1/chat/completions",
                                json={"model": model, "stream": True, "max_tokens": 8,
                                      "messages": [{"role": "user", "content": PROMPT}]}) as r:
                async for line in r.aiter_lines():
                    if line.startswith("data:") and '"content"' in line:
                        first = time.perf_counter() - t0
                        break
            if first:
                out.append(first)
    return out


async def _agg_throughput(port: int, model: str, concurrency: int,
                          max_tokens: int = 128, trials: int = 3) -> tuple[float, int]:
    """Aggregate tok/s under `concurrency` concurrent non-stream requests: exact
    completion_tokens summed / wall-clock span. Warms the concurrency level first, then
    takes the best of `trials` (best = least thermal/scheduling noise). Mirrors
    bench_serve's sys@N but lighter."""
    import httpx
    body = {"model": model, "stream": False, "max_tokens": max_tokens, "temperature": 0.0,
            "messages": [{"role": "user", "content": PROMPT}]}
    async with httpx.AsyncClient(timeout=600) as c:
        async def _one():
            r = await c.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body)
            return (r.json().get("usage", {}) or {}).get("completion_tokens", 0)
        await asyncio.gather(*[_one() for _ in range(concurrency)])  # warm the batch level
        best = 0.0
        total = 0
        for _ in range(trials):
            t0 = time.perf_counter()
            toks = await asyncio.gather(*[_one() for _ in range(concurrency)])
            dt = time.perf_counter() - t0
            tps = sum(toks) / dt if dt else 0.0
            if tps > best:
                best, total = tps, sum(toks)
    return (round(best, 1), total)


def _inproc_ttft(model_path: str, n: int) -> list[float]:
    """In-process fast-path TTFT for the same prompt (no HTTP, no gateway)."""
    sys.path.insert(0, os.path.join(REPO, "python"))
    import asyncio as _a

    from yunshu_engine.batched_engine import BatchedEngine

    eng = BatchedEngine(model_name=model_path)

    async def _run():
        await eng.start()
        msgs = [{"role": "user", "content": PROMPT}]
        # warmup
        for _ in range(2):
            async for _o in eng.stream_chat(msgs, max_tokens=8):
                break
        res: list[float] = []
        for _ in range(n):
            t0 = time.perf_counter()
            async for _o in eng.stream_chat(msgs, max_tokens=8):
                res.append(time.perf_counter() - t0)
                break
        await eng.stop()
        return res

    return _a.run(_run())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-0.8B-MLX-bf16")
    ap.add_argument("--port", type=int, default=8013)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--inproc", action="store_true", help="also measure in-process baseline")
    ap.add_argument("--loop", action="store_true", help="enable engine-loop (continuous batching)")
    ap.add_argument("--concurrency", default="", help="comma list e.g. 8,16,32 → aggregate tok/s")
    a = ap.parse_args()
    model_id = os.path.basename(a.model)

    env = dict(os.environ)
    env["YUNSHU_MODEL"] = a.model
    env["YUNSHU_AUTH_DISABLED"] = "true"
    env["YUNSHU_ENGINE_LOOP"] = "1" if a.loop else "0"
    env["PYTHONPATH"] = os.path.join(REPO, "python")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "yunshu_gateway.main:app",
         "--host", "127.0.0.1", "--port", str(a.port), "--log-level", "warning"],
        env=env, cwd=REPO,
    )
    try:
        import httpx
        ready = False
        for _ in range(120):
            try:
                r = httpx.get(f"http://127.0.0.1:{a.port}/v1/models", timeout=2)
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(1)
        if not ready:
            print("server did not become ready"); return
        time.sleep(2)
        http = asyncio.run(_http_ttft(a.port, model_id, a.n))
        print(f"\n=== TTFT probe: {model_id} (loop={a.loop}, n={a.n}) ===")
        print(f"  HTTP served TTFT  median: {_median_ms(http)} ms   "
              f"(min {_median_ms([min(http)]) if http else 0}, samples={len(http)})")
        if a.concurrency:
            for N in [int(x) for x in a.concurrency.split(",")]:
                tps, tot = asyncio.run(_agg_throughput(a.port, model_id, N))
                print(f"  aggregate tok/s @ N={N}: {tps}  ({tot} tokens)")
        if a.inproc:
            ip = _inproc_ttft(a.model, a.n)
            print(f"  in-process  TTFT  median: {_median_ms(ip)} ms")
            if http and ip:
                print(f"  >>> gateway overhead ≈ {_median_ms(http) - _median_ms(ip):.1f} ms")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
        # ensure no orphan
        subprocess.run(["pkill", "-f", f"uvicorn.*--port {a.port}"], capture_output=True)


if __name__ == "__main__":
    main()
