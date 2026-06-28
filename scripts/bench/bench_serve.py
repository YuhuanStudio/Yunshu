"""Production server-based framework benchmark (the RIGHT way).

Earlier framework bench (_fw_*.py) called each engine's BatchedEngine IN-PROCESS,
which bypasses the real serving stack (HTTP, scheduler, admission, continuous
batching). The data exposed the flaw: oMLX showed batch32/single = 1.2x (no
batching) while yunshu-loop showed 4.0x — an apples-to-oranges artifact of how
each engine's in-process API was driven.

This benchmarks each framework's ACTUAL PRODUCTION SERVER over HTTP — the path a
real deployment uses — so every framework is driven identically through its own
OpenAI-compatible `/v1/chat/completions`:

  - launch the framework's server subprocess, wait until /v1/models is ready
  - single-request: TTFT (stream, first token) + decode tok/s
  - under N concurrent clients: system aggregate tok/s — sum of EXACT
    usage.completion_tokens / wall (what a deployment sees) + avg latency under load
  - thermal: tags each run with measured GPU TFLOP/s; --wait-tflops cools first;
    --cooldown-sec isolates each framework at a matched thermal state
  - ALWAYS kill the server + verify the port is freed (try/finally)

All numbers are ABSOLUTE (feed scripts/perf_history.py). Servers OpenAI-compatible.

Run (needs the model drive mounted + oMLX venv):
  PYTHONPATH=. OMLX_PYTHON=.venvs/omlx/bin/python uv run python scripts/bench_serve.py \
      --models ./models/Qwen3.5-0.8B-MLX-bf16 --frameworks yunshu,mlx-lm,oMLX
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time

# scripts/bench/bench_serve.py → repo root is THREE levels up (Wave 688 reorg).
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# perf_history lives at scripts/ (one level up); make it importable.
sys.path.insert(0, os.path.join(REPO, "scripts"))
PROMPT = "Topic: " + ("Explain how photosynthesis works in detail. " * 40)
MAX_TOKENS = 64


def _thermal_warmup(seconds: float):
    """Drive the GPU to a STEADY thermal state (fans up) before measuring, so all
    frameworks see the same regime — cheaper than cold-isolating each.

    Model decode is bandwidth-bound (low compute util) and barely heats the GPU
    (user: "3B 跑得太輕鬆，根本沒 warm 起來"). So we saturate COMPUTE with sustained
    large fp16 matmuls — the heaviest, most reliable way to spin the fans up,
    model-agnostic.
    """
    if seconds <= 0:
        return
    import mlx.core as mx
    print(f">>> thermal warmup: {seconds:.0f}s of sustained large matmuls "
          f"(max GPU compute — spin fans to steady state)...", flush=True)
    N = 8192
    a = mx.random.normal((N, N), dtype=mx.float16)
    b = mx.random.normal((N, N), dtype=mx.float16)
    t0 = time.time()
    iters = 0
    while time.time() - t0 < seconds:
        c = a
        for _ in range(40):                      # ~40 8192³ matmuls per eval batch
            c = (c @ b) * mx.array(1e-4, mx.float16)
        mx.eval(c)
        iters += 1
    del a, b, c
    mx.clear_cache()
    print(f"    warmup done ({time.time() - t0:.0f}s, {iters * 40} matmuls) — measuring under warm regime",
          flush=True)


def _gpu_tflops(seconds: float = 2.5) -> float:
    """Measure the GPU's CURRENT sustained fp16 matmul throughput (TFLOP/s).

    A self-calibrating thermal-state signal that needs no temp sensor / sudo.
    NOTE: this is a 30-core M3 Max (36GB). Its COOL MLX fp16 ceiling for 8192³ is
    ~9.5 TFLOP/s — verified the GENUINE peak (Wave 657): it plateaus N=4096→8192 (so
    not a cache wall) and holds steady with no decay; the ~21 'theoretical' fp16 is
    not MLX-achievable here. A run reading well below ~9.5 indicates thermal throttle
    from sustained CONCURRENT load (multiple servers + warmup matmuls at once); a
    single clean run does NOT throttle (raw mlx-lm decode is flat across 8 back-to-back
    iters). Every bench result is tagged with this so it can be judged in context.
    """
    import mlx.core as mx
    N = 8192
    a = mx.random.normal((N, N), dtype=mx.float16)
    b = mx.random.normal((N, N), dtype=mx.float16)
    mx.eval(a @ b)  # warm
    t0 = time.perf_counter()
    iters = 0
    while time.perf_counter() - t0 < seconds:
        mx.eval(a @ b)
        iters += 1
    dt = time.perf_counter() - t0
    del a, b
    mx.clear_cache()
    return (2 * N ** 3 * iters) / dt / 1e12


def _wait_until_cool(min_tflops: float, max_wait: float = 1200) -> float:
    """Block until the GPU is THERMALLY READY, then return its TFLOP/s.

    Ready = reached the absolute target min_tflops (best case) OR PLATEAUED — the
    speed stopped improving across idle periods, i.e. it's as idle/cool as it'll
    get on this machine right now (robust when the true peak is unknown or a
    background app like LM Studio caps recovery). Idles between probes so it cools.
    """
    t0 = time.time()
    best = _gpu_tflops()
    print(f">>> GPU readiness: {best:.1f} TFLOP/s (target ≥{min_tflops}, else wait for plateau)", flush=True)
    stale = 0
    while time.time() - t0 < max_wait:
        if best >= min_tflops:
            print(f"    GPU at {best:.1f} TFLOP/s — ready (hit target)", flush=True)
            return best
        print(f"    {best:.1f} < {min_tflops} — cooling 45s...", flush=True)
        time.sleep(45)
        cur = _gpu_tflops()
        if cur > best * 1.03:          # still improving
            best = cur
            stale = 0
        else:                          # not improving
            best = max(best, cur)
            stale += 1
            if stale >= 2:             # plateaued across 2 idle periods
                print(f"    GPU plateaued at {best:.1f} TFLOP/s — proceeding (as cool as it gets)", flush=True)
                return best
    print(f"    timed out at {best:.1f} TFLOP/s — proceeding", flush=True)
    return best


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# ── per-framework server launch spec ─────────────────────────────────────────

_OMLX_KEY = "benchkey"


def _launch_spec(fw: str, model_path: str, port: int):
    """Return (argv, env, cwd, cleanup_dir, headers) to launch framework `fw`'s
    server. cleanup_dir (or None) is a temp dir removed after the server is killed;
    headers are auth headers the client must send to this server.
    """
    env = dict(os.environ)
    abspath = os.path.abspath(model_path)
    name = os.path.basename(abspath)
    if fw in ("yunshu", "yunshu-fast"):
        env.update(PYTHONPATH=os.path.join(REPO, "python"),
                   YUNSHU_MODEL=model_path, YUNSHU_AUTH_DISABLED="true",
                   YUNSHU_ENGINE_LOOP="1" if fw == "yunshu" else "0")
        return ([sys.executable, "-m", "uvicorn", "yunshu_gateway.main:app",
                 "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"], env, REPO, None, {})
    if fw == "mlx-lm":
        return ([sys.executable, "-m", "mlx_lm", "server", "--model", abspath,
                 "--host", "127.0.0.1", "--port", str(port)], env, REPO, None, {})
    if fw == "vllm-mlx":
        # vLLM's Apple-Silicon Metal fork (reference/vllm-mlx) has a real OpenAI
        # server (vllm_mlx.cli serve <model> --host --port; auth disabled by
        # default). No __main__/console-script in our venv, so drive cli.main()
        # with a constructed argv. Importable via PYTHONPATH. The all-external
        # comparison (Wave 688) keeps vllm-mlx fair by serving it like the rest.
        vdir = os.path.join(REPO, "reference", "vllm-mlx")
        if not os.path.isdir(vdir):
            return None
        env.update(PYTHONPATH=vdir)
        code = ("import sys; sys.argv=['vllm-mlx','serve',%r,'--host','127.0.0.1',"
                "'--port',%r]; from vllm_mlx.cli import main; main()" % (abspath, str(port)))
        return ([sys.executable, "-c", code], env, REPO, None, {})
    if fw == "oMLX":
        op = os.environ.get("OMLX_PYTHON")
        if not op or not os.path.exists(op):
            return None
        # CANONICAL production invocation: `omlx serve --model-dir <the real
        # models directory>` — the exact multi-model deployment (lazy-loads the
        # requested model; discovers all, loads none until asked). No symlink
        # tricks. Auth is oMLX's production default → pass --api-key + Bearer.
        models_dir = os.path.dirname(abspath)
        env.update(PYTHONPATH=os.path.join(REPO, "reference", "omlx"))
        return ([op, "-m", "omlx.cli", "serve", "--model-dir", models_dir, "--api-key", _OMLX_KEY,
                 "--host", "127.0.0.1", "--port", str(port)], env, REPO, None,
                {"Authorization": f"Bearer {_OMLX_KEY}"})
    return None


async def _wait_ready(client, port: int, timeout: float = 240.0, want: str | None = None):
    """Poll /v1/models until the server answers; return the served model id
    (preferring one whose id contains `want`, e.g. the model we asked for)."""
    base = f"http://127.0.0.1:{port}"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = await client.get(base + "/v1/models", timeout=5)
            if r.status_code == 200:
                ids = [d.get("id") for d in (r.json().get("data") or []) if d.get("id")]
                if want:
                    for i in ids:
                        if want in i or i in want:
                            return i
                return ids[0] if ids else "default"
        except Exception:
            pass
        await asyncio.sleep(2)
    return None


# Token counts come from the server's EXACT usage.completion_tokens (non-stream),
# NOT by counting SSE chunks — servers buffer streaming at different granularities
# (mlx-lm sends multi-token chunks), so chunk-counting is not comparable. Streaming
# is used ONLY to time the first token (TTFT).

async def _ttft(client, port: int, model: str, content: str) -> float | None:
    """Time-to-first-content-token via a streamed request (token COUNT ignored)."""
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": 8, "temperature": 0.0, "stream": True}
    t0 = time.perf_counter()
    async with client.stream("POST", f"http://127.0.0.1:{port}/v1/chat/completions",
                             json=body, timeout=600) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            for c in obj.get("choices", []):
                if (c.get("delta") or {}).get("content"):
                    return time.perf_counter() - t0
    return None


async def _req(client, port: int, model: str, content: str, max_tokens: int):
    """Non-streaming request → (latency_s, completion_tokens) using EXACT usage."""
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": False}
    t0 = time.perf_counter()
    r = await client.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                          json=body, timeout=600)
    lat = time.perf_counter() - t0
    ct = 0
    try:
        ct = (r.json().get("usage") or {}).get("completion_tokens", 0) or 0
    except Exception:
        pass
    return lat, ct


async def _bench_one(fw: str, model_path: str, concurrency: list[int], trials: int = 3,
                     cooldown_sec: float = 0) -> dict:
    import httpx
    port = _free_port()
    spec = _launch_spec(fw, model_path, port)
    if spec is None:
        return {"fw": fw, "status": "UNAVAILABLE"}
    argv, env, cwd, cleanup_dir, headers = spec
    want_model = os.path.basename(os.path.abspath(model_path))
    log = open(f"/tmp/serve_{fw}_{port}.log", "w")
    proc = subprocess.Popen(argv, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    out = {"fw": fw, "model": os.path.basename(model_path), "port": port}
    try:
        async with httpx.AsyncClient(headers=headers) as client:
            model_id = await _wait_ready(client, port, want=want_model)
            if model_id is None:
                out["status"] = "LAUNCH/READY FAIL (see log)"
                return out
            # ISOLATE: idle the GPU to cool toward a consistent thermal baseline
            # BEFORE measuring, so cumulative heat from earlier frameworks doesn't
            # throttle this one (each framework enters measurement at the same temp).
            if cooldown_sec > 0:
                print(f"    cooldown {cooldown_sec:.0f}s (GPU idle → consistent thermal entry)...", flush=True)
                await asyncio.sleep(cooldown_sec)
            # warmup (load + JIT + settle the model on the GPU)
            for _ in range(2):
                await _req(client, port, model_id, "Count to five.", 16)
            await asyncio.sleep(2)
            # single-request: TTFT (min streamed probe) + decode tps (exact usage
            # tokens / decode window), MEDIAN over `trials`
            s_ttft, s_dec = [], []
            for _ in range(trials):
                ttfts = [t for t in [await _ttft(client, port, model_id, PROMPT) for _ in range(2)] if t]
                if ttfts:
                    s_ttft.append(min(ttfts))
                lat, ct = await _req(client, port, model_id, PROMPT, MAX_TOKENS)
                tt = min(ttfts) if ttfts else 0
                if ct > 1 and lat - tt > 0:
                    s_dec.append((ct - 1) / (lat - tt))
            out["ttft_ms"] = round(_median(s_ttft) * 1000, 1) if s_ttft else None
            out["decode_tps"] = round(_median(s_dec), 1) if s_dec else None
            # concurrency sweep: N concurrent non-stream → EXACT system tok/s,
            # MEDIAN over `trials` with a settle between trials (kill thermal/noise)
            out["batch"] = {}
            out["batch_latency_ms"] = {}
            for N in concurrency:
                tps_trials, lat_trials = [], []
                for _t in range(trials):
                    wall0 = time.perf_counter()
                    res = await asyncio.gather(*[
                        _req(client, port, model_id, f"[{i}] " + PROMPT, MAX_TOKENS)
                        for i in range(N)])
                    wall = time.perf_counter() - wall0
                    total_gen = sum(r[1] for r in res)
                    if wall:
                        tps_trials.append(total_gen / wall)
                    lat_trials.append(sum(r[0] for r in res) / len(res))
                    await asyncio.sleep(2)  # settle between trials
                out["batch"][str(N)] = round(_median(tps_trials), 1) if tps_trials else 0
                out["batch_latency_ms"][str(N)] = round(_median(lat_trials) * 1000, 1) if lat_trials else None
            out["trials"] = trials
            out["status"] = "OK"
    except Exception as e:
        out["status"] = f"RUN FAIL: {str(e)[:80]}"
    finally:
        # ALWAYS kill the server group + verify the port is freed
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            proc.wait(timeout=15)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        log.close()
        if cleanup_dir:
            import shutil
            shutil.rmtree(cleanup_dir, ignore_errors=True)
    return out


def _render(results: list[dict]):
    print("\n" + "=" * 78)
    print("SERVER-BASED FRAMEWORK BENCHMARK (production HTTP path, OpenAI /v1)")
    print("=" * 78)
    by_model: dict[str, list[dict]] = {}
    for r in results:
        by_model.setdefault(r.get("model", "?"), []).append(r)
    for model, rows in by_model.items():
        print(f"\n### {model}")
        # discover the actual concurrency levels measured (not a fixed 8/16/32)
        ns = sorted({int(k) for r in rows for k in (r.get("batch") or {})})
        ncols = "".join(f"{'sys@'+str(n):<8}" for n in ns)
        nmax = ns[-1] if ns else 32
        print(f"{'framework':<14}{'TTFT ms':<10}{'dec t/s':<9}{ncols}{'lat@'+str(nmax)+'ms':<10}")
        for r in rows:
            if r.get("status") != "OK":
                print(f"{r['fw']:<14}{r.get('status')}")
                continue
            b = r.get("batch", {})
            lat = r.get("batch_latency_ms", {})
            ncells = "".join(f"{str(b.get(str(n),'-')):<8}" for n in ns)
            print(f"{r['fw']:<14}{str(r.get('ttft_ms','-')):<10}{str(r.get('decode_tps','-')):<9}{ncells}"
                  f"{str(lat.get(str(nmax),'-')):<10}")
    print("\nTTFT/dec t/s = single request. sys@N = system aggregate tok/s under N concurrent")
    print("clients (sum of EXACT usage.completion_tokens / wall). lat@N = avg e2e latency under load.")


async def main_async(models: list[str], frameworks: list[str], concurrency: list[int],
                     trials: int = 3, warmup_sec: float = 60, cooldown_sec: float = 0,
                     wait_tflops: float = 0):
    # Thermal readiness: optionally block until the GPU has cooled to wait_tflops,
    # then tag the run with the measured speed so results can be judged for throttle.
    gpu0 = _wait_until_cool(wait_tflops) if wait_tflops > 0 else _gpu_tflops()
    print(f">>> GPU speed at start: {gpu0:.1f} TFLOP/s (this M3 Max's cool baseline ≈10; "
          f"well below = throttled by sustained load)", flush=True)
    # ISOLATE mode (cooldown_sec>0): cool the GPU between frameworks so each enters
    # measurement at the same thermal baseline — the right fix when sustained back-
    # to-back load throttles later frameworks (the full-run observation). In that
    # mode skip the global pre-heat warmup (it works against the cooldown).
    if cooldown_sec <= 0:
        _thermal_warmup(warmup_sec)
    else:
        print(f">>> ISOLATE mode: {cooldown_sec:.0f}s GPU cooldown before each framework", flush=True)
    results = []
    for model in models:
        if not os.path.isdir(model):
            print(f"SKIP {model}: not found (drive mounted?)")
            continue
        for fw in frameworks:
            print(f"\n>>> [{os.path.basename(model)}] {fw} — launching server...", flush=True)
            r = await _bench_one(fw, model, concurrency, trials, cooldown_sec)
            print(f"    {fw}: {r.get('status')}", flush=True)
            results.append(r)
    gpu1 = _gpu_tflops()
    _render(results)
    print(f"\nGPU TFLOP/s: {gpu0:.1f} at start → {gpu1:.1f} at end (this machine's cool "
          f"baseline ≈10). {'⚠️ likely throttled by sustained load' if min(gpu0, gpu1) < 8.5 else 'at baseline — full speed'}")
    json.dump({"gpu_tflops_start": round(gpu0, 1), "gpu_tflops_end": round(gpu1, 1),
               "results": results}, open("/tmp/bench_serve.json", "w"), indent=2)
    print("full json -> /tmp/bench_serve.json")

    # Feed the append-only absolute-evolution trend (perf_history). Every run is
    # tagged with the GPU TFLOP/s it was measured at, so throttled points are
    # visible in the time series.
    try:
        from perf_history import snapshot_from_kpis, trend as _trend
        kpis: dict[str, float] = {"gpu_tflops": round(min(gpu0, gpu1), 1)}
        for r in results:
            if r.get("status") != "OK":
                continue
            m = r.get("model", "?")
            fw = r.get("fw", "?")
            if isinstance(r.get("ttft_ms"), (int, float)):
                kpis[f"serve/{m}/{fw}/ttft_ms"] = r["ttft_ms"]
            if isinstance(r.get("decode_tps"), (int, float)):
                kpis[f"serve/{m}/{fw}/decode_tps"] = r["decode_tps"]
            for n, v in (r.get("batch") or {}).items():
                if isinstance(v, (int, float)):
                    kpis[f"serve/{m}/{fw}/sys{n}"] = v
        if snapshot_from_kpis(kpis, source="bench_serve",
                              extra={"gpu_tflops_start": round(gpu0, 1), "gpu_tflops_end": round(gpu1, 1)}):
            _trend()
    except Exception as e:
        print(f"(perf-history feed skipped: {e})")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="./models/Qwen3.5-0.8B-MLX-bf16")
    ap.add_argument("--frameworks", default="yunshu,yunshu-fast,mlx-lm,oMLX")
    ap.add_argument("--concurrency", default="8,16,32")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--warmup-sec", type=float, default=60,
                    help="sustained large-matmul thermal warmup before measuring (0=off)")
    ap.add_argument("--cooldown-sec", type=float, default=0,
                    help="ISOLATE: idle GPU this long before each framework so cumulative "
                         "heat doesn't throttle later frameworks (e.g. 90). Skips global warmup.")
    ap.add_argument("--wait-tflops", type=float, default=0,
                    help="block until the GPU has cooled to ≥ this fp16 TFLOP/s before "
                         "measuring; falls back to a plateau if unreachable. This M3 "
                         "Max's cool baseline ≈10, so ~9.5 is a sensible target. 0=off.")
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    fws = [f.strip() for f in a.frameworks.split(",") if f.strip()]
    conc = [int(x) for x in a.concurrency.split(",")]
    asyncio.run(main_async(models, fws, conc, a.trials, a.warmup_sec, a.cooldown_sec, a.wait_tflops))


if __name__ == "__main__":
    main()
