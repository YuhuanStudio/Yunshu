"""Per-model performance sweep (full tier) — TTFT / decode tok-s / peak RAM, isolated.

Honest hardware-ceiling numbers per LLM, each model in its OWN subprocess (never
co-loaded — 36GB safety). Reports TTFT (first-token latency), steady decode
throughput (tok/s), and peak process RSS. These are EXPECTED to be 80-95% of the
M3 Max memory-bandwidth ceiling (decode is bandwidth-bound) — the point is a
living, reproducible record, not a target.

Run:  PYTHONPATH=. uv run python scripts/sweep_perf.py
Child: PYTHONPATH=. uv run python scripts/sweep_perf.py --child <model_path>
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time

MODELS = [
    "./models/Qwen2.5-3B-Instruct-4bit",
    "./models/Qwen2.5-3B-Instruct-bf16",
    "./models/Qwen3.5-0.8B-MLX-bf16",
    "./models/Qwen3.5-2B-MLX-bf16",
    "./models/Qwen3.5-9B-MLX-4bit",
]


def _rss_mb() -> float:
    try:
        import resource
        # ru_maxrss is bytes on macOS, kB on Linux.
        m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return m / (1024 * 1024) if m > 1e9 else m / 1024
    except Exception:
        return 0.0


async def _child(model: str) -> int:
    if not os.path.exists(model):
        print(json.dumps({"model": model, "skip": True}))
        return 0
    from yunshu_engine.batched_engine import BatchedEngine
    eng = BatchedEngine(model_name=model)
    await eng.start()
    q = [{"role": "user", "content": "Write a detailed paragraph about the history of computing."}]
    try:
        # warm
        await eng.chat(messages=q, max_tokens=8, temperature=0.0, enable_thinking=False)
        # TTFT
        t0 = time.time()
        await eng.chat(messages=q, max_tokens=1, temperature=0.0, enable_thinking=False)
        ttft = time.time() - t0
        # steady decode
        N = 160
        t0 = time.time()
        r = await eng.chat(messages=q, max_tokens=N, temperature=0.0, enable_thinking=False)
        dt = time.time() - t0
        ct = getattr(r, "completion_tokens", 0) or (r.get("completion_tokens", 0) if isinstance(r, dict) else 0)
    finally:
        await eng.stop()
    tps = (ct / dt) if dt > 0 else 0.0
    print(json.dumps({"model": model, "ttft_s": round(ttft, 3),
                      "decode_tok_s": round(tps, 1), "tokens": ct,
                      "peak_rss_mb": round(_rss_mb(), 0)}))
    return 0


def _parent() -> int:
    rows = []
    for m in MODELS:
        if not os.path.exists(m):
            rows.append({"model": m, "status": "absent"})
            continue
        env = dict(os.environ, PYTHONPATH=os.environ.get("PYTHONPATH", "."))
        try:
            p = subprocess.run([sys.executable, __file__, "--child", m],
                               capture_output=True, text=True, timeout=420, env=env)
            line = [l for l in p.stdout.splitlines() if l.startswith("{")]
            rows.append(json.loads(line[-1]) if line else {"model": m, "status": "no-output"})
        except Exception as e:
            rows.append({"model": m, "status": f"FAIL ({type(e).__name__})"})

    print(f"  {'model':34s} {'TTFT':>7s} {'tok/s':>8s} {'peakRSS':>9s}")
    measured = 0
    for r in rows:
        name = os.path.basename(r["model"])
        if r.get("skip") or r.get("status"):
            print(f"  {name:34s}  {r.get('status','skip')}")
            continue
        measured += 1
        print(f"  {name:34s} {r['ttft_s']:6.2f}s {r['decode_tok_s']:7.1f} {r['peak_rss_mb']:7.0f}MB")
    # write structured artifact for perf_history
    art = "/tmp/yunshu_perf_sweep.json"
    json.dump(rows, open(art, "w"))
    print(f"full json -> {art}")
    print(f"RESULT: {measured} models measured")
    print("PASS" if measured > 0 else "FAIL")
    return 0 if measured > 0 else 1


def main() -> int:
    if "--child" in sys.argv:
        return asyncio.run(_child(sys.argv[sys.argv.index("--child") + 1]))
    return _parent()


if __name__ == "__main__":
    sys.exit(main())
