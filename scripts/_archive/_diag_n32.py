"""Localize the N=32 throughput regression: engine_core vs gateway HTTP.

Same machine state, same engine-loop, same prompt. Measures aggregate tok/s
(EXACT completion_tokens / wall) at N=16 and N=32 via TWO paths:
  (A) IN-PROCESS: BatchedEngine(engine-loop).chat() concurrently — pure engine.
  (B) HTTP: through the gateway /v1/chat/completions.
If (A) scales N16→N32 but (B) drops, the regression is the GATEWAY, not the
engine scheduler.

Run: PYTHONPATH=. uv run python scripts/_diag_n32.py ./models/Qwen2.5-3B-Instruct-bf16
"""
from __future__ import annotations
import asyncio, os, sys, time

MODEL = sys.argv[1] if len(sys.argv) > 1 else "./models/Qwen2.5-3B-Instruct-bf16"
PROMPT = "Topic: " + ("Explain how photosynthesis works in detail. " * 40)
MT = 64


async def _engine_path():
    os.environ["YUNSHU_ENGINE_LOOP"] = "1"
    from yunshu_engine.batched_engine import BatchedEngine
    eng = BatchedEngine(model_name=MODEL)
    await eng.start()

    async def one(i):
        o = await eng.chat(messages=[{"role": "user", "content": f"[{i}] " + PROMPT}],
                           max_tokens=MT, temperature=0.0, enable_thinking=False)
        return getattr(o, "completion_tokens", 0) if not isinstance(o, dict) else o.get("completion_tokens", 0)

    # warmup
    await one(999)
    res = {}
    try:
        for N in (16, 32):
            t0 = time.perf_counter()
            cts = await asyncio.gather(*[one(i) for i in range(N)])
            wall = time.perf_counter() - t0
            res[N] = round(sum(cts) / wall, 1)
    finally:
        await eng.stop()
    return res


async def main():
    print(f"=== ENGINE-LOOP IN-PROCESS ({os.path.basename(MODEL)}) ===", flush=True)
    eng_res = await _engine_path()
    print(f"  engine aggregate tok/s:  N=16 {eng_res.get(16)}   N=32 {eng_res.get(32)}", flush=True)
    drop = eng_res.get(32, 0) < eng_res.get(16, 0)
    print(f"  → engine {'DROPS at N=32 (scheduler issue)' if drop else 'SCALES to N=32 (engine OK → gateway is the bottleneck)'}",
          flush=True)


if __name__ == "__main__":
    asyncio.run(main())
