"""Sampling-correctness gate (small LLM).

Three robust invariants: (1) temperature=0 is deterministic (same prompt → identical
output); (2) a fixed seed is reproducible (same seed → identical sampled output);
(3) different seeds vary (different seed → different output). Guards the sampler +
seed plumbing.

Run: PYTHONPATH=. uv run python scripts/verify_sampling.py
"""
from __future__ import annotations

import asyncio
import os
import sys

MODEL = os.environ.get("YUNSHU_BENCH_MODEL", "./models/Qwen3.5-0.8B-MLX-bf16")


async def main() -> int:
    if not os.path.exists(MODEL):
        print("SKIP: model not mounted")
        return 0
    from yunshu_engine.batched_engine import BatchedEngine
    eng = BatchedEngine(model_name=MODEL)
    await eng.start()

    async def gen(temp, seed):
        o = await eng.chat(messages=[{"role": "user", "content": "Write one short sentence about a river."}],
                           max_tokens=32, temperature=temp, seed=seed, enable_thinking=False)
        return (o["text"] if isinstance(o, dict) else o.text).strip()
    try:
        det0, det1 = await gen(0.0, None), await gen(0.0, None)
        s7a, s7b = await gen(1.0, 7), await gen(1.0, 7)
        s99 = await gen(1.0, 99)
    finally:
        await eng.stop()

    checks = {
        "temp=0 deterministic": det0 == det1,
        "seed reproducible (seed=7 ×2 identical)": s7a == s7b,
        "seed variation (seed 7 ≠ 99)": s7a != s99,
    }
    for k, v in checks.items():
        print(f"  {'OK ' if v else 'BAD'} {k}")
    print(f"RESULT: {sum(checks.values())}/{len(checks)} | t0={det0[:40]!r} s7={s7a[:40]!r} s99={s99[:40]!r}")
    ok = all(checks.values())
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
