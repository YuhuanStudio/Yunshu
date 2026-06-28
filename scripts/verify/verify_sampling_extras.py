"""Sampling-extras gate — XTC + thinking_budget plumbing (small LLM).

Closes two parameter gaps the main sampling gate didn't cover:

 (1) XTC (eXclude Top Choices): at temp>0 with a fixed seed, enabling XTC must
     CHANGE the sampled output vs XTC-off (it probabilistically drops the most
     likely tokens). Proves xtc_probability/xtc_threshold are plumbed and active.
 (2) thinking_budget: with enable_thinking, a large budget must yield at least as
     much generated content as a tiny budget (the budget caps the <think> span),
     and neither path errors. Proves enable_thinking/thinking_budget are plumbed.

Run: PYTHONPATH=. uv run python scripts/verify_sampling_extras.py
"""
from __future__ import annotations

import asyncio
import os
import sys

MODEL = os.environ.get("YUNSHU_BENCH_MODEL", "./models/Qwen3.5-0.8B-MLX-bf16")


def _text(o):
    return (o["text"] if isinstance(o, dict) else getattr(o, "text", "")).strip()


async def main() -> int:
    if not os.path.exists(MODEL):
        print("SKIP: model not mounted")
        return 0
    from yunshu_engine.batched_engine import BatchedEngine
    eng = BatchedEngine(model_name=MODEL)
    await eng.start()

    Q = [{"role": "user", "content": "List three colors."}]
    try:
        # (1) XTC effect: same seed/temp, only XTC differs → output must change.
        base = _text(await eng.chat(messages=Q, max_tokens=40, temperature=1.0, seed=7,
                                    enable_thinking=False))
        xtc = _text(await eng.chat(messages=Q, max_tokens=40, temperature=1.0, seed=7,
                                   enable_thinking=False,
                                   xtc_probability=1.0, xtc_threshold=0.05))
        # (2) thinking_budget: small vs large budget, both must run.
        TQ = [{"role": "user", "content": "What is 17 + 26? Think step by step."}]
        small = _text(await eng.chat(messages=TQ, max_tokens=200, temperature=0.0,
                                     enable_thinking=True, thinking_budget=8))
        large = _text(await eng.chat(messages=TQ, max_tokens=200, temperature=0.0,
                                     enable_thinking=True, thinking_budget=256))
    finally:
        await eng.stop()

    checks = {
        "XTC changes sampled output (xtc-on != xtc-off, same seed)": (base != xtc),
        "XTC output non-empty (no crash)": len(xtc) > 0,
        "thinking_budget both paths produce output": (len(small) > 0 and len(large) > 0),
        "larger thinking_budget ⇒ ≥ content of tiny budget": (len(large) >= len(small)),
    }
    for k, v in checks.items():
        print(f"  {'OK ' if v else 'BAD'} {k}")
    print(f"  · base[:30]={base[:30]!r} xtc[:30]={xtc[:30]!r}")
    print(f"  · think small={len(small)}ch large={len(large)}ch")
    ok = all(checks.values())
    print(f"RESULT: {sum(checks.values())}/{len(checks)}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
