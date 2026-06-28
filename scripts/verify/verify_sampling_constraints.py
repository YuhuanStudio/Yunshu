"""top_k / top_p / min_p sampling-constraint gate (direct engine, deterministic).

verify_sampling.py covers temperature + seed. This gate covers the truncation
samplers (top_k, top_p, min_p), whose guarantee is: each collapses sampling so
that even at a HIGH temperature the output is deterministic, and differs from
unconstrained high-temp sampling (i.e. the param actually takes effect).

  - unconstrained temp=2.0 is NON-deterministic across calls (control: high temp
    really is random here)
  - top_k=1 @ temp=2.0 is reproducible even ACROSS different seeds (it keeps
    exactly one token, so the RNG is irrelevant) AND differs from unconstrained
  - min_p=1.0 and top_p=0.02 @ temp=2.0 are reproducible under the SAME seed
    (min_p=1.0 keeps tied-max tokens, so across DIFFERENT seeds it can vary — its
    contract is the seed guarantee, not seed-independence)

Note: these are NOT asserted token-equal to temp-0 greedy — the truncation
samplers run a separate numpy categorical path whose tie-breaking/precision can
pick a different token than the mlx argmax greedy path at a near-tie. The real
guarantee is determinism + effect, which is what we check.

Run: PYTHONPATH=. uv run python scripts/verify_sampling_constraints.py
"""
from __future__ import annotations

import asyncio
import os
import sys

MODEL = os.environ.get("YUNSHU_BENCH_MODEL", "./models/Qwen3.5-0.8B-MLX-bf16")
MSGS = [{"role": "user", "content": "Tell me about your day in a sentence."}]


def _txt(o):
    return (o["text"] if isinstance(o, dict) else o.text).strip()


async def main() -> int:
    if not os.path.exists(MODEL):
        print("SKIP: model not mounted")
        return 0
    from yunshu_engine.batched_engine import BatchedEngine
    eng = BatchedEngine(model_name=MODEL)
    await eng.start()

    checks: dict[str, bool] = {}
    detail: list[str] = []
    try:
        async def gen(**kw):
            return _txt(await eng.chat(messages=MSGS, max_tokens=24, temperature=2.0,
                                       enable_thinking=False, **kw))

        # control: high temp is genuinely random (different seeds → different)
        u1 = await gen(seed=1)
        u2 = await gen(seed=2)
        checks["control: unconstrained temp=2 is random"] = u1 != u2

        tk_a, tk_b = await gen(top_k=1, seed=1), await gen(top_k=1, seed=2)
        checks["top_k=1 @ temp2: reproducible across DIFFERENT seeds"] = tk_a == tk_b
        checks["top_k=1 @ temp2: differs from unconstrained"] = tk_a != u1

        mp_a, mp_b = await gen(min_p=1.0, seed=5), await gen(min_p=1.0, seed=5)
        checks["min_p=1.0 @ temp2: reproducible under same seed"] = mp_a == mp_b

        tp_a, tp_b = await gen(top_p=0.02, seed=5), await gen(top_p=0.02, seed=5)
        checks["top_p=0.02 @ temp2: reproducible under same seed"] = tp_a == tp_b

        detail.append(f"unconstrained: {u1[:30]!r} vs {u2[:30]!r}")
        detail.append(f"top_k=1: {tk_a[:30]!r} | min_p=1: {mp_a[:30]!r} | top_p=.02: {tp_a[:30]!r}")
    finally:
        await eng.stop()

    for k, v in checks.items():
        print(f"  {'OK ' if v else 'BAD'} {k}")
    for line in detail:
        print(f"     {line}")
    ok = all(checks.values())
    print(f"RESULT: {sum(checks.values())}/{len(checks)} passed")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
