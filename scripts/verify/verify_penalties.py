"""frequency/presence penalty effect gate (direct engine, deterministic).

The penalties subtract from a token's logits based on prior occurrences. A past
bug silently dropped them (output identical to penalty=0). This gate proves they
measurably reduce repetition, deterministically at temp 0.

  - baseline (penalty 0) is reproducible
  - frequency_penalty=2.0 strictly REDUCES the repeat count of a repetition-prone
    prompt vs penalty 0, and changes the output (escalating per-count penalty)
  - presence_penalty=8.0 also reduces repetition / changes the output. Note: this
    is a FLAT one-shot penalty, so a small value (2.0) cannot overcome a strongly
    dominant token's logit margin — verified pp=2 is a no-op but pp=8 takes effect
    (the param IS plumbed; it is not a constant-output bug)

Run: PYTHONPATH=. uv run python scripts/verify_penalties.py
"""
from __future__ import annotations

import asyncio
import os
import sys

MODEL = os.environ.get("YUNSHU_BENCH_MODEL", "./models/Qwen3.5-0.8B-MLX-bf16")
MSGS = [{"role": "user", "content": "Say the word banana over and over, many times in a row."}]


def _txt(o):
    return (o["text"] if isinstance(o, dict) else o.text)


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
        async def gen(fp=0.0, pp=0.0):
            return _txt(await eng.chat(messages=MSGS, max_tokens=40, temperature=0.0,
                                       enable_thinking=False, frequency_penalty=fp, presence_penalty=pp))

        base0, base1 = await gen(), await gen()
        fp2 = await gen(fp=2.0)
        pp8 = await gen(pp=8.0)
        c0, cf, cp = (base0.lower().count("banana"), fp2.lower().count("banana"),
                      pp8.lower().count("banana"))

        checks["penalty 0 reproducible (control)"] = base0 == base1
        checks["frequency_penalty=2 reduces repetition"] = cf < c0
        checks["frequency_penalty=2 changes output"] = fp2 != base0
        checks["presence_penalty=8 reduces repetition + changes output"] = cp < c0 and pp8 != base0
        detail.append(f"banana count: base={c0} fp2={cf} pp8={cp}")
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
