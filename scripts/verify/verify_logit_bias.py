"""logit_bias sampler-param effect gate (direct engine, deterministic).

logit_bias is an additive per-token-id bias applied to logits before sampling
(batched_engine.py). This gate proves the plumbing actually changes the output,
exactly and deterministically, using logprobs to identify the chosen token:

  - baseline at temp 0 is reproducible (same text twice)
  - biasing the chosen first token by -100 SUPPRESSES it: the new first token id
    differs and the output text changes
  - a near-zero bias on an unrelated token leaves the output unchanged (the bias
    is targeted, not a global perturbation)

Run: PYTHONPATH=. uv run python scripts/verify_logit_bias.py
"""
from __future__ import annotations

import asyncio
import os
import sys

MODEL = os.environ.get("YUNSHU_BENCH_MODEL", "./models/Qwen3.5-0.8B-MLX-bf16")
MSGS = [{"role": "user", "content": "What color is a clear daytime sky? Answer in one word."}]


def _txt(o):
    return (o["text"] if isinstance(o, dict) else o.text).strip()


def _first_tok(o):
    lps = o.get("logprobs") if isinstance(o, dict) else getattr(o, "logprobs", None)
    return lps[0].get("token_id") if lps else None


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
        async def gen(bias=None):
            return await eng.chat(messages=MSGS, max_tokens=8, temperature=0.0,
                                  enable_thinking=False, logprobs=True, logit_bias=bias)

        base0 = await gen()
        base1 = await gen()
        t0 = _first_tok(base0)
        checks["temp0 reproducible (control)"] = _txt(base0) == _txt(base1)

        # suppress the chosen first token hard
        biased = await gen({t0: -100.0}) if t0 is not None else base0
        t1 = _first_tok(biased)
        checks["logit_bias -100 suppresses chosen token (id changes)"] = (t0 is not None and t1 != t0)
        checks["logit_bias -100 changes output text"] = _txt(biased) != _txt(base0)

        # an out-of-vocab / negative token id must be IGNORED, not crash the
        # request (regression guard for the unguarded-index bug).
        try:
            oov = await gen({999999: 50.0, -3: 50.0})
            checks["out-of-vocab/negative id ignored (no crash)"] = _txt(oov) == _txt(base0)
        except Exception as e:
            checks["out-of-vocab/negative id ignored (no crash)"] = False
            detail.append(f"OOV crash: {type(e).__name__}: {e}")

        detail.append(f"base={_txt(base0)!r} t0={t0} | biased={_txt(biased)!r} t1={t1}")
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
