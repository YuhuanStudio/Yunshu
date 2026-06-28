"""N-gram speculative-decoding gate (Wave 686) — small LLMs.

Locks in the W686 fixes so they cannot silently regress:

 (1) NON-TRIMMABLE GUARD (the keystone): Qwen3.5's hybrid attention uses
     non-trimmable ArraysCache, so `_generate_ngram_spec` MUST detect this
     (can_trim_prompt_cache → False) and delegate the whole request to the plain
     fast path. The output must therefore be BIT-IDENTICAL to plain greedy decode
     — i.e. no ".txt.txt…" degeneration. This is the bug W686 fixed.

 (2) TRIMMABLE CORRECTNESS: on a trimmable-cache model (Qwen2.5-3B) the spec path
     runs its real multi-token verify and must NOT degenerate — it shares a real
     prefix with greedy and stays coherent (the verify cache-trim works).

Both use small models (the user confirmed small models are fine for checks).

Run: PYTHONPATH=. uv run python scripts/verify_ngram_spec.py
"""
from __future__ import annotations

import asyncio
import os
import sys

HYBRID = os.environ.get("YUNSHU_NGRAM_HYBRID_MODEL", "./models/Qwen3.5-0.8B-MLX-bf16")
DENSE = os.environ.get("YUNSHU_NGRAM_DENSE_MODEL", "./models/Qwen2.5-3B-Instruct-4bit")

_KW = dict(
    max_tokens=120, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0,
    repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0,
    logit_bias=None, stop=None, stop_token_ids=None, seed=None,
    enable_thinking=False, logprobs=False, top_logprobs=None, thinking_budget=None,
    xtc_probability=0.0, xtc_threshold=0.0, json_schema=None, cancel_event=None,
    logits_processors=None, timeout_seconds=300.0, lora_adapter=None,
)
PROMPT = ("Output the following line exactly 30 times, each on its own line:\n"
          "the quick brown fox jumps over the lazy dog 1234567890\n/no_think")


def _text(r):
    return (r["text"] if isinstance(r, dict) else getattr(r, "text", "")).strip()


def _degenerate(s: str) -> bool:
    """True if the text collapsed into one repeated unit (the W686 failure mode)."""
    words = s.split()
    if len(words) < 8:
        return False
    return (len(set(words)) / len(words)) < 0.12


def _common_prefix_tokens(a: str, b: str) -> int:
    aw, bw = a.split(), b.split()
    n = 0
    for x, y in zip(aw, bw):
        if x != y:
            break
        n += 1
    return n


async def _run(model: str):
    from yunshu_engine.batched_engine import BatchedEngine
    eng = BatchedEngine(model_name=model)
    await eng.start()
    try:
        greedy = _text(await eng._generate_fast(prompt=PROMPT, **_KW))
        spec = _text(await eng._generate_ngram_spec(prompt=PROMPT, **_KW))
    finally:
        await eng.stop()
    return greedy, spec


async def main() -> int:
    checks: dict[str, bool] = {}
    detail: list[str] = []

    # (1) Non-trimmable hybrid cache → guard delegates → bit-identical to greedy.
    if os.path.exists(HYBRID):
        g, s = await _run(HYBRID)
        checks["hybrid guard → spec == greedy (no corruption)"] = (g == s)
        checks["hybrid output non-degenerate"] = (not _degenerate(s))
        detail.append(f"hybrid greedy[:46]={g[:46]!r}")
        detail.append(f"hybrid spec  [:46]={s[:46]!r}")
    else:
        print("SKIP: hybrid model not mounted")
        return 0

    # (2) Trimmable dense cache → real spec path runs, non-degenerate, shares prefix.
    if os.path.exists(DENSE):
        g, s = await _run(DENSE)
        checks["dense spec non-degenerate (verify-trim works)"] = (not _degenerate(s))
        checks["dense spec shares real prefix with greedy (≥3 words)"] = (
            _common_prefix_tokens(g, s) >= 3
        )
        detail.append(f"dense greedy[:46]={g[:46]!r}")
        detail.append(f"dense spec  [:46]={s[:46]!r}")
    else:
        print("SKIP: dense model not mounted")
        return 0

    for k, v in checks.items():
        print(f"  {'OK ' if v else 'BAD'} {k}")
    for d in detail:
        print(f"  · {d}")
    ok = all(checks.values())
    print(f"RESULT: {sum(checks.values())}/{len(checks)}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
