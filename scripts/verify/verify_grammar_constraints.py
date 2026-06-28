"""Grammar-constraint gate (choice + regex), small LLM.

The constrained decoder supports json_schema types beyond object schemas:
{"type":"choice","choices":[...]} and {"type":"regex","pattern":"..."}. These had
NO gate. Verifies the output is forced into the choice set / to fully match the
regex, across a few prompts.

Run: PYTHONPATH=. uv run python scripts/verify_grammar_constraints.py
"""
from __future__ import annotations

import asyncio
import os
import re
import sys

MODEL = os.environ.get("YUNSHU_BENCH_MODEL", "./models/Qwen3.5-0.8B-MLX-bf16")


async def main() -> int:
    if not os.path.exists(MODEL):
        print("SKIP: model not mounted")
        return 0
    from yunshu_engine.batched_engine import BatchedEngine
    eng = BatchedEngine(model_name=MODEL)
    await eng.start()

    async def gen(prompt, schema, mt=16):
        o = await eng.chat(messages=[{"role": "user", "content": prompt}],
                           max_tokens=mt, temperature=0.0, enable_thinking=False,
                           json_schema=schema)
        return (o["text"] if isinstance(o, dict) else o.text).strip()
    fails = 0
    try:
        # choice
        for prompt, choices in [("Is the sky blue? Answer.", ["yes", "no"]),
                                ("Pick a primary color.", ["red", "green", "blue"])]:
            out = await gen(prompt, {"type": "choice", "choices": choices})
            ok = out in choices
            print(f"  {'OK ' if ok else 'BAD'} choice {choices}: {out!r}")
            fails += not ok
        # regex
        for prompt, pat in [("Give a phone number.", r"\d{3}-\d{4}"),
                            ("Output a 2-letter uppercase code.", r"[A-Z]{2}")]:
            out = await gen(prompt, {"type": "regex", "pattern": pat})
            ok = bool(re.fullmatch(pat, out))
            print(f"  {'OK ' if ok else 'BAD'} regex /{pat}/: {out!r}")
            fails += not ok
    finally:
        await eng.stop()
    print(f"RESULT: {4 - fails} passed, {fails} failed")
    print("PASS" if fails == 0 else "FAIL")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
