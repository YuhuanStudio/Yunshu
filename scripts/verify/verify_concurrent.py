"""Concurrent-request safety gate (in-process ASGI, real engine).

Default serving runs on a single-threaded MLX executor; concurrent requests must
serialize cleanly without corrupting each other's KV/state (a historical failure
mode — see the n>1 KV-corruption note in chat.py). This gate fires several
distinct chat requests concurrently via asyncio.gather and checks they all
complete correctly and independently.

  - all N concurrent requests return HTTP 200 with non-empty content
  - each answer is correct for ITS OWN distinct question (no cross-talk /
    response mix-up under concurrency)
  - usage is well-formed on every response

Run: PYTHONPATH=. uv run python scripts/verify_concurrent.py
"""
from __future__ import annotations

import asyncio
import os
import sys

MODEL = os.environ.get("YUNSHU_BENCH_MODEL", "./models/Qwen3.5-0.8B-MLX-bf16")

# (prompt, substring that MUST appear in a correct answer). Chosen so a 0.8B
# answers reliably at temp 0 — the gate is about CONCURRENCY SAFETY (each answer
# maps to its own question = no cross-talk), not model knowledge.
QUESTIONS = [
    ("What is the capital of France? One word.", "paris"),
    ("What is 6 multiplied by 7? Reply with just the number.", "42"),
    ("What is the capital of Italy? One word.", "rome"),
    ("What is the capital of Japan? One word.", "tokyo"),
    ("How many days are in a week? Just the number.", "7"),
    ("What is the capital of Egypt? One word.", "cairo"),
]


def _norm(s: str) -> str:
    """Lowercase + strip unicode subscripts/spaces for robust substring match."""
    import unicodedata
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


async def main() -> int:
    if not os.path.exists(MODEL):
        print("SKIP: model not mounted")
        return 0

    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    os.environ["YUNSHU_DRAIN_TIMEOUT"] = "0"

    import httpx

    from yunshu_engine.batched_engine import BatchedEngine
    from yunshu_gateway.engine import set_engine
    from yunshu_gateway.main import create_app

    eng = BatchedEngine(model_name=MODEL)
    await eng.start()
    set_engine(eng)

    checks: dict[str, bool] = {}
    detail: list[str] = []
    try:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=180) as client:
            async def ask(q):
                return await client.post("/v1/chat/completions", json={
                    "model": MODEL, "messages": [{"role": "user", "content": q}],
                    "temperature": 0.0, "max_tokens": 24, "enable_thinking": False})

            # fire ALL concurrently
            responses = await asyncio.gather(*[ask(q) for q, _ in QUESTIONS])

            all_200 = all(r.status_code == 200 for r in responses)
            checks["all concurrent requests → HTTP 200"] = all_200
            contents = []
            usage_ok = True
            correct = 0
            for (q, want), r in zip(QUESTIONS, responses):
                if r.status_code != 200:
                    contents.append(f"[{r.status_code}]")
                    continue
                d = r.json()
                c = (d["choices"][0]["message"]["content"] or "")
                contents.append(c[:24])
                u = d.get("usage") or {}
                if not (u.get("prompt_tokens", 0) > 0 and u.get("completion_tokens", 0) >= 0):
                    usage_ok = False
                if _norm(want) in _norm(c):
                    correct += 1
            checks["all responses non-empty"] = all(c.strip() for c in contents)
            checks["usage well-formed on every response"] = usage_ok
            # each answer correct for ITS OWN question → no cross-talk under
            # concurrency. Allow 1 small-model slip; the mapping must clearly hold.
            checks["answers correct for own question (≥5/6, no cross-talk)"] = correct >= 5
            detail.append(f"correct={correct}/{len(QUESTIONS)}")
            for (q, _), c in zip(QUESTIONS, contents):
                detail.append(f"  {q[:34]:34s} → {c!r}")
    finally:
        set_engine(None)
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
