"""Gateway sampling-param forwarding gate (in-process ASGI, real engine).

verify_sampling.py exercises sampling at the ENGINE level. This gate verifies the
GATEWAY forwards seed / stop / max_tokens correctly through the HTTP layer to the
engine — a frequent source of param-dropping regressions.

  - seed is forwarded: same seed (temp>0) → identical text twice; different seed
    → different text
  - stop is forwarded: a stop string truncates the output (finish_reason=="stop")
  - max_tokens is forwarded: a tiny cap yields finish_reason=="length" and
    completion_tokens <= the cap

Run: PYTHONPATH=. uv run python scripts/verify_gateway_sampling.py
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
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=120) as client:
            msgs = [{"role": "user", "content": "Write one short sentence about a mountain."}]

            async def chat(**kw):
                body = {"model": MODEL, "messages": msgs, "enable_thinking": False, **kw}
                r = await client.post("/v1/chat/completions", json=body)
                return r

            # ── seed forwarding ──────────────────────────────────────────────
            r7a = await chat(seed=7, temperature=1.0, max_tokens=24)
            r7b = await chat(seed=7, temperature=1.0, max_tokens=24)
            r99 = await chat(seed=99, temperature=1.0, max_tokens=24)
            t7a = r7a.json()["choices"][0]["message"]["content"]
            t7b = r7b.json()["choices"][0]["message"]["content"]
            t99 = r99.json()["choices"][0]["message"]["content"]
            checks["seed forwarded: same seed → identical"] = t7a == t7b
            checks["seed forwarded: different seed → different"] = t7a != t99

            # ── stop forwarding ──────────────────────────────────────────────
            base = await chat(temperature=0.0, max_tokens=60)
            base_txt = base.json()["choices"][0]["message"]["content"]
            words = [w.strip(".,") for w in base_txt.split() if len(w.strip(".,")) >= 4]
            stop_word = words[len(words) // 2] if words else None
            if stop_word:
                cut = await chat(temperature=0.0, max_tokens=60, stop=[stop_word])
                cj = cut.json()["choices"][0]
                checks["stop forwarded: finish_reason==stop"] = cj.get("finish_reason") == "stop"
                checks["stop forwarded: output truncated"] = len(cj["message"]["content"]) < len(base_txt)
                detail.append(f"stop={stop_word!r} base_len={len(base_txt)} cut_len={len(cj['message']['content'])}")
            else:
                checks["stop forwarded: usable stop word"] = False

            # ── max_tokens forwarding ────────────────────────────────────────
            cap = await chat(temperature=0.0, max_tokens=3)
            cj = cap.json()
            ch = cj["choices"][0]
            ct = (cj.get("usage") or {}).get("completion_tokens", 999)
            checks["max_tokens forwarded: finish_reason==length"] = ch.get("finish_reason") == "length"
            checks["max_tokens forwarded: completion_tokens <= cap"] = ct <= 3
            detail.append(f"max_tokens=3 → fr={ch.get('finish_reason')} completion_tokens={ct}")
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
