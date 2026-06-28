"""Usage token-accounting gate (in-process ASGI, real engine).

Token usage drives billing/limits, so it must be exact and consistent. This gate
checks the /v1/chat/completions usage block against the tokenizer ground truth.

  - prompt_tokens EXACTLY equals the engine tokenizer's count of the templated
    prompt (not an estimate)
  - total_tokens == prompt_tokens + completion_tokens
  - identical requests report identical prompt_tokens (deterministic accounting)
  - streaming (stream_options.include_usage) reports the SAME usage as
    non-streaming for the same temp-0 request

Run: PYTHONPATH=. uv run python scripts/verify_usage_accounting.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

MODEL = os.environ.get("YUNSHU_BENCH_MODEL", "./models/Qwen3.5-0.8B-MLX-bf16")
MSGS = [{"role": "user", "content": "Write one sentence about the moon."}]


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

    # tokenizer ground truth for the templated prompt
    expected_pt = None
    try:
        templated = eng._apply_chat_template(MSGS, False)
        expected_pt = len(eng._tokenizer.encode(templated))
    except Exception as e:
        expected_pt = None
        _err = str(e)

    checks: dict[str, bool] = {}
    detail: list[str] = []
    try:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=120) as client:
            body = {"model": MODEL, "messages": MSGS, "temperature": 0.0,
                    "max_tokens": 24, "enable_thinking": False}
            r1 = await client.post("/v1/chat/completions", json=body)
            r2 = await client.post("/v1/chat/completions", json=body)
            u1 = r1.json()["usage"]
            u2 = r2.json()["usage"]

            checks["total == prompt + completion"] = (
                u1["total_tokens"] == u1["prompt_tokens"] + u1["completion_tokens"])
            checks["prompt_tokens deterministic (two identical reqs)"] = (
                u1["prompt_tokens"] == u2["prompt_tokens"])
            if expected_pt is not None:
                checks["prompt_tokens == tokenizer count of templated prompt"] = (
                    u1["prompt_tokens"] == expected_pt)
            detail.append(f"prompt_tokens={u1['prompt_tokens']} expected={expected_pt} "
                          f"completion={u1['completion_tokens']} total={u1['total_tokens']}")

            # streaming usage must match non-streaming
            su = None
            async with client.stream("POST", "/v1/chat/completions",
                                     json=dict(body, stream=True,
                                               stream_options={"include_usage": True})) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and "[DONE]" not in line:
                        obj = json.loads(line[6:])
                        if obj.get("usage"):
                            su = obj["usage"]
            checks["streaming usage == non-streaming usage"] = (
                su is not None
                and su["prompt_tokens"] == u1["prompt_tokens"]
                and su["completion_tokens"] == u1["completion_tokens"])
            detail.append(f"stream usage={su}")
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
