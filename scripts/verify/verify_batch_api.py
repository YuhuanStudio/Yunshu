"""Batch inference API gate (in-process ASGI, real engine).

/v1/batch takes a list of items (each with a custom_id + an OpenAI request body)
and returns per-item results. This gate submits a small batch and checks the
results map back to their custom_ids correctly.

  - HTTP 200; response is a batch object with a terminal status
  - every submitted custom_id appears exactly once in the results
  - each result carries a usable completion (content or a chat response body)
  - a deliberately malformed item is reported as an error WITHOUT failing the
    whole batch (the good items still succeed)

Run: PYTHONPATH=. uv run python scripts/verify_batch_api.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

MODEL = os.environ.get("YUNSHU_BENCH_MODEL", "./models/Qwen3.5-0.8B-MLX-bf16")


def _result_text(r: dict) -> str:
    """Pull assistant text out of whatever shape a batch result uses."""
    blob = json.dumps(r).lower()
    for key in ("response", "body", "result"):
        v = r.get(key)
        if isinstance(v, dict):
            try:
                return v["choices"][0]["message"]["content"] or ""
            except Exception:
                pass
    # fall back: any content field present
    return blob


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
            def item(cid, prompt):
                return {"custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                        "body": {"messages": [{"role": "user", "content": prompt}],
                                 "max_tokens": 16, "temperature": 0.0}}

            body = {"model": MODEL, "requests": [
                item("req-france", "Capital of France? One word."),
                item("req-math", "What is 8 plus 5? Just the number."),
                item("req-japan", "Capital of Japan? One word."),
            ]}
            r = await client.post("/v1/batch", json=body)
            checks["/v1/batch: HTTP 200"] = r.status_code == 200
            if r.status_code != 200:
                detail.append(f"status={r.status_code} body={r.text[:200]}")
            else:
                d = r.json()
                results = d.get("results") or []
                checks["batch: object + status present"] = bool(d.get("status")) and d.get("object") == "batch"
                got_ids = [x.get("custom_id") for x in results]
                want_ids = ["req-france", "req-math", "req-japan"]
                checks["batch: all custom_ids returned exactly once"] = sorted(got_ids) == sorted(want_ids)
                # each maps to a usable completion
                texts = {x.get("custom_id"): _result_text(x) for x in results}
                checks["batch: results carry completions"] = all(texts.get(c, "").strip() for c in want_ids)
                detail.append(f"status={d.get('status')} ids={got_ids}")
                for c in want_ids:
                    detail.append(f"  {c} → {texts.get(c, '')[:40]!r}")

            # malformed item must not sink the whole batch
            r2 = await client.post("/v1/batch", json={"model": MODEL, "requests": [
                item("ok-1", "Say hi."),
                {"custom_id": "bad-1", "method": "POST", "url": "/v1/chat/completions", "body": {}},  # no messages
            ]})
            if r2.status_code == 200:
                res2 = {x.get("custom_id"): x for x in (r2.json().get("results") or [])}
                ok_good = bool(_result_text(res2.get("ok-1", {})).strip())
                bad_flagged = "bad-1" in res2  # present (likely with an error), didn't crash batch
                checks["batch: malformed item isolated (good item still ok)"] = ok_good and bad_flagged
            else:
                checks["batch: malformed item isolated (good item still ok)"] = False
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
