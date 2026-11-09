"""Batch inference API gate (in-process ASGI, real engine).

OpenAI Batch API over the real route: upload a JSONL file (purpose=batch) to
``/v1/files``, create a batch with ``POST /v1/batches`` (input_file_id +
endpoint + completion_window), poll ``GET /v1/batches/{id}`` until terminal,
then read the output / error files from ``/v1/files/{id}/content``.

  - HTTP 200 on upload and create; batch object with a terminal status
  - every submitted custom_id appears exactly once in the output file
  - each output line carries a usable completion (response.body)
  - a line that fails per-item generation lands in the error file WITHOUT
    failing the good items
  - a line that is not valid JSON fails the whole batch at validation
    (status "failed" with an errors list), as the real API does

Run: PYTHONPATH=. uv run python scripts/verify/verify_batch_api.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

MODEL = os.environ.get("YUNSHU_BENCH_MODEL", "./models/Qwen3.5-0.8B-MLX-bf16")
TERMINAL = ("completed", "failed", "expired", "cancelled")


def _line(cid: str, prompt: str) -> dict:
    return {
        "custom_id": cid,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.0,
        },
    }


def _content(out: dict) -> str:
    try:
        return out["response"]["body"]["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def _jsonl(lines: list[dict | str]) -> bytes:
    return "\n".join(x if isinstance(x, str) else json.dumps(x) for x in lines).encode()


async def _upload(client, data: bytes) -> str:
    r = await client.post(
        "/v1/files",
        data={"purpose": "batch"},
        files={"file": ("batch.jsonl", data, "application/jsonl")},
    )
    assert r.status_code == 200, f"upload status={r.status_code} body={r.text[:200]}"
    return r.json()["id"]


async def _run_batch(client, file_id: str, timeout_s: float = 900) -> dict:
    r = await client.post(
        "/v1/batches",
        json={
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    assert r.status_code == 200, f"create status={r.status_code} body={r.text[:200]}"
    bid = r.json()["id"]
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        d = (await client.get(f"/v1/batches/{bid}")).json()
        if d.get("status") in TERMINAL or loop.time() > deadline:
            return d
        await asyncio.sleep(0.5)


async def _read_jsonl(client, file_id: str | None) -> list[dict]:
    if not file_id:
        return []
    r = await client.get(f"/v1/files/{file_id}/content")
    assert r.status_code == 200, f"content status={r.status_code}"
    return [json.loads(x) for x in r.text.splitlines() if x.strip()]


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
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", timeout=180
        ) as client:
            want_ids = ["req-france", "req-math", "req-japan"]
            good = [
                _line("req-france", "Capital of France? One word."),
                _line("req-math", "What is 8 plus 5? Just the number."),
                _line("req-japan", "Capital of Japan? One word."),
            ]
            d = await _run_batch(client, await _upload(client, _jsonl(good)))
            checks["batch: object + terminal status"] = (
                d.get("object") == "batch" and d.get("status") == "completed"
            )
            out = await _read_jsonl(client, d.get("output_file_id"))
            got_ids = [x.get("custom_id") for x in out]
            checks["batch: all custom_ids returned exactly once"] = sorted(
                got_ids
            ) == sorted(want_ids)
            texts = {x.get("custom_id"): _content(x) for x in out}
            checks["batch: results carry completions"] = all(
                texts.get(c, "").strip() for c in want_ids
            )
            detail.append(f"status={d.get('status')} ids={got_ids}")
            for c in want_ids:
                detail.append(f"  {c} → {texts.get(c, '')[:40]!r}")

            # one generation-level failure (no messages) must not sink the good item
            d2 = await _run_batch(
                client,
                await _upload(
                    client,
                    _jsonl(
                        [
                            _line("ok-1", "Say hi."),
                            {
                                "custom_id": "bad-1",
                                "method": "POST",
                                "url": "/v1/chat/completions",
                                "body": {"model": MODEL},
                            },
                        ]
                    ),
                ),
            )
            ok_out = {
                x["custom_id"]: x
                for x in await _read_jsonl(client, d2.get("output_file_id"))
            }
            err_out = {
                x["custom_id"]: x
                for x in await _read_jsonl(client, d2.get("error_file_id"))
            }
            checks["batch: malformed item isolated (good item still ok)"] = bool(
                _content(ok_out.get("ok-1", {})).strip() and "bad-1" in err_out
            )

            # a non-JSON line fails the batch at validation with an errors list
            d3 = await _run_batch(
                client,
                await _upload(client, _jsonl([_line("ok-2", "Hi."), "not json"])),
            )
            checks["batch: invalid JSON line fails the batch with errors"] = d3.get(
                "status"
            ) == "failed" and bool((d3.get("errors") or {}).get("data"))
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
