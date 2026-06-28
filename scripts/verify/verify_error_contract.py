"""HTTP error-contract gate (in-process ASGI — no model, no GPU).

Error paths are a classic silent-regression source (a refactor turns a clean 400
into a 500, or lets a malformed body through). This gate pins the request-
validation contract of /v1/chat/completions with NO engine loaded — so it is
fully deterministic, instant, and runs even when the model drive is absent.

  - malformed/invalid bodies are rejected with 4xx (not 200, not 500):
    missing messages, empty messages, missing model, bad role, negative
    max_tokens, n=0
  - a well-formed request to an unknown model (no engine) returns 404
  - every error body is JSON carrying an "error" object (not an HTML 500 page)

Run: PYTHONPATH=. uv run python scripts/verify_error_contract.py
"""
from __future__ import annotations

import asyncio
import sys


async def main() -> int:
    import os
    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    os.environ["YUNSHU_DRAIN_TIMEOUT"] = "0"

    import httpx

    from yunshu_gateway.main import create_app

    # Bad requests must be 4xx client errors (reject), NOT 200 and NOT 5xx.
    bad_bodies = {
        "missing messages": {"model": "m"},
        "empty messages": {"model": "m", "messages": []},
        "missing model": {"messages": [{"role": "user", "content": "hi"}]},
        "bad role": {"model": "m", "messages": [{"role": "alien", "content": "hi"}]},
        "negative max_tokens": {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": -5},
        "n=0": {"model": "m", "messages": [{"role": "user", "content": "hi"}], "n": 0},
    }

    checks: dict[str, bool] = {}
    detail: list[str] = []
    transport = httpx.ASGITransport(app=create_app())  # NO engine set
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as client:
        for name, body in bad_bodies.items():
            r = await client.post("/v1/chat/completions", json=body)
            is_4xx = 400 <= r.status_code < 500
            json_err = False
            try:
                json_err = isinstance(r.json().get("error") or r.json().get("detail"), (str, dict, list))
            except Exception:
                json_err = False
            checks[f"reject: {name} → 4xx"] = is_4xx
            checks[f"reject: {name} → JSON error body"] = json_err
            detail.append(f"{name}: {r.status_code}")

        # Well-formed request, unknown model, no engine → 404 (not 500/200).
        r = await client.post("/v1/chat/completions", json={
            "model": "definitely-not-loaded",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
        })
        checks["unknown model (no engine) → 404"] = r.status_code == 404
        detail.append(f"unknown-model: {r.status_code}")

    for k, v in checks.items():
        print(f"  {'OK ' if v else 'BAD'} {k}")
    print("     " + " | ".join(detail))
    ok = all(checks.values())
    print(f"RESULT: {sum(checks.values())}/{len(checks)} passed")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
