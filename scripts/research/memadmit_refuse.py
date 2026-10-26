"""Send one request too large for the Metal working set and record how the server answers.

usage: memadmit_refuse.py --url URL --model NAME --ctx-chars N --out FILE.json

Expects a memory_limit refusal (HTTP 507, error.code memory_limit) and the server still
answering a small request afterwards; exits non-zero otherwise (measurement fails closed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from conc_probe import body  # noqa: E402


def judge(status: int, payload: dict, small_ok: bool) -> list[str]:
    """Problems with the observed answer (empty = the refusal behaved)."""
    problems = []
    if status != 507:
        problems.append(f"expected HTTP 507, got {status}")
    code = (payload.get("error") or {}).get("code")
    if code != "memory_limit":
        problems.append(f"expected error.code memory_limit, got {code!r}")
    if not small_ok:
        problems.append("the server did not answer a small request afterwards")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx-chars", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    b = body(a.model, 7, 16, a.ctx_chars, True)
    b["stream"] = False
    b.pop("stream_options", None)
    r = httpx.post(a.url + "/v1/chat/completions", json=b, timeout=1800.0)
    try:
        payload = r.json()
    except ValueError:
        payload = {}
    small = httpx.post(
        a.url + "/v1/chat/completions",
        json={
            "model": a.model,
            "messages": [{"role": "user", "content": "Say ok."}],
            "max_tokens": 8,
            "temperature": 0,
        },
        timeout=300.0,
    )
    problems = judge(r.status_code, payload, small.status_code == 200)
    Path(a.out).write_text(
        json.dumps(
            {
                "status": r.status_code,
                "payload": payload,
                "small_status": small.status_code,
                "problems": problems,
            },
            indent=1,
        )
    )
    print("refusal:", r.status_code, json.dumps(payload)[:300], "problems:", problems)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
