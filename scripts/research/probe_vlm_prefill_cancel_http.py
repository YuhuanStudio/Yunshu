"""Disconnect during long VLM prefill, then measure same-process recovery."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx


def _body(model: str, content: str, max_tokens: int = 16) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "stream": True,
    }


def _collect(url: str, body: dict, timeout: float) -> dict:
    start = time.perf_counter()
    parts = []
    finish = None
    with httpx.stream("POST", url, json=body, timeout=timeout) as response:
        for line in response.iter_lines():
            if not line.startswith("data: ") or line[6:] == "[DONE]":
                continue
            event = json.loads(line[6:])
            choice = (event.get("choices") or [{}])[0]
            parts.append(choice.get("delta", {}).get("content") or "")
            finish = choice.get("finish_reason") or finish
    return {
        "status": response.status_code,
        "text": "".join(parts),
        "finish_reason": finish,
        "complete_s": round(time.perf_counter() - start, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18764/v1/chat/completions")
    parser.add_argument("--model", default="Qwen3.8-27B-oQ4e-mtp")
    parser.add_argument("--reference-repeats", type=int, default=2300)
    parser.add_argument("--disconnect-after", type=float, default=2.0)
    parser.add_argument("--recovery-timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Warm model loading and the Metal owner before the cancellation probe.
    warmup = _collect(args.url, _body(args.model, "Reply WARM only."), 120)
    reference = (
        "Reference paragraph for a document summary. It does not modify the code. "
        * args.reference_repeats
    )
    long_body = _body(
        args.model,
        reference + "\nThe current code is ALPHA. Reply with the current code only.",
    )
    begin = time.perf_counter()
    with httpx.stream("POST", args.url, json=long_body, timeout=120) as response:
        status = response.status_code
        time.sleep(args.disconnect_after)
    disconnected = time.perf_counter()
    try:
        recovery = _collect(
            args.url,
            _body(args.model, "Reply RECOVERED only."),
            args.recovery_timeout,
        )
    except Exception as exc:
        recovery = {"error": f"{type(exc).__name__}: {exc}"}
    result = {
        "warmup": warmup,
        "long_prompt_reference_repeats": args.reference_repeats,
        "long_request_status": status,
        "disconnect_after_s": round(disconnected - begin, 6),
        "recovery": recovery,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
