"""Grow an 8K cached Qwen VLM history to 32K without copying the model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx


def _call(url: str, model: str, messages: list[dict], expected: str) -> dict:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 32,
        "enable_thinking": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    first = None
    parts = []
    usage = None
    finish = None
    with httpx.stream("POST", url, json=body, timeout=180) as response:
        for line in response.iter_lines():
            if not line.startswith("data: ") or line[6:] == "[DONE]":
                continue
            event = json.loads(line[6:])
            usage = event.get("usage") or usage
            choice = (event.get("choices") or [{}])[0]
            segment = choice.get("delta", {}).get("content") or ""
            if segment and first is None:
                first = time.perf_counter() - start
            parts.append(segment)
            finish = choice.get("finish_reason") or finish
    text = "".join(parts)
    return {
        "status": response.status_code,
        "text": text,
        "expected": expected,
        "correct": text.strip() == expected,
        "first_content_s": first,
        "complete_s": time.perf_counter() - start,
        "usage": usage,
        "finish_reason": finish,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18764/v1/chat/completions")
    parser.add_argument("--model", default="Qwen3.8-27B-oQ4e-mtp")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = (
        "Reference paragraph for a document summary. It does not modify the code. "
    )
    first_user = {
        "role": "user",
        "content": reference * 600
        + "\nThe current code is ALPHA. Reply with the current code only.",
    }
    rows = [
        _call(args.url, args.model, [first_user], "ALPHA"),
        _call(
            args.url,
            args.model,
            [
                first_user,
                {"role": "assistant", "content": "ALPHA"},
                {
                    "role": "user",
                    "content": reference * 1700
                    + "\nUpdate the current code to COBALT. Reply with the current code only.",
                },
            ],
            "COBALT",
        ),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    for row in rows:
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
