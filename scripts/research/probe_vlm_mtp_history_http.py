"""HTTP multi-turn Qwen3.8 cache/latency probe; run once per server mode."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import httpx


def _rss(pid: int | None) -> int | None:
    if pid is None:
        return None
    try:
        value = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "rss="], text=True
        ).strip()
        return int(value) * 1024
    except (subprocess.CalledProcessError, ValueError):
        return None


def _call(url: str, body: dict) -> dict:
    start = time.perf_counter()
    first = None
    parts = []
    finish = None
    with httpx.stream("POST", url, json=body, timeout=120) as response:
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            if line[6:] == "[DONE]":
                break
            event = json.loads(line[6:])
            choice = event.get("choices", [{}])[0]
            segment = choice.get("delta", {}).get("content") or ""
            if segment:
                if first is None:
                    first = time.perf_counter() - start
                parts.append(segment)
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    return {
        "status": response.status_code,
        "first_content_s": round(first, 6) if first is not None else None,
        "complete_s": round(time.perf_counter() - start, 6),
        "text": "".join(parts),
        "finish_reason": finish,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18764/v1/chat/completions")
    parser.add_argument("--model", default="Qwen3.8-27B-oQ4e-mtp")
    parser.add_argument("--mode", choices=["ar", "mtp"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-pid", type=int)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reference = (
        "Reference paragraph for a document summary. It does not modify the code. "
        * 100
    )
    messages = []
    with args.output.open("w") as file:
        for turn in range(12):
            expected = "COBALT" if turn >= 6 else "ALPHA"
            if turn == 0:
                user = reference + "\nThe current code is ALPHA. Reply with the current code only."
            elif turn == 6:
                user = "Update the current code to COBALT. Reply with the current code only."
            else:
                user = "What is the current code? Reply with that code only."
            messages.append({"role": "user", "content": user})
            result = _call(
                args.url,
                {
                    "model": args.model,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 32,
                    "enable_thinking": False,
                    "stream": True,
                },
            )
            row = {
                "mode": args.mode,
                "turn": turn + 1,
                "expected": expected,
                **result,
                "correct": result["text"].strip() == expected,
                "rss_bytes": _rss(args.server_pid),
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
            print(
                args.mode,
                turn + 1,
                row["correct"],
                row["first_content_s"],
                row["complete_s"],
                flush=True,
            )
            # Keep the next request's history identical between AR and MTP even
            # if either mode makes a mistake on this turn.
            messages.append({"role": "assistant", "content": expected})


if __name__ == "__main__":
    main()
