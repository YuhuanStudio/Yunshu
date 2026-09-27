"""Sequential task-latency probe for an already-running local chat endpoint.

Keeps output and usage for correctness review. Repeated measurements are not
an official benchmark; compare backends only under documented model settings.
This script never resolves or downloads a model.
"""

import argparse
import base64
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def image_data(red_on_left):
    from io import BytesIO

    from PIL import Image, ImageDraw

    im = Image.new("RGB", (256, 128), "blue")
    if red_on_left:
        box = (0, 0, 127, 127)
    else:
        box = (128, 0, 255, 127)
    ImageDraw.Draw(im).rectangle(box, fill="red")
    buffer = BytesIO()
    im.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def make_cases():
    long_prefix = "The archive record is a neutral entry with no code.\n" * 480
    history = []
    for turn in range(10):
        history.append({"role": "user", "content": f"Progress note {turn}: continue tracking the room code ORCHID."})
        history.append({"role": "assistant", "content": "The room code remains ORCHID."})
    return [
        ("short", [{"role": "user", "content": "Reply with only ORCHID."}], 24, "ORCHID"),
        (
            "code",
            [{"role": "user", "content": "Write a Python function named sum_even that sums even integers from a list. Return code only."}],
            256,
            "def sum_even",
        ),
        (
            "long_cold",
            [{"role": "user", "content": long_prefix + "\nThe final code is ALPHA. What is the final code? Reply with the code only."}],
            32,
            "ALPHA",
        ),
        (
            "long_repeat",
            [{"role": "user", "content": long_prefix + "\nThe final code is ALPHA. What is the final code? Reply with the code only."}],
            32,
            "ALPHA",
        ),
        (
            "long_edited_tail",
            [{"role": "user", "content": long_prefix + "\nThe final code is COBALT. What is the final code? Reply with the code only."}],
            32,
            "COBALT",
        ),
        (
            "multiturn",
            [
                {"role": "user", "content": "Remember that the room code is ORCHID."},
                {"role": "assistant", "content": "I will remember the room code."},
                {"role": "user", "content": "A visitor first said BLUE, then corrected themselves. The room code remains unchanged. What is the room code? Reply with just the code."},
            ],
            32,
            "ORCHID",
        ),
        (
            "history_20",
            history + [{"role": "user", "content": "What is the current room code? Reply with only the code."}],
            32,
            "ORCHID",
        ),
        (
            "history_20_edited_tail",
            history + [{"role": "user", "content": "The room code changed to COBALT. What is the current room code? Reply with only the code."}],
            32,
            "COBALT",
        ),
        (
            "vision_left",
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_data(True)}}, {"type": "text", "text": "Which half is red? Reply left or right."}]}],
            32,
            "left",
        ),
        (
            "vision_right",
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_data(False)}}, {"type": "text", "text": "Which half is red? Reply left or right."}]}],
            32,
            "right",
        ),
    ]


def run_request(url, model, messages, max_tokens, raw_path):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    result = {
        "input_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
        "first_content_s": None,
        "first_reasoning_s": None,
        "first_event_s": None,
        "content": "",
        "reasoning": "",
        "finish_reason": None,
        "usage": None,
        "done_received": False,
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as response, raw_path.open("wb") as raw:
        result["http_status"] = response.status
        for line in response:
            raw.write(line)
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                result["done_received"] = True
                break
            event = json.loads(data)
            if result["first_event_s"] is None:
                result["first_event_s"] = time.perf_counter() - started
            if event.get("error"):
                result["stream_error"] = event["error"]
            if event.get("usage"):
                result["usage"] = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content"):
                    if result["first_content_s"] is None:
                        result["first_content_s"] = time.perf_counter() - started
                    result["content"] += delta["content"]
                if delta.get("reasoning_content"):
                    if result["first_reasoning_s"] is None:
                        result["first_reasoning_s"] = time.perf_counter() - started
                    result["reasoning"] += delta["reasoning_content"]
                if choice.get("finish_reason"):
                    result["finish_reason"] = choice["finish_reason"]
    result["wall_s"] = time.perf_counter() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--case", nargs="+", dest="selected_cases")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    cases = make_cases()
    if args.selected_cases:
        cases = [case for case in cases if case[0] in args.selected_cases]
        if not cases:
            parser.error("No matching cases")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output.with_suffix("")
    raw_dir.mkdir(parents=True, exist_ok=True)
    for repeat in range(args.repeats):
        for name, messages, limit, expected in cases:
            row = {
                "case": name,
                "repeat": repeat,
                "model": args.model,
                "expected": expected,
                "process_load_state": "not_controlled",
                "first_in_sequence": repeat == 0 and name == cases[0][0],
                "prefix_first_exposure": repeat == 0 and name == "long_cold",
            }
            raw_path = raw_dir / f"{repeat:03d}-{name}.sse"
            try:
                row.update(run_request(args.url, args.model, messages, limit, raw_path))
                out = row["content"].strip()
                row["task_ok"] = expected in out if name == "code" else out.lower() == expected.lower()
            except Exception as exc:
                row["error"] = repr(exc)
                if hasattr(exc, "read"):
                    row["error_body"] = exc.read().decode(errors="replace")
                row["task_ok"] = False
            with args.output.open("a") as file:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k not in ("content", "reasoning")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
