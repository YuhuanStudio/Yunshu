"""Sequential real HTTP requests, preserving raw SSE and failures beside timings."""

import argparse
import base64
import hashlib
import json
import time
import traceback
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image, ImageDraw

    image_path = args.output.parent / "red-left-blue-right.png"
    im = Image.new("RGB", (256, 128), "blue")
    ImageDraw.Draw(im).rectangle((0, 0, 127, 127), fill="red")
    im.save(image_path)
    short = "List the integers from 1 to 30, separated by commas. No explanation."
    long = (
        "The archive entry has code ALPHA and status OPEN.\n" * 300
    ) + "\nWhat is the code? Reply only with the code."
    cases = [
        ("text_cold", short, False, 96),
        ("text_repeat", short, False, 96),
        ("long_prefill", long, False, 24),
        ("long_repeat", long, False, 24),
        ("vision", "Which half is red, left or right? Reply with one word.", True, 24),
    ]
    for name, text, vision, maximum in cases:
        content = text
        if vision:
            data = base64.b64encode(image_path.read_bytes()).decode()
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + data},
                },
                {"type": "text", "text": text},
            ]
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": maximum,
            "stream": True,
            "stream_options": {"include_usage": True},
            "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False},
        }
        row = {
            "case": name,
            "url": args.url,
            "model": args.model,
            "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        (
            args.output.parent / (args.output.stem + "-" + name + "-request.json")
        ).write_text(json.dumps(body, ensure_ascii=False) + "\n")
        start = time.perf_counter()
        first = None
        chunks = []
        finishes = []
        usage = None
        try:
            req = urllib.request.Request(
                args.url.rstrip("/") + "/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with (
                urllib.request.urlopen(req, timeout=600) as response,
                (args.output.parent / (args.output.stem + "-" + name + ".sse")).open(
                    "wb"
                ) as raw,
            ):
                row["http_status"] = response.status
                for line in response:
                    raw.write(line)
                    raw.flush()
                    if not line.startswith(b"data: "):
                        continue
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        row["done_received"] = True
                        break
                    event = json.loads(data)
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        if delta.get("content"):
                            if first is None:
                                first = time.perf_counter() - start
                            chunks.append(delta["content"])
                        if choice.get("finish_reason"):
                            finishes.append(choice["finish_reason"])
            row.update(
                first_text_s=first,
                wall_s=time.perf_counter() - start,
                output="".join(chunks),
                usage=usage,
                finish_reasons=finishes,
            )
        except Exception as exc:
            row.update(error=traceback.format_exc(), wall_s=time.perf_counter() - start)
            if hasattr(exc, "read"):
                row["error_body"] = exc.read().decode(errors="replace")
        with args.output.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
