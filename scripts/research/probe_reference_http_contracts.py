"""Real-model HTTP cache isolation, constraints, tools and disconnect recovery probes."""

import argparse
import base64
import io
import json
import time
import traceback
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)

    def record(row):
        with a.output.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def request(name, messages, expected=None, **extra):
        body = dict(
            model=a.model,
            messages=messages,
            temperature=0,
            max_tokens=128,
            reasoning_effort="none",
            chat_template_kwargs={"enable_thinking": False},
        )
        body.update(extra)
        row = dict(case=name, expected=expected)
        (a.output.parent / (a.output.stem + "-" + name + "-request.json")).write_text(
            json.dumps(body) + "\n"
        )
        t = time.perf_counter()
        try:
            req = urllib.request.Request(
                a.url + "/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=600) as r:
                data = json.load(r)
            row.update(wall_s=time.perf_counter() - t, response=data)
            if expected is not None:
                row["exact_match"] = (
                    data["choices"][0]["message"].get("content", "").strip() == expected
                )
        except Exception as e:
            row.update(wall_s=time.perf_counter() - t, error=traceback.format_exc())
            if hasattr(e, "read"):
                row["error_body"] = e.read().decode(errors="replace")
        record(row)

    prefix = (
        "Reference material: this paragraph is irrelevant to the final passphrase.\n"
        * 300
    )
    for label, value in [("a", "ORCHID"), ("b", "COBALT"), ("a_again", "ORCHID")]:
        request(
            "prefix_branch_" + label,
            [
                {
                    "role": "user",
                    "content": prefix
                    + "\nThe final passphrase is "
                    + value
                    + ". Reply only with that passphrase.",
                }
            ],
            value,
        )

    from PIL import Image, ImageDraw

    for label, left, right in [
        ("original", "red", "blue"),
        ("swapped", "blue", "red"),
        ("original_again", "red", "blue"),
    ]:
        im = Image.new("RGB", (256, 128), right)
        ImageDraw.Draw(im).rectangle((0, 0, 127, 127), fill=left)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode()
        request(
            "vision_" + label,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + data},
                        },
                        {
                            "type": "text",
                            "text": "Which half is red, left or right? Reply with one lowercase word.",
                        },
                    ],
                }
            ],
            "left" if left == "red" else "right",
        )

    request(
        "json_schema",
        [{"role": "user", "content": "Return the code COBALT and count 7 in JSON."}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "probe",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "enum": ["COBALT"]},
                        "count": {"type": "integer", "enum": [7]},
                    },
                    "required": ["code", "count"],
                    "additionalProperties": False,
                },
            },
        },
    )
    request(
        "tool_call",
        [{"role": "user", "content": "Call lookup for city Taipei."}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "lookup"}},
    )

    body = dict(
        model=a.model,
        messages=[
            {
                "role": "user",
                "content": "Write a very long story about a mountain expedition, at least 2000 words.",
            }
        ],
        temperature=0,
        max_tokens=4096,
        stream=True,
        reasoning_effort="none",
        chat_template_kwargs={"enable_thinking": False},
    )
    t = time.perf_counter()
    try:
        req = urllib.request.Request(
            a.url + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            for line in r:
                if line.startswith(b"data: ") and line[6:].strip() != b"[DONE]":
                    data = json.loads(line[6:])
                    if any(
                        c.get("delta", {}).get("content")
                        for c in data.get("choices", [])
                    ):
                        record(
                            dict(
                                case="disconnect",
                                first_content_s=time.perf_counter() - t,
                                event=data,
                            )
                        )
                        break
        time.sleep(0.2)
        try:
            with urllib.request.urlopen(a.url + "/status", timeout=5) as r:
                record(dict(case="status_after_disconnect", status=json.load(r)))
        except Exception as e:
            record(dict(case="status_after_disconnect", error=str(e)))
    except Exception:
        record(dict(case="disconnect", error=traceback.format_exc()))
    request(
        "after_disconnect",
        [{"role": "user", "content": "Reply only with READY."}],
        "READY",
    )


if __name__ == "__main__":
    main()
