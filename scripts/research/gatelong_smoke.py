"""Engine-path smoke for the in-process footprint sampler and tools on the text fast path.

Run via scripts/realmodel/serve_and_run.sh (server env YUNSHU_FOOTPRINT_SAMPLE_MS=20):
    python gatelong_smoke.py PORT
Exits nonzero unless /metrics reports a footprint peak >= current > 0 with samples, and a
tools request answers in both streaming and non-streaming modes.
"""

import json
import re
import sys
import urllib.request

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def parse_footprint(text: str) -> dict:
    out = {}
    for m in re.finditer(
        r'^yunshu_process_footprint_bytes\{type="(\w+)"\} (\d+)$', text, re.M
    ):
        out[m.group(1)] = int(m.group(2))
    m = re.search(r"^yunshu_process_footprint_samples_total (\d+)$", text, re.M)
    out["samples"] = int(m.group(1)) if m else 0
    return out


def check_footprint(fp: dict) -> list:
    bad = []
    if not fp.get("current"):
        bad.append("no current footprint")
    if fp.get("peak", 0) < fp.get("current", 1):
        bad.append("peak below current")
    if fp.get("samples", 0) < 10:
        bad.append(f"too few samples: {fp.get('samples')}")
    return bad


def post(port, body, stream=False):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=300).read().decode()


def main(port):
    body = dict(
        model="m",
        messages=[
            {"role": "user", "content": "What is the weather in Paris? Use the tool."}
        ],
        tools=TOOLS,
        max_tokens=96,
        temperature=0,
    )
    d = json.loads(post(port, body))
    ch = d["choices"][0]
    print(
        "non-stream finish",
        ch["finish_reason"],
        "tool_calls",
        ch["message"].get("tool_calls"),
        flush=True,
    )
    s = post(port, dict(body, stream=True))
    assert "data:" in s and "[DONE]" in s, "stream incomplete"
    print("stream ok", len(s), flush=True)
    text = (
        urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10)
        .read()
        .decode()
    )
    fp = parse_footprint(text)
    print("footprint", fp, flush=True)
    bad = check_footprint(fp)
    if bad:
        print("FAIL", bad)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1])))
