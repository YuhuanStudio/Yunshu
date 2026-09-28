"""Concurrent-request check for any OpenAI-compatible server.

1. Correctness under concurrency: N simultaneous short questions with known
   answers (arithmetic, JSON schema, tool call), greedy, non-thinking.
2. Throughput: one streaming 256-token generation alone, then N at once.
   Records per-request TTFT and decode rate, and the aggregate tok/s, so a
   server that serializes requests shows up as aggregate ~= single.

    python scripts/research/probe_concurrency.py --url http://127.0.0.1:18764 \
        --model Qwen3.8-27B --n 8 --output runs/concurrency.jsonl
"""

import argparse
import concurrent.futures
import http.client
import json
import time
import urllib.parse
from pathlib import Path

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather forecast for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city", "days"],
        },
    },
}
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def post(url, body, stream=False, timeout=1800):
    u = urllib.parse.urlparse(url)
    body = {
        "temperature": 0.0,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
        **body,
    }
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    t0 = time.perf_counter()
    conn.request(
        "POST",
        "/v1/chat/completions",
        json.dumps(body),
        {"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    if not stream:
        data = json.loads(resp.read() or b"{}")
        conn.close()
        return data, time.perf_counter() - t0
    first = last = None
    text, usage = "", None
    for line in resp:
        if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
            continue
        ev = json.loads(line[6:])
        usage = ev.get("usage") or usage
        for ch in ev.get("choices") or []:
            piece = (ch.get("delta") or {}).get("content")
            if piece:
                now = time.perf_counter()
                first = first or now
                last = now
                text += piece
    conn.close()
    n = (usage or {}).get("completion_tokens") or 0
    return {
        "ttft_s": round(first - t0, 3) if first else None,
        "wall_s": round(time.perf_counter() - t0, 3),
        "tokens": n,
        "decode_tps": round((n - 1) / (last - first), 1)
        if first and last and last > first and n > 1
        else None,
        "text": text,
    }, None


def qa_cases(model, n):
    cases = []
    for i in range(n):
        kind = ("math", "schema", "tool")[i % 3]
        a, b = 17 + 3 * i, 25 + 7 * i
        if kind == "math":
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"What is {a}+{b}? Reply with only the number.",
                    }
                ],
                "max_tokens": 16,
            }
        elif kind == "schema":
            body = {
                "messages": [
                    {"role": "user", "content": f"What is {a}+{b}? Reply as JSON."}
                ],
                "max_tokens": 32,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "a", "schema": SCHEMA},
                },
            }
        else:
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Get the weather for Oslo for {i % 5 + 1} days.",
                    }
                ],
                "max_tokens": 128,
                "tools": [TOOL],
                "tool_choice": "auto",
            }
        cases.append(
            (kind, a + b if kind != "tool" else i % 5 + 1, {"model": model, **body})
        )
    return cases


def check(kind, expect, data):
    msg = ((data.get("choices") or [{}])[0]).get("message") or {}
    content = (msg.get("content") or "").strip()
    try:
        if kind == "math":
            return content.replace(",", "").strip(".") == str(expect), content
        if kind == "schema":
            return json.loads(content).get("answer") == expect, content
        calls = msg.get("tool_calls") or []
        args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
        return args.get("city", "").lower() == "oslo" and args.get(
            "days"
        ) == expect, args
    except Exception as e:  # noqa: BLE001
        return False, f"{content!r} {e}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--note", default="")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out = a.output.open("a")

    def emit(row):
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps(row)[:400], flush=True)

    emit(
        {
            "kind": "meta",
            "url": a.url,
            "model": a.model,
            "n": a.n,
            "note": a.note,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    cases = qa_cases(a.model, a.n)
    serial = {}
    for i, (kind, expect, body) in enumerate(cases):
        data, wall = post(a.url, body)
        serial[i] = check(kind, expect, data)
    with concurrent.futures.ThreadPoolExecutor(a.n) as pool:
        futs = {
            i: pool.submit(post, a.url, body) for i, (_, _, body) in enumerate(cases)
        }
        conc = {
            i: check(cases[i][0], cases[i][1], f.result()[0])
            + (round(f.result()[1], 2),)
            for i, f in futs.items()
        }
    for i in range(a.n):
        emit(
            {
                "kind": "qa",
                "i": i,
                "type": cases[i][0],
                "serial_ok": serial[i][0],
                "concurrent_ok": conc[i][0],
                "same_answer": str(serial[i][1]) == str(conc[i][1]),
                "concurrent_wall_s": conc[i][2],
                "serial": str(serial[i][1])[:80],
                "concurrent": str(conc[i][1])[:80],
            }
        )
    prompt = (
        "Write a Python module implementing an LRU cache with docstrings and tests."
    )
    body = {
        "model": a.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": a.tokens,
    }
    single, _ = post(a.url, body, stream=True)
    emit({"kind": "single", **{k: v for k, v in single.items() if k != "text"}})
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(a.n) as pool:
        rs = list(pool.map(lambda _: post(a.url, body, stream=True)[0], range(a.n)))
    wall = time.perf_counter() - t0
    total = sum(r["tokens"] for r in rs)
    emit(
        {
            "kind": "concurrent",
            "n": a.n,
            "wall_s": round(wall, 2),
            "aggregate_tps": round(total / wall, 1),
            "single_decode_tps": single["decode_tps"],
            "ttft_s": [r["ttft_s"] for r in rs],
            "decode_tps": [r["decode_tps"] for r in rs],
            "same_text_as_single": sum(r["text"] == single["text"] for r in rs),
        }
    )
    ok = sum(bool(conc[i][0]) for i in range(a.n))
    emit(
        {
            "kind": "summary",
            "qa_concurrent_ok": ok,
            "qa_n": a.n,
            "aggregate_tps": round(total / wall, 1),
            "single_decode_tps": single["decode_tps"],
        }
    )


if __name__ == "__main__":
    main()
