"""Real-server correctness probe for native JSON calls and lazy structural tags.

Run only through gpuq. --base-url allows the CPU fake-server harness.
No timing claims; first failed request stops the probe.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SCHEMA = {
    "type": "object",
    "properties": {"city": {"const": "Taipei"}},
    "required": ["city"],
    "additionalProperties": False,
}
TOOLS = [{"type": "function", "function": {"name": "weather", "parameters": SCHEMA}}]
RF = {
    "type": "structural_tag",
    "structures": [{"begin": "<result>", "end": "</result>", "schema": SCHEMA}],
    "triggers": ["<result>"],
}

RF_TOOL = {
    "type": "structural_tag",
    "structures": [
        {
            "begin": "<tool_call>",
            "end": "</tool_call>",
            "schema": {
                "type": "object",
                "properties": {"name": {"const": "weather"}, "arguments": SCHEMA},
                "required": ["name", "arguments"],
                "additionalProperties": False,
            },
        }
    ],
    "triggers": ["<tool_call>"],
}


def requests(model: str):
    base = {"model": model, "temperature": 0, "max_tokens": 384}
    yield (
        "forced",
        {
            **base,
            "messages": [{"role": "user", "content": "Call weather for Taipei."}],
            "tools": TOOLS,
            "tool_choice": {"type": "function", "function": {"name": "weather"}},
            "stream": False,
        },
    )
    yield (
        "stream",
        {
            **base,
            "messages": [{"role": "user", "content": "Call weather for Taipei."}],
            "tools": TOOLS,
            "tool_choice": "required",
            "stream": True,
        },
    )
    yield (
        "structural",
        {
            **base,
            "messages": [
                {
                    "role": "user",
                    "content": 'Write a brief explanation, then output exactly <result>{"city":"Tokyo"}</result>. Do not use markdown fences.',
                }
            ],
            "response_format": RF,
            "thinking_budget": 96,
            "enable_thinking": True,
        },
    )

    yield (
        "structural_tool",
        {
            **base,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "parameters": {
                            **SCHEMA,
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "response_format": RF_TOOL,
            "messages": [
                {
                    "role": "user",
                    "content": 'Call weather for Tokyo using <tool_call>{"name":"weather","arguments":{"city":"Tokyo"}}</tool_call>. You may reason briefly first.',
                }
            ],
            "thinking_budget": 96,
            "enable_thinking": True,
        },
    )


def judge(kind: str, response: dict | str) -> dict:
    if kind == "stream":
        slots = {}
        for line in str(response).splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if "error" in event:
                raise ValueError(str(event["error"]))
            for choice in event.get("choices", []):
                for call in choice.get("delta", {}).get("tool_calls", []):
                    slot = slots.setdefault(
                        call["index"], {"name": "", "arguments": "", "id": ""}
                    )
                    fn = call.get("function", {})
                    slot["name"] += fn.get("name", "")
                    slot["arguments"] += fn.get("arguments", "")
                    slot["id"] += call.get("id", "")
        calls = list(slots.values())
    elif kind in ("forced", "structural_tool"):
        calls = [
            dict(c["function"], id=c["id"])
            for c in response["choices"][0]["message"]["tool_calls"]
        ]
    else:
        message = response["choices"][0]["message"]
        content = message.get("content") or ""
        if "<result>" not in content:
            if not content and not message.get("reasoning_content"):
                raise ValueError("empty free response")
            return {
                "kind": kind,
                "ok": True,
                "engaged": False,
                "content": content,
                "reasoning_chars": len(message.get("reasoning_content") or ""),
            }
        if "</result>" not in content:
            raise ValueError("structural tag started but did not close")
        body = content.split("<result>", 1)[1].split("</result>", 1)[0]
        if json.loads(body) != {"city": "Taipei"}:
            raise ValueError("structural schema mismatch")
        return {
            "kind": kind,
            "ok": True,
            "engaged": True,
            "reasoning_chars": len(message.get("reasoning_content") or ""),
            "content": content,
        }
    # A structural tag may legitimately fire more than once (parallel calls);
    # forced tool_choice is exactly one. Every call must be valid with its own id.
    many = kind == "structural_tool"
    if (not calls if many else len(calls) != 1) or any(
        c["name"] != "weather" or not c["id"] for c in calls
    ):
        raise ValueError("wrong tool count, name or missing id")
    if len({c["id"] for c in calls}) != len(calls):
        raise ValueError("duplicate tool call ids")
    if any(json.loads(c["arguments"]) != {"city": "Taipei"} for c in calls):
        raise ValueError("wrong tool arguments")
    return {"kind": kind, "ok": True, "calls": calls}


def run(url: str, model: str) -> list[dict]:
    rows = []
    for kind, payload in requests(model):
        req = urllib.request.Request(
            url + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as reply:
            raw = reply.read().decode()
        parsed = raw if kind == "stream" else json.loads(raw)
        try:
            row = judge(kind, parsed)
        except (ValueError, KeyError) as exc:
            raise ValueError(f"{kind}: {exc}; response={parsed!r}") from exc
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return rows


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base-url")
    p.add_argument("--port", type=int, default=18996)
    return p


def available_port(preferred: int, wait_s: float = 600.0) -> int:
    """Wait for the shared server pool; paused gpuq jobs retain their listeners."""
    deadline = time.monotonic() + wait_s
    ports = [preferred, *[p for p in range(18990, 19000) if p != preferred]]
    while True:
        for port in ports:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(("127.0.0.1", port))
                except OSError:
                    continue
                return port
        left = deadline - time.monotonic()
        if left <= 0:
            raise RuntimeError("server port pool busy after waiting")
        print("toolparse: server port pool busy; waiting", flush=True)
        time.sleep(min(5.0, left))


def main() -> int:
    args = parser().parse_args()
    if not 18990 <= args.port <= 18999:
        raise ValueError("probe ports must be 18990–18999")
    proc = None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log = args.out.with_suffix(".server.log").open("w")
    try:
        url = args.base_url or f"http://127.0.0.1:{args.port}"
        if not args.base_url:
            args.port = available_port(args.port)
            url = f"http://127.0.0.1:{args.port}"
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "yunshu_cli",
                    "serve",
                    "--model",
                    args.model,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.port),
                    "--auth-token",
                    "",
                    "--set",
                    f"YUNSHU_SSD_CACHE_DIR={args.out.parent / 'toolparse-cache'}",
                ],
                stdout=log,
                stderr=log,
            )
            for _ in range(120):
                if proc.poll() is not None:
                    raise RuntimeError("server exited before ready")
                try:
                    with urllib.request.urlopen(url + "/health", timeout=2):
                        break
                except OSError:
                    time.sleep(1)
            else:
                raise RuntimeError("server health timed out")
        rows = run(url, args.model)
        args.out.write_text(
            json.dumps(
                {
                    "complete": True,
                    "rows": rows,
                    "server_log_tail": args.out.with_suffix(".server.log")
                    .read_text(errors="replace")
                    .splitlines()[-80:],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        return 0
    except Exception as exc:
        args.out.write_text(
            json.dumps(
                {
                    "complete": False,
                    "error": str(exc),
                    "server_log_tail": args.out.with_suffix(".server.log")
                    .read_text(errors="replace")
                    .splitlines()[-80:],
                }
            )
            + "\n"
        )
        print(str(exc), file=sys.stderr, flush=True)
        return 1
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        log.close()


if __name__ == "__main__":
    sys.exit(main())
