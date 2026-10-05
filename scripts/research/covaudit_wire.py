"""Chat Completions (opencode) and Responses (Codex) renderings of the coverage-audit session.

`covaudit_session.py` builds one canonical history in Anthropic Messages shape. These helpers render
the same history as /v1/chat/completions or /v1/responses requests and fold each stream back into
the shape `parse_sse` returns (text, tool_calls with parsed input, stop_reason, usage with
cache_read_input_tokens), so one judge covers all three client wires.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path


def _blocks(content) -> list:
    return (
        [{"type": "text", "text": content}]
        if isinstance(content, str)
        else list(content)
    )


def system_text(body: dict) -> str:
    s = body.get("system") or ""
    return s if isinstance(s, str) else "".join(b.get("text", "") for b in s)


def to_chat(body: dict) -> dict:
    msgs = [{"role": "system", "content": system_text(body)}]
    for m in body["messages"]:
        text, calls = "", []
        for b in _blocks(m["content"]):
            if b["type"] == "text":
                text += b["text"]
            elif b["type"] == "tool_use":
                calls.append(
                    {
                        "id": b["id"],
                        "type": "function",
                        "function": {
                            "name": b["name"],
                            "arguments": json.dumps(b["input"]),
                        },
                    }
                )
            elif b["type"] == "tool_result":
                msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": b["tool_use_id"],
                        "content": b["content"],
                    }
                )
        if m["role"] == "assistant":
            msg = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            msgs.append(msg)
        elif text:
            msgs.append({"role": "user", "content": text})
    return {
        "model": body["model"],
        "messages": msgs,
        "max_tokens": body["max_tokens"],
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": "thinking" not in body},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in body["tools"]
        ],
    }


def to_responses(body: dict) -> dict:
    items: list = []
    for m in body["messages"]:
        for b in _blocks(m["content"]):
            if b["type"] == "text":
                if m["role"] == "assistant":
                    items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": b["text"]}],
                        }
                    )
                else:
                    items.append(
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": b["text"]}],
                        }
                    )
            elif b["type"] == "tool_use":
                items.append(
                    {
                        "type": "function_call",
                        "call_id": b["id"],
                        "name": b["name"],
                        "arguments": json.dumps(b["input"]),
                    }
                )
            elif b["type"] == "tool_result":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": b["tool_use_id"],
                        "output": b["content"],
                    }
                )
    return {
        "model": body["model"],
        "instructions": system_text(body),
        "input": items,
        "max_output_tokens": body["max_tokens"],
        "temperature": 0,
        "stream": True,
        "store": False,
        "reasoning": {"effort": "medium" if "thinking" not in body else "none"},
        "tools": [
            {
                "type": "function",
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            }
            for t in body["tools"]
        ],
    }


def _data_lines(lines):
    for raw in lines:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        raw = raw.strip()
        if raw.startswith("data:"):
            payload = raw[5:].strip()
            if payload == "[DONE]":
                yield None
            elif payload:
                yield json.loads(payload)


def _args(raw: str):
    try:
        return json.loads(raw or "{}")
    except ValueError:
        return None


def parse_chat_sse(lines) -> dict:
    text, calls, usage, stop, ended = "", {}, {}, None, False
    for d in _data_lines(lines):
        if d is None:
            ended = True
            continue
        if "error" in d:
            raise RuntimeError(f"stream error: {d}")
        if d.get("usage"):
            u = d["usage"]
            cached = int(
                (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            )
            usage = {
                "input_tokens": int(u.get("prompt_tokens") or 0) - cached,
                "cache_read_input_tokens": cached,
                "output_tokens": u.get("completion_tokens"),
            }
        for ch in d.get("choices") or []:
            x = ch.get("delta") or {}
            text += x.get("content") or ""
            for tc in x.get("tool_calls") or []:
                c = calls.setdefault(
                    tc.get("index", 0), {"name": "", "id": None, "json": ""}
                )
                c["id"] = tc.get("id") or c["id"]
                f = tc.get("function") or {}
                c["name"] = f.get("name") or c["name"]
                c["json"] += f.get("arguments") or ""
            if ch.get("finish_reason"):
                stop = {
                    "tool_calls": "tool_use",
                    "stop": "end_turn",
                    "length": "max_tokens",
                }.get(ch["finish_reason"], ch["finish_reason"])
    return {
        "text": text,
        "thinking": "",
        "tool_calls": [
            {
                "name": c["name"],
                "id": c["id"],
                "input": _args(c["json"]),
                "raw": c["json"],
            }
            for _, c in sorted(calls.items())
        ],
        "stop_reason": stop,
        "usage": usage,
        "ended": ended and stop is not None,
    }


def parse_responses_sse(lines) -> dict:
    text, items, usage, stop, ended = "", {}, {}, None, False
    for d in _data_lines(lines):
        if d is None:
            continue
        t = d.get("type", "")
        if t == "error":
            raise RuntimeError(f"stream error: {d}")
        if t == "response.output_text.delta":
            text += d.get("delta", "")
        elif (
            t == "response.output_item.added"
            and d["item"].get("type") == "function_call"
        ):
            it = d["item"]
            items[it.get("id") or it["call_id"]] = {
                "name": it["name"],
                "id": it["call_id"],
                "json": it.get("arguments") or "",
            }
        elif t == "response.function_call_arguments.delta":
            if d["item_id"] in items:
                items[d["item_id"]]["json"] += d.get("delta", "")
        elif t == "response.function_call_arguments.done":
            if d["item_id"] in items:
                items[d["item_id"]]["json"] = d.get(
                    "arguments", items[d["item_id"]]["json"]
                )
        elif t in ("response.completed", "response.incomplete"):
            r = d["response"]
            ended = True
            u = r.get("usage") or {}
            cached = int(
                (u.get("input_tokens_details") or {}).get("cached_tokens") or 0
            )
            usage = {
                "input_tokens": int(u.get("input_tokens") or 0) - cached,
                "cache_read_input_tokens": cached,
                "output_tokens": u.get("output_tokens"),
            }
            has_call = any(
                o.get("type") == "function_call" for o in r.get("output") or []
            )
            stop = (
                "tool_use"
                if has_call
                else ("max_tokens" if t == "response.incomplete" else "end_turn")
            )
    return {
        "text": text,
        "thinking": "",
        "tool_calls": [
            {
                "name": c["name"],
                "id": c["id"],
                "input": _args(c["json"]),
                "raw": c["json"],
            }
            for c in items.values()
        ],
        "stop_reason": stop,
        "usage": usage,
        "ended": ended,
    }


def send_api(url: str, body: dict, api: str, timeout=900) -> dict:
    """POST the canonical (Messages-shaped) body rendered for `api` (chat | responses); parsed reply."""
    path, payload, parse = {
        "chat": ("/v1/chat/completions", to_chat(body), parse_chat_sse),
        "responses": ("/v1/responses", to_responses(body), parse_responses_sse),
    }[api]
    req = urllib.request.Request(
        url + path,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json", "authorization": "Bearer k"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        lines = list(r)
    raw = os.environ.get(
        "COVAUDIT_RAW"
    )  # a directory: keep every raw stream for diagnosis
    if raw:
        Path(raw).mkdir(parents=True, exist_ok=True)
        n = len(list(Path(raw).glob("*.sse")))
        (Path(raw) / f"{api}-{n:03d}.sse").write_bytes(b"".join(lines))
    res = parse(lines)
    res["wall_s"] = round(time.monotonic() - t0, 2)
    return res
