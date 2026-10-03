"""Bounded tool-loop reasoning continuity, including clients that rename call IDs.

Only complete successful replies are retained. Keys include the authenticated caller,
model, tools and entire visible history, with IDs removed and JSON arguments canonicalized.
An explicit reasoning field (even empty) always wins. No disk persistence.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
from collections import OrderedDict


def _canonical(messages):
    out = []
    for message in messages:
        m = copy.deepcopy(message)
        for key in ("reasoning", "reasoning_content", "tool_call_id", "id"):
            m.pop(key, None)
        if m.get("role") == "assistant":
            m["content"] = m.get("content") or ""
        for call in m.get("tool_calls") or []:
            call.pop("id", None)
            fn = call.get("function", {})
            args = fn.get("arguments")
            if isinstance(args, str):
                with contextlib.suppress(ValueError):
                    fn["arguments"] = json.loads(args or "{}")
        if isinstance(m.get("content"), list):
            m["content"] = [
                b
                for b in m["content"]
                if b.get("type") not in ("thinking", "redacted_thinking")
            ]
            for block in m["content"]:
                if block.get("type") in ("tool_use", "tool_result"):
                    block.pop("id", None)
                    block.pop("tool_use_id", None)
        out.append(m)
    return out


class ToolReasoningCache:
    def __init__(self, entries=256, max_bytes=8 * 1024 * 1024):
        self.entries, self.max_bytes = entries, max_bytes
        self.data: OrderedDict[str, str] = OrderedDict()
        self.size = 0

    @staticmethod
    def key(body, messages, caller):
        identity = [
            caller,
            body.get("model"),
            body.get("system"),
            body.get("tools"),
            _canonical(messages),
        ]
        return hashlib.sha256(
            json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

    def remember(self, body, message, reasoning, caller):
        if not reasoning or not isinstance(reasoning, str):
            return
        key = self.key(body, [*body["messages"], message], caller)
        # Identical visible trajectories with different hidden reasoning are ambiguous.
        if key in self.data and self.data[key] != reasoning:
            reasoning = ""
        previous = self.data.pop(key, "")
        self.size -= len(previous.encode())
        if len(reasoning.encode()) > self.max_bytes:
            return
        self.data[key] = reasoning
        self.size += len(reasoning.encode())
        while len(self.data) > self.entries or self.size > self.max_bytes:
            _, old = self.data.popitem(last=False)
            self.size -= len(old.encode())

    def restore(self, body, caller):
        body = copy.deepcopy(body)
        for i, message in enumerate(body.get("messages", [])):
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            anthropic = isinstance(content, list)
            has_tools = message.get("tool_calls") or (
                anthropic and any(b.get("type") == "tool_use" for b in content)
            )
            explicit = (
                message.get("reasoning_content") is not None
                or message.get("reasoning") is not None
                or (
                    anthropic
                    and any(
                        b.get("type") in ("thinking", "redacted_thinking")
                        for b in content
                    )
                )
            )
            if not has_tools or explicit:
                continue
            key = self.key(body, body["messages"][: i + 1], caller)
            reasoning = self.data.get(key)
            if reasoning:
                self.data.move_to_end(key)
                if anthropic:
                    content.insert(0, {"type": "thinking", "thinking": reasoning})
                else:
                    message["reasoning_content"] = reasoning
        return body


class ToolReasoningMiddleware:
    def __init__(self, app):
        self.app = app
        self.cache = ToolReasoningCache()

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in ("/v1/chat/completions", "/v1/messages")
        ):
            return await self.app(scope, receive, send)
        chunks = []
        while True:
            event = await receive()
            if event["type"] != "http.request":
                return
            chunks.append(event.get("body", b""))
            if not event.get("more_body"):
                break
        raw = b"".join(chunks)
        try:
            original = json.loads(raw)
            if not isinstance(original, dict) or not isinstance(
                original.get("messages"), list
            ):
                raise ValueError("not a chat request")
            has_tools = original.get("tools") or any(
                isinstance(m, dict)
                and (
                    m.get("tool_calls")
                    or (
                        isinstance(m.get("content"), list)
                        and any(
                            isinstance(b, dict) and b.get("type") == "tool_use"
                            for b in m["content"]
                        )
                    )
                )
                for m in original["messages"]
            )
            if not has_tools:
                raise ValueError("tool-free request")
            headers = dict(scope.get("headers", []))
            caller = hashlib.sha256(
                headers.get(b"authorization", b"")
                + b"\0"
                + headers.get(b"x-api-key", b"")
            ).hexdigest()
            body = self.cache.restore(original, caller)
            raw = json.dumps(body).encode()
            # Request size accounting must see the rewritten size.
            scope = dict(
                scope,
                headers=[
                    (k, v)
                    for k, v in scope.get("headers", [])
                    if k != b"content-length"
                ]
                + [(b"content-length", str(len(raw)).encode())],
            )
        except (ValueError, TypeError, AttributeError):
            body = None
        consumed = False

        async def replay():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": raw, "more_body": False}
            return await receive()

        # Observing bytes must never delay or change the streamed response. A bounded
        # capture keeps failures/cancelled streams out and avoids unbounded buffering.
        response = bytearray()
        status = 0
        capture = True

        async def observe(event):
            nonlocal status, capture
            if event["type"] == "http.response.start":
                status = event["status"]
            elif event["type"] == "http.response.body" and body and status == 200:
                if capture:
                    response.extend(event.get("body", b""))
                if len(response) > self.cache.max_bytes:
                    capture = False
                    response.clear()
                if capture and not event.get("more_body"):
                    self._remember_response(body, bytes(response), caller)
            await send(event)

        await self.app(scope, replay, observe)

    def _remember_response(self, body, raw, caller):
        try:
            text = raw.decode()
            if not body.get("stream"):
                result = json.loads(text)
                if "choices" in result:
                    for choice in result["choices"]:
                        m = choice["message"]
                        if m.get("tool_calls"):
                            self.cache.remember(
                                body,
                                m,
                                m.get("reasoning_content") or m.get("reasoning"),
                                caller,
                            )
                elif result.get("stop_reason") == "tool_use":
                    m = {"role": "assistant", "content": result["content"]}
                    r = "\n".join(
                        b.get("thinking", "")
                        for b in m["content"]
                        if b.get("type") == "thinking"
                    )
                    self.cache.remember(body, m, r, caller)
                return
            events = [
                json.loads(line[6:])
                for line in text.splitlines()
                if line.startswith("data: ") and line[6:] != "[DONE]"
            ]
            if "choices" in (events[0] if events else {}):
                choices = {}
                finished = set()
                for event in events:
                    for choice in event.get("choices", []):
                        index = choice.get("index", 0)
                        m = choices.setdefault(
                            index,
                            {
                                "role": "assistant",
                                "content": "",
                                "reasoning_content": "",
                                "tool_calls": [],
                            },
                        )
                        delta = choice.get("delta", {})
                        m["content"] += delta.get("content") or ""
                        m["reasoning_content"] += (
                            delta.get("reasoning_content")
                            or delta.get("reasoning")
                            or ""
                        )
                        for tc in delta.get("tool_calls") or []:
                            i = tc.get("index", 0)
                            while len(m["tool_calls"]) <= i:
                                m["tool_calls"].append(
                                    {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                )
                            call = m["tool_calls"][i]
                            call["id"] += tc.get("id") or ""
                            for k in ("name", "arguments"):
                                call["function"][k] += (
                                    tc.get("function", {}).get(k) or ""
                                )
                        if choice.get("finish_reason") == "tool_calls":
                            finished.add(index)
                for index in finished:
                    m = choices[index]
                    self.cache.remember(body, m, m["reasoning_content"], caller)
            else:
                blocks = {}
                complete = False
                for event in events:
                    if event.get("type") == "content_block_start":
                        blocks[event["index"]] = event["content_block"]
                    elif event.get("type") == "content_block_delta":
                        b = blocks[event["index"]]
                        d = event["delta"]
                        for src, dst in (
                            ("thinking", "thinking"),
                            ("text", "text"),
                            ("partial_json", "_json"),
                        ):
                            if src in d:
                                b[dst] = b.get(dst, "") + d[src]
                    elif event.get("type") == "message_delta":
                        complete = (
                            event.get("delta", {}).get("stop_reason") == "tool_use"
                        )
                if complete and any(e.get("type") == "message_stop" for e in events):
                    for b in blocks.values():
                        if "_json" in b:
                            b["input"] = json.loads(b.pop("_json"))
                    m = {"role": "assistant", "content": list(blocks.values())}
                    r = "\n".join(
                        b.get("thinking", "")
                        for b in blocks.values()
                        if b.get("type") == "thinking"
                    )
                    self.cache.remember(body, m, r, caller)
        except (ValueError, KeyError, TypeError, AttributeError):
            return  # malformed/incomplete responses never populate the cache
