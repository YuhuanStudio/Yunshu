"""ToolReasoningMiddleware: same keys and restored bodies, no whole-body copy or re-dump."""

import asyncio
import contextlib
import copy
import hashlib
import json

from yunshu_gateway.middleware.tool_reasoning import (
    ToolReasoningCache,
    ToolReasoningMiddleware,
)


def _old_canonical(messages):
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


def _old_key(body, messages, caller):
    identity = [
        caller,
        body.get("model"),
        body.get("system"),
        body.get("tools"),
        _old_canonical(messages),
    ]
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _old_restore(cache, body, caller):
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
                    b.get("type") in ("thinking", "redacted_thinking") for b in content
                )
            )
        )
        if not has_tools or explicit:
            continue
        key = _old_key(body, body["messages"][: i + 1], caller)
        reasoning = cache.data.get(key)
        if reasoning:
            cache.data.move_to_end(key)
            if anthropic:
                content.insert(0, {"type": "thinking", "thinking": reasoning})
            else:
                message["reasoning_content"] = reasoning
    return body


def _openai_body():
    return {
        "model": "m",
        "system": "sys ✓",
        "tools": [
            {
                "type": "function",
                "function": {"name": "f", "parameters": {"b": 1, "a": 2}},
            }
        ],
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "id": "x1",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "f", "arguments": '{"z": 1, "a": "é"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "g", "arguments": "not json"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_2", "content": "ok"},
        ],
    }


def _anthropic_body():
    return {
        "model": "m",
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "let me"},
                    {"type": "tool_use", "id": "t1", "name": "f", "input": {"q": 1}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "r"}
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t2", "name": "f", "input": {"q": 2}},
                ],
            },
        ],
    }


def _seed(cache, body, caller):
    # remember the reasoning that belongs to each tool turn of the body
    for i, message in enumerate(body["messages"]):
        if (
            message.get("role") == "assistant"
            and message.get("tool_calls")
            or (
                isinstance(message.get("content"), list)
                and any(b.get("type") == "tool_use" for b in message["content"])
            )
        ):
            key = ToolReasoningCache.key(body, body["messages"][: i + 1], caller)
            cache.data[key] = f"reason {i}"


def test_keys_match_the_previous_algorithm():
    for body in (_openai_body(), _anthropic_body()):
        msgs = body["messages"]
        for i in range(len(msgs)):
            assert ToolReasoningCache.key(body, msgs[: i + 1], "c") == _old_key(
                body, msgs[: i + 1], "c"
            )
        assert ToolReasoningCache.key(body, [], "c") == _old_key(body, [], "c")


def test_restore_matches_the_previous_algorithm_and_leaves_input_alone():
    for body in (_openai_body(), _anthropic_body()):
        cache = ToolReasoningCache()
        _seed(cache, body, "caller")
        before = copy.deepcopy(body)
        new = cache.restore(body, "caller")
        assert body == before
        assert new == _old_restore(cache, body, "caller")
        assert new != body  # reasoning was restored into the copy


def test_restore_returns_same_object_when_nothing_is_cached():
    body = _openai_body()
    assert ToolReasoningCache().restore(body, "caller") is body


def _run_middleware(raw, cache=None):
    seen = {}

    async def app(scope, receive, send):
        chunks = []
        while True:
            event = await receive()
            chunks.append(event["body"])
            if not event.get("more_body"):
                break
        seen["raw"] = b"".join(chunks)
        seen["length"] = dict(scope["headers"]).get(b"content-length")
        await send({"type": "http.response.start", "status": 500})
        await send({"type": "http.response.body", "body": b""})

    mw = ToolReasoningMiddleware(app)
    if cache is not None:
        mw.cache = cache
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(b"authorization", b"Bearer x")],
    }
    sent = []
    fed = [{"type": "http.request", "body": raw, "more_body": False}]

    async def receive():
        return fed.pop(0) if fed else {"type": "http.disconnect"}

    async def send(event):
        sent.append(event)

    asyncio.run(mw(scope, receive, send))
    return seen


def test_unchanged_body_is_forwarded_byte_for_byte():
    raw = json.dumps(_openai_body(), indent=3, ensure_ascii=True).encode()
    seen = _run_middleware(raw)
    assert seen["raw"] == raw
    assert seen["length"] == str(len(raw)).encode()


def test_tool_free_body_is_forwarded_byte_for_byte():
    raw = b'{ "model" : "m",  "messages" : [ {"role":"user","content":"hi"} ] }'
    assert _run_middleware(raw)["raw"] == raw


def test_restored_body_is_reserialized_and_length_updated():
    body = _openai_body()
    raw = json.dumps(body, indent=2).encode()
    cache = ToolReasoningCache()
    _seed(cache, body, hashlib.sha256(b"Bearer x" + b"\0" + b"").hexdigest())
    seen = _run_middleware(raw, cache)
    forwarded = json.loads(seen["raw"])
    assert forwarded["messages"][1]["reasoning_content"] == "reason 1"
    assert seen["length"] == str(len(seen["raw"])).encode()
