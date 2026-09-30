"""Claude Code's tool-calling shapes through the real /v1/messages route (fake engine, official
anthropic SDK): malformed Qwen tool calls become tool_use blocks, no markup leaks, and the
Anthropic-specific shapes Claude Code relies on hold (stop_reason, block ordering, parallel
tool_use, tool_result with is_error / images, cache_control)."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

anthropic = pytest.importorskip("anthropic")
pytest.importorskip("mlx_vlm")

from yunshu_engine.batched_engine import BatchedEngine, GenerationOutput  # noqa: E402
from yunshu_engine.tool_format import (  # noqa: E402
    INJECTED_JSON,
    JSON_MESSAGE,
    _upstream,
)

TOOLS = [
    {
        "name": "Read",
        "description": "Read a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "Bash",
        "description": "Run a command",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
]

WELL_FORMED = (
    "<tool_call>\n<function=Read>\n<parameter=file_path>\n/a.py\n</parameter>\n"
    "<parameter=limit>\n50\n</parameter>\n</function>\n</tool_call>"
)
NO_FUNCTION_CLOSE = "<tool_call>\n<function=Read>\n<parameter=file_path>\n/b.py\n</parameter>\n</tool_call>"
SLIP = '<tool_call>\n{"function": "Bash": "arguments": {"command": "ls"}}\n</tool_call>'

SHAPES = {
    "well_formed": (WELL_FORMED, [("Read", {"file_path": "/a.py", "limit": 50})]),
    "no_function_close": (NO_FUNCTION_CLOSE, [("Read", {"file_path": "/b.py"})]),
    "key_value_slip": (SLIP, [("Bash", {"command": "ls"})]),
    "parallel": (
        WELL_FORMED + "\n" + NO_FUNCTION_CLOSE + "\n" + SLIP,
        [
            ("Read", {"file_path": "/a.py", "limit": 50}),
            ("Read", {"file_path": "/b.py"}),
            ("Bash", {"command": "ls"}),
        ],
    ),
}


@pytest.fixture
def engine():
    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    from yunshu_gateway.engine import set_engine

    eng = BatchedEngine()
    eng._model = object()
    eng._loaded = True
    eng.model_name = "claude-test"
    eng._running = True
    eng._yunshu_tool_formats = (_upstream("qwen3_coder"), INJECTED_JSON, JSON_MESSAGE)
    set_engine(eng)
    yield eng
    set_engine(None)
    os.environ.pop("YUNSHU_AUTH_DISABLED", None)


def _sdk() -> anthropic.Anthropic:
    from yunshu_gateway.main import create_app

    http = TestClient(create_app(), raise_server_exceptions=False)
    return anthropic.Anthropic(
        base_url="http://testserver", api_key="k", http_client=http
    )


def _tokens(text: str, n: int):
    """Fixture token stream: ``text`` cut every ``n`` characters (n=1 splits every marker)."""
    pieces = [text[i : i + n] for i in range(0, len(text), n)]

    async def gen(*_a, **_k):
        done = ""
        for i, p in enumerate(pieces):
            done += p
            yield GenerationOutput(
                text=done,
                new_text=p,
                prompt_tokens=5,
                completion_tokens=i + 1,
                finished=False,
            )
        yield GenerationOutput(
            text=done,
            new_text="",
            prompt_tokens=5,
            completion_tokens=len(pieces),
            finished=True,
            finish_reason="stop",
        )

    return gen


async def _whole(text: str):
    async def fn(*_a, **_k):
        return GenerationOutput(
            text=text,
            new_text=text,
            prompt_tokens=5,
            completion_tokens=9,
            finished=True,
            finish_reason="stop",
        )

    return fn


def _check(message, expected):
    tool_blocks = [b for b in message.content if b.type == "tool_use"]
    assert [(b.name, b.input) for b in tool_blocks] == expected
    assert all(b.id.startswith("toolu_") for b in tool_blocks)
    assert len({b.id for b in tool_blocks}) == len(tool_blocks)
    for b in message.content:
        if b.type == "text":
            assert "tool_call" not in b.text and "<function" not in b.text, b.text
    assert message.stop_reason == "tool_use"


@pytest.mark.parametrize("n", [1, 4, 10_000])
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_streaming_shapes_become_tool_use(engine, shape, n):
    text, expected = SHAPES[shape]
    with patch.object(engine, "stream_chat", _tokens("Let me look.\n" + text, n)):
        with _sdk().messages.stream(
            model="claude-test",
            max_tokens=256,
            tools=TOOLS,
            messages=[{"role": "user", "content": "go"}],
        ) as stream:
            final = stream.get_final_message()
    _check(final, expected)
    assert (
        final.content[0].type == "text"
        and final.content[0].text.strip() == "Let me look."
    )


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_non_streaming_shapes_become_tool_use(engine, shape):
    text, expected = SHAPES[shape]

    async def chat(*_a, **_k):
        return GenerationOutput(
            text="Let me look.\n" + text,
            new_text="Let me look.\n" + text,
            prompt_tokens=5,
            completion_tokens=9,
            finished=True,
            finish_reason="stop",
        )

    with patch.object(engine, "chat", chat), patch.object(engine, "generate", chat):
        msg = _sdk().messages.create(
            model="claude-test",
            max_tokens=256,
            tools=TOOLS,
            messages=[{"role": "user", "content": "go"}],
        )
    _check(msg, expected)


def test_unreadable_call_never_leaks_streaming(engine):
    text = 'Sure.\n<tool_call>\n{"broken": [1,\n</tool_call>\nDone.'
    with patch.object(engine, "stream_chat", _tokens(text, 3)):
        with _sdk().messages.stream(
            model="claude-test",
            max_tokens=64,
            tools=TOOLS,
            messages=[{"role": "user", "content": "go"}],
        ) as stream:
            final = stream.get_final_message()
    joined = "".join(b.text for b in final.content if b.type == "text")
    assert "tool_call" not in joined and "broken" not in joined
    assert final.stop_reason == "end_turn"


def _raw_events(engine, text, n=3, **extra):
    """SSE events as (event name, data) from a raw streaming request."""
    from yunshu_gateway.main import create_app

    client = TestClient(create_app(), raise_server_exceptions=False)
    with patch.object(engine, "stream_chat", _tokens(text, n)):
        r = client.post(
            "/v1/messages",
            json={
                "model": "claude-test",
                "max_tokens": 256,
                "stream": True,
                "tools": TOOLS,
                "messages": [{"role": "user", "content": "go"}],
                **extra,
            },
        )
    assert r.status_code == 200, r.text
    events = []
    for block in r.text.split("\n\n"):
        lines = [ln for ln in block.splitlines() if not ln.startswith(":")]
        if len(lines) >= 2 and lines[0].startswith("event: "):
            events.append((lines[0][7:], json.loads(lines[1][6:])))
    return events


def test_block_ordering_and_input_json_delta(engine):
    events = _raw_events(engine, "Ok.\n" + WELL_FORMED + "\n" + SLIP)
    names = [e for e, _ in events]
    assert names[0] == "message_start" and names[-1] == "message_stop"
    assert names[-2] == "message_delta"
    open_blocks: dict[int, str] = {}
    next_index = 0
    tool_json: dict[int, str] = {}
    for name, data in events:
        if name == "content_block_start":
            assert data["index"] == next_index  # dense, in order
            assert not open_blocks  # one block open at a time
            open_blocks[data["index"]] = data["content_block"]["type"]
            if data["content_block"]["type"] == "tool_use":
                assert data["content_block"]["input"] == {}
                tool_json[data["index"]] = ""
        elif name == "content_block_delta":
            kind = open_blocks[data["index"]]
            if kind == "tool_use":
                assert data["delta"]["type"] == "input_json_delta"
                tool_json[data["index"]] += data["delta"]["partial_json"]
            else:
                assert data["delta"]["type"] in ("text_delta", "thinking_delta")
        elif name == "content_block_stop":
            del open_blocks[data["index"]]
            next_index += 1
    assert not open_blocks
    assert [json.loads(v) for v in tool_json.values()] == [
        {"file_path": "/a.py", "limit": 50},
        {"command": "ls"},
    ]
    delta = next(d for n, d in events if n == "message_delta")
    assert delta["delta"]["stop_reason"] == "tool_use"


def test_disable_parallel_tool_use_keeps_one_call(engine):
    events = _raw_events(
        engine,
        WELL_FORMED + "\n" + SLIP,
        tool_choice={"type": "auto", "disable_parallel_tool_use": True},
    )
    starts = [d for n, d in events if n == "content_block_start"]
    assert [
        s["content_block"]["name"]
        for s in starts
        if s["content_block"]["type"] == "tool_use"
    ] == ["Read"]
    assert not any(
        "tool_call" in d.get("delta", {}).get("text", "")
        for n, d in events
        if n == "content_block_delta"
    )


def test_tool_result_is_error_and_image_reach_the_model(engine, monkeypatch):
    from yunshu_gateway import model_guards as router

    monkeypatch.setattr(router, "reject_images_for_text_model", lambda *_a, **_k: None)
    seen = {}

    async def chat(*_a, **kwargs):
        seen["messages"] = kwargs.get("messages") or _a[0]
        return GenerationOutput(
            text="ok",
            new_text="ok",
            prompt_tokens=3,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
        )

    png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    with patch.object(engine, "chat", chat), patch.object(engine, "generate", chat):
        _sdk().messages.create(
            model="claude-test",
            max_tokens=32,
            tools=TOOLS,
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "Read",
                            "input": {"file_path": "/x"},
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_2",
                            "name": "Read",
                            "input": {"file_path": "/y"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "is_error": True,
                            "content": "ENOENT: no such file",
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_2",
                            "content": [
                                {"type": "text", "text": "screenshot"},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": png,
                                    },
                                },
                            ],
                        },
                    ],
                },
            ],
        )
    blob = json.dumps(seen["messages"], default=str)
    assert "ENOENT: no such file" in blob
    assert "screenshot" in blob
    assert "toolu_1" in blob or "tool_call_id" in blob


def test_cache_control_is_accepted_everywhere_claude_code_sends_it(engine):
    async def chat(*_a, **_k):
        return GenerationOutput(
            text="ok",
            new_text="ok",
            prompt_tokens=3,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
        )

    cc = {"type": "ephemeral"}
    tools = [dict(t) for t in TOOLS]
    tools[-1]["cache_control"] = cc
    with patch.object(engine, "chat", chat), patch.object(engine, "generate", chat):
        msg = _sdk().messages.create(
            model="claude-test",
            max_tokens=32,
            system=[
                {"type": "text", "text": "You are Claude Code.", "cache_control": cc}
            ],
            tools=tools,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hi", "cache_control": cc}],
                }
            ],
        )
    assert msg.content[0].text == "ok"


def test_engine_error_after_stream_start_is_an_error_event_not_an_empty_message(engine):
    """A template / engine failure used to end the stream as a normal empty `end_turn` message."""
    from yunshu_gateway.main import create_app

    async def failing(*_a, **_k):
        yield GenerationOutput(
            text="",
            new_text="",
            finished=True,
            finish_reason="error",
            error="System message must be at the beginning.",
        )

    client = TestClient(create_app(), raise_server_exceptions=False)
    with patch.object(engine, "stream_chat", failing):
        r = client.post(
            "/v1/messages",
            json={
                "model": "claude-test",
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": "go"}],
            },
        )
    assert "event: error" in r.text
    assert "System message must be at the beginning" in r.text


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
def test_engine_error_reaches_openai_streams(engine, path):
    from yunshu_gateway.main import create_app

    async def failing(*_a, **_k):
        yield GenerationOutput(
            text="",
            new_text="",
            finished=True,
            finish_reason="error",
            error="template exploded",
        )

    body = {"model": "m", "stream": True, "max_tokens": 16}
    body["input" if "responses" in path else "messages"] = (
        "go" if "responses" in path else [{"role": "user", "content": "go"}]
    )
    client = TestClient(create_app(), raise_server_exceptions=False)
    with patch.object(engine, "stream_chat", failing):
        r = client.post(path, json=body)
    assert "template exploded" in r.text
