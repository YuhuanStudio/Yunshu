"""_thinking_events: frames with no thinking content pass through without a JSON parse."""

import asyncio
import json
from types import SimpleNamespace

from yunshu_gateway.routers import anthropic as anthropic_router


def _frame(event_type, data):
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def _text_stream():
    return [
        _frame("message_start", {"type": "message_start", "message": {"id": "m"}}),
        _frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        _frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "héllo ✓"},
            },
        ),
        _frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _frame("message_stop", {"type": "message_stop"}),
    ]


def _thinking_stream():
    return [
        _frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            },
        ),
        _frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "secret plan"},
            },
        ),
        _frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _frame("message_stop", {"type": "message_stop"}),
    ]


async def _chunks(frames, split):
    raw = b"".join(frames)
    if split:  # split mid-frame to exercise the buffer
        for i in range(0, len(raw), 37):
            yield raw[i : i + 37]
    else:
        yield raw


async def _collect(frames, display, split=False):
    req = SimpleNamespace(thinking={"display": display} if display else None)
    return b"".join(
        [
            c
            async for c in anthropic_router._thinking_events(
                _chunks(frames, split), req
            )
        ]
    )


class _CountingJson:
    def __init__(self, real):
        self.real = real
        self.loads_calls = 0

    def loads(self, *a, **k):
        self.loads_calls += 1
        return self.real.loads(*a, **k)

    def dumps(self, *a, **k):
        return self.real.dumps(*a, **k)


def test_text_only_stream_is_byte_identical_and_not_parsed(monkeypatch):
    counting = _CountingJson(json)
    monkeypatch.setattr(anthropic_router, "json", counting)
    frames = _text_stream()
    for display in (None, "omitted", "updates"):
        for split in (False, True):
            out = asyncio.run(_collect(frames, display, split))
            assert out == b"".join(frames)
    assert counting.loads_calls == 0


def test_thinking_stream_is_still_filtered_when_hidden():
    out = asyncio.run(_collect(_thinking_stream(), "omitted"))
    assert b"secret plan" not in out
    assert b"signature_delta" in out


def test_visible_thinking_stream_keeps_its_text_and_signature_frame():
    out = asyncio.run(_collect(_thinking_stream(), None))
    assert b"secret plan" in out
    assert out.count(b"signature_delta") == 1
