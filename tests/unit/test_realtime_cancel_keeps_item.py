"""response.cancel / barge-in keeps the interrupted reply as an incomplete assistant item.

The item used to be added to the conversation only on the happy path, so the
conversation.item.truncate a client sends after a barge-in answered "Item not found" and
the next turn's prompt forgot the reply the user had just heard part of.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from yunshu_gateway.routers.realtime import ConversationItem

from .test_realtime_audio import _make_session


class _SlowLLM:
    """Streams two tokens, then waits (the consumer is cancelled while it waits)."""

    async def generate_stream(self, **kw):
        yield SimpleNamespace(
            token_text="Hello there, ", finish_reason=None, error=None
        )
        yield SimpleNamespace(token_text="friend", finish_reason=None, error=None)
        await asyncio.sleep(30)
        yield SimpleNamespace(token_text="!", finish_reason="stop", error=None)


def _types(session):
    return [e["type"] for e in session.ws.sent]


async def _interrupt(modalities):
    s = _make_session()
    s._initial_engine = _SlowLLM()
    s._initial_model = s.session.model
    s.session.modalities = modalities
    s.conversation.add_item(
        ConversationItem(
            "item_user",
            "message",
            role="user",
            content=[{"type": "input_text", "text": "hi"}],
        )
    )
    await s._handle_response_create({})
    for _ in range(200):  # let the stream reach the slow wait
        await asyncio.sleep(0.01)
        if sum(".delta" in t for t in _types(s)) >= 2:
            break
    item_id = s._active_response._item_id
    await s._handle_response_cancel({})
    return s, item_id


@pytest.mark.asyncio
async def test_cancel_keeps_partial_text_item_and_truncate_finds_it():
    s, item_id = await _interrupt(["text"])
    item = next(it for it in s.conversation.items if it.item_id == item_id)
    assert item.role == "assistant" and item.status == "incomplete"
    assert item.content[0]["text"] == "Hello there, friend"
    assert "conversation.item.created" in _types(s)
    # the closing output_item.done carries the text and the incomplete status
    done = [e for e in s.ws.sent if e["type"] == "response.output_item.done"][-1]
    assert done["item"]["status"] == "incomplete"
    # the model remembers the interrupted reply
    assert any("Hello there, friend" in str(m) for m in s._build_messages())
    s.ws.sent.clear()
    await s._handle_conversation_item_truncate({"item_id": item_id, "audio_end_ms": 10})
    assert "error" not in _types(s)


@pytest.mark.asyncio
async def test_cancel_audio_response_keeps_audio_part_with_transcript():
    s, item_id = await _interrupt(["audio", "text"])
    item = next(it for it in s.conversation.items if it.item_id == item_id)
    part = item.content[0]
    assert part["type"] == "audio" and part["transcript"] == "Hello there, friend"
    assert part["duration_ms"] > 0
    s.ws.sent.clear()
    await s._handle_conversation_item_truncate({"item_id": item_id, "audio_end_ms": 10})
    assert "error" not in _types(s)
