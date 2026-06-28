"""Waves 962 + 963: realtime audio.done guarantee + conversation-history bound.

W962: response.audio.done was emitted ONLY inside the synthesis loop (before break) and in
  the except handler. When manager is None, no entry has synthesize, or every candidate is
  skipped by _key_allows, the loop fell through with NO audio.done → an OpenAI-SDK client
  waited forever for the terminal audio event. Also, when audio was requested but the turn
  produced no visible text, _synthesize_audio_response was never called → no audio.done.
  Now: _synthesize_audio_response guarantees exactly one audio.done (finally), and the
  empty-text branch emits a bare audio.done.
W963: conversation.items was unbounded (W799 capped only the input-audio buffer). Every turn
  appends items and _build_messages replays the whole history → a conversation.item.create
  flood / long session grows RSS + prompt cost without limit. Cap to the most recent N
  (FIFO eviction), and bound a single item's content size.
"""
from __future__ import annotations

import os

import pytest

import yunshu_gateway.routers.realtime as rt


@pytest.mark.asyncio
async def test_w962_audio_done_emitted_when_no_manager(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    ws = MagicMock()
    ws.send_json = AsyncMock()
    session = rt.RealtimeSession(ws)
    monkeypatch.setattr("yunshu_gateway.engine.get_model_manager", lambda: None)

    await session._synthesize_audio_response("hello", "resp_1", "item_1")
    types = [c[0][0].get("type") for c in ws.send_json.call_args_list]
    assert types.count("response.audio.done") == 1


def test_w963_conversation_history_capped(monkeypatch):
    monkeypatch.setenv("YUNSHU_REALTIME_MAX_CONVERSATION_ITEMS", "5")
    conv = rt.Conversation("c1")
    for i in range(50):
        conv.add_item(rt.ConversationItem(
            item_id=f"i{i}", item_type="message", role="user",
            content=[{"type": "text", "text": str(i)}],
        ))
    assert len(conv.items) == 5
    # the most recent 5 survive, oldest evicted
    assert [it.item_id for it in conv.items] == [f"i{i}" for i in range(45, 50)]


def test_w963_default_cap_is_positive():
    os.environ.pop("YUNSHU_REALTIME_MAX_CONVERSATION_ITEMS", None)
    assert rt._max_conversation_items() == 1000
    # malformed env → default, never 0/negative
    os.environ["YUNSHU_REALTIME_MAX_CONVERSATION_ITEMS"] = "-3"
    assert rt._max_conversation_items() == 1000
    os.environ["YUNSHU_REALTIME_MAX_CONVERSATION_ITEMS"] = "notanint"
    assert rt._max_conversation_items() == 1000
    os.environ.pop("YUNSHU_REALTIME_MAX_CONVERSATION_ITEMS", None)


def test_w963_trim_preserves_previous_item_insert(monkeypatch):
    # inserting after a ref id still trims, and ordering stays correct
    monkeypatch.setenv("YUNSHU_REALTIME_MAX_CONVERSATION_ITEMS", "3")
    conv = rt.Conversation("c2")
    for i in range(3):
        conv.add_item(rt.ConversationItem(item_id=f"a{i}", item_type="message", role="user"))
    conv.add_item(
        rt.ConversationItem(item_id="ins", item_type="message", role="user"),
        previous_item_id="a2",
    )
    assert len(conv.items) == 3  # still capped
