"""Exactly one terminal audio.done / response.done per cancelled response.

* Cancelled during TTS, the synthesis task's own cleanup sent response.audio.done and the
  cancel handler sent a second one.
* response.cancel right after response.create (the task had not started) saw the previous
  response's stale "terminal event sent" flag and returned without any response.done.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from yunshu_gateway.routers.realtime import ConversationItem

from .test_realtime_audio import _make_session


class _LLM:
    def __init__(self, hang=False):
        self.hang = hang

    async def generate_stream(self, **kw):
        yield SimpleNamespace(token_text="Hello there", finish_reason=None, error=None)
        if self.hang:
            await asyncio.sleep(30)
        yield SimpleNamespace(token_text=".", finish_reason="stop", error=None)


class _StallingTTS:
    async def synthesize(self, text, voice=None):
        raise AssertionError

    async def synthesize_stream(self, text, voice=None):
        yield {"audio": b"\x00\x01" * 480, "is_final": False}
        await asyncio.sleep(30)


def _manager(engine):
    entry = SimpleNamespace(is_loaded=True, engine=engine)
    return SimpleNamespace(list_entries=lambda: [entry])


def _session(llm, modalities):
    s = _make_session()
    s._initial_engine = llm
    s._initial_model = s.session.model
    s.session.modalities = modalities
    s.conversation.add_item(
        ConversationItem(
            "item_user",
            "message",
            role="user",
            content=[{"type": "text", "text": "hi"}],
        )
    )
    return s


def _count(s, typ):
    return sum(e["type"] == typ for e in s.ws.sent)


@pytest.mark.asyncio
async def test_cancel_during_tts_sends_one_audio_done(monkeypatch):
    import yunshu_gateway.engine as gw_engine

    monkeypatch.setattr(
        gw_engine, "get_model_manager", lambda: _manager(_StallingTTS())
    )
    s = _session(_LLM(), ["audio", "text"])
    await s._handle_response_create({})
    for _ in range(300):  # until the first audio chunk went out
        await asyncio.sleep(0.01)
        if _count(s, "response.audio.delta"):
            break
    assert _count(s, "response.audio.delta")
    await s._handle_response_cancel({})
    assert _count(s, "response.audio.done") == 1
    assert _count(s, "response.done") == 1
    last = [e for e in s.ws.sent if e["type"] == "response.done"][-1]
    assert last["response"]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_right_after_create_still_ends_the_response():
    s = _session(_LLM(), ["text"])
    await s._handle_response_create({})
    for _ in range(300):  # first response completes: its terminal flag stays set
        await asyncio.sleep(0.01)
        if _count(s, "response.done"):
            break
    assert _count(s, "response.done") == 1
    await s._handle_response_create({})
    await s._handle_response_cancel({})  # no loop iteration in between
    dones = [e for e in s.ws.sent if e["type"] == "response.done"]
    assert len(dones) == 2
    assert dones[-1]["response"]["status"] == "cancelled"
