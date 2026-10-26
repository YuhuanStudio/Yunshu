"""speech_started / speech_stopped / committed carry the id of the user item the utterance
becomes (and committed its previous_item_id), and the item created after the ASR uses that
same id, so a client can correlate the VAD events with the transcription (OpenAI Realtime)."""

from __future__ import annotations

import asyncio
import base64
import struct
from types import SimpleNamespace

import pytest

from yunshu_gateway.routers.realtime import ConversationItem

from .test_realtime_audio import _make_session

SPEECH = struct.pack("<2400h", *([10000] * 2400))  # 100 ms, loud
SILENCE = b"\x00" * 4800


async def _speak(s):
    s.session.turn_detection = {"type": "server_vad", "silence_duration_ms": 300}
    for chunk in (SPEECH, SPEECH, SILENCE, SILENCE, SILENCE, SILENCE):
        audio = base64.b64encode(chunk).decode()
        await s._handle_event({"type": "input_audio_buffer.append", "audio": audio})
    await asyncio.wait_for(s._lane, 2)  # the auto-commit runs in the lane


def _asr(monkeypatch):
    class ASR:
        async def transcribe(self, path):
            return {"text": "hello"}

    entry = SimpleNamespace(is_loaded=True, engine=ASR(), model_id="asr")
    mgr = SimpleNamespace(list_entries=lambda: [entry])
    monkeypatch.setattr("yunshu_gateway.engine.get_model_manager", lambda: mgr)


@pytest.mark.asyncio
async def test_vad_events_commit_and_item_share_one_id(monkeypatch):
    _asr(monkeypatch)
    s = _make_session()
    s.conversation.add_item(
        ConversationItem(
            "item_prev", "message", role="assistant", content=[{"type": "text"}]
        )
    )
    await _speak(s)
    by = {e["type"]: e for e in s.ws.sent}
    started = by["input_audio_buffer.speech_started"]
    stopped = by["input_audio_buffer.speech_stopped"]
    committed = by["input_audio_buffer.committed"]
    created = next(
        e
        for e in s.ws.sent
        if e["type"] == "conversation.item.created" and e["item"]["role"] == "user"
    )
    item_id = started["item_id"]
    assert item_id.startswith("item_")
    assert stopped["item_id"] == committed["item_id"] == item_id
    assert committed["previous_item_id"] == "item_prev"
    assert created["item"]["id"] == item_id


@pytest.mark.asyncio
async def test_manual_commit_event_and_item_share_one_id(monkeypatch):
    _asr(monkeypatch)
    s = _make_session()
    s.session.turn_detection = None
    s._audio_buffer.extend(b"\x01\x00" * 4800)
    await s._handle_input_audio_buffer_commit({})
    committed = next(
        e for e in s.ws.sent if e["type"] == "input_audio_buffer.committed"
    )
    created = next(e for e in s.ws.sent if e["type"] == "conversation.item.created")
    assert committed["item_id"] == created["item"]["id"]
    assert committed["previous_item_id"] is None
