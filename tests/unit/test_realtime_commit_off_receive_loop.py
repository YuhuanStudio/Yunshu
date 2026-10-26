"""input_audio_buffer.commit transcribes off the receive loop.

The ASR was awaited inside the event handler, so a long transcription stopped the session
from reading anything else (audio, response.cancel) and, over WebRTC, overflowed the
256-slot data-channel queue until the call was torn down. The commit now runs in an ordered
lane: later events that need its user item (response.create) wait behind it, everything
else keeps flowing, and audio appended meanwhile belongs to the next turn.
"""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import pytest

from .test_realtime_audio import _make_session


class _SlowASR:
    def __init__(self):
        self.release = asyncio.Event()
        self.started = asyncio.Event()

    async def transcribe(self, path):
        self.started.set()
        await self.release.wait()
        return {"text": "hello world"}


def _b64(n):
    return base64.b64encode(b"\x01\x00" * n).decode()


def _install(monkeypatch, asr):
    entry = SimpleNamespace(is_loaded=True, engine=asr)
    manager = SimpleNamespace(list_entries=lambda: [entry])
    monkeypatch.setattr("yunshu_gateway.engine.get_model_manager", lambda: manager)


@pytest.mark.asyncio
async def test_commit_does_not_block_the_receive_loop(monkeypatch):
    asr = _SlowASR()
    _install(monkeypatch, asr)
    s = _make_session()
    s.session.turn_detection = None
    await s._handle_event({"type": "input_audio_buffer.append", "audio": _b64(100)})
    # returns while the transcription is still running
    await asyncio.wait_for(s._handle_event({"type": "input_audio_buffer.commit"}), 1)
    await asyncio.wait_for(asr.started.wait(), 1)
    # the loop keeps handling events during the ASR; new audio is the next turn's
    await s._handle_event({"type": "input_audio_buffer.append", "audio": _b64(50)})
    assert len(s._audio_buffer) == 100
    # an event that needs the user item queues behind the commit instead of racing it
    await asyncio.wait_for(s._handle_event({"type": "response.create"}), 1)
    types = [e["type"] for e in s.ws.sent]
    assert "response.created" not in types
    assert "conversation.item.created" not in types

    asr.release.set()
    await asyncio.wait_for(s._lane, 2)
    types = [e["type"] for e in s.ws.sent]
    assert types.index("conversation.item.created") < types.index(
        "conversation.item.input_audio_transcription.completed"
    )
    assert [it.role for it in s.conversation.items][:1] == ["user"]
