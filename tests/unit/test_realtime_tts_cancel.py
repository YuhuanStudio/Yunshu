"""R28: the realtime cascade hands the response's cancel event to the TTS stream, so a
barge-in reaches the engine thread without waiting for the consumer."""

from __future__ import annotations

import asyncio

import pytest

from .test_realtime_audio import _make_session


class _TTS:
    def __init__(self):
        self.cancel_event = "unset"

    async def synthesize(self, text, voice=None):  # presence marks it a TTS engine
        raise AssertionError

    async def synthesize_stream(self, text, voice=None, cancel_event=None):
        self.cancel_event = cancel_event
        yield {"audio": b"\x00\x01" * 64, "is_final": False}
        yield {"audio": b"", "is_final": True}


class _NoCancelTTS(_TTS):
    async def synthesize_stream(self, text, voice=None):  # an engine without the knob
        yield {"audio": b"\x00\x01" * 64, "is_final": False}
        yield {"audio": b"", "is_final": True}


def _manager(engine):
    class _Entry:
        is_loaded = True

    entry = _Entry()
    entry.engine = engine

    class _Manager:
        def list_entries(self):
            return [entry]

    return _Manager()


@pytest.mark.asyncio
async def test_synthesis_gets_the_response_cancel_event(monkeypatch):
    import yunshu_gateway.engine as gw_engine

    tts = _TTS()
    monkeypatch.setattr(gw_engine, "get_model_manager", lambda: _manager(tts))
    session = _make_session()
    session._cancel_event = asyncio.Event()
    await session._synthesize_audio_response("hello", "resp", "item")
    assert tts.cancel_event is session._cancel_event
    types = [e.get("type") for e in session.ws.sent]
    assert "response.audio.delta" in types and types[-1] == "response.audio.done"


@pytest.mark.asyncio
async def test_engine_without_the_parameter_still_streams(monkeypatch):
    import yunshu_gateway.engine as gw_engine

    monkeypatch.setattr(
        gw_engine, "get_model_manager", lambda: _manager(_NoCancelTTS())
    )
    session = _make_session()
    session._cancel_event = asyncio.Event()
    await session._synthesize_audio_response("hello", "resp", "item")
    types = [e.get("type") for e in session.ws.sent]
    assert "response.audio.delta" in types
