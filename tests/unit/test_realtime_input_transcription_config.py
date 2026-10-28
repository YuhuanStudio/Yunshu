"""session.input_audio_transcription is stored, echoed, and drives the ASR choice.

The field was silently dropped (not a SessionConfig attribute), so session objects always
reported null and the first loaded ASR model transcribed every turn whatever the client
asked for (model / language)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from yunshu_gateway import realtime_ga as ga

from .test_realtime_gaps import _session


class _ASR:
    def __init__(self, name, calls):
        self.name, self.calls = name, calls

    async def transcribe(self, path, language=None):
        self.calls.append((self.name, language))
        return {"text": "hi"}


class _PlainASR:
    def __init__(self, calls):
        self.calls = calls

    async def transcribe(self, path):  # no language hint
        self.calls.append(("plain", None))
        return {"text": "hi"}


def _mgr(*pairs):
    entries = [
        SimpleNamespace(is_loaded=True, engine=e, model_id=name) for name, e in pairs
    ]
    return SimpleNamespace(list_entries=lambda: entries)


@pytest.mark.asyncio
async def test_session_update_stores_and_echoes_transcription_config():
    s, sent = _session("beta")
    cfg = {"model": "asr-b", "language": "de"}
    await s._handle_event(
        {"type": "session.update", "session": {"input_audio_transcription": cfg}}
    )
    assert s.session.input_audio_transcription == cfg
    assert sent[-1]["session"]["input_audio_transcription"] == cfg


def test_ga_session_update_maps_audio_input_transcription():
    out = ga._session_from_ga(
        {"audio": {"input": {"transcription": {"model": "asr-b", "language": "de"}}}}
    )
    assert out["input_audio_transcription"] == {"model": "asr-b", "language": "de"}


@pytest.mark.asyncio
async def test_commit_uses_the_requested_asr_model_and_language(monkeypatch):
    calls: list = []
    mgr = _mgr(("asr-a", _ASR("a", calls)), ("asr-b", _ASR("b", calls)))
    monkeypatch.setattr("yunshu_gateway.engine.get_model_manager", lambda: mgr)
    s, _ = _session("beta")
    s.session.input_audio_transcription = {"model": "asr-b", "language": "de"}
    s._audio_buffer.extend(b"\x01\x00" * 4800)
    await s._handle_input_audio_buffer_commit({})
    assert calls == [("b", "de")]


@pytest.mark.asyncio
async def test_without_a_config_the_first_asr_is_used_and_no_language_is_sent(
    monkeypatch,
):
    calls: list = []
    mgr = _mgr(("asr-a", _PlainASR(calls)), ("asr-b", _ASR("b", calls)))
    monkeypatch.setattr("yunshu_gateway.engine.get_model_manager", lambda: mgr)
    s, _ = _session("beta")
    s._audio_buffer.extend(b"\x01\x00" * 4800)
    await s._handle_input_audio_buffer_commit({})
    assert calls == [("plain", None)]
    # a language hint for an engine that takes none is ignored, not an error
    s.session.input_audio_transcription = {"language": "de"}
    s._audio_buffer.extend(b"\x01\x00" * 4800)
    await s._handle_input_audio_buffer_commit({})
    assert calls[-1] == ("plain", None)
