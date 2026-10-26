"""ASR on a clip with no samples returns an empty transcription (mlx-audio 0.6 raises
ValueError on zero samples, which surfaced as a 500)."""

from __future__ import annotations

import asyncio
import struct
import wave
from unittest.mock import MagicMock

import pytest

from yunshu_engine import audio_engine
from yunshu_engine.audio_engine import ASREngine


def _engine(side_effect):
    engine = ASREngine("whisper-small")
    engine._model = MagicMock()
    engine._model.generate = MagicMock(side_effect=side_effect)
    return engine


def _empty_wav(path):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"")


def test_empty_wav_is_an_empty_transcription(tmp_path):
    p = tmp_path / "empty.wav"
    _empty_wav(p)
    engine = _engine(ValueError("audio has zero samples"))
    out = asyncio.run(engine.transcribe(str(p)))
    assert out["text"] == "" and out["segments"] == [] and out["duration"] == 0.0
    engine._model.generate.assert_not_called()


def test_zero_sample_decode_error_on_another_format_is_empty(tmp_path, monkeypatch):
    p = tmp_path / "empty.flac"
    p.write_bytes(b"fLaC" + struct.pack("<I", 0))  # not a WAV: no header shortcut
    monkeypatch.setattr(audio_engine, "_decoded_audio_duration_s", lambda path: 0.0)
    engine = _engine(ValueError("audio has zero samples"))
    out = asyncio.run(engine.transcribe(str(p), language="en"))
    assert out["text"] == "" and out["language"] == "en"


def test_other_value_errors_still_surface(tmp_path, monkeypatch):
    p = tmp_path / "x.flac"
    p.write_bytes(b"fLaC" + struct.pack("<I", 0))
    monkeypatch.setattr(audio_engine, "_decoded_audio_duration_s", lambda path: 1.5)
    engine = _engine(ValueError("bad option"))
    with pytest.raises(ValueError, match="bad option"):
        asyncio.run(engine.transcribe(str(p)))
