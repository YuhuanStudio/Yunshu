"""CPU coverage for the audio gate's bounded header read and file lifetime."""

import importlib.util
import io
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "data, expected", [(b"RIFFpayload", True), (b"NOPE", False), (b"", False)]
)
def test_header_read_closes_stream(monkeypatch, data, expected):
    path = Path(__file__).parents[2] / "scripts/verify/verify_tts_asr_roundtrip.py"
    spec = importlib.util.spec_from_file_location("audio_header_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stream = io.BytesIO(data)
    calls = []

    def open_stream(path, mode):
        calls.append((path, mode))
        return stream

    monkeypatch.setattr("builtins.open", open_stream)
    assert module._has_riff_header("fixture.wav") is expected
    assert calls == [("fixture.wav", "rb")]
    assert stream.closed
