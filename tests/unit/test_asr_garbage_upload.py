"""A transcription upload that is not audio is a 400, not a 500 (real Qwen3-ASR server, route check)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import yunshu_gateway.engine as eng_mod
from yunshu_engine.audio_engine import ASREngine
from yunshu_gateway.main import create_app
from yunshu_gateway.routers import audio as A  # noqa: N812


class Boom(ASREngine):
    def __init__(self):
        pass

    model_name = "asr"
    is_loaded = True

    async def transcribe(self, **kw):
        raise RuntimeError("ffmpeg could not decode")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    for mod in (eng_mod, A):
        monkeypatch.setattr(mod, "get_model_manager", lambda: None, raising=False)
    eng = Boom()
    monkeypatch.setattr(A, "_select_audio_engine", lambda m, model, t: eng)
    return TestClient(create_app())


def post(client, data):
    return client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr"},
        files={"file": ("a.wav", data, "audio/wav")},
    )


def test_not_audio_is_400(client):
    r = post(client, b"this is not audio at all")
    assert r.status_code == 400 and "not audio" in r.json()["error"]["message"]


def test_real_audio_failure_stays_500(client):
    assert post(client, b"RIFF....WAVEfmt ").status_code == 500
