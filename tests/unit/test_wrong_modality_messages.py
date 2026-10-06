"""A chat model asked for speech, transcription or images answers what it is and what to start
instead. It said "ASR model 'X' not found" for the very model /v1/models lists (and logged a
traceback from `None.get_engine`) -- found by the real-server route checks on a chat checkpoint."""

from __future__ import annotations

import io
import types
import wave

import pytest
from fastapi.testclient import TestClient

import yunshu_gateway.engine as eng_mod
from yunshu_gateway.main import create_app
from yunshu_gateway.routers import audio as A  # noqa: N812
from yunshu_gateway.routers import images as I  # noqa: E741,N812


def _wav():
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    return b.getvalue()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    chat = types.SimpleNamespace(
        model_name="/models/Qwen3.5-0.8B-MLX-bf16", is_loaded=True
    )
    for mod in (eng_mod, A, I):
        monkeypatch.setattr(mod, "get_model_manager", lambda: None, raising=False)
    monkeypatch.setattr(eng_mod, "get_engine", lambda: chat)
    monkeypatch.setattr(A, "get_engine", lambda: chat, raising=False)
    monkeypatch.setattr(I, "get_engine", lambda: chat, raising=False)
    return TestClient(create_app())


def _msg(r):
    assert r.status_code == 404, r.text
    return r.json()["error"]["message"]


def test_transcription_names_the_served_model(client):
    r = client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1"},
        files={"file": ("a.wav", _wav(), "audio/wav")},
    )
    m = _msg(r)
    assert "Qwen3.5-0.8B-MLX-bf16" in m and "cannot transcribe audio" in m
    assert "not found" not in m and "yunshu serve -m" in m


def test_translation_speech_and_stream(client):
    m = _msg(
        client.post(
            "/v1/audio/translations",
            data={"model": "whisper-1"},
            files={"file": ("a.wav", _wav(), "audio/wav")},
        )
    )
    assert "cannot transcribe audio" in m
    for path in ("/v1/audio/speech", "/v1/audio/speech/stream"):
        m = _msg(
            client.post(path, json={"model": "tts-1", "input": "hi", "voice": "alloy"})
        )
        assert "cannot synthesize speech" in m and "Qwen3.5-0.8B-MLX-bf16" in m, path


def test_image_generation(client):
    for path, body in (
        ("/v1/images/generations", {"model": "dall-e-3", "prompt": "a cat"}),
        ("/v1/images/generations/stream", {"model": "dall-e-3", "prompt": "a cat"}),
    ):
        m = _msg(client.post(path, json=body))
        assert "cannot generate or edit images" in m, path


def test_multi_model_mode_keeps_not_found(monkeypatch):
    from yunshu_gateway.model_guards import wrong_modality_detail

    monkeypatch.setattr(eng_mod, "get_model_manager", lambda: object())
    assert wrong_modality_detail("asr", "x") is None
    monkeypatch.setattr(eng_mod, "get_model_manager", lambda: None)
    monkeypatch.setattr(eng_mod, "get_engine", lambda: None)
    assert wrong_modality_detail("asr", "x") is None
    assert wrong_modality_detail("nope", "x") is None
