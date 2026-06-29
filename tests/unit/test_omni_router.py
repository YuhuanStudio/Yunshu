"""Omni router — lightweight coverage (no model needed).

The model-backed streaming path is an integration test (needs a resident
Qwen3-Omni ~22GB). These cover the routing/config surface so CI stays fast.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_gateway.routers import omni


@pytest.fixture(autouse=True)
def _reset_engine():
    omni._omni_engine = None
    yield
    omni._omni_engine = None


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(omni.router)
    return app


def test_503_when_no_omni_model_configured(monkeypatch):
    monkeypatch.delenv("YUNSHU_OMNI_MODEL", raising=False)
    client = TestClient(_app())
    r = client.post("/v1/omni/speech/stream", json={"text": "hi"})
    assert r.status_code == 503
    assert "YUNSHU_OMNI_MODEL" in r.json()["detail"]


def test_request_validation_rejects_empty_text():
    client = TestClient(_app())
    r = client.post("/v1/omni/speech/stream", json={"text": ""})
    assert r.status_code == 422  # min_length=1


class _FakeOmni:
    """No-model stand-in: validates speakers, streams nothing."""

    valid_speakers = {"ethan", "chelsie", "aiden"}
    sample_rate = 24000

    def resolve_speaker(self, requested):
        if not requested:
            return "Ethan"
        return "Chelsie" if requested.lower() in ("chelsie", "alloy") else None

    async def stream(self, *a, **k):  # empty async generator
        return
        yield


def test_unknown_voice_returns_400(monkeypatch):
    monkeypatch.setattr(omni, "_get_omni_engine", lambda: _FakeOmni())
    client = TestClient(_app())
    r = client.post("/v1/omni/speech/stream", json={"text": "hi", "speaker": "bogus"})
    assert r.status_code == 400
    assert "Unknown speaker" in r.json()["detail"]
    # the error lists the valid set so the caller can correct it
    assert "chelsie" in r.json()["detail"]


def test_valid_voice_passes_validation(monkeypatch):
    monkeypatch.setattr(omni, "_get_omni_engine", lambda: _FakeOmni())
    client = TestClient(_app())
    r = client.post("/v1/omni/speech/stream", json={"text": "hi", "speaker": "chelsie"})
    assert r.status_code == 200  # validation passed → SSE stream opened


def test_no_speaker_skips_validation(monkeypatch):
    monkeypatch.setattr(omni, "_get_omni_engine", lambda: _FakeOmni())
    client = TestClient(_app())
    r = client.post("/v1/omni/speech/stream", json={"text": "hi"})
    assert r.status_code == 200  # no speaker → default, no 400


def test_preload_noop_without_model(monkeypatch):
    """No YUNSHU_OMNI_MODEL → preload must return without constructing an engine."""
    import asyncio

    monkeypatch.delenv("YUNSHU_OMNI_MODEL", raising=False)
    asyncio.run(omni.preload_and_warmup())
    assert omni._omni_engine is None  # never touched the 22GB load path


def test_preload_respects_optout(monkeypatch):
    """YUNSHU_OMNI_PRELOAD=0 → preload must skip even when a model is configured."""
    import asyncio

    monkeypatch.setenv("YUNSHU_OMNI_MODEL", "/nonexistent/model")
    monkeypatch.setenv("YUNSHU_OMNI_PRELOAD", "0")
    asyncio.run(omni.preload_and_warmup())
    assert omni._omni_engine is None  # opt-out honored before any load attempt


def test_pcm16_b64_roundtrip_is_real_audio():
    import base64

    import numpy as np

    wav = np.linspace(-0.5, 0.5, 24000, dtype=np.float32)  # 1s sweep
    b64 = omni._pcm16_b64(wav)
    back = np.frombuffer(base64.b64decode(b64), "<i2").astype(np.float32) / 32767.0
    # int16 quantization loses precision but the signal is preserved
    assert back.shape == wav.shape
    assert np.abs(back.mean() - wav.mean()) < 0.01
