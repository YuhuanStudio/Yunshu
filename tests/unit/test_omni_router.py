"""Omni router — lightweight coverage (no model needed).

The model-backed streaming path is an integration test (needs a resident
Qwen3-Omni ~22GB). These cover the routing/config surface so CI stays fast.
"""
from __future__ import annotations

import os

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


def test_pcm16_b64_roundtrip_is_real_audio():
    import base64

    import numpy as np

    wav = np.linspace(-0.5, 0.5, 24000, dtype=np.float32)  # 1s sweep
    b64 = omni._pcm16_b64(wav)
    back = np.frombuffer(base64.b64decode(b64), "<i2").astype(np.float32) / 32767.0
    # int16 quantization loses precision but the signal is preserved
    assert back.shape == wav.shape
    assert np.abs(back.mean() - wav.mean()) < 0.01
