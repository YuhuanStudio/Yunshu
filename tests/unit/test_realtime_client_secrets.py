"""Realtime client secrets / beta sessions through the official openai SDK; the secret opens the socket."""

from __future__ import annotations

import openai
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from yunshu_gateway import realtime_secrets as rs
from yunshu_gateway.routers import realtime as realtime_mod
from yunshu_gateway.routers import realtime_secrets as secrets_router


@pytest.fixture
def world(monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    rs.reset()
    app = FastAPI()
    app.include_router(secrets_router.router, prefix="/v1")
    app.include_router(realtime_mod.router)
    tc = TestClient(app)
    sdk = openai.OpenAI(base_url="http://testserver/v1", api_key="x", http_client=tc)
    yield sdk, tc
    rs.reset()


def test_client_secret_realtime(world):
    sdk, _ = world
    r = sdk.realtime.client_secrets.create(
        expires_after={"anchor": "created_at", "seconds": 120},
        session={
            "type": "realtime",
            "model": "my-model",
            "instructions": "be brief",
            "output_modalities": ["text"],
            "audio": {"output": {"voice": "echo"}},
        },
    )
    assert r.value.startswith("ek_")
    assert 100 < r.expires_at - __import__("time").time() <= 121
    assert r.session.type == "realtime"
    assert r.session.instructions == "be brief"
    assert r.session.model == "my-model"
    assert r.session.audio.output.voice == "echo"
    assert r.session.id.startswith("sess_")


def test_client_secret_transcription(world):
    sdk, _ = world
    r = sdk.realtime.client_secrets.create(
        session={
            "type": "transcription",
            "audio": {
                "input": {"transcription": {"model": "whisper-1", "language": "en"}}
            },
        }
    )
    assert r.session.type == "transcription"
    assert r.session.audio.input.transcription.language == "en"


def test_client_secret_defaults_and_bad_ttl(world):
    sdk, _ = world
    r = sdk.realtime.client_secrets.create()
    assert r.session.type == "realtime" and r.value
    for bad in (5, 7201):
        with pytest.raises(openai.BadRequestError):
            sdk.realtime.client_secrets.create(
                expires_after={"anchor": "created_at", "seconds": bad}
            )


def test_beta_sessions(world):
    sdk, _ = world
    s = sdk.beta.realtime.sessions.create(
        model="my-model", instructions="hi", modalities=["text"], voice="sage"
    )
    assert s.client_secret.value.startswith("ek_") and s.instructions == "hi"
    assert s.voice == "sage" and s.modalities == ["text"]
    t = sdk.beta.realtime.transcription_sessions.create(
        input_audio_transcription={"model": "whisper-1"}
    )
    assert t.client_secret.value.startswith("ek_")
    assert t.input_audio_transcription.model == "whisper-1"


def test_secret_authenticates_the_socket_and_applies_its_session(world, monkeypatch):
    sdk, tc = world
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "static-token")
    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "false")
    sdk = openai.OpenAI(
        base_url="http://testserver/v1", api_key="static-token", http_client=tc
    )
    r = sdk.realtime.client_secrets.create(
        session={
            "type": "realtime",
            "instructions": "from the secret",
            "audio": {"output": {"voice": "echo"}},
        }
    )
    with pytest.raises(WebSocketDisconnect):
        with tc.websocket_connect(
            "/v1/realtime", headers={"authorization": "Bearer nope"}
        ):
            pass
    with tc.websocket_connect(
        "/v1/realtime", headers={"authorization": f"Bearer {r.value}"}
    ) as ws:
        ev = ws.receive_json()
        assert ev["type"] == "session.created"
        assert ev["session"]["instructions"] == "from the secret"
        assert ev["session"]["id"] == r.session.id
        assert ev["session"]["audio"]["output"]["voice"] == "echo"
    # a secret can open another session until it expires
    with tc.websocket_connect(
        "/v1/realtime", headers={"authorization": f"Bearer {r.value}"}
    ) as ws:
        assert ws.receive_json()["type"] == "session.created"


def test_expired_secret_is_refused(world, monkeypatch):
    sdk, tc = world
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "static-token")
    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "false")
    sec = rs.mint({}, "realtime", 10, now=1.0)  # expired long ago
    assert rs.lookup(sec.value) is None
    with pytest.raises(WebSocketDisconnect):
        with tc.websocket_connect(
            "/v1/realtime", headers={"authorization": f"Bearer {sec.value}"}
        ):
            pass
