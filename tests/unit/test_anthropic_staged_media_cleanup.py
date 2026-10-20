"""Images staged on disk for a /v1/messages or count_tokens request are unlinked on every
path that does not hand them to a running generation (early error returns, a conversion
that fails after staging an earlier image)."""

import os

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

_PNG = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
}
_BLOCKED = {
    "type": "image",
    "source": {"type": "url", "url": "http://127.0.0.1/blocked.png"},
}


@pytest.fixture
def client_and_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "true")
    monkeypatch.setenv("YUNSHU_MEDIA_DIR", str(tmp_path))
    from yunshu_engine.batched_engine import BatchedEngine
    from yunshu_gateway.engine import set_engine
    from yunshu_gateway.main import create_app
    from yunshu_gateway.routers import anthropic

    engine = BatchedEngine()
    engine._model = object()
    engine._loaded = True
    engine._running = True
    set_engine(engine)

    async def _unknown(model_id):
        raise HTTPException(status_code=404, detail="model not found")

    monkeypatch.setattr(anthropic, "_resolve_engine", _unknown)
    yield TestClient(create_app(), raise_server_exceptions=False), tmp_path
    set_engine(None)


def _body(*images):
    return {
        "model": "nope",
        "max_tokens": 10,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}, *images]}
        ],
    }


def test_unknown_model_does_not_leak_staged_image(client_and_dir):
    client, media = client_and_dir
    r = client.post("/v1/messages", json=_body(_PNG))
    assert r.status_code == 404, r.text
    assert os.listdir(media) == []


def test_partial_conversion_failure_unlinks_first_image(client_and_dir):
    client, media = client_and_dir
    r = client.post("/v1/messages", json=_body(_PNG, _BLOCKED))
    assert r.status_code >= 400, r.text
    assert os.listdir(media) == []


def test_count_tokens_failing_image_does_not_leak(client_and_dir):
    client, media = client_and_dir
    r = client.post("/v1/messages/count_tokens", json=_body(_PNG, _BLOCKED))
    assert r.status_code >= 400, r.text
    assert os.listdir(media) == []
