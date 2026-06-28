"""/v1/images/generations decoded the optional `control_image` base64
OUTSIDE the try block, so a malformed control image raised binascii.Error → the global
handler turned it into a 500 "Internal server error". Every sibling image route
(variations/edits/inpaint/controlnet/depth) correctly returns 400 for bad base64.
Now the control_image decode is guarded the same way."""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


class _FakeEntry:
    def __init__(self, engine):
        self.is_loaded = True
        self.model_id = "test"
        self.engine = engine


class _FakeManager:
    def __init__(self, engine):
        self._e = engine

    def list_entries(self):
        return [_FakeEntry(self._e)]

    async def get_engine(self, model):
        return self._e


@pytest.fixture
def _client_with_image_engine(monkeypatch):
    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    from yunshu_engine.image_engine import ImageGenEngine
    from yunshu_gateway.routers import images as images_mod
    # bare instance — passes isinstance() without loading a model; we never reach
    # generation because the malformed base64 must 400 first.
    fake_engine = ImageGenEngine.__new__(ImageGenEngine)
    monkeypatch.setattr(images_mod, "get_model_manager", lambda: _FakeManager(fake_engine))
    from yunshu_gateway.main import create_app
    yield TestClient(create_app(), raise_server_exceptions=False)
    os.environ.pop("YUNSHU_AUTH_DISABLED", None)


def test_malformed_control_image_returns_400(_client_with_image_engine):
    resp = _client_with_image_engine.post("/v1/images/generations", json={
        "model": "test",
        "prompt": "a cat",
        "size": "512x512",
        "control_image": "!!!not-valid-base64!!!",
    })
    assert resp.status_code == 400, f"got {resp.status_code}: {resp.text}"
    assert "control_image" in resp.text.lower() or "base64" in resp.text.lower()


def test_source_guards_control_image_decode():
    import inspect

    from yunshu_gateway.routers import images
    src = inspect.getsource(images.generate_image_endpoint) if hasattr(images, "generate_image_endpoint") else inspect.getsource(images)
    assert "Invalid base64 control_image data" in src
