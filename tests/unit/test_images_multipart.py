"""POST /v1/images/edits and /v1/images/variations take multipart/form-data like the OpenAI API
and its SDKs send. They only accepted JSON with a base64 image, so `client.images.edit(image=file,
...)` got a 400 "Input should be a valid dictionary" (found by the real-server route checks, which
use the official SDK)."""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from yunshu_gateway.main import create_app
from yunshu_gateway.routers import images as I  # noqa: E741,N812

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class FakeImages:
    def __init__(self):
        self.calls = []

    async def generate(self, **kw):
        self.calls.append(kw)
        return [PNG]


@pytest.fixture
def env(monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    eng = FakeImages()
    monkeypatch.setattr(I, "_select_image_engine", lambda manager, model: eng)
    return TestClient(create_app()), eng


def test_edit_multipart_like_the_openai_sdk(env):
    client, eng = env
    r = client.post(
        "/v1/images/edits",
        data={"model": "m", "prompt": "make it blue", "n": "1", "size": "256x256"},
        files={"image": ("a.png", PNG, "image/png")},
    )
    assert r.status_code == 200, r.text
    assert base64.b64decode(r.json()["data"][0]["b64_json"]) == PNG
    call = eng.calls[0]
    assert call["image"] == PNG and call["prompt"] == "make it blue"
    assert (call["width"], call["height"]) == (256, 256)


def test_edit_multipart_array_field_and_mask_ignored(env):
    client, eng = env
    r = client.post(
        "/v1/images/edits",
        data={"prompt": "p", "size": "256x256"},
        files=[
            ("image[]", ("a.png", PNG, "image/png")),
            ("mask", ("m.png", b"ignored", "image/png")),
        ],
    )
    assert r.status_code == 200, r.text
    assert eng.calls[0]["image"] == PNG


def test_variation_multipart(env):
    client, eng = env
    r = client.post(
        "/v1/images/variations",
        data={"model": "m", "n": "2", "size": "256x256"},
        files={"image": ("a.png", PNG, "image/png")},
    )
    assert r.status_code == 200, r.text
    assert len(eng.calls) == 2 and eng.calls[0]["image"] == PNG


def test_json_body_still_works(env):
    client, eng = env
    r = client.post(
        "/v1/images/edits",
        json={
            "image": base64.b64encode(PNG).decode(),
            "prompt": "p",
            "size": "256x256",
        },
    )
    assert r.status_code == 200, r.text
    assert eng.calls[0]["image"] == PNG


def test_validation_errors_keep_the_openai_shape(env):
    client, _ = env
    r = client.post(
        "/v1/images/edits", data={"size": "256x256"}, files={"image": ("a.png", PNG)}
    )
    err = r.json()["error"]
    assert (
        r.status_code == 400
        and "prompt" in err["message"]
        and err["type"] == "invalid_request_error"
    )
    r = client.post("/v1/images/edits", data={"prompt": "p"})  # no image at all
    assert r.status_code == 400 and "image" in r.json()["error"]["message"]
    r = client.post(
        "/v1/images/variations",
        content=b"{no",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400
