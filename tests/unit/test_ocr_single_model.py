"""POST /v1/ocr in single-model mode. It dereferenced a missing model manager (500 "OCR extraction
failed") for every served model: a vision model never reached its OCR fallback and a text model got
a 500 instead of the 503 that says what to start (found by the real-server route checks)."""

from __future__ import annotations

import types

import pytest
from fastapi.testclient import TestClient

import yunshu_gateway.engine as eng_mod
from yunshu_gateway.main import create_app


class FakeVLM:
    pass


@pytest.fixture
def make_client(monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)

    def make(engine):
        monkeypatch.setattr(eng_mod, "get_model_manager", lambda: None)
        monkeypatch.setattr(eng_mod, "get_engine", lambda: engine)
        return TestClient(create_app())

    return make


PNG = __import__("base64").b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_text_model_gets_a_503_that_names_the_fix(make_client):
    chat = types.SimpleNamespace(
        model_name="/m/Qwen2.5-3B-Instruct-4bit", is_loaded=True
    )
    r = make_client(chat).post("/v1/ocr", files={"file": ("a.png", PNG, "image/png")})
    assert r.status_code == 503, r.text
    msg = r.json()["error"]["message"]
    assert "Qwen2.5-3B-Instruct-4bit" in msg and "cannot read text from images" in msg


def test_vision_model_serves_the_ocr_fallback(make_client, monkeypatch, tmp_path):
    monkeypatch.setenv("YUNSHU_MEDIA_DIR", str(tmp_path / "media"))
    from yunshu_engine.vlm_engine import VLMEngine

    calls = []

    class V(VLMEngine):
        def __init__(self):
            pass

        model_name = "/m/Qwen3.5-0.8B"
        is_loaded = True

        async def generate(self, **kw):
            calls.append(kw["messages"])
            return {"text": " HELLO ", "prompt_tokens": 40, "completion_tokens": 2}

    r = make_client(V()).post(
        "/v1/ocr", files={"file": ("a.png", PNG, "image/png")}, data={"model": "any"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "HELLO" and body["usage"]["completion_tokens"] == 2
    parts = calls[0][0]["content"]
    urls = [p["image_url"]["url"] for p in parts if p.get("type") == "image_url"]
    # the VLM refuses local files outside YUNSHU_MEDIA_DIR: the upload must be staged there
    assert urls and urls[0].startswith("file://" + str(tmp_path / "media")), urls


def test_upload_that_is_not_an_image_is_400(make_client):
    chat = types.SimpleNamespace(model_name="/m/x", is_loaded=True)
    r = make_client(chat).post(
        "/v1/ocr", files={"file": ("a.png", b"not an image", "image/png")}
    )
    assert r.status_code == 400 and "not an image" in r.json()["error"]["message"]
