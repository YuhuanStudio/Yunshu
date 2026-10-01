"""Requests a model cannot serve are a 400, not garbage or silence.

- embedding-only checkpoints (sentence-transformers export) on chat / completions / messages
- Anthropic image blocks sent to a text-only model
- the version string comes from pyproject in a checkout
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from yunshu_engine.version import yunshu_version
from yunshu_gateway import model_guards
from yunshu_gateway.routers import anthropic, chat, completions
from yunshu_gateway.routers import responses as responses_mod


class _TextEngine:
    is_loaded = True
    model_name = "text-model"
    _tokenizer = None


def _embedder(tmp_path):
    (tmp_path / "1_Pooling").mkdir()
    (tmp_path / "1_Pooling" / "config.json").write_text("{}")
    return SimpleNamespace(is_loaded=True, model_name=str(tmp_path))


def test_embedding_only_detected_by_pooling_config(tmp_path):
    assert model_guards.is_embedding_only(_embedder(tmp_path)) is True
    assert model_guards.is_embedding_only(_TextEngine()) is False


def test_reject_embedding_only_message(tmp_path):
    with pytest.raises(HTTPException) as e:
        model_guards.reject_embedding_only(_embedder(tmp_path), "emb")
    assert e.value.status_code == 400
    assert "/v1/embeddings" in e.value.detail and "/api/embed" in e.value.detail


def test_reject_images_for_text_engine():
    with pytest.raises(HTTPException) as e:
        model_guards.reject_images_for_text_model(_TextEngine(), True)
    assert e.value.status_code == 400
    assert "image" in e.value.detail
    model_guards.reject_images_for_text_model(_TextEngine(), False)


def test_images_allowed_for_vlm_engine():
    from yunshu_engine.vlm_engine import VLMEngine

    vlm = object.__new__(VLMEngine)
    model_guards.reject_images_for_text_model(vlm, True)


def _client(monkeypatch, engine):
    app = FastAPI()
    app.include_router(chat.router, prefix="/v1")
    app.include_router(completions.router, prefix="/v1")
    app.include_router(anthropic.router, prefix="/v1")
    app.include_router(responses_mod.router, prefix="/v1")
    for mod in (chat, completions, responses_mod):
        monkeypatch.setattr(mod, "get_engine", lambda: engine)
    monkeypatch.setattr(chat, "get_model_manager", lambda: None)
    monkeypatch.setattr(anthropic, "get_engine", lambda: engine)
    return TestClient(app)


def test_chat_and_completions_on_embedding_model_are_400(monkeypatch, tmp_path):
    c = _client(monkeypatch, _embedder(tmp_path))
    r = c.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 400 and "/v1/embeddings" in r.text
    r = c.post("/v1/completions", json={"model": "m", "prompt": "hi"})
    assert r.status_code == 400 and "/v1/embeddings" in r.text


def test_anthropic_messages_on_embedding_model_are_400(monkeypatch, tmp_path):
    c = _client(monkeypatch, _embedder(tmp_path))
    r = c.post(
        "/v1/messages",
        json={
            "model": "m",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 400
    assert r.json()["type"] == "error"
    assert "/v1/embeddings" in r.json()["error"]["message"]


def test_anthropic_image_block_on_text_model_is_400(monkeypatch):
    c = _client(monkeypatch, _TextEngine())
    png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    r = c.post(
        "/v1/messages",
        json={
            "model": "m",
            "max_tokens": 8,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": png,
                            },
                        },
                    ],
                }
            ],
        },
    )
    assert r.status_code == 400
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert "image" in body["error"]["message"]


def test_version_reads_pyproject_in_a_checkout():
    assert yunshu_version() == "0.1.2"


def test_responses_on_embedding_model_is_400(monkeypatch, tmp_path):
    c = _client(monkeypatch, _embedder(tmp_path))
    for stream in (False, True):
        r = c.post(
            "/v1/responses", json={"model": "m", "input": "hi", "stream": stream}
        )
        assert r.status_code == 400 and "/v1/embeddings" in r.text
