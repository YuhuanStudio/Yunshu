"""ModelCards served through /v1/models, /v1/models/{id} and the Ollama /api/show layer."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from yunshu_engine.model_manager import ModelManager
from yunshu_gateway import model_cards
from yunshu_gateway.main import create_app
from yunshu_gateway.routers import models as models_router
from yunshu_gateway.routers import ollama

MODELS = Path(os.environ.get("YUNSHU_TEST_MODELS", "~/.yunshu/models")).expanduser()


@pytest.fixture
def served(monkeypatch):
    for name in ("Qwen2.5-3B-Instruct-4bit", "GLM-OCR-bf16"):
        if not (MODELS / name).exists():
            pytest.skip(f"{name} not available")
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    mgr = ModelManager(max_memory_bytes=None)
    mgr.register_model("qwen25-3b", str(MODELS / "Qwen2.5-3B-Instruct-4bit"))
    mgr.register_model("glm-ocr", str(MODELS / "GLM-OCR-bf16"))
    monkeypatch.setattr(models_router, "get_model_manager", lambda: mgr)
    monkeypatch.setattr(model_cards, "get_model_manager", lambda: mgr)
    app = create_app()
    # The Ollama layer reads /v1/models over loopback: point it back at this app.
    monkeypatch.setattr(
        ollama,
        "_client",
        lambda request: httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://loopback"
        ),
    )
    return TestClient(app)


def test_list_and_retrieve_carry_cards(served):
    body = served.get("/v1/models").json()
    by_id = {m["id"]: m for m in body["data"]}
    assert set(by_id) == {"qwen25-3b", "glm-ocr"}
    m = by_id["qwen25-3b"]
    assert m["context_length"] == 32768 and m["model_type"] == "chat"
    assert m["yunshu"]["state"]["status"] == "not-loaded"
    assert "path" not in m["yunshu"]  # anonymous callers never see filesystem paths
    assert by_id["glm-ocr"]["yunshu"]["kind"] == "ocr"
    assert isinstance(m["created"], int) and m["created"] > 0

    one = served.get("/v1/models/qwen25-3b").json()
    assert one["id"] == "qwen25-3b"
    assert one["yunshu"]["parameters"] == m["yunshu"]["parameters"]
    # same resolution rules as inference (case-insensitive)
    assert served.get("/v1/models/QWEN25-3B").json()["id"] == "qwen25-3b"
    assert served.get("/v1/models/nope").status_code == 404


def test_ollama_show_uses_the_card(served):
    r = served.post("/api/show", json={"model": "qwen25-3b"})
    assert r.status_code == 200
    assert r.json()["capabilities"] == ["completion", "tools"]
    assert r.json()["model_info"]["qwen2.context_length"] == 32768
    r = served.post("/api/show", json={"model": "glm-ocr"})
    assert r.json()["capabilities"] == ["ocr"]
