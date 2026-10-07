"""The real static-token owner can inspect residency; anonymous lists cannot."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_engine import settings
from yunshu_gateway.middleware.auth import AuthMiddleware
from yunshu_gateway.routers import models


@pytest.mark.parametrize("path", ["/v1/models", "/v1/models/embedding"])
def test_static_owner_gets_loaded_state_and_anonymous_does_not(monkeypatch, path):
    from yunshu_gateway import model_card_formats, model_cards

    settings.clear_overrides()
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "fixture-token")
    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "0")
    entry = SimpleNamespace(
        is_loaded=True,
        estimated_bytes=1024,
        engine=SimpleNamespace(get_stats=lambda: {"requests": 0}),
    )
    manager = SimpleNamespace(
        list_entries=lambda: [entry],
        get_entry=lambda _: entry,
        resolve_model_id=lambda name: name,
    )
    monkeypatch.setattr(models, "get_model_manager", lambda: manager)
    monkeypatch.setattr(model_cards, "entry_card", lambda _: object())
    monkeypatch.setattr(model_cards, "find_card", lambda _: object())
    monkeypatch.setattr(
        model_card_formats, "openai_model", lambda card, detailed: {"id": "embedding"}
    )
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(models.router, prefix="/v1")
    with TestClient(app) as client:
        response = client.get(path, headers={"Authorization": "Bearer fixture-token"})
        assert response.status_code == 200
        body = response.json()
        item = body["data"][0] if path == "/v1/models" else body
        assert item["loaded"] is True and item["size_gb"] >= 0
        assert client.get(path).status_code == 401
        monkeypatch.delenv("YUNSHU_AUTH_TOKEN")
        public = client.get(path)
        assert public.status_code == 200
        public_body = public.json()
        public_item = public_body["data"][0] if path == "/v1/models" else public_body
        assert "loaded" not in public_item and "size_gb" not in public_item
    settings.clear_overrides()
