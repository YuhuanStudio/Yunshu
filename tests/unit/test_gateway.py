"""Unit tests for Yunshu gateway."""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from yunshu_gateway.main import create_app


@pytest.fixture
def client():
    with patch.dict(os.environ, {"YUNSHU_AUTH_DISABLED": "true"}):
        app = create_app()
        yield TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "engine" in data


def test_list_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)


def test_chat_completions_no_model(client):
    """No model loaded → 503."""
    # Ensure no engine is set for this test
    from yunshu_gateway.engine import set_engine

    set_engine(None)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert resp.status_code in (404, 503)
