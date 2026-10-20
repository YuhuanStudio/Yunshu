"""Ollama loopback calls forward both credential headers."""

from __future__ import annotations

from types import SimpleNamespace

from yunshu_gateway.routers import ollama


def _req(headers):
    return SimpleNamespace(
        headers=headers,
        state=SimpleNamespace(request_id=None),
        url=SimpleNamespace(scheme="http", netloc="h"),
        scope={"server": ("127.0.0.1", 18990)},
    )


def test_x_api_key_forwarded():
    c = ollama._client(_req({"x-api-key": "sk-1"}))
    assert c.headers["x-api-key"] == "sk-1"


def test_authorization_still_forwarded():
    c = ollama._client(_req({"authorization": "Bearer t"}))
    assert c.headers["authorization"] == "Bearer t"
