"""Text WebSocket endpoints accept stored API keys, like realtime does."""

from __future__ import annotations

import asyncio

from yunshu_gateway import api_keys
from yunshu_gateway.routers import stream_ws


class _WS:
    def __init__(self, headers):
        self.headers = headers
        self.query_params = {}
        self.closed = None

    async def close(self, code=1000):
        self.closed = code


class _Store:
    def has_keys(self):
        return True

    def authenticate(self, secret):
        if secret != "sk-stored":
            raise api_keys.AuthFailureError(401, "bad")
        return api_keys.Principal("key", frozenset({"infer"}), "k1", "n")


def _admit(headers, monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_DISABLED", raising=False)
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "static-token")
    monkeypatch.setattr(api_keys, "get_store", lambda: _Store())
    ws = _WS(headers)
    ok = asyncio.run(stream_ws._admit(ws))
    return ok, ws


def test_stored_key_is_admitted(monkeypatch):
    ok, _ = _admit({"authorization": "Bearer sk-stored"}, monkeypatch)
    assert ok


def test_stored_key_via_x_api_key(monkeypatch):
    ok, _ = _admit({"x-api-key": "sk-stored"}, monkeypatch)
    assert ok


def test_unknown_key_rejected(monkeypatch):
    ok, ws = _admit({"authorization": "Bearer nope"}, monkeypatch)
    assert not ok and ws.closed == 1008


def test_static_token_still_admitted(monkeypatch):
    ok, _ = _admit({"authorization": "Bearer static-token"}, monkeypatch)
    assert ok
