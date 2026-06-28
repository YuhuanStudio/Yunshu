"""b: the /mcp endpoint must return JSON-RPC 2.0 error objects for malformed
envelopes (it used to take `req: JSONRPCRequest`, so FastAPI rejected bad bodies with
an OpenAI-style HTTP error a JSON-RPC client can't parse). Also: an explicit
`id: null` is a REQUEST, not a notification (only an omitted id is a notification)."""
from __future__ import annotations

import asyncio
import json
import types

from yunshu_gateway.routers import (
    mcp as M,  # noqa: N812  # intentional short module alias
)


class _FakeRequest:
    def __init__(self, raw: bytes | object):
        self._raw = raw
        self.state = types.SimpleNamespace(rbac_key=None, role="admin")

    async def json(self):
        if isinstance(self._raw, (bytes, str)):
            return json.loads(self._raw)  # raises on invalid JSON → PARSE_ERROR
        return self._raw


def _call(raw, monkeypatch):
    # _check_permission is imported inside the endpoint from .models — neutralize it.
    monkeypatch.setattr("yunshu_gateway.routers.models._check_permission", lambda *a, **k: None)
    resp = asyncio.run(M.mcp_endpoint(_FakeRequest(raw)))
    return resp


def _body(resp):
    return json.loads(bytes(resp.body).decode())


def test_invalid_json_returns_parse_error(monkeypatch):
    resp = _call(b"{not json", monkeypatch)
    b = _body(resp)
    assert b["jsonrpc"] == "2.0"
    assert b["error"]["code"] == -32700  # PARSE_ERROR
    assert b["id"] is None


def test_non_object_body_returns_invalid_request(monkeypatch):
    resp = _call([1, 2, 3], monkeypatch)  # JSON array (batch) not supported
    b = _body(resp)
    assert b["error"]["code"] == -32600  # INVALID_REQUEST
    assert b["id"] is None


def test_missing_method_returns_invalid_request_with_id(monkeypatch):
    resp = _call({"jsonrpc": "2.0", "id": 7}, monkeypatch)  # no `method`
    b = _body(resp)
    assert b["error"]["code"] == -32600
    assert b["id"] == 7  # echoes the request id


def test_missing_method_notification_is_silent(monkeypatch):
    resp = _call({"jsonrpc": "2.0"}, monkeypatch)  # no method, no id → notification
    assert resp.status_code == 204


def test_explicit_id_null_is_a_request_not_notification(monkeypatch):
    # initialize works without an engine; explicit id:null must get a response.
    resp = _call({"jsonrpc": "2.0", "method": "initialize", "id": None}, monkeypatch)
    assert resp.status_code == 200
    b = _body(resp)
    assert "result" in b and b["id"] is None


def test_omitted_id_is_a_notification(monkeypatch):
    resp = _call({"jsonrpc": "2.0", "method": "initialize"}, monkeypatch)
    assert resp.status_code == 204


def test_valid_request_dispatches(monkeypatch):
    resp = _call({"jsonrpc": "2.0", "method": "initialize", "id": 1}, monkeypatch)
    assert resp.status_code == 200
    b = _body(resp)
    assert b["result"]["serverInfo"]["name"] == "yunshu"
    assert b["id"] == 1
