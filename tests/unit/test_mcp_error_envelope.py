"""MCP is JSON-RPC, so EVERY error on /v1/mcp must return a JSON-RPC 2.0 error
object — not the OpenAI HTTP envelope a JSON-RPC client can't parse. W785b fixed only the
in-router body-parse paths; this covers the auth/permission/model-access denials (router
+ middleware) and stops the internal-error info leak."""
from __future__ import annotations

import asyncio
import json
import types

from fastapi import HTTPException

from yunshu_gateway.middleware.tenant_auth import _ErrorFormatter
from yunshu_gateway.routers import (
    mcp as M,  # noqa: N812  # intentional short module alias
)


def _body(resp):
    return json.loads(bytes(resp.body).decode())


def _req(path, raw=None):
    s = types.SimpleNamespace(rbac_key=None, role="admin")
    r = types.SimpleNamespace(state=s, url=types.SimpleNamespace(path=path))

    async def _json():
        return raw

    r.json = _json
    return r


def test_error_formatter_mcp_returns_jsonrpc():
    resp = _ErrorFormatter.auth_error(_req("/v1/mcp"), "Invalid API key", status_code=401)
    b = _body(resp)
    assert b["jsonrpc"] == "2.0"
    assert b["error"]["code"] == -32600
    assert b["id"] is None
    assert "error" in b and "type" not in b  # not the OpenAI/Anthropic shape


def test_error_formatter_openai_unchanged():
    resp = _ErrorFormatter.auth_error(_req("/v1/chat/completions"), "Invalid API key", 401)
    b = _body(resp)
    assert "jsonrpc" not in b
    assert b["error"]["type"] == "authentication_error"


def test_mcp_endpoint_permission_denial_is_jsonrpc(monkeypatch):
    def _deny(*a, **k):
        raise HTTPException(status_code=403, detail="no can_infer")

    monkeypatch.setattr("yunshu_gateway.routers.models._check_permission", _deny)
    resp = asyncio.run(M.mcp_endpoint(_req("/v1/mcp", {"jsonrpc": "2.0", "method": "initialize", "id": 1})))
    b = _body(resp)
    assert b["jsonrpc"] == "2.0"
    assert b["error"]["code"] == -32600
    assert "Forbidden" in b["error"]["message"]


def test_mcp_internal_error_does_not_leak(monkeypatch):
    # Force a handler to raise an exception carrying a "secret path" — the client copy
    # must be generic, not the raw exception string.
    monkeypatch.setattr("yunshu_gateway.routers.models._check_permission", lambda *a, **k: None)

    async def _boom(params, req_id):
        raise RuntimeError("/secret/internal/path/model.safetensors")

    monkeypatch.setitem(M._METHODS, "initialize", _boom)
    resp = asyncio.run(M.mcp_endpoint(_req("/v1/mcp", {"jsonrpc": "2.0", "method": "initialize", "id": 1})))
    b = _body(resp)
    assert b["error"]["code"] == -32603  # INTERNAL_ERROR
    assert "secret" not in b["error"]["message"]  # no leak
    assert b["error"]["message"] == "Internal error"
