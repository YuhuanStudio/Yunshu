"""middleware-level denials (rate-limit, per-key concurrency/TPM cap, auth
lockout) returned the OpenAI HTTP error envelope even for the JSON-RPC /v1/mcp endpoint —
a JSON-RPC client can't parse {"error":{...}}. W805 had made only the auth-error path
MCP-aware. Centralize the path-aware envelope (OpenAI / Anthropic / JSON-RPC) in
format_error_response and route rate_limit.py's 429s + tenant_auth's lockout/concurrency/
TPM 429s through it."""
from __future__ import annotations

import inspect
import json

from yunshu_gateway.error_envelope import format_error_response


def _body(resp):
    return json.loads(bytes(resp.body))


def test_mcp_path_gets_jsonrpc_envelope():
    r = format_error_response("/v1/mcp", "Rate limit exceeded", 429, retry_after=30)
    b = _body(r)
    assert b["jsonrpc"] == "2.0"
    assert b["error"]["code"] == -32600
    assert b["error"]["message"] == "Rate limit exceeded"
    assert b["id"] is None
    assert r.status_code == 429
    assert r.headers.get("Retry-After") == "30"


def test_anthropic_path_envelope():
    b = _body(format_error_response("/v1/messages", "Rate limit exceeded", 429))
    assert b["type"] == "error"
    assert b["error"]["type"] == "rate_limit_error"


def test_openai_default_envelope_429_and_401():
    b429 = _body(format_error_response("/v1/chat/completions", "x", 429, code="rate_limit_exceeded"))
    assert b429["error"]["type"] == "rate_limit_error"
    assert b429["error"]["code"] == "rate_limit_exceeded"
    b401 = _body(format_error_response("/v1/chat/completions", "bad", 401))
    assert b401["error"]["type"] == "authentication_error"
    assert b401["error"]["code"] == "invalid_api_key"


def test_custom_code_preserved():
    b = _body(format_error_response("/v1/chat/completions", "conc", 429, code="concurrency_limit"))
    assert b["error"]["code"] == "concurrency_limit"


def test_rate_limit_middleware_uses_shared_envelope():
    from yunshu_gateway.middleware import rate_limit
    src = inspect.getsource(rate_limit)
    assert "format_error_response" in src
    # the old hardcoded OpenAI/Anthropic 429 branches are gone
    assert '"code": "rate_limit_exceeded",' not in src


def test_tenant_auth_uses_shared_envelope():
    from yunshu_gateway.middleware import tenant_auth
    src = inspect.getsource(tenant_auth)
    # Single-consumer model: the simplified middleware routes its 401 denial
    # through the in-module _ErrorFormatter.auth_error helper, which produces
    # the path-aware envelope (JSON-RPC for /v1/mcp, Anthropic shape for
    # /v1/messages, OpenAI shape otherwise) — same shared contract as
    # format_error_response.
    assert "_ErrorFormatter.auth_error(" in src
