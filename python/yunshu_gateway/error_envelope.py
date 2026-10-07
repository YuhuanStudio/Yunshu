"""Path-aware error envelope for MIDDLEWARE-level denials.

A request that never reaches its route handler — rejected by auth, rate limiting,
concurrency/TPM caps — still gets an error response from the middleware. That response
must speak the protocol family the *endpoint* would have: OpenAI `{"error":{...}}`,
Anthropic `{"type":"error","error":{...}}`, or — for the JSON-RPC `/v1/mcp` endpoint —
a JSON-RPC 2.0 error object the client can actually parse. Previously the auth-error path
was MCP-aware but the rate-limit / concurrency / TPM / lockout denials returned the
OpenAI envelope to MCP clients. This centralizes the branching so every middleware denial
is consistent.
"""

from __future__ import annotations

import json

from starlette.responses import JSONResponse


class EngineStreamError(Exception):
    """The engine failed after a stream started (e.g. a chat-template error). Streaming routers
    surface it as a protocol error event carrying the engine's message, never as an empty
    successful completion."""


def server_error_body(message: str = "Internal server error") -> dict:
    """The OpenAI error object for a server-side failure (every OpenAI-shaped route, stream or not)."""
    return {
        "error": {
            "message": message,
            "type": "server_error",
            "param": None,
            "code": "internal_error",
        }
    }


def server_error_sse(message: str = "Internal server error") -> bytes:
    """The same error as a terminal SSE ``data:`` event."""
    body = json.dumps(server_error_body(message), ensure_ascii=False)
    return f"data: {body}\n\n".encode()


# Paths served by the Anthropic router (exact match — don't overmatch /admin/.../messages).
_ANTHROPIC_PATHS = frozenset(
    {
        "/v1/messages",
        "/messages",
        "/v1/messages/count_tokens",
        "/messages/count_tokens",
    }
)
_MCP_PATH = "/v1/mcp"


def format_error_response(
    path: str,
    message: str,
    status_code: int,
    *,
    code: str | None = None,
    retry_after: int | str | None = None,
    extra_headers: dict | None = None,
    jsonrpc_code: int = -32600,  # INVALID_REQUEST
    error_type: str | None = None,
    request_id: str | None = None,
    x_yunshu: dict | None = None,
) -> JSONResponse:
    """Build a JSONResponse error in the envelope matching `path`'s API family.

    - `/v1/mcp`            → JSON-RPC 2.0 error (id null; no parsed request body at the
                             middleware layer).
    - Anthropic paths      → {"type":"error","error":{type,message}}.
    - everything else      → OpenAI {"error":{message,type,code}}.

    `code` overrides the OpenAI `code` field (defaults to rate_limit_exceeded on 429,
    else omitted). `retry_after` sets the Retry-After header (429/503). `error_type`
    overrides the dialect's error type; 5xx get `server_error` (OpenAI) / `overloaded_error`
    (503, 529) / `timeout_error` (504) / `api_error` (Anthropic). `x_yunshu` is merged into
    the OpenAI `error.x_yunshu` object (next to the hint and `request_id`).
    """
    headers = dict(extra_headers or {})
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)

    if path.startswith("/tavily/"):
        return JSONResponse(
            status_code=status_code,
            content={"detail": {"error": message}},
            headers=headers,
        )

    if path == _MCP_PATH:
        return JSONResponse(
            status_code=status_code,
            content={
                "jsonrpc": "2.0",
                "error": {"code": jsonrpc_code, "message": message},
                "id": None,
            },
            headers=headers,
        )

    is_429 = status_code == 429
    if path in _ANTHROPIC_PATHS:
        a_type = error_type or _anthropic_type(status_code)
        return JSONResponse(
            status_code=status_code,
            content={"type": "error", "error": {"type": a_type, "message": message}},
            headers=headers,
        )

    o_type = error_type or (
        "rate_limit_error"
        if is_429
        else (
            "authentication_error"
            if status_code == 401
            else ("server_error" if status_code >= 500 else "invalid_request_error")
        )
    )
    o_code = (
        code
        if code is not None
        else (
            "rate_limit_exceeded"
            if is_429
            else ("invalid_api_key" if status_code == 401 else None)
        )
    )
    err: dict = {"message": message, "type": o_type}
    if o_code is not None:
        err["code"] = o_code
    from .error_hints import add_hint

    add_hint(err, status_code, request_id)
    if x_yunshu:
        err.setdefault("x_yunshu", {}).update(x_yunshu)
    return JSONResponse(
        status_code=status_code, content={"error": err}, headers=headers
    )


def _anthropic_type(status_code: int) -> str:
    if status_code == 429:
        return "rate_limit_error"
    if status_code == 401:
        return "authentication_error"
    if status_code in (503, 529):
        return "overloaded_error"
    if status_code == 504:
        return "timeout_error"
    if status_code >= 500:
        return "api_error"
    return "invalid_request_error"
