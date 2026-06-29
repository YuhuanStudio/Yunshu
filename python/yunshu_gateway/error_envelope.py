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

from starlette.responses import JSONResponse

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
) -> JSONResponse:
    """Build a JSONResponse error in the envelope matching `path`'s API family.

    - `/v1/mcp`            → JSON-RPC 2.0 error (id null; no parsed request body at the
                             middleware layer).
    - Anthropic paths      → {"type":"error","error":{type,message}}.
    - everything else      → OpenAI {"error":{message,type,code}}.

    `code` overrides the OpenAI `code` field (defaults to rate_limit_exceeded on 429,
    else omitted). `retry_after` sets the Retry-After header (429/503).
    """
    headers = dict(extra_headers or {})
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)

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
        a_type = (
            "rate_limit_error"
            if is_429
            else (
                "authentication_error"
                if status_code == 401
                else "invalid_request_error"
            )
        )
        return JSONResponse(
            status_code=status_code,
            content={"type": "error", "error": {"type": a_type, "message": message}},
            headers=headers,
        )

    o_type = (
        "rate_limit_error"
        if is_429
        else ("authentication_error" if status_code == 401 else "invalid_request_error")
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
    return JSONResponse(
        status_code=status_code, content={"error": err}, headers=headers
    )
