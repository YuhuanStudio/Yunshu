"""Host / Origin validation for a loopback-bound server (pure ASGI).

A server on 127.0.0.1 without a token trusts whoever can reach the socket, and a web page
can reach it: DNS rebinding makes ``http://attacker.example:8000`` resolve to 127.0.0.1
(same origin, so CORS does not apply), and a cross-site ``text/plain`` POST needs no
preflight and still runs the route. Two checks close both:

- **Host**: when the server is bound to a loopback address, a request whose ``Host`` is not
  a loopback name or IP (``localhost``, ``127.0.0.1``, ``[::1]``, any port) gets 403.
  ``YUNSHU_ALLOWED_HOSTS`` adds names (and turns the check on for any bind address).
- **Origin**: a state-changing request carrying an ``Origin`` that is not loopback, not the
  request's own origin, and not in ``YUNSHU_CORS_ORIGINS`` gets 403. Requests that carry
  credentials (``Authorization`` / ``x-api-key``) are exempt: a cross-site page cannot add
  those headers without a preflight, which the CORS layer already decides.
"""

from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlsplit

from yunshu_engine import settings
from yunshu_gateway.error_envelope import format_error_response

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_LOOPBACK_BINDS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


def _split_host(value: str) -> str:
    """The host part of a ``Host`` header / URL netloc: lower case, no port, no brackets."""
    value = value.strip().lower()
    if value.startswith("["):
        return value[1 : value.find("]")] if "]" in value else value[1:]
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def is_loopback_host(value: str) -> bool:
    """``localhost`` (and ``*.localhost``) or a loopback IP literal."""
    host = _split_host(value)
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _server_is_loopback_socket(scope: dict) -> bool:
    server = scope.get("server")
    if not server or not isinstance(server[0], str):
        return False
    try:
        return ipaddress.ip_address(server[0]).is_loopback
    except ValueError:
        return False


def _allowed_hosts() -> frozenset[str]:
    return frozenset(
        _split_host(h)
        for h in (settings.get("YUNSHU_ALLOWED_HOSTS") or ())
        if h.strip()
    )


def host_ok(host_header: str | None, scope: dict) -> bool:
    """Whether the ``Host`` header passes (True when no check applies)."""
    extra = _allowed_hosts()
    if "*" in extra:
        return True
    if not extra:
        bound = (settings.get("YUNSHU_HOST") or "127.0.0.1").strip().lower()
        # The socket test keeps in-process clients (TestClient, Unix sockets) out of the check.
        if bound not in _LOOPBACK_BINDS or not _server_is_loopback_socket(scope):
            return True
    if not host_header:
        return True  # HTTP/1.0 without Host: not a browser
    return is_loopback_host(host_header) or _split_host(host_header) in extra


def origin_ok(origin: str, host_header: str | None) -> bool:
    """Whether a cross-origin-capable request from ``origin`` may change state."""
    from .live_cors import current_origins

    allowed = current_origins()
    if "*" in allowed or origin in allowed:
        return True
    netloc = urlsplit(origin).netloc
    if not netloc:  # "null" (sandboxed iframe, file://) and garbage
        return False
    if is_loopback_host(netloc):
        return True
    return bool(host_header) and netloc.lower() == host_header.lower()


class HostGuardMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        kind = scope["type"]
        if kind not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = {k: v for k, v in scope["headers"] if k in (b"host", b"origin")}
        host = headers.get(b"host", b"").decode("latin-1") or None
        reason = None
        if not host_ok(host, scope):
            reason = "Invalid Host header"
        elif kind == "http" and scope["method"] not in _SAFE_METHODS:
            origin = headers.get(b"origin", b"").decode("latin-1")
            if (
                origin
                and not any(
                    k in (b"authorization", b"x-api-key") for k, _ in scope["headers"]
                )
                and not origin_ok(origin, host)
            ):
                reason = "Cross-origin request refused"
        if reason is None:
            await self.app(scope, receive, send)
        elif kind == "websocket":
            await send({"type": "websocket.close", "code": 1008})
        else:
            response = format_error_response(
                scope["path"], reason, 403, code="forbidden_origin"
            )
            await response(scope, receive, send)
