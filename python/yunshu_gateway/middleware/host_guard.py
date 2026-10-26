"""Host / Origin validation for a loopback-bound server (pure ASGI).

A server on 127.0.0.1 without a token trusts whoever can reach the socket, and a web page
can reach it: DNS rebinding makes ``http://attacker.example:8000`` resolve to 127.0.0.1
(same origin, so CORS does not apply), and a cross-site ``text/plain`` POST needs no
preflight and still runs the route. Two checks close both:

- **Host**: when the server is bound to a loopback address, a request whose ``Host`` is not
  a loopback name or IP gets 403. Accepted: ``localhost`` (also ``localhost.`` and
  ``*.localhost``), any 127.0.0.0/8 form (``127.0.0.1``, ``127.1``, ``0x7f.1``), ``[::1]`` and
  its mapped form, and the unspecified address (``0.0.0.0``, ``[::]``: connecting to it reaches
  this machine, and a rebinding page cannot make a browser send it for an attacker name), any
  port. ``host.docker.internal`` is NOT accepted: a container reaches the host's loopback
  server through it only if the server listens beyond loopback, and then this check is off.
  A reverse proxy, ``tailscale serve`` or ``ssh -L`` that forwards a public ``Host`` to a
  loopback server needs that name in ``YUNSHU_ALLOWED_HOSTS`` (which also turns the check on for
  any bind address).
- **Origin**: a state-changing request carrying an ``Origin`` that is not loopback, not the
  request's own origin, and not in ``YUNSHU_CORS_ORIGINS`` gets 403. An entry of that setting
  matches exactly, or as ``scheme://*`` for every origin of a scheme (``vscode-webview://*``,
  ``app://*`` for Electron renderers). ``Origin: null`` (sandboxed iframe, ``file://``) is always
  refused. Requests that carry credentials (``Authorization`` / ``x-api-key``) are exempt: a
  cross-site page cannot add those headers without a preflight, which the CORS layer already
  decides.
"""

from __future__ import annotations

import ipaddress
import re
import socket
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


_LEGACY_IPV4 = re.compile(r"^(?:0x[0-9a-f]+|[0-9]+)(?:\.(?:0x[0-9a-f]+|[0-9]+)){0,3}$")


def _ip_of(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """An IP literal, including the short IPv4 forms browsers and ``inet_aton`` accept (``127.1``)."""
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    if _LEGACY_IPV4.match(host):
        try:
            return ipaddress.IPv4Address(socket.inet_aton(host))
        except (OSError, ValueError):
            return None
    return None


def is_loopback_host(value: str) -> bool:
    """``localhost`` (also ``localhost.`` and ``*.localhost``) or a loopback IP literal."""
    host = _split_host(value).rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        return True
    ip = _ip_of(host)
    if ip is None:
        return False
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_loopback


def is_local_host(value: str) -> bool:
    """A loopback host, or the unspecified address (``0.0.0.0``, ``[::]``) that reaches this machine."""
    if is_loopback_host(value):
        return True
    ip = _ip_of(_split_host(value).rstrip("."))
    return ip is not None and ip.is_unspecified


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
        _split_host(h).rstrip(".")
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
    return is_local_host(host_header) or _split_host(host_header).rstrip(".") in extra


def origin_ok(origin: str, host_header: str | None) -> bool:
    """Whether a cross-origin-capable request from ``origin`` may change state."""
    from .live_cors import current_origins

    if origin.strip().lower() == "null":  # sandboxed iframe, file://: never a known origin
        return False
    allowed = current_origins()
    if "*" in allowed or origin in allowed:
        return True
    parts = urlsplit(origin)
    # custom schemes (Electron app://, vscode-webview://<random id>) listed as "scheme://*"
    if parts.scheme and f"{parts.scheme.lower()}://*" in (o.lower() for o in allowed):
        return True
    netloc = parts.netloc
    if not netloc:  # garbage
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
            shown = (host or "")[:100]
            reason = (
                f"Invalid Host header {shown!r}: this server answers loopback names only; "
                "add the name to YUNSHU_ALLOWED_HOSTS to serve it (reverse proxy, "
                "tailscale serve, ssh -L)"
            )
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
