"""Calls the gateway makes to itself on a caller's behalf."""

from __future__ import annotations

from fastapi import Request


def self_base(request: Request) -> str:
    """This server's own socket address. Never the Host header: loopback calls carry the
    caller's credentials (and the in-process secret), so a client-chosen authority would
    receive them."""
    server = request.scope.get("server")
    if not server:
        raise RuntimeError("no server address in the ASGI scope")
    host, port = server[0], server[1]
    if ":" in host:
        host = f"[{host}]"
    return f"{request.url.scheme}://{host}:{port}"
