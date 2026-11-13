"""Who counts as the person at this machine when no token proves it."""

from __future__ import annotations

from fastapi import Request


def _is_loopback_host(host: str | None) -> bool:
    import ipaddress

    h = (host or "").strip().strip("[]").lower()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h.split("%")[0]).is_loopback
    except ValueError:
        return False


def local_operator(request: Request) -> bool:
    """The caller is the person at this machine, as far as one can tell without a token.

    True only for a direct loopback socket peer whose forwarded-for chain (the console process adds
    the browser's address), Host and Origin (a page on another site, or a DNS-rebound name) are all
    loopback. A LAN or remote browser behind the console, or a hostile web page, is not."""
    from urllib.parse import urlsplit

    client = getattr(request, "client", None)
    if client is None or not _is_loopback_host(client.host):
        return False
    headers = request.headers
    xff = headers.get("x-forwarded-for")
    if xff and not all(_is_loopback_host(h) for h in xff.split(",")):
        return False
    real = headers.get("x-real-ip")
    if headers.get("forwarded") or (real and not _is_loopback_host(real)):
        return False
    host = headers.get("host", "")
    if host:
        name = host.rsplit(":", 1)[0] if not host.endswith("]") else host
        if not _is_loopback_host(name):
            return False
    origin = headers.get("origin")
    if origin and origin != "null":
        if not _is_loopback_host(urlsplit(origin).hostname):
            return False
    elif origin == "null":
        return False
    return True


_CONSOLE_PREFIXES = ("/v1/yunshu/", "/v1/models/load", "/v1/models/unload/")


def local_console_route(request: Request, auth_token: str) -> bool:
    """Stored keys alone switch auth on, but must not lock the person at this machine out of the
    console (which has no key until they type one): a credential-less loopback caller reaches the
    console's own routes. Inference stays key-gated, and a static token gates everything."""
    if auth_token or not request.url.path.startswith(_CONSOLE_PREFIXES):
        return False
    return local_operator(request)
