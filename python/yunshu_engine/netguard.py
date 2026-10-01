"""One network policy for every outbound fetch: web_fetch, the MCP connector, VLM media download.

These fetch URLs that a model (and so, through prompt injection, a web page) or a request chose.
They must not reach the host's own network: loopback, private, link-local (cloud metadata),
carrier-grade NAT, multicast, reserved, IPv4-mapped IPv6 of any of those, and unspecified
addresses are refused unless a setting allows them.

The check runs on the *resolved* addresses, and the connection is then pinned to a checked
address (Host header + SNI keep the original name), so a DNS answer that changes between the
check and the connect cannot slip through. Redirects are followed hop by hop with the same check
on every hop. Credentials are bound to an origin (:func:`same_origin`): a URL on another origin
never receives the headers meant for the first one.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx


class UrlNotAllowedError(Exception):
    """The URL is malformed, uses a forbidden scheme, or resolves to a forbidden address."""


@dataclass(frozen=True)
class Target:
    url: str  # original URL
    host: str
    port: int
    scheme: str
    ip: str  # a checked, resolved address to connect to


def is_forbidden_ip(ip: str) -> bool:
    a = ipaddress.ip_address(ip)
    if isinstance(a, ipaddress.IPv6Address) and a.ipv4_mapped:
        a = a.ipv4_mapped
    return (
        a.is_private
        or a.is_loopback
        or a.is_link_local
        or a.is_multicast
        or a.is_reserved
        or a.is_unspecified
        or (
            isinstance(a, ipaddress.IPv4Address)
            and a in ipaddress.ip_network("100.64.0.0/10")
        )
    )


def parse_url(url: str):
    """Split and validate an http(s) URL; every malformed input is a :class:`UrlNotAllowedError`."""
    if not isinstance(url, str) or not url.strip():
        raise UrlNotAllowedError("empty url")
    url = url.strip()
    if any(ord(c) <= 0x20 or ord(c) == 0x7F for c in url):
        raise UrlNotAllowedError("control characters or spaces in url")
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port
    except ValueError as e:
        raise UrlNotAllowedError("malformed url") from e
    if parts.scheme not in ("http", "https"):
        raise UrlNotAllowedError(f"scheme {parts.scheme or '(none)'!r} not allowed")
    if not host:
        raise UrlNotAllowedError("no host")
    if "@" in parts.netloc or parts.username or parts.password:
        raise UrlNotAllowedError("credentials in url not allowed")
    if port is not None and not 1 <= port <= 65535:
        raise UrlNotAllowedError("port out of range")
    return parts


def origin_of(url: str) -> tuple[str, str, int]:
    """(scheme, host, effective port) of a validated URL."""
    p = parse_url(url)
    return (
        p.scheme,
        (p.hostname or "").lower().rstrip("."),
        p.port or (443 if p.scheme == "https" else 80),
    )


def same_origin(a: str, b: str) -> bool:
    try:
        return origin_of(a) == origin_of(b)
    except UrlNotAllowedError:
        return False


def join_url(base: str, ref: str) -> str:
    try:
        return urljoin(base, ref)
    except ValueError as e:
        raise UrlNotAllowedError("malformed redirect url") from e


async def resolve_target(url: str, *, allow_private: bool) -> Target:
    """Resolve once and return a checked address. The caller connects to ``Target.ip`` only."""
    parts = parse_url(url)
    host = parts.hostname or ""
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        # A literal address needs no lookup.
        ipaddress.ip_address(host)
        infos = [host]
    except ValueError:
        loop = asyncio.get_running_loop()
        try:
            res = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except (socket.gaierror, UnicodeError) as e:
            raise UrlNotAllowedError(f"cannot resolve {host}: {e}") from e
        infos = [r[4][0] for r in res]
    if not infos:
        raise UrlNotAllowedError(f"cannot resolve {host}")
    if not allow_private:
        bad = [ip for ip in infos if is_forbidden_ip(ip)]
        if bad:
            raise UrlNotAllowedError(
                f"{host} resolves to a private or reserved address"
            )
    return Target(url=url, host=host, port=port, scheme=parts.scheme, ip=infos[0])


def pin_request(tgt: Target, url: str | None = None) -> tuple[str, dict, dict]:
    """(url, extra headers, httpx extensions) that connect to ``tgt.ip`` while the Host header
    and TLS SNI / certificate check keep the original name. ``url`` is another URL on the same
    origin as ``tgt`` (default: the target's own)."""
    parts = urlsplit(url or tgt.url)
    ip = f"[{tgt.ip}]" if ":" in tgt.ip else tgt.ip
    pinned = f"{tgt.scheme}://{ip}:{tgt.port}{parts.path or '/'}" + (
        f"?{parts.query}" if parts.query else ""
    )
    default_port = 443 if tgt.scheme == "https" else 80
    host = f"[{tgt.host}]" if ":" in tgt.host else tgt.host
    hostport = host if tgt.port == default_port else f"{host}:{tgt.port}"
    ext = {"sni_hostname": tgt.host} if tgt.scheme == "https" else {}
    return pinned, {"Host": hostport}, ext


def domain_matches(host: str, patterns: list[str]) -> bool:
    """Anthropic domain filters: a listed domain covers its subdomains ('example.com' matches
    'docs.example.com'); an entry with a path only matches by host here."""
    host = host.lower().rstrip(".")
    for p in patterns:
        p = p.lower().strip().split("/", 1)[0].lstrip(".")
        if p and (host == p or host.endswith("." + p)):
            return True
    return False


async def download_to_file(
    url: str,
    dest: str,
    *,
    max_bytes: int,
    timeout: float,
    allow_private: bool = False,
    client: httpx.AsyncClient | None = None,
    verify: bool = True,
    headers: dict | None = None,
    max_redirects: int = 3,
) -> int:
    """Download ``url`` into ``dest`` under the network policy and return the byte count.

    One deadline covers DNS, every redirect hop and the body. Each hop is resolved, checked and
    pinned. No credentials are sent. Raises ``ValueError`` (message contains "size limit" for an
    oversized body, "SSRF blocked" for a refused address) for every failure the caller maps to a
    clean 400.
    """
    own = client is None
    if own:
        client = httpx.AsyncClient(
            verify=verify, trust_env=False, follow_redirects=False
        )
    try:
        async with asyncio.timeout(timeout):
            cur = url
            for _ in range(max_redirects + 1):
                try:
                    tgt = await resolve_target(cur, allow_private=allow_private)
                except UrlNotAllowedError as e:
                    raise ValueError(f"SSRF blocked: {e}") from e
                pinned, extra, ext = pin_request(tgt)
                async with client.stream(
                    "GET",
                    pinned,
                    headers={**(headers or {}), **extra},
                    extensions=ext,
                ) as resp:
                    if resp.status_code in (
                        301,
                        302,
                        303,
                        307,
                        308,
                    ) and resp.headers.get("location"):
                        try:
                            cur = join_url(cur, resp.headers["location"])
                        except UrlNotAllowedError as e:
                            raise ValueError(f"SSRF blocked: {e}") from e
                        continue
                    if resp.status_code >= 300:
                        raise ValueError(f"HTTP {resp.status_code}")
                    clen = resp.headers.get("Content-Length")
                    if (
                        clen is not None
                        and str(clen).isdigit()
                        and int(clen) > max_bytes
                    ):
                        raise ValueError(
                            f"remote file exceeds size limit ({clen} > {max_bytes} bytes)"
                        )
                    written = 0
                    with open(dest, "wb") as fh:
                        async for chunk in resp.aiter_bytes(65536):
                            written += len(chunk)
                            if written > max_bytes:
                                raise ValueError(
                                    f"remote file exceeds size limit (> {max_bytes} bytes)"
                                )
                            fh.write(chunk)
                    return written
            raise ValueError("too many redirects")
    except TimeoutError as e:
        raise ValueError(f"download timed out after {timeout:g}s") from e
    except httpx.HTTPError as e:
        raise ValueError(f"{type(e).__name__}: {e}") from e
    finally:
        if own:
            await client.aclose()
