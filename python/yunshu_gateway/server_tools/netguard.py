"""SSRF guard for server-side fetches.

``web_fetch`` follows URLs the model (and so, through prompt injection, a web page) chose. It
must not reach the host's own network: loopback, private, link-local (cloud metadata),
carrier-grade NAT, multicast and unspecified addresses are refused unless a setting allows them.

The check runs on the *resolved* addresses, and the connection is then pinned to a checked
address (Host header + SNI keep the original name), so a DNS answer that changes between the
check and the connect cannot slip through. Redirects are followed by the caller hop by hop with
the same check on every hop.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit


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
    if not isinstance(url, str) or not url.strip():
        raise UrlNotAllowedError("empty url")
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https"):
        raise UrlNotAllowedError(f"scheme {parts.scheme or '(none)'!r} not allowed")
    if not parts.hostname:
        raise UrlNotAllowedError("no host")
    if parts.username or parts.password:
        raise UrlNotAllowedError("credentials in url not allowed")
    return parts


async def resolve_target(url: str, *, allow_private: bool) -> Target:
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
        except socket.gaierror as e:
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


def domain_matches(host: str, patterns: list[str]) -> bool:
    """Anthropic domain filters: a listed domain covers its subdomains ('example.com' matches
    'docs.example.com'); an entry with a path only matches by host here."""
    host = host.lower().rstrip(".")
    for p in patterns:
        p = p.lower().strip().split("/", 1)[0].lstrip(".")
        if p and (host == p or host.endswith("." + p)):
            return True
    return False
