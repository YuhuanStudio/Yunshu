"""Loopback calls go to this server's socket, never to the Host header's authority."""

from starlette.requests import Request

from yunshu_gateway.loopback import self_base


def _request(host_header, server):
    scope = {
        "type": "http",
        "scheme": "http",
        "method": "GET",
        "path": "/api/chat",
        "query_string": b"",
        "headers": [(b"host", host_header.encode())],
        "server": server,
    }
    return Request(scope)


def test_self_base_ignores_the_host_header():
    assert self_base(_request("attacker.example:8080", ("127.0.0.1", 18990))) == (
        "http://127.0.0.1:18990"
    )
    assert self_base(_request("x", ("::1", 18991))) == "http://[::1]:18991"
