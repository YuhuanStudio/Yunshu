"""Shared network policy (B13-B19): pinning, reply validation, cleanup, origin-bound credentials,
deadlines / byte budgets, URL parsing. Mocked DNS and HTTP only; nothing here opens a real socket."""

from __future__ import annotations

import asyncio
import json
import socket
import time

import httpx
import pytest

from yunshu_engine import netguard
from yunshu_engine.netguard import UrlNotAllowedError
from yunshu_gateway.server_tools import mcp_connector, webfetch
from yunshu_gateway.server_tools.mcp_connector import McpConnection, McpError

PUBLIC = "93.184.216.34"


def fake_dns(monkeypatch, *answers):
    """socket.getaddrinfo returns the next answer per call (the last repeats)."""
    calls: list[str] = []

    def gai(host, port, *a, **k):
        calls.append(host)
        ip = answers[min(len(calls) - 1, len(answers) - 1)]
        fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(fam, socket.SOCK_STREAM, 0, "", (ip, port or 0))]

    monkeypatch.setattr(socket, "getaddrinfo", gai)
    return calls


def rpc(rid, result=None, **extra):
    return {"jsonrpc": "2.0", "id": rid, "result": result or {}, **extra}


def mcp_handler(log, *, init_reply=None, status=200):
    def handler(req: httpx.Request) -> httpx.Response:
        log.append(req)
        body = json.loads(req.content or b"{}") if req.method == "POST" else {}
        if req.method == "DELETE":
            return httpx.Response(200)
        if "id" not in body:
            return httpx.Response(202)
        if status != 200:
            return httpx.Response(status)
        if body["method"] == "initialize":
            return httpx.Response(
                200, json=init_reply or rpc(body["id"], {"serverInfo": {"name": "s"}})
            )
        return httpx.Response(200, json=rpc(body["id"], {"tools": []}))

    return handler


def conn_for(handler, url="https://example.com/mcp", **kw):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return McpConnection(url, client=client, timeout=kw.pop("timeout", 5), **kw), client


# ── B19 url parsing ───────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "url",
    [
        "https://example.com:abc/a",
        "http://[broken/a",
        "https://example.com:0/a",
        "https://example.com:65536/a",
        "https://u:p@example.com/a",
        "https://@example.com/a",
        "https:///a",
        "https://exa\x00mple.com/a",
        "https://example.com/a\r\nX: y",
        "https://exa mple.com/",
    ],
)
async def test_b19_bad_urls_are_url_not_allowed(url):
    with pytest.raises(UrlNotAllowedError):
        netguard.parse_url(url)
    with pytest.raises(UrlNotAllowedError):
        await netguard.resolve_target(url, allow_private=True)


async def test_b19_web_fetch_and_mcp_report_bad_urls():
    with pytest.raises(webfetch.FetchError) as e:
        await webfetch.fetch_url("https://example.com:abc/a")
    assert e.value.code == "url_not_allowed"
    with pytest.raises(McpError):
        await McpConnection("http://[broken/a").connect()


# ── B13 pinning ───────────────────────────────────────────────────────────────
async def test_b13_mcp_pins_checked_ip_on_every_method(monkeypatch):
    calls = fake_dns(monkeypatch, PUBLIC, "127.0.0.1")  # second lookup would rebind
    log: list[httpx.Request] = []
    c, client = conn_for(mcp_handler(log), allow_private=False)
    async with c:
        await c.list_tools()
    await client.aclose()
    assert {r.url.host for r in log} == {PUBLIC}
    assert {r.headers["host"] for r in log} == {"example.com"}
    assert {r.method for r in log} >= {"POST"}
    assert len(calls) == 1  # resolved once per connection


async def test_b13_private_policy(monkeypatch):
    fake_dns(monkeypatch, "127.0.0.1")
    c, _ = conn_for(mcp_handler([]), allow_private=False)
    with pytest.raises(McpError, match="not allowed"):
        await c.connect()
    log: list[httpx.Request] = []
    c, _ = conn_for(mcp_handler(log), allow_private=True)
    await c.connect()
    assert log and log[0].url.host == "127.0.0.1"


async def test_b13_own_client_ignores_proxy_env(monkeypatch):
    fake_dns(monkeypatch, PUBLIC)
    c = McpConnection("https://example.com/mcp", allow_private=False)
    c._client = None
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:1")
    seen = {}
    real = httpx.AsyncClient

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    def spy(*a, **k):  # noqa: F811
        seen.update(k)
        return real(transport=httpx.MockTransport(mcp_handler([], status=500)), **k)

    monkeypatch.setattr(mcp_connector.httpx, "AsyncClient", spy)
    with pytest.raises(McpError):
        await asyncio.wait_for(c.connect(), 3)
    assert seen.get("trust_env") is False


# ── B14 media download ────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "ip",
    ["::ffff:127.0.0.1", "::ffff:10.0.0.1", "100.64.0.1", "224.0.0.1", "240.0.0.1"],
)
async def test_b14_media_blocks_mapped_cgnat_multicast_reserved(
    monkeypatch, tmp_path, ip
):
    from yunshu_engine.vlm_engine import _resolve_media_target

    fake_dns(monkeypatch, ip)
    with pytest.raises(ValueError, match="SSRF blocked"):
        await _resolve_media_target("http://img.example.com/a.png")


async def test_b14_media_download_pins_and_caps(monkeypatch, tmp_path):
    fake_dns(monkeypatch, PUBLIC, "127.0.0.1")
    seen: list[httpx.Request] = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, content=b"x" * 100)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    dest = tmp_path / "a.png"
    n = await netguard.download_to_file(
        "https://img.example.com/a.png",
        str(dest),
        max_bytes=1000,
        timeout=5,
        allow_private=False,
        client=client,
    )
    assert n == 100 and dest.read_bytes() == b"x" * 100
    assert seen[0].url.host == PUBLIC and seen[0].headers["host"] == "img.example.com"
    fake_dns(monkeypatch, PUBLIC)
    with pytest.raises(ValueError, match="size limit"):
        await netguard.download_to_file(
            "https://img.example.com/a.png",
            str(dest),
            max_bytes=10,
            timeout=5,
            allow_private=False,
            client=client,
        )
    await client.aclose()


async def test_b14_media_redirect_to_private_is_rechecked(monkeypatch, tmp_path):
    fake_dns(monkeypatch, PUBLIC, "169.254.169.254")
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(302, headers={"location": "http://meta.example/x"})
        )
    )
    with pytest.raises(ValueError, match="SSRF blocked"):
        await netguard.download_to_file(
            "https://img.example.com/a.png",
            str(tmp_path / "a"),
            max_bytes=10,
            timeout=5,
            allow_private=False,
            client=client,
        )


# ── B15 reply validation ──────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "reply",
    [
        {"jsonrpc": "2.0", "id": 999, "result": {"serverInfo": {"name": "wrong-id"}}},
        {"jsonrpc": "2.0", "result": {}},  # missing id (a notification-shaped reply)
        {"jsonrpc": "2.0", "id": "1", "result": {}},  # string id
        {"jsonrpc": "2.0", "method": "notifications/x", "params": {}},
        {"jsonrpc": "2.0", "id": 1, "result": {}, "error": {"message": "m"}},
        {"jsonrpc": "2.0", "id": 1},
        {"id": 1, "result": {}},  # no jsonrpc marker
        ["scalar", 3],
        [{"jsonrpc": "2.0", "id": 5, "result": {}}],
        "scalar",
        7,
        None,
    ],
)
async def test_b15_invalid_replies_raise_mcp_error(monkeypatch, reply):
    fake_dns(monkeypatch, PUBLIC)
    c, _ = conn_for(lambda r: httpx.Response(200, json=reply), allow_private=False)
    with pytest.raises(McpError):
        await c.connect()


async def test_b15_valid_forms_still_work(monkeypatch):
    fake_dns(monkeypatch, PUBLIC)

    def handler(req):
        body = json.loads(req.content)
        if "id" not in body:
            return httpx.Response(202)
        ok = rpc(body["id"], {"serverInfo": {"name": "good"}})
        if req.headers.get("x-mode") == "never":
            return httpx.Response(500)
        # batch reply plus an SSE reply with an unrelated notification first
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                'data: {"jsonrpc":"2.0","method":"notifications/message"}\n\n'
                'data: {"jsonrpc":"2.0","id":77,"result":{}}\n\n'
                f"data: {json.dumps(ok)}\n\n"
            ).encode(),
        )

    c, _ = conn_for(handler, allow_private=False)
    await c.connect()
    assert c.server_info == {"name": "good"}
    c2, _ = conn_for(
        lambda r: (
            httpx.Response(
                200,
                json=[
                    {"jsonrpc": "2.0", "id": 9, "result": {}},
                    rpc(1, {"serverInfo": {"name": "b"}}),
                ]
                if json.loads(r.content).get("id")
                else {},
            )
            if json.loads(r.content).get("id")
            else httpx.Response(202)
        ),
        allow_private=False,
    )
    await c2.connect()
    assert c2.server_info == {"name": "b"}


# ── B16 cleanup ───────────────────────────────────────────────────────────────
async def test_b16_failed_initialize_closes_owned_client(monkeypatch):
    fake_dns(monkeypatch, PUBLIC)
    made: list[httpx.AsyncClient] = []
    real = httpx.AsyncClient

    def factory(*a, **k):
        cl = real(transport=httpx.MockTransport(mcp_handler([], status=400)), **k)
        made.append(cl)
        return cl

    monkeypatch.setattr(mcp_connector.httpx, "AsyncClient", factory)
    c = McpConnection("https://example.com/mcp", allow_private=False, timeout=5)
    with pytest.raises(McpError):
        await c.connect()
    assert made and made[0].is_closed and c._client is None


async def test_b16_failed_initialize_keeps_injected_client(monkeypatch):
    fake_dns(monkeypatch, PUBLIC)
    c, client = conn_for(mcp_handler([], status=400), allow_private=False)
    with pytest.raises(McpError):
        await c.connect()
    assert not client.is_closed  # not ours to close


async def test_b16_cancel_during_connect_cleans_up(monkeypatch):
    fake_dns(monkeypatch, PUBLIC)
    started = asyncio.Event()

    async def handler(req):
        started.set()
        await asyncio.sleep(30)

    c, _ = conn_for(handler, allow_private=False, timeout=60)
    t = asyncio.create_task(c.connect())
    await started.wait()
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert c._client is None or c._client.is_closed or not c._own


def sse_server_transport(endpoint_event: bytes, tail: bytes = b"", hang=False):
    """Streamable POST /mcp is 405 (-> legacy); GET returns an SSE stream."""

    def handler(req):
        if req.method == "GET":
            if hang:

                async def gen():
                    yield endpoint_event
                    await asyncio.sleep(30)

                return httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, content=gen()
                )
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=endpoint_event + tail,
            )
        if req.url.path == "/mcp":
            return httpx.Response(405)
        return httpx.Response(202)

    return handler


async def test_b16_legacy_no_endpoint_event_leaves_no_task(monkeypatch):
    fake_dns(monkeypatch, PUBLIC)

    async def never():
        await asyncio.sleep(30)
        yield b""

    def handler(req):
        if req.method == "GET":
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=never()
            )
        return httpx.Response(405)

    c, _ = conn_for(handler, allow_private=False, timeout=0.5)
    with pytest.raises(McpError):
        await c.connect()
    assert c._legacy_task is None or c._legacy_task.done()


async def test_b16_legacy_eof_wakes_waiters_and_drops_them(monkeypatch):
    fake_dns(monkeypatch, PUBLIC)
    h = sse_server_transport(b"event: endpoint\ndata: /messages?s=1\n\n")
    c, _ = conn_for(h, allow_private=False, timeout=10)
    t0 = time.monotonic()
    with pytest.raises(McpError):
        await c.connect()  # initialize waits on a stream that already hit EOF
    assert time.monotonic() - t0 < 3  # woken by EOF, not by the 10 s deadline
    assert not c._legacy_waiters


# ── B17 legacy endpoint origin ────────────────────────────────────────────────
async def test_b17_cross_origin_legacy_endpoint_gets_no_request(monkeypatch):
    fake_dns(monkeypatch, PUBLIC)
    log: list[httpx.Request] = []

    def handler(req):
        log.append(req)
        return sse_server_transport(
            b"event: endpoint\ndata: https://evil.example.net/steal\n\n", hang=True
        )(req)

    c, _ = conn_for(handler, allow_private=False, timeout=3, authorization="sekret")
    with pytest.raises(McpError, match="origin"):
        await c.connect()
    assert not [r for r in log if r.url.host == "evil.example.net"]
    assert not [r for r in log if r.url.path == "/steal"]


async def test_b17_relative_endpoint_still_works(monkeypatch):
    fake_dns(monkeypatch, PUBLIC)
    log: list[httpx.Request] = []

    def handler(req):
        log.append(req)
        return sse_server_transport(
            b"event: endpoint\ndata: /messages?s=1\n\n"
            b'event: message\ndata: {"jsonrpc":"2.0","id":2,"result":{"serverInfo":{"name":"L"}}}\n\n',
            hang=False,
        )(req)

    # the stream replies before the POST registers; use a hanging stream feeding the reply late
    async def gen():
        yield b"event: endpoint\ndata: /messages?s=1\n\n"
        for _ in range(100):
            await asyncio.sleep(0.02)
            if any(r.url.path == "/messages" for r in log):
                break
        yield b'event: message\ndata: {"jsonrpc":"2.0","id":2,"result":{"serverInfo":{"name":"L"}}}\n\n'
        await asyncio.sleep(30)

    def handler2(req):
        log.append(req)
        if req.method == "GET":
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=gen()
            )
        if req.url.path == "/mcp":
            return httpx.Response(405)
        return httpx.Response(202)

    c, _ = conn_for(handler2, allow_private=False, timeout=5)
    await c.connect()
    assert c.server_info == {"name": "L"}
    assert {r.url.host for r in log} == {PUBLIC}
    await c.close()


# ── B18 deadlines and byte budgets ────────────────────────────────────────────
async def test_b18_mcp_slow_dns_is_inside_deadline(monkeypatch):
    def slow(host, port, *a, **k):
        time.sleep(2)
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (PUBLIC, port))]

    monkeypatch.setattr(socket, "getaddrinfo", slow)
    c, _ = conn_for(mcp_handler([]), allow_private=False, timeout=0.5)
    t0 = time.monotonic()
    with pytest.raises(McpError, match="timed out"):
        await c.connect()
    assert time.monotonic() - t0 < 1.5


async def test_b18_mcp_json_reply_byte_budget(monkeypatch):
    fake_dns(monkeypatch, PUBLIC)
    big = json.dumps(rpc(1, {"serverInfo": {"name": "a" * 50000}})).encode()
    c, _ = conn_for(
        lambda r: httpx.Response(
            200, content=big, headers={"content-type": "application/json"}
        ),
        allow_private=False,
        max_bytes=10_000,
    )
    with pytest.raises(McpError, match="too large"):
        await c.connect()


async def test_b18_mcp_unterminated_sse_event_budget(monkeypatch):
    fake_dns(monkeypatch, PUBLIC)

    async def endless():
        while True:
            yield b"data: " + b"a" * 1000  # never a blank line

    c, _ = conn_for(
        lambda r: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=endless()
        ),
        allow_private=False,
        max_bytes=20_000,
        timeout=5,
    )
    with pytest.raises(McpError, match="too large"):
        await c.connect()


async def test_b18_mcp_legacy_stream_event_budget(monkeypatch):
    fake_dns(monkeypatch, PUBLIC)

    async def gen():
        yield b"event: endpoint\ndata: /messages\n\n"
        while True:
            yield b"data: " + b"a" * 1000

    def handler(req):
        if req.method == "GET":
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=gen()
            )
        return httpx.Response(405) if req.url.path == "/mcp" else httpx.Response(202)

    c, _ = conn_for(handler, allow_private=False, max_bytes=20_000, timeout=5)
    t0 = time.monotonic()
    with pytest.raises(McpError):
        await c.connect()
    assert time.monotonic() - t0 < 4


async def test_b18_web_fetch_slow_dns_has_total_deadline(monkeypatch):
    monkeypatch.setenv("YUNSHU_WEB_FETCH_TIMEOUT", "1")

    def slow(host, port, *a, **k):
        time.sleep(3)
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (PUBLIC, port))]

    monkeypatch.setattr(socket, "getaddrinfo", slow)
    t0 = time.monotonic()
    with pytest.raises(webfetch.FetchError) as e:
        await webfetch.fetch_url("https://example.com/")
    assert e.value.code == "url_not_accessible"
    assert time.monotonic() - t0 < 2.5


async def test_b18_web_fetch_redirect_chain_shares_one_deadline(monkeypatch):
    monkeypatch.setenv("YUNSHU_WEB_FETCH_TIMEOUT", "1")
    fake_dns(monkeypatch, PUBLIC)

    async def handler(req):
        await asyncio.sleep(0.4)
        return httpx.Response(302, headers={"location": "/n"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    t0 = time.monotonic()
    with pytest.raises(webfetch.FetchError):
        await webfetch.fetch_url("https://example.com/", client=client)
    assert time.monotonic() - t0 < 1.8
