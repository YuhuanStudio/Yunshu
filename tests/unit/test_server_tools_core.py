"""Server-side tool building blocks: SSRF guard, web_fetch, HTML to text, search providers, MCP client."""

from __future__ import annotations

import http.server
import threading

import httpx
import pytest
from server_tools_helpers import FakeMcpHttp, FakeSearch

from yunshu_gateway.server_tools import mcp_connector, netguard, search, webfetch
from yunshu_gateway.server_tools.mcp_connector import McpConnection, McpError
from yunshu_gateway.server_tools.runtime import ServerToolDef, ServerToolRuntime

# ── SSRF guard ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.1.2.3",
        "192.168.0.9",
        "172.16.5.5",
        "169.254.169.254",
        "100.64.0.1",
        "0.0.0.0",
        "::1",
        "fe80::1",
        "fd00::1",
        "::ffff:127.0.0.1",
        "224.0.0.1",
    ],
)
def test_forbidden_ips(ip):
    assert netguard.is_forbidden_ip(ip)


@pytest.mark.parametrize(
    "ip", ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700:4700::1111"]
)
def test_public_ips_allowed(ip):
    assert not netguard.is_forbidden_ip(ip)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://x.test/",
        "gopher://x",
        "http://user:pw@example.com/",
        "http://",
        "",
        "javascript:1",
    ],
)
def test_bad_urls_rejected(url):
    with pytest.raises(netguard.UrlNotAllowedError):
        netguard.parse_url(url)


async def test_private_targets_blocked_and_allowed():
    with pytest.raises(netguard.UrlNotAllowedError):
        await netguard.resolve_target("http://127.0.0.1:9/", allow_private=False)
    with pytest.raises(netguard.UrlNotAllowedError):
        await netguard.resolve_target(
            "http://169.254.169.254/latest/meta-data", allow_private=False
        )
    with pytest.raises(netguard.UrlNotAllowedError):
        await netguard.resolve_target("http://localhost:9/", allow_private=False)
    t = await netguard.resolve_target("http://127.0.0.1:9/", allow_private=True)
    assert t.ip == "127.0.0.1" and t.port == 9


def test_domain_matches():
    assert netguard.domain_matches("docs.example.com", ["example.com"])
    assert netguard.domain_matches("example.com", ["example.com/path"])
    assert not netguard.domain_matches("notexample.com", ["example.com"])
    assert not netguard.domain_matches("example.com.evil.io", ["example.com"])


# ── HTML to text ──────────────────────────────────────────────────────────────


def test_html_to_text_structure():
    html = """<html><head><title> My  Page </title><style>p{}</style></head><body>
    <script>alert(1)</script><h1>Head</h1><p>Hello <a href="/x">link</a> world.</p>
    <ul><li>one</li><li>two</li></ul><pre>a  b\nc</pre><img alt="pic" src="x.png"></body></html>"""
    title, text = webfetch.html_to_text(html, "https://ex.com/a/b")
    assert title == "My Page"
    assert "alert" not in text and "p{}" not in text
    assert "# Head" in text
    assert "Hello link (https://ex.com/x) world." in text
    assert "- one" in text and "- two" in text
    assert "a b\nc" in text
    assert "[image: pic]" in text


# ── web_fetch against a loopback server ───────────────────────────────────────


class _Site(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        routes = {
            "/page": (
                200,
                "text/html; charset=utf-8",
                b"<html><title>T</title><body><p>Body text</p></body></html>",
            ),
            "/plain": (200, "text/plain", b"plain text here"),
            "/big": (200, "text/plain", b"x" * 5000),
            "/pdf": (200, "application/pdf", b"%PDF-1.4"),
            "/missing": (404, "text/plain", b"nope"),
        }
        if self.path == "/redir":
            self.send_response(302)
            self.send_header("Location", "/page")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/redir-meta":
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        code, ct, body = routes.get(self.path, (404, "text/plain", b"nope"))
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def site():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Site)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


async def test_fetch_blocks_loopback_by_default(site):
    with pytest.raises(webfetch.FetchError) as e:
        await webfetch.fetch_url(site + "/page")
    assert e.value.code == "url_not_allowed"


async def test_fetch_allowed_by_setting(site, monkeypatch):
    monkeypatch.setenv("YUNSHU_WEB_FETCH_ALLOW_PRIVATE", "1")
    r = await webfetch.fetch_url(site + "/page")
    assert r.title == "T" and "Body text" in r.text and r.media_type == "text/plain"
    r = await webfetch.fetch_url(site + "/redir")
    assert r.url.endswith("/page") and r.redirects
    r = await webfetch.fetch_url(site + "/plain")
    assert r.text == "plain text here"


async def test_fetch_errors(site, monkeypatch):
    monkeypatch.setenv("YUNSHU_WEB_FETCH_ALLOW_PRIVATE", "1")
    for path, code in (
        ("/missing", "url_not_accessible"),
        ("/pdf", "unsupported_content_type"),
    ):
        with pytest.raises(webfetch.FetchError) as e:
            await webfetch.fetch_url(site + path)
        assert e.value.code == code
    with pytest.raises(webfetch.FetchError) as e:
        await webfetch.fetch_url("http://x.test/" + "a" * 300)
    assert e.value.code == "url_too_long"
    with pytest.raises(webfetch.FetchError) as e:
        await webfetch.fetch_url("")
    assert e.value.code == "invalid_tool_input"


async def test_fetch_redirect_to_private_is_blocked(site, monkeypatch):
    # loopback is allowed for the first hop only through the setting; the redirect hop targets
    # the metadata address, which must be refused when private access is off.
    monkeypatch.setenv("YUNSHU_WEB_FETCH_ALLOW_PRIVATE", "0")
    with pytest.raises(webfetch.FetchError) as e:
        await webfetch.fetch_url(site + "/redir-meta")
    assert e.value.code == "url_not_allowed"


async def test_fetch_size_and_text_limits(site, monkeypatch):
    monkeypatch.setenv("YUNSHU_WEB_FETCH_ALLOW_PRIVATE", "1")
    monkeypatch.setenv("YUNSHU_WEB_FETCH_MAX_BYTES", "2048")
    r = await webfetch.fetch_url(site + "/big")
    assert r.truncated and r.bytes_read == 2048
    monkeypatch.setenv("YUNSHU_WEB_FETCH_MAX_BYTES", "2000000")
    r = await webfetch.fetch_url(site + "/big", max_content_chars=1000)
    assert r.truncated and len(r.text) == 1000


async def test_fetch_domain_filters(site, monkeypatch):
    monkeypatch.setenv("YUNSHU_WEB_FETCH_ALLOW_PRIVATE", "1")
    with pytest.raises(webfetch.FetchError) as e:
        await webfetch.fetch_url(site + "/page", allowed_domains=["example.com"])
    assert e.value.code == "url_not_allowed"
    with pytest.raises(webfetch.FetchError) as e:
        await webfetch.fetch_url(site + "/page", blocked_domains=["127.0.0.1"])
    assert e.value.code == "url_not_allowed"


async def test_fetch_disabled(monkeypatch):
    monkeypatch.setenv("YUNSHU_WEB_FETCH", "0")
    with pytest.raises(webfetch.FetchError) as e:
        await webfetch.fetch_url("https://example.com/")
    assert e.value.code == "unavailable"


# ── search providers ──────────────────────────────────────────────────────────


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_searxng_provider():
    def h(req: httpx.Request):
        assert req.url.path == "/search" and req.url.params["format"] == "json"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "A",
                        "url": "https://a.com/x",
                        "content": "<b>alpha</b> text",
                        "publishedDate": "2026-01-01",
                    },
                    {"title": "B", "url": "https://b.org/y", "content": "beta"},
                ]
            },
        )

    async with _client(h) as c:
        rows = await search.SearXNG("http://sx:8080/").search("q", limit=5, client=c)
    assert [r.url for r in rows] == ["https://a.com/x", "https://b.org/y"]
    assert rows[0].snippet == "alpha text" and rows[0].page_age == "2026-01-01"


async def test_brave_tavily_exa_shapes():
    def brave(req):
        assert (
            req.headers["x-subscription-token"] == "K"
            and req.url.params["country"] == "US"
        )
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {"title": "T", "url": "https://t.io", "description": "d"}
                    ]
                }
            },
        )

    async with _client(brave) as c:
        r = await search.Brave("K").search(
            "q", limit=3, user_location={"country": "US"}, client=c
        )
    assert r[0].title == "T"

    def tav(req):
        import json as _j

        body = _j.loads(req.content)
        assert req.headers["authorization"] == "Bearer K" and body[
            "include_domains"
        ] == ["a.com"]
        return httpx.Response(
            200,
            json={"results": [{"title": "T", "url": "https://a.com", "content": "c"}]},
        )

    async with _client(tav) as c:
        r = await search.Tavily("K").search(
            "q", limit=3, allowed_domains=["a.com"], client=c
        )
    assert r[0].snippet == "c"

    def exa(req):
        assert req.headers["x-api-key"] == "K"
        return httpx.Response(
            200,
            json={"results": [{"title": "T", "url": "https://e.com", "text": "txt"}]},
        )

    async with _client(exa) as c:
        r = await search.Exa("K").search("q", limit=3, client=c)
    assert r[0].url == "https://e.com"


async def test_provider_http_errors_map_to_spec_codes():
    async with _client(lambda r: httpx.Response(429)) as c:
        with pytest.raises(search.SearchError) as e:
            await search.Brave("K").search("q", limit=3, client=c)
    assert e.value.code == "too_many_requests"
    async with _client(lambda r: httpx.Response(500)) as c:
        with pytest.raises(search.SearchError) as e:
            await search.Brave("K").search("q", limit=3, client=c)
    assert e.value.code == "unavailable"


def test_provider_selection(monkeypatch):
    search.set_provider_for_tests(None)
    for k in (
        "YUNSHU_SEARXNG_URL",
        "YUNSHU_BRAVE_API_KEY",
        "YUNSHU_TAVILY_API_KEY",
        "YUNSHU_EXA_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)
    assert search.get_provider() is None
    monkeypatch.setenv("YUNSHU_EXA_API_KEY", "e")
    assert search.get_provider().name == "exa"
    monkeypatch.setenv("YUNSHU_SEARXNG_URL", "http://sx")
    assert search.get_provider().name == "searxng"  # auto prefers the private option
    monkeypatch.setenv("YUNSHU_WEB_SEARCH_PROVIDER", "exa")
    assert search.get_provider().name == "exa"
    monkeypatch.setenv("YUNSHU_WEB_SEARCH_PROVIDER", "none")
    assert search.get_provider() is None
    monkeypatch.setenv("YUNSHU_WEB_SEARCH_PROVIDER", "brave")
    assert search.get_provider() is None  # brave chosen but no key


async def test_run_search_validation_and_filters():
    fake = FakeSearch()
    search.set_provider_for_tests(fake)
    try:
        with pytest.raises(search.SearchError) as e:
            await search.run_search("")
        assert e.value.code == "invalid_input"
        with pytest.raises(search.SearchError) as e:
            await search.run_search("x" * 500)
        assert e.value.code == "query_too_long"
        async with httpx.AsyncClient() as c:
            _, rows = await search.run_search(
                "q", allowed_domains=["example.com"], client=c
            )
            assert [r.domain for r in rows] == ["example.com"]
            _, rows = await search.run_search(
                "q", blocked_domains=["example.com"], client=c
            )
            assert [r.domain for r in rows] == ["ml-explore.github.io"]
    finally:
        search.set_provider_for_tests(None)


async def test_no_provider_gives_unavailable_with_hint(monkeypatch):
    search.set_provider_for_tests(None)
    for k in (
        "YUNSHU_SEARXNG_URL",
        "YUNSHU_BRAVE_API_KEY",
        "YUNSHU_TAVILY_API_KEY",
        "YUNSHU_EXA_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)
    rt = ServerToolRuntime([ServerToolDef("web_search", "web_search", "", {}, spec={})])
    out = await rt.execute("web_search", {"query": "hi"})
    await rt.aclose()
    assert (
        out.is_error
        and out.error_code == "unavailable"
        and "YUNSHU_SEARXNG_URL" in (out.hint or "")
    )


# ── MCP connector client ──────────────────────────────────────────────────────


@pytest.mark.parametrize("sse_reply", [False, True])
async def test_mcp_streamable_http(sse_reply):
    srv = FakeMcpHttp(sse_reply=sse_reply)
    try:
        async with McpConnection(srv.url) as c:
            tools = await c.list_tools()
            assert [t.name for t in tools] == ["echo", "add"]
            r = await c.call_tool("add", {"a": 2, "b": 3})
            assert r.text == "5" and not r.is_error
        methods = [m.get("method") for m in srv.requests]
        assert methods[:2] == ["initialize", "notifications/initialized"]
        # the session id is echoed after initialize
        assert (
            srv.headers[-1].get("Mcp-Session-Id") == "s1"
            or srv.headers[2].get("Mcp-Session-Id") == "s1"
        )
    finally:
        srv.stop()


async def test_mcp_legacy_sse_transport():
    srv = FakeMcpHttp(mode="sse")
    try:
        async with McpConnection(srv.url) as c:
            assert (await c.call_tool("echo", {"text": "hey"})).text == "hey"
    finally:
        srv.stop()


async def test_mcp_auth_and_errors():
    srv = FakeMcpHttp(token="sekret")
    try:
        with pytest.raises(McpError) as e:
            await McpConnection(srv.url).connect()
        assert "authorization" in e.value.message.lower()
        async with McpConnection(srv.url, authorization="sekret") as c:
            assert (await c.call_tool("echo", {"text": "ok"})).text == "ok"
            with pytest.raises(McpError):
                await c.call_tool("nope", {})
    finally:
        srv.stop()


async def test_mcp_private_blocked_when_disallowed():
    srv = FakeMcpHttp()
    try:
        with pytest.raises(McpError) as e:
            await McpConnection(srv.url, allow_private=False).connect()
        assert "not allowed" in e.value.message
    finally:
        srv.stop()


async def test_mcp_timeout():
    class Slow(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            import time

            time.sleep(2)

    s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Slow)
    s.daemon_threads = True
    threading.Thread(target=s.serve_forever, daemon=True).start()
    try:
        with pytest.raises(McpError) as e:
            await McpConnection(
                f"http://127.0.0.1:{s.server_address[1]}/mcp", timeout=0.5
            ).connect()
        assert "timed out" in e.value.message
    finally:
        s.shutdown()
        s.server_close()


def test_mcp_module_exports():
    assert mcp_connector.PROTOCOL_VERSION
