"""CPU tests: Chromium resources never bypass the origin fetch guard."""

from types import SimpleNamespace

import pytest

from yunshu_engine import settings
from yunshu_gateway.server_tools.webfetch import FetchError, FetchResult
from yunshu_gateway.tavily.models import CrawlRequest, ExtractRequest
from yunshu_gateway.tavily.render import Resources
from yunshu_gateway.tavily.service import TavilyService


class Route:
    def __init__(self, url, method="GET", resource_type="script"):
        self.request = SimpleNamespace(
            url=url, method=method, resource_type=resource_type
        )
        self.aborted = False
        self.fulfilled = None

    async def abort(self):
        self.aborted = True

    async def fulfill(self, **kwargs):
        self.fulfilled = kwargs


@pytest.mark.parametrize(
    "url,method,kind",
    [
        ("http://127.0.0.1/private", "GET", "fetch"),
        ("https://other.example/script.js", "GET", "script"),
        ("http://example.org/script.js", "GET", "script"),
        ("https://example.org:444/script.js", "GET", "script"),
        ("https://example.org/send", "POST", "fetch"),
        ("https://example.org/image.png", "GET", "image"),
        ("file:///etc/passwd", "GET", "document"),
    ],
)
async def test_resource_policy_rejects_before_fetch(url, method, kind):
    async def no_network(*args, **kwargs):
        pytest.fail("rejected resource reached fetcher")

    resources = Resources("https://example.org/", True, 2, no_network)
    route = Route(url, method, kind)
    await resources.handle(route)
    assert route.aborted and not route.fulfilled


async def test_resources_guarded_bounded_and_redirects_refused():
    calls = []

    async def fetch(url, **kwargs):
        calls.append(kwargs)
        return FetchResult(url, "", "window.fixture=42", "text/plain", "")

    resources = Resources("https://example.org/", True, 2, fetch)
    route = Route("https://example.org/a.js")
    await resources.handle(route)
    assert route.fulfilled["body"] == "window.fixture=42"
    assert calls[0]["automated"] and calls[0]["cache_namespace"] == "tavily-render-raw"
    assert calls[0]["timeout_seconds"] <= 2
    with pytest.raises(FetchError, match="redirect refused"):
        calls[0]["redirect_guard"]("https://other.example/secret")
    resources.count = 32
    rejected = Route(route.request.url)
    await resources.handle(rejected)
    assert rejected.aborted and len(calls) == 1
    resources.count = 0

    async def redirect(url, **kwargs):
        return FetchResult(
            "https://example.org/other", "", "wrong base", "text/plain", ""
        )

    resources.fetcher = redirect
    await resources.handle(rejected)
    assert not rejected.fulfilled


async def test_resource_truncated_or_denied_body_is_not_executed():
    for denied in (False, True):

        async def fetch(url, **kwargs):
            if denied:
                raise FetchError("url_not_allowed", "DNS guard rejected")
            return FetchResult(
                url, "", "partial script", "text/plain", "", truncated=True
            )

        route = Route("https://example.org/a.js")
        await Resources("https://example.org/", False, 2, fetch).handle(route)
        assert route.aborted and not route.fulfilled


async def test_advanced_extract_crawl_map_render_only_shells(monkeypatch):
    original = settings.get
    monkeypatch.setattr(
        settings,
        "get",
        lambda key: True if key == "YUNSHU_WEB_RENDER" else original(key),
    )
    calls = []

    async def fetch(url, **kwargs):
        return FetchResult(
            url, "", "Loading", "text/html", "", metadata={"js_shell": True}
        )

    async def renderer(url, **kwargs):
        calls.append(kwargs)
        return FetchResult(
            url,
            "Rendered",
            "A genuine rendered article " * 20,
            "text/html",
            "",
            metadata={"links": []},
        )

    srv = TavilyService(fetcher=fetch, renderer=renderer)
    try:
        basic = await srv.extract(ExtractRequest(urls=["https://example.org"]))
        assert basic["results"][0]["raw_content"] == "Loading" and not calls
        advanced = await srv.extract(
            ExtractRequest(urls=["https://example.org"], extract_depth="advanced")
        )
        assert "rendered article" in advanced["results"][0]["raw_content"]
        assert calls[-1]["automated"] is False
        for map_only in (False, True):
            await srv.crawl(
                CrawlRequest(url="https://example.org", extract_depth="advanced"),
                map_only=map_only,
            )
            assert calls[-1]["automated"] is True

        async def unavailable(*args, **kwargs):
            raise FetchError("unavailable", "missing browser")

        srv.renderer = unavailable
        fallback = await srv.extract(
            ExtractRequest(urls=["https://example.org"], extract_depth="advanced")
        )
        assert fallback["results"][0]["raw_content"] == "Loading"
    finally:
        monkeypatch.undo()


async def test_rejected_static_fetch_never_starts_browser():
    async def fetch(*args, **kwargs):
        raise FetchError("url_not_allowed", "robots denied")

    async def renderer(*args, **kwargs):
        pytest.fail("browser bypassed static fetch rejection")

    srv = TavilyService(fetcher=fetch, renderer=renderer)
    with pytest.raises(FetchError):
        await srv.fetched(
            "https://example.org", automated=True, timeout=3, advanced=True
        )


async def test_redirect_guard_runs_before_next_origin_connection(monkeypatch):
    import httpx

    from yunshu_engine.netguard import Target
    from yunshu_gateway.server_tools import webfetch
    from yunshu_gateway.server_tools.research.fetcher import page

    resolved, requests = [], []

    async def resolve(url, **kwargs):
        resolved.append(url)
        return Target(
            url=url, host="example.org", port=443, scheme="https", ip="93.184.216.34"
        )

    def respond(req):
        requests.append(str(req.url))
        return httpx.Response(
            302, headers={"location": "https://other.example/private"}
        )

    def guard(target):
        raise FetchError("url_not_allowed", "redirect refused before connection")

    monkeypatch.setattr(webfetch, "resolve_target", resolve)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(FetchError, match="before connection"):
            await page(
                "https://example.org/redirect-probe",
                automated=False,
                redirect_guard=guard,
                client=client,
                timeout=3,
            )
    assert resolved == ["https://example.org/redirect-probe"]
    assert len(requests) == 1


async def test_cached_redirect_cannot_bypass_render_guard(monkeypatch):
    from yunshu_gateway.server_tools.research import fetcher
    from yunshu_gateway.server_tools.research.cache import PageCache

    cache = PageCache()
    cache.put(
        "raw::https://example.org/",
        FetchResult(
            "https://other.example/", "", "redirected raw body", "text/plain", ""
        ),
    )
    monkeypatch.setattr(fetcher, "pages", cache)

    def guard(target):
        raise FetchError("url_not_allowed", "cached redirect refused")

    with pytest.raises(FetchError, match="cached redirect"):
        await fetcher.page(
            "https://example.org/",
            automated=False,
            cache_namespace="raw",
            redirect_guard=guard,
        )
