"""Image vertical adapters use transport fixtures, not live providers."""

import httpx
import pytest

from yunshu_gateway.server_tools.search import Brave, SearXNG
from yunshu_gateway.tavily import images


@pytest.mark.parametrize(
    "provider", [Brave("fixture-key"), SearXNG("https://search.example")]
)
async def test_image_vertical_params_and_domain_filters(provider, monkeypatch):
    monkeypatch.setattr(images, "get_provider", lambda: provider)
    seen = []

    def respond(req):
        seen.append(req)
        assert req.url.params["q"] == "fixture images"
        rows = [
            {
                "url": "https://docs.example/page",
                "img_src": "https://cdn.example/a.png",
                "properties": {"url": "https://cdn.example/a.png"},
            },
            {
                "url": "https://docs.example/page",
                "img_src": "https://cdn.example/a.png",
                "properties": {"url": "https://cdn.example/a.png"},
            },
            {
                "url": "https://excluded.example/page",
                "img_src": "https://cdn.example/b.png",
                "properties": {"url": "https://cdn.example/b.png"},
            },
            {
                "url": "https://docs.example/page",
                "img_src": "file:///secret",
                "properties": {"url": "file:///secret"},
            },
        ]
        return httpx.Response(200, json={"results": rows})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await images.run_images(
            "fixture images",
            client=client,
            allowed_domains=["docs.example"],
            blocked_domains=["excluded.example"],
            options={"safe_search": True, "language": "en"},
            user_location={"country": "US"},
        )
    assert result == [{"url": "https://cdn.example/a.png"}]
    if provider.name == "brave":
        assert seen[0].headers["x-subscription-token"] == "fixture-key"
        assert seen[0].url.params["safesearch"] == "strict"
        assert seen[0].url.params["country"] == "US"
    else:
        assert seen[0].url.params["categories"] == "images"
        assert seen[0].url.params["safesearch"] == "2"


async def test_image_failure_and_disabled_provider_are_optional(monkeypatch):
    monkeypatch.setattr(
        images, "get_provider", lambda: SearXNG("https://search.example")
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(429))
    ) as client:
        assert await images.run_images("fixture", client=client) == []
    monkeypatch.setattr(images, "get_provider", lambda: None)
    assert await images.run_images("fixture") == []


async def test_service_requests_images_only_when_requested():
    from yunshu_gateway.server_tools.search import SearchResult
    from yunshu_gateway.tavily.models import SearchRequest
    from yunshu_gateway.tavily.service import TavilyService

    calls = []

    async def search(*args, **kwargs):
        return "fixture", [SearchResult("Title", "https://example.org", "snippet")]

    async def image_search(query, **kwargs):
        calls.append((query, kwargs))
        return [{"url": "https://example.org/a.png"}]

    async def fetch(url, **kwargs):
        from yunshu_gateway.server_tools.webfetch import FetchResult

        return FetchResult(url, "", "fixture page", "text/plain", "")

    srv = TavilyService(searcher=search, image_searcher=image_search, fetcher=fetch)
    await srv.search(SearchRequest(query="fixture", search_depth="ultra-fast"))
    assert not calls
    result = await srv.search(
        SearchRequest(query="fixture", search_depth="ultra-fast", include_images=True)
    )
    assert result["images"] == [{"url": "https://example.org/a.png"}]
    assert len(calls) == 1
