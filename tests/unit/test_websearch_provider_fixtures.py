"""Frozen parser contracts: no network and no reliance on parser-generated fixtures."""

import json
from pathlib import Path

import httpx
import pytest

from yunshu_engine import settings
from yunshu_gateway.server_tools import search
from yunshu_gateway.server_tools.metasearch import Mwmbl

FIXTURES = json.loads(
    (
        Path(__file__).resolve().parents[1] / "fixtures/websearch/providers.json"
    ).read_text()
)


@pytest.mark.parametrize(
    "name",
    [
        "ddg_html",
        "wikipedia",
        "mwmbl",
        "brave",
        "serper",
        "exa",
        "tavily",
        "perplexity",
        "searxng",
    ],
)
async def test_provider_parser(name):
    classes = {
        "ddg_html": search.DuckDuckGo,
        "wikipedia": search.Wikipedia,
        "mwmbl": Mwmbl,
        "brave": search.Brave,
        "serper": search.Serper,
        "exa": search.Exa,
        "tavily": search.Tavily,
        "perplexity": search.Perplexity,
        "searxng": search.SearXNG,
    }
    provider = (
        classes[name]()
        if name in ("ddg_html", "wikipedia", "mwmbl")
        else classes[name](
            "https://fixture.example" if name == "searxng" else "fixture-key"
        )
    )
    search.DuckDuckGo._blocked_until = search.DuckDuckGo._next = 0

    def reply(request):
        sample = FIXTURES[name]
        return (
            httpx.Response(sample["status"], text=sample["body"])
            if name == "ddg_html"
            else httpx.Response(200, json=sample)
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        rows = await provider.search("guide", limit=5, client=client)
    assert len(rows) == 1 and rows[0].title == "Guide title"
    assert rows[0].snippet == "An exact answer." and rows[0].url.startswith("https://")


async def test_pushdown_and_cache_partition(monkeypatch, tmp_path):
    settings.set_override(
        "YUNSHU_WEB_SEARCH_HEALTH_FILE", str(tmp_path / "health.json")
    )
    captured = []

    def reply(request):
        captured.append(request)
        name = "serper" if request.url.host == "google.serper.dev" else "brave"
        sample = FIXTURES[name]
        if request.url.path.endswith("/news") and name == "serper":
            sample = {"news": sample["organic"]}
        if request.url.path.endswith("/news/search") and name == "brave":
            sample = sample["web"]
        return httpx.Response(200, json=sample)

    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
            options = {
                "topic": "news",
                "time_range": "week",
                "language": "en",
                "safe_search": True,
            }
            for provider in (search.Brave("key"), search.Serper("key")):
                await provider.search(
                    "q",
                    limit=5,
                    client=client,
                    options=options,
                    user_location={"country": "tw"},
                )
        assert captured[0].url.path.endswith("/news/search")
        assert (
            captured[0].url.params["freshness"] == "pw"
            and captured[0].url.params["country"] == "tw"
        )
        body = json.loads(captured[1].content)
        assert (
            captured[1].url.path == "/news"
            and body["tbs"] == "qdr:w"
            and body["gl"] == "tw"
        )
    finally:
        settings.clear_overrides()


async def test_ddg_legitimate_empty_is_distinct_from_parser_failure():
    search.DuckDuckGo._blocked_until = search.DuckDuckGo._next = 0
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, text="<p>No results found.</p>")
        )
    ) as client:
        assert await search.DuckDuckGo().search("q", limit=5, client=client) == []
    search.DuckDuckGo._next = 0
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, text="<html>Changed layout</html>")
        )
    ) as client:
        with pytest.raises(search.SearchError, match="layout"):
            await search.DuckDuckGo().search("q", limit=5, client=client)


def test_mwmbl_is_explicit_opt_in(monkeypatch):
    search.set_provider_for_tests(None)
    monkeypatch.setenv("YUNSHU_WEB_SEARCH_PROVIDER", "auto")
    monkeypatch.setenv("YUNSHU_WEB_KEYLESS", "1")
    monkeypatch.setenv("YUNSHU_WEB_MWMBL", "0")
    assert "mwmbl" not in {
        provider.name for provider in search.get_provider().providers
    }
    monkeypatch.setenv("YUNSHU_WEB_MWMBL", "1")
    assert "mwmbl" in {provider.name for provider in search.get_provider().providers}
    monkeypatch.setenv("YUNSHU_WEB_MWMBL", "0")
    monkeypatch.setenv("YUNSHU_WEB_SEARCH_PROVIDER", "mwmbl")
    assert search.get_provider().name == "mwmbl"


async def test_metasearch_returns_after_first_rows_plus_grace(tmp_path, monkeypatch):
    import asyncio
    import time

    import httpx

    from yunshu_gateway.server_tools import metasearch as ms
    from yunshu_gateway.server_tools.search import SearchProvider, SearchResult

    monkeypatch.setenv("YUNSHU_WEB_SEARCH_HEALTH_FILE", str(tmp_path / "h.json"))
    ms._health.clear()

    class Fast(SearchProvider):
        name = "fast_fake"

        async def search(self, query, *, limit, client, **kw):
            await asyncio.sleep(0.01)
            return [SearchResult("a", "https://a.example/", "alpha")]

    class Slow(SearchProvider):
        name = "slow_fake"

        async def search(self, query, *, limit, client, **kw):
            await asyncio.sleep(5)
            return [SearchResult("b", "https://b.example/", "beta")]

    chain = ms.Metasearch([Fast(), Slow()])
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        rows = await chain.search(
            "q", limit=5, client=client, options={"serp_grace": 0.05}
        )
    assert time.perf_counter() - t0 < 1.0
    assert [r.url for r in rows] == ["https://a.example/"]
    slow = [h for k, h in ms._health.items() if k.startswith("slow_fake")][0]
    assert slow.consecutive_failures == 0 and not slow.probing
