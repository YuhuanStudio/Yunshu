"""Official independent-index contracts, with no live network or model."""

import logging

import httpx
import pytest

from yunshu_engine import settings
from yunshu_gateway.server_tools import search
from yunshu_gateway.server_tools.metasearch import Marginalia, Mojeek


@pytest.mark.parametrize("name", ["mojeek", "marginalia"])
def test_open_index_requires_explicit_key(monkeypatch, name):
    settings.clear_overrides()
    search.set_provider_for_tests(None)
    for provider in search.PROVIDER_ORDER:
        monkeypatch.delenv(f"YUNSHU_{provider.upper()}_API_KEY", raising=False)
    monkeypatch.setenv("YUNSHU_WEB_SEARCH_PROVIDER", "auto")
    assert name not in {p.name for p in search.get_provider().providers}
    monkeypatch.setenv(
        f"YUNSHU_{name.upper()}_API_KEY",
        "public" if name == "marginalia" else "fixture-key",
    )
    assert name in {p.name for p in search.get_provider().providers}
    monkeypatch.setenv("YUNSHU_WEB_SEARCH_PROVIDER", name)
    assert search.get_provider().name == name
    assert settings.REGISTRY[f"YUNSHU_{name.upper()}_API_KEY"].secret


async def test_marginalia_contract_and_safe_search():
    def reply(req):
        assert req.url.host == "api2.marginalia-search.com"
        assert req.headers["API-Key"] == "fixture-key"
        assert req.url.params == httpx.QueryParams(query="sqlite", count=6, nsfw=1)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "SQLite",
                        "url": "https://sqlite.org/",
                        "description": "<b>Database</b> guide",
                    },
                    {"title": "missing URL"},
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        rows = await Marginalia("fixture-key").search(
            "sqlite", limit=3, client=client, options={"safe_search": True}
        )
    assert len(rows) == 1 and rows[0].snippet == "Database guide"


async def test_mojeek_contract_and_error_redaction():
    def reply(req):
        assert req.url.host == "api.mojeek.com"
        for key, value in {
            "fmt": "json",
            "t": "10",
            "api_key": "fixture-key",
            "safe": "1",
            "lb": "EN",
            "rb": "TW",
            "since": "20261001",
            "before": "20261009",
        }.items():
            assert req.url.params[key] == value
        return httpx.Response(
            200,
            json={
                "response": {
                    "status": "OK",
                    "results": [
                        {
                            "title": "Guide",
                            "url": "https://example.org/",
                            "desc": "An exact answer.",
                            "date": "Thu Oct 8 00:00:00 2026",
                        }
                    ],
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        rows = await Mojeek("fixture-key").search(
            "guide",
            limit=20,
            client=client,
            options={
                "safe_search": True,
                "language": "en",
                "start_date": "2026-10-01",
                "end_date": "2026-10-08",
            },
            user_location={"country": "tw"},
        )
    assert rows[0].snippet == "An exact answer." and rows[0].page_age
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200, json={"response": {"status": "ERROR: fixture-key"}}
            )
        )
    ) as client:
        with pytest.raises(search.SearchError, match="API error") as caught:
            await Mojeek("fixture-key").search("q", limit=1, client=client)
    assert "fixture-key" not in str(caught.value)


@pytest.mark.parametrize("provider", [Marginalia, Mojeek])
async def test_open_index_rate_limit(provider):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(429))
    ) as client:
        with pytest.raises(search.SearchError) as caught:
            await provider("fixture-key").search("q", limit=1, client=client)
    assert caught.value.code == "too_many_requests"


async def test_mojeek_search_does_not_log_query_or_key(monkeypatch, caplog):
    monkeypatch.setenv("YUNSHU_WEB_SEARCH_PROVIDER", "mojeek")
    monkeypatch.setenv("YUNSHU_MOJEEK_API_KEY", "private-fixture-key")
    search.set_provider_for_tests(None)
    search._search_cache.clear()
    with caplog.at_level(logging.INFO, logger="httpx"):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(
                    200, json={"response": {"status": "OK", "results": []}}
                )
            )
        ) as client:
            await search.run_search(
                "private-query-unique", client=client, enrich_results=False
            )
    assert "private-query-unique" not in caplog.text
    assert "private-fixture-key" not in caplog.text
