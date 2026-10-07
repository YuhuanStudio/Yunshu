"""Offline contract, safety and task lifecycle tests; generation and all network are fakes."""

import asyncio
import base64
import json
import uuid

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_engine import settings
from yunshu_gateway.routers import tavily
from yunshu_gateway.server_tools.search import SearchResult
from yunshu_gateway.server_tools.webfetch import FetchError, FetchResult
from yunshu_gateway.tavily.content import page_content
from yunshu_gateway.tavily.models import (
    ResearchRequest,
    SearchRequest,
)
from yunshu_gateway.tavily.service import TavilyService


@pytest.fixture
def srv():
    calls = []

    async def fetch(url, **kwargs):
        calls.append((url, kwargs))
        if "fail" in url:
            raise FetchError("url_not_accessible", "fixture failure")
        return FetchResult(
            url,
            "Guide",
            "# Guide\nParis weather is sunny, 21 degrees Celsius. " * 20,
            "text/html",
            "2026-10-07",
            published_at="2026-10-07",
            metadata={
                "links": [
                    "https://example.org/docs",
                    "https://example.org/fail",
                    "https://external.org/guide",
                ],
                "images": ["https://example.org/a.png"],
                "language": "en",
            },
        )

    async def search(query, **kwargs):
        calls.append((query, kwargs))
        return "fixture", [
            SearchResult(
                "Weather", "https://example.org", "Paris is sunny.", "2026-10-07"
            )
        ][: kwargs["limit"]]

    async def generate(system, user, schema, max_tokens):
        calls.append(("generate", (system, user, schema, max_tokens)))
        if schema and "queries" in schema["properties"]:
            return json.dumps({"queries": ["Paris weather"]})
        return (
            json.dumps({"answer": "Paris is sunny [1]."})
            if schema
            else "Paris is sunny [1]."
        )

    service = TavilyService(generator=generate, fetcher=fetch, searcher=search)
    service.calls = calls
    return service


@pytest.fixture
def client(srv):
    app = FastAPI()
    app.state.tavily_service = srv
    app.include_router(tavily.router)
    with TestClient(app) as client:
        yield client


def test_every_sdk_search_field_and_empty_values(client, srv):
    body = {
        "query": "Paris weather",
        "search_depth": "basic",
        "chunks_per_source": 3,
        "max_results": 20,
        "topic": "general",
        "time_range": "",
        "start_date": "",
        "end_date": [],
        "days": 7,
        "max_age_hours": 0,
        "fetch_timeout": 1,
        "cache_fallback": True,
        "include_published_date": True,
        "filter_by_published_date": True,
        "include_answer": False,
        "include_raw_content": "text",
        "include_images": True,
        "include_image_descriptions": True,
        "include_favicon": True,
        "include_domains": ["https://www.example.org"],
        "exclude_domains": [],
        "include_domains_mode": "restrict",
        "country": "united states",
        "language": "english",
        "filter_by_language": True,
        "auto_parameters": True,
        "exact_match": False,
        "safe_search": True,
        "api_key": "tvly-fixture",
        "include_usage": True,
        "unknown_future_field": 1,
    }
    response = client.post("/tavily/search", json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["answer"] is None and result["follow_up_questions"] is None
    assert isinstance(result["images"], list) and result["usage"]["credits"] == 1
    uuid.UUID(result["request_id"])
    row = result["results"][0]
    assert 0 <= row["score"] <= 1 and row["images"]
    assert row["favicon"] == "https://example.org/favicon.ico"
    assert "GMT" in row["published_date"] and row["raw_content"]
    assert all(len(chunk) <= 500 for chunk in row["content"].split(" [...] "))
    assert "serp;dur=" in response.headers["server-timing"]
    assert not any(call[0] == "generate" for call in srv.calls)


@pytest.mark.parametrize(
    "body,status",
    [
        ({"query": "q", "include_domains_mode": "prefer"}, 400),
        ({"query": "q", "filter_by_language": True}, 400),
        ({"query": "q", "time_range": "day", "start_date": "2026-10-01"}, 400),
        ({"query": "q", "max_results": 21}, 422),
        ({"query": []}, 422),
    ],
)
def test_validation(client, body, status):
    response = client.post("/tavily/search", json=body)
    assert response.status_code == status
    assert "detail" in response.json()


def test_answer_only_requested_and_zero_results(client, srv):
    response = client.post(
        "/tavily/search", json={"query": "Paris", "include_answer": "advanced"}
    )
    assert response.json()["answer"] == "Paris is sunny [1]."
    assert len([call for call in srv.calls if call[0] == "generate"]) == 1
    response = client.post("/tavily/search", json={"query": "Paris", "max_results": 0})
    assert response.json()["results"] == []


def test_extract_failures_and_single_url(client):
    response = client.post(
        "/tavily/extract",
        json={
            "urls": ["https://example.org", "https://example.org/fail"],
            "format": "text",
            "query": "weather",
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert len(result["results"]) == len(result["failed_results"]) == 1
    assert isinstance(result["results"][0]["images"], list)
    result = client.post(
        "/tavily/extract", json={"urls": "https://example.org/fail"}
    ).json()
    assert result["results"] == [] and result["usage"]["credits"] == 0
    assert (
        client.post(
            "/tavily/extract", json={"urls": ["https://example.org"] * 21}
        ).status_code
        == 400
    )


def test_crawl_map_selectors_caps_external_not_followed(client, srv):
    body = {
        "url": "example.org",
        "max_depth": 2,
        "max_breadth": 4,
        "limit": 5,
        "select_paths": ["/docs|^$|^/$"],
        "allow_external": True,
        "instructions": "weather",
        "include_images": True,
    }
    crawl = client.post("/tavily/crawl", json=body).json()
    assert len(crawl["results"]) == 2
    assert all(isinstance(row["images"], list) for row in crawl["results"])
    mapped = client.post(
        "/tavily/map", json={"url": "example.org", "max_depth": 1, "limit": 4}
    ).json()
    assert all(isinstance(url, str) for url in mapped["results"])
    assert "https://external.org/guide" in mapped["results"]
    assert not any(call[0].startswith("https://external.org") for call in srv.calls)
    assert client.post("/tavily/map", json={}).status_code == 400
    assert (
        client.post("/tavily/map", json={"url": "file:///etc/passwd"}).status_code
        == 403
    )


def test_feedback_usage_logs(client):
    result = client.post("/tavily/search", json={"query": "q"}).json()
    feedback = client.post(
        "/tavily/feedback",
        json={
            "request_id": result["request_id"],
            "human_score": 1,
            "extra_scores": [{"label": "relevant", "value": 1}],
        },
    ).json()
    assert feedback["success"] and feedback["feedback_id"]
    assert client.post("/tavily/feedback", json={}).status_code == 400
    assert (
        client.post(
            "/tavily/feedback", json={"request_id": "x", "unknown": "x" * 262144}
        ).status_code
        == 413
    )
    assert client.get("/tavily/usage").json()["key"]["search_usage"] == 1
    logs = client.post("/tavily/logs", json={"endpoints": ["search"]}).json()
    assert logs["count"] == 1 and "query" not in logs["logs"][0]


async def test_research_async_registry_schema_sse_failure_and_queue(srv):
    req = ResearchRequest(
        input="Paris weather",
        output_schema={
            "properties": {
                "answer": {"type": "string", "description": "Grounded answer"}
            },
            "required": ["answer"],
        },
        files=[
            {
                "name": "notes.md",
                "type": "base64",
                "data": base64.b64encode(b"Paris weather notes").decode(),
            }
        ],
    )
    result = srv.start_research(req, {})
    assert result["status"] == "pending"
    await srv.tasks[result["request_id"]]["_task"]
    completed = srv.research(result["request_id"])
    assert completed["status"] == "completed" and isinstance(completed["content"], dict)
    assert any(source["url"] == "file:notes.md" for source in completed["sources"])
    frames = "".join([frame async for frame in srv.stream(result["request_id"])])
    assert "chat.completion.chunk" in frames and "event: done" in frames
    assert "WebSearch" in frames and "sources" in frames

    async def fail(*args):
        raise RuntimeError("fixture generation failure")

    srv.generator = fail
    result = srv.start_research(ResearchRequest(input="q"), {})
    await srv.tasks[result["request_id"]]["_task"]
    assert srv.research(result["request_id"])["status"] == "failed"
    await srv.close()


def test_research_poll_and_mcp(client):
    assert client.get("/tavily/research/missing").status_code == 404
    init = client.post(
        "/tavily/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"}
    ).json()
    assert init["result"]["serverInfo"]["name"] == "yunshu-tavily"
    tools = client.post(
        "/tavily/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    ).json()["result"]["tools"]
    assert {"tavily_search", "tavily-search", "tavily_research"} <= {
        tool["name"] for tool in tools
    }
    result = client.post(
        "/tavily/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "tavily_search", "arguments": {"query": "Paris"}},
        },
    ).json()
    assert not result["result"]["isError"]


def test_page_metadata_markdown_hidden_text():
    _, text, date, metadata = page_content(
        '<html lang="en"><head><meta property="article:published_time" content="2026-10-07"><link rel="icon" href="/icon.png"></head><body><article><h1>Guide</h1><p>Real content</p><a href="/docs">Docs</a><img src="/img.png"><p hidden>HIDDEN</p></article></body></html>',
        "https://example.org",
    )
    assert "HIDDEN" not in text
    assert date == "2026-10-07" and metadata["language"] == "en"
    assert metadata["links"] == ["https://example.org/docs"]
    assert metadata["images"] == ["https://example.org/img.png"]


async def test_metasearch_rrf_deadline_and_health(tmp_path):
    from yunshu_gateway.server_tools.metasearch import Metasearch, _health, read_health
    from yunshu_gateway.server_tools.search import SearchProvider

    _health.clear()
    settings.set_override(
        "YUNSHU_WEB_SEARCH_HEALTH_FILE", str(tmp_path / "health.json")
    )
    settings.set_override("YUNSHU_WEB_SEARCH_PROVIDER_TIMEOUT", 0.1)

    class Provider(SearchProvider):
        def __init__(self, name, urls):
            self.name, self.urls = name, urls

        async def search(self, query, **kwargs):
            if self.name == "slow":
                await asyncio.sleep(1)
            return [SearchResult(url, url) for url in self.urls]

    try:
        meta = Metasearch(
            [
                Provider("a", ["https://a.org", "https://shared.org?utm_x=1"]),
                Provider("b", ["https://b.org", "https://shared.org"]),
                Provider("slow", []),
            ]
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: httpx.Response(500))
        ) as client:
            for _ in range(3):
                rows = await meta.search("q", limit=5, client=client)
            assert rows[0].url.startswith("https://shared.org")
        state = next(value for key, value in _health.items() if key.startswith("slow:"))
        assert state.consecutive_failures == 3 and state.disabled_until > 0
        assert read_health()["providers"] and "q" not in json.dumps(read_health())[0:0]
    finally:
        settings.clear_overrides()


def test_tavily_auth_bearer_body_and_errors(monkeypatch):
    import sys
    from pathlib import Path

    from yunshu_gateway.error_envelope import format_error_response

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
    from tavily_fixture import create_fixture

    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "false")
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "tvly-local")
    with TestClient(create_fixture()) as client:
        denied = client.post("/tavily/search", json={"query": "q"})
        assert denied.status_code == 401 and isinstance(
            denied.json()["detail"]["error"], str
        )
        assert (
            client.post(
                "/tavily/search",
                json={"query": "q"},
                headers={"Authorization": "Bearer tvly-local"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/tavily/search", json={"query": "q", "api_key": "tvly-local"}
            ).status_code
            == 200
        )
    limited = format_error_response("/tavily/search", "limited", 429, retry_after=3)
    assert json.loads(limited.body) == {"detail": {"error": "limited"}}
    assert limited.headers["retry-after"] == "3"


def test_cache_namespaces_and_metadata_budget():
    from yunshu_gateway.server_tools.research.cache import PageCache

    value = FetchResult(
        "https://example.org/",
        "",
        "small",
        "text/html",
        "",
        metadata={"links": ["x" * 2000]},
    )
    cache = PageCache(max_bytes=1024)
    cache.put("tavily::https://example.org/", value)
    assert not cache.rows
    cache = PageCache()
    cache.put("tavily::https://example.org/", value)
    assert cache.get("https://example.org/") is None
    assert cache.get("tavily::https://example.org/").metadata["links"]


async def test_requested_image_descriptions_are_bounded_and_cached(srv):
    calls = []

    async def describe(url):
        calls.append(url)
        return "A weather map"

    srv.image_describer = describe
    result = await srv.search(SearchRequest(query="weather", include_images=True))
    assert not calls and "description" not in result["images"][0]
    req = SearchRequest(
        query="weather", include_images=True, include_image_descriptions=True
    )
    one = await srv.search(req)
    two = await srv.search(req)
    assert calls == ["https://example.org/a.png"]
    assert (
        one["images"][0]["description"]
        == two["images"][0]["description"]
        == "A weather map"
    )


def test_pdf_extract_uses_guarded_byte_cap(monkeypatch):
    import asyncio
    import io

    from pypdf import PdfWriter

    from yunshu_engine.netguard import Target
    from yunshu_gateway.server_tools import webfetch

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    data = io.BytesIO()
    writer.write(data)

    async def resolve(url, **kwargs):
        return Target(url, "example.org", 443, "https", "1.1.1.1")

    # Exercise the actual shared downloader, with only DNS/transport injected.
    monkeypatch.setattr(webfetch, "resolve_target", resolve)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(
                    200,
                    content=data.getvalue(),
                    headers={"content-type": "application/pdf"},
                )
            )
        ) as client:
            return await webfetch._fetch_url(
                "https://example.org/a.pdf", client=client, allow_pdf=True
            )

    assert asyncio.run(run()).media_type == "application/pdf"


def test_invalid_research_files_and_rate_limit(client, srv):
    response = client.post(
        "/tavily/research",
        json={
            "input": "q",
            "files": [{"name": "notes.md", "data": "!!!", "type": "base64"}],
        },
    )
    assert response.status_code == 400 and "detail" in response.json()
    for i in range(4):
        srv.tasks[str(i)] = {"status": "in_progress"}
    response = client.post("/tavily/research", json={"input": "q"})
    assert response.status_code == 429 and response.headers["retry-after"] == "10"


def test_main_gateway_body_limit_and_errors(monkeypatch, srv):
    from yunshu_gateway.main import create_app

    settings.set_override("YUNSHU_MAX_REQUEST_SIZE", 1024)
    try:
        app = create_app()
        app.state.tavily_service = srv
        client = TestClient(app)
        bad = client.post("/tavily/search", json={"query": "x" * 2000})
        assert bad.status_code == 413 and bad.json() == {
            "detail": {"error": "Request body too large"}
        }
        unknown = client.get("/tavily/unknown")
        assert unknown.status_code == 404 and "error" in unknown.json()["detail"]
    finally:
        settings.clear_overrides()


def test_extract_url_validation_and_partial_failure(client):
    bad = client.post(
        "/tavily/extract", json={"urls": ["file:///etc/passwd", "not-a-url"]}
    )
    assert (
        bad.status_code == 400
        and bad.json()["detail"]["error"] == "All URLs failed validation"
    )
    partial = client.post(
        "/tavily/extract", json={"urls": ["https://example.org", "not-a-url"]}
    ).json()
    assert len(partial["results"]) == len(partial["failed_results"]) == 1


async def test_irrelevant_page_does_not_demote_a_better_provider_excerpt(srv):
    original = srv.fetcher

    async def unrelated(url, **kwargs):
        result = await original(url, **kwargs)
        result.text = "Unrelated orange orchard content"
        return result

    srv.fetcher = unrelated
    result = await srv.search(
        SearchRequest(query="Paris sunny", include_raw_content=True)
    )
    assert result["results"][0]["content"] == "Paris is sunny."
    assert result["results"][0]["raw_content"] == "Unrelated orange orchard content"


async def test_exact_match_checks_the_page_not_the_provider_snippet(srv):
    original = srv.fetcher

    async def unrelated(url, **kwargs):
        result = await original(url, **kwargs)
        result.text = "Unrelated content"
        return result

    srv.fetcher = unrelated
    result = await srv.search(SearchRequest(query='"Paris is sunny"', exact_match=True))
    assert result["results"] == []


async def test_pro_has_serial_subtopic_tool_traces(srv):
    created = srv.start_research(
        ResearchRequest(input="Paris weather and travel", model="pro"), {}
    )
    await srv.tasks[created["request_id"]]["_task"]
    frames = "".join([frame async for frame in srv.stream(created["request_id"])])
    assert "ResearchSubtopic" in frames and "parent_tool_call_id" in frames
    assert srv.research(created["request_id"])["usage"]["credits"] >= 15
