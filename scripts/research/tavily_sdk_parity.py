"""Official Python and LangChain clients against offline ASGI fixtures or a real server.

Use an isolated SDK environment, never the engine venv. Fixture mode makes no
network requests. Real-server mode runs through gpuq/yv only.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url")
    p.add_argument("--api-key", default="tvly-fixture")
    p.add_argument("--out", type=Path, required=True)
    return p


def valid(rows):
    checks = [row for row in rows if "check" in row]
    return bool(
        rows
        and rows[-1].get("complete") is True
        and checks
        and all(row.get("pass") is True for row in checks)
    )


def check_response(response, endpoint):
    assert isinstance(response, dict)
    assert isinstance(response.get("response_time"), (int, float))
    assert isinstance(response.get("request_id"), str)
    if "usage" in response:
        assert isinstance(response["usage"], dict)
    if endpoint in ("search", "extract", "crawl", "map"):
        assert isinstance(response["results"], list)
    if endpoint == "search":
        assert isinstance(response["images"], list)
        assert all(
            isinstance(row["score"], (int, float)) and isinstance(row["content"], str)
            for row in response["results"]
        )
    if endpoint == "extract":
        assert isinstance(response["failed_results"], list)


def fixture_session(client):
    import requests

    class Session(requests.Session):
        def request(self, method, url, **kwargs):
            kwargs.pop("stream", None)
            kwargs.pop("timeout", None)
            if "allow_redirects" in kwargs:
                kwargs["follow_redirects"] = kwargs.pop("allow_redirects")
            kwargs["headers"] = {**self.headers, **(kwargs.get("headers") or {})}
            if "data" in kwargs:
                kwargs["content"] = kwargs.pop("data")
            response = client.request(method, url, **kwargs)
            result = requests.Response()
            result.status_code = response.status_code
            result.headers.update(response.headers)
            result._content = response.content
            result._content_consumed = True
            result.url = str(response.url)
            return result

    return Session()


def main(argv=None):
    args = parser().parse_args(argv)
    import httpx
    from langchain_tavily._utilities import TavilySearchAPIWrapper
    from tavily import AsyncTavilyClient, TavilyClient

    rows = []
    app = None
    fixture_client = None
    if not args.base_url:
        from fastapi.testclient import TestClient
        from tavily_fixture import create_fixture

        app = create_fixture()
        fixture_client = TestClient(app)
        fixture_client.__enter__()
        base = "http://testserver/tavily"
    else:
        base = args.base_url.rstrip("/")

    def record(name, endpoint, value):
        check_response(value, endpoint)
        rows.append({"check": name, "pass": True, "request_id": value["request_id"]})

    # SDK's actual session injection; the client package and request serialization are unchanged.
    session = fixture_session(fixture_client) if fixture_client else None
    client = TavilyClient(api_key=args.api_key, api_base_url=base, session=session)
    try:
        record(
            "python.sync.search",
            "search",
            client.search(
                "Paris weather",
                include_usage=True,
                include_images=True,
                chunks_per_source=2,
            ),
        )
        record(
            "python.sync.extract",
            "extract",
            client.extract(
                ["https://fixture.example/", "https://fixture.example/fail"],
                include_usage=True,
            ),
        )
        record(
            "python.sync.map",
            "map",
            client.map("https://fixture.example/", limit=2, include_usage=True),
        )
        record(
            "python.sync.crawl",
            "crawl",
            client.crawl("https://fixture.example/", limit=2, include_usage=True),
        )
        assert isinstance(client.get_search_context("Paris weather"), str)
        assert isinstance(client.qna_search("Paris weather"), str)
        rows.append({"check": "python.sync.helpers", "pass": True})

        async def async_checks():
            transport = httpx.ASGITransport(app=app) if app else None
            async with httpx.AsyncClient(transport=transport, base_url=base) as http:
                async_client = AsyncTavilyClient(
                    api_key=args.api_key, api_base_url=base, client=http
                )
                record(
                    "python.async.search",
                    "search",
                    await async_client.search("Paris weather", include_usage=True),
                )
                record(
                    "python.async.extract",
                    "extract",
                    await async_client.extract(
                        "https://fixture.example/", include_usage=True
                    ),
                )
                record(
                    "python.async.crawl",
                    "crawl",
                    await async_client.crawl(
                        "https://fixture.example/", limit=2, include_usage=True
                    ),
                )
                record(
                    "python.async.map",
                    "map",
                    await async_client.map(
                        "https://fixture.example/", limit=2, include_usage=True
                    ),
                )
                task = await async_client.research("Paris weather", model="mini")
                record("python.async.research.create", "research", task)
                for _ in range(200):
                    result = await async_client.get_research(
                        task["request_id"], include_usage=True
                    )
                    if result["status"] in ("completed", "failed"):
                        break
                    await asyncio.sleep(0.02)
                assert result["status"] == "completed", result
                record("python.async.research.poll", "research", result)
                data = b""
                async for chunk in await async_client.research(
                    "Paris weather", stream=True
                ):
                    data += chunk
                assert b"event: done" in data and b"chat.completion.chunk" in data
                rows.append({"check": "python.async.research.sse", "pass": True})

        asyncio.run(async_checks())
        created = client.research("Paris weather", model="mini")
        record("python.sync.research.create", "research", created)
        import time

        for _ in range(200):
            completed = client.get_research(created["request_id"], include_usage=True)
            if completed["status"] in ("completed", "failed"):
                break
            time.sleep(0.01)
        assert completed["status"] == "completed", completed
        record("python.sync.research.poll", "research", completed)
        raw = b"".join(client.research("Paris weather", stream=True))
        assert b"event: done" in raw
        rows.append({"check": "python.sync.research.sse", "pass": True})
        assert client.feedback(request_id=created["request_id"], human_score=1)[
            "success"
        ]
        rows.append({"check": "python.sync.feedback", "pass": True})

        lc_options = {
            "search_depth": "basic",
            "include_domains": [],
            "exclude_domains": [],
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
            "include_image_descriptions": False,
            "include_favicon": False,
            "topic": "general",
            "time_range": None,
            "country": None,
            "auto_parameters": False,
            "start_date": None,
            "end_date": None,
            "exact_match": False,
            "include_usage": True,
        }
        wrapper = TavilySearchAPIWrapper(tavily_api_key=args.api_key, api_base_url=base)
        if fixture_client:
            from unittest.mock import patch

            with patch("requests.post", session.post):
                result = wrapper.raw_results(
                    "Paris weather", max_results=2, **lc_options
                )
        else:
            result = wrapper.raw_results("Paris weather", max_results=2, **lc_options)
        record("langchain.search", "search", result)
        if fixture_client:
            import inspect
            from unittest.mock import patch

            from langchain_tavily import _utilities as utilities

            wrappers = [
                (
                    "extract",
                    utilities.TavilyExtractAPIWrapper,
                    {"urls": ["https://fixture.example/"]},
                ),
                (
                    "map",
                    utilities.TavilyMapAPIWrapper,
                    {"url": "https://fixture.example/", "limit": 2},
                ),
                (
                    "crawl",
                    utilities.TavilyCrawlAPIWrapper,
                    {"url": "https://fixture.example/", "limit": 2},
                ),
                (
                    "research",
                    utilities.TavilyResearchAPIWrapper,
                    {"input": "Paris weather", "research_model": "mini"},
                ),
            ]
            with (
                patch("requests.post", session.post),
                patch("requests.get", session.get),
            ):
                for endpoint, cls, given in wrappers:
                    wrapper = cls(tavily_api_key=args.api_key, api_base_url=base)
                    parameters = inspect.signature(wrapper.raw_results).parameters
                    values = {
                        name: None
                        for name, parameter in parameters.items()
                        if parameter.kind
                        not in (
                            inspect.Parameter.VAR_KEYWORD,
                            inspect.Parameter.VAR_POSITIONAL,
                        )
                    }
                    values.update(given)
                    values["include_usage"] = True
                    result = wrapper.raw_results(**values)
                    record("langchain." + endpoint, endpoint, result)
                    if endpoint == "research":
                        streamed = wrapper.raw_results(
                            input="Paris weather",
                            research_model="mini",
                            output_schema=None,
                            stream=True,
                            citation_format="numbered",
                        )
                        assert b"event: done" in b"".join(streamed)
                        rows.append({"check": "langchain.research.sse", "pass": True})

        rows.append(
            {
                "complete": True,
                "pass": True,
                "fixture": app is not None,
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ("tavily-python", "langchain-tavily")
                },
            }
        )
    finally:
        if fixture_client:
            fixture_client.__exit__(None, None, None)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return 0 if valid(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
