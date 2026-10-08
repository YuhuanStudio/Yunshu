"""Offline ASGI fixture: real Tavily router/service, fake provider/pages/generation.

Only intended for unit and SDK-parity tests. Never mounted by the serving gateway.
"""

import json

from fastapi import FastAPI

from yunshu_gateway.middleware.auth import AuthMiddleware
from yunshu_gateway.routers.tavily import router
from yunshu_gateway.server_tools.search import SearchResult
from yunshu_gateway.server_tools.webfetch import FetchError, FetchResult
from yunshu_gateway.tavily.service import TavilyService

ROOT = "https://fixture.example"


def create_fixture():
    async def search(query, **kwargs):
        return "recorded", [
            SearchResult(
                "Paris weather",
                ROOT + "/",
                "Paris is sunny, 21 degrees Celsius.",
                "2026-10-07",
            )
        ][: kwargs["limit"]]

    async def fetch(url, **kwargs):
        if "fail" in url:
            raise FetchError("url_not_accessible", "Recorded fixture: failed URL")
        return FetchResult(
            url,
            "Paris weather",
            "# Paris weather\nParis is sunny, 21 degrees Celsius. Dry all week.",
            "text/html",
            "2026-10-07",
            published_at="2026-10-07",
            metadata={"links": [ROOT + "/forecast"], "images": [], "language": "en"},
        )

    async def generate(system, user, schema, max_tokens):
        if schema:
            if "queries" in schema.get("properties", {}):
                return json.dumps({"queries": ["Paris weather"]})
            return json.dumps(
                {key: "Paris is sunny [1]." for key in schema.get("properties", {})}
            )
        return "Paris is sunny, 21 degrees Celsius [1]."

    async def image_search(query, **kwargs):
        return [{"url": ROOT + "/authored-fixture.png"}]

    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.state.tavily_service = TavilyService(
        generator=generate, searcher=search, fetcher=fetch, image_searcher=image_search
    )
    app.include_router(router)
    return app
