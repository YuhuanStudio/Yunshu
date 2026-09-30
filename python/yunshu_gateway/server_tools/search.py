"""Pluggable web search backends for the server-side ``web_search`` tool.

Providers: a self-hosted SearXNG (the privacy-friendly default recommendation), Brave, Tavily
and Exa (API keys). Nothing is on unless configured (``YUNSHU_WEB_SEARCH_PROVIDER`` /
``YUNSHU_SEARXNG_URL`` / ``YUNSHU_*_API_KEY``); with no provider a request gets the API's own
``unavailable`` error plus a hint that says how to configure one.

Error codes follow Anthropic's ``web_search_tool_result_error``: ``too_many_requests``,
``invalid_input``, ``max_uses_exceeded``, ``query_too_long``, ``unavailable``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from yunshu_engine import settings

from .netguard import domain_matches

MAX_QUERY_LEN = 400
PROVIDER_ORDER = ("searxng", "brave", "tavily", "exa")

SETUP_HINT = (
    "Server-side web search is off. Point Yunshu at a search backend: run a SearXNG instance and "
    "set YUNSHU_SEARXNG_URL (recommended, private), or set one of YUNSHU_BRAVE_API_KEY, "
    "YUNSHU_TAVILY_API_KEY, YUNSHU_EXA_API_KEY. `yunshu config` shows the effective values."
)


class SearchError(Exception):
    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code
        self.message = message or code


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    page_age: str | None = None

    @property
    def domain(self) -> str:
        return (urlsplit(self.url).hostname or "").lower()


class SearchProvider:
    name = "base"

    async def search(
        self,
        query: str,
        *,
        limit: int,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        user_location: dict | None = None,
        client: httpx.AsyncClient,
    ) -> list[SearchResult]:
        raise NotImplementedError


def _raise_http(resp: httpx.Response, who: str):
    if resp.status_code == 429:
        raise SearchError("too_many_requests", f"{who} rate-limited the search")
    if resp.status_code >= 400:
        raise SearchError("unavailable", f"{who} returned HTTP {resp.status_code}")


def _clean(s: str | None, n: int = 600) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or "")).strip()[:n]


class SearXNG(SearchProvider):
    name = "searxng"

    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    async def search(
        self,
        query,
        *,
        limit,
        allowed_domains=None,
        blocked_domains=None,
        user_location=None,
        client,
    ):
        params = {
            "q": query,
            "format": "json",
            "categories": "general",
            "safesearch": "0",
        }
        if user_location and user_location.get("country"):
            params["language"] = f"en-{user_location['country']}"
        r = await client.get(
            f"{self.base}/search", params=params, headers={"Accept": "application/json"}
        )
        _raise_http(r, "SearXNG")
        try:
            data = r.json()
        except ValueError as e:
            raise SearchError(
                "unavailable", "SearXNG did not return JSON (enable the json format)"
            ) from e
        return [
            SearchResult(
                _clean(x.get("title"), 300),
                x["url"],
                _clean(x.get("content")),
                x.get("publishedDate"),
            )
            for x in data.get("results", [])
            if x.get("url")
        ]


class Brave(SearchProvider):
    name = "brave"

    def __init__(self, key: str):
        self.key = key

    async def search(
        self,
        query,
        *,
        limit,
        allowed_domains=None,
        blocked_domains=None,
        user_location=None,
        client,
    ):
        params = {"q": query, "count": min(20, max(limit * 2, limit))}
        if user_location and user_location.get("country"):
            params["country"] = user_location["country"]
        r = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params=params,
            headers={"X-Subscription-Token": self.key, "Accept": "application/json"},
        )
        _raise_http(r, "Brave")
        return [
            SearchResult(
                _clean(x.get("title"), 300),
                x["url"],
                _clean(x.get("description")),
                x.get("age"),
            )
            for x in (r.json().get("web") or {}).get("results", [])
            if x.get("url")
        ]


class Tavily(SearchProvider):
    name = "tavily"

    def __init__(self, key: str):
        self.key = key

    async def search(
        self,
        query,
        *,
        limit,
        allowed_domains=None,
        blocked_domains=None,
        user_location=None,
        client,
    ):
        body: dict = {"query": query, "max_results": min(20, limit * 2)}
        if allowed_domains:
            body["include_domains"] = allowed_domains
        if blocked_domains:
            body["exclude_domains"] = blocked_domains
        r = await client.post(
            "https://api.tavily.com/search",
            json=body,
            headers={"Authorization": f"Bearer {self.key}"},
        )
        _raise_http(r, "Tavily")
        return [
            SearchResult(
                _clean(x.get("title"), 300),
                x["url"],
                _clean(x.get("content")),
                x.get("published_date"),
            )
            for x in r.json().get("results", [])
            if x.get("url")
        ]


class Exa(SearchProvider):
    name = "exa"

    def __init__(self, key: str):
        self.key = key

    async def search(
        self,
        query,
        *,
        limit,
        allowed_domains=None,
        blocked_domains=None,
        user_location=None,
        client,
    ):
        body: dict = {
            "query": query,
            "numResults": min(25, limit * 2),
            "contents": {"text": {"maxCharacters": 800}},
        }
        if allowed_domains:
            body["includeDomains"] = allowed_domains
        if blocked_domains:
            body["excludeDomains"] = blocked_domains
        r = await client.post(
            "https://api.exa.ai/search", json=body, headers={"x-api-key": self.key}
        )
        _raise_http(r, "Exa")
        return [
            SearchResult(
                _clean(x.get("title"), 300),
                x["url"],
                _clean(x.get("text")),
                x.get("publishedDate"),
            )
            for x in r.json().get("results", [])
            if x.get("url")
        ]


_override: SearchProvider | None = None  # tests inject a fake provider here


def set_provider_for_tests(p: SearchProvider | None) -> None:
    global _override
    _override = p


def get_provider() -> SearchProvider | None:
    """The configured provider, or None when web search is not set up."""
    if _override is not None:
        return _override
    want = settings.get("YUNSHU_WEB_SEARCH_PROVIDER")
    if want == "none":
        return None
    url = settings.get("YUNSHU_SEARXNG_URL")
    keys = {
        "brave": settings.get("YUNSHU_BRAVE_API_KEY"),
        "tavily": settings.get("YUNSHU_TAVILY_API_KEY"),
        "exa": settings.get("YUNSHU_EXA_API_KEY"),
    }
    order = PROVIDER_ORDER if want == "auto" else (want,)
    for n in order:
        if n == "searxng" and url:
            return SearXNG(url)
        if n in keys and keys[n]:
            return {"brave": Brave, "tavily": Tavily, "exa": Exa}[n](keys[n])
    return None


async def run_search(
    query: str,
    *,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    user_location: dict | None = None,
    client: httpx.AsyncClient | None = None,
) -> tuple[str, list[SearchResult]]:
    """Run one search. Returns (provider name, filtered results) or raises :class:`SearchError`."""
    if not isinstance(query, str) or not query.strip():
        raise SearchError("invalid_input", "query is required")
    if len(query) > MAX_QUERY_LEN:
        raise SearchError(
            "query_too_long", f"query longer than {MAX_QUERY_LEN} characters"
        )
    prov = get_provider()
    if prov is None:
        raise SearchError("unavailable", SETUP_HINT)
    limit = int(settings.get("YUNSHU_WEB_SEARCH_RESULTS"))
    own = client is None
    client = client or httpx.AsyncClient(
        timeout=float(settings.get("YUNSHU_WEB_FETCH_TIMEOUT"))
    )
    try:
        try:
            rows = await prov.search(
                query.strip(),
                limit=limit,
                allowed_domains=allowed_domains,
                blocked_domains=blocked_domains,
                user_location=user_location,
                client=client,
            )
        except SearchError:
            raise
        except httpx.HTTPError as e:
            raise SearchError("unavailable", f"{prov.name}: {type(e).__name__}") from e
    finally:
        if own:
            await client.aclose()
    if allowed_domains:
        rows = [r for r in rows if domain_matches(r.domain, allowed_domains)]
    if blocked_domains:
        rows = [r for r in rows if not domain_matches(r.domain, blocked_domains)]
    seen, out = set(), []
    for r in rows:
        if r.url not in seen:
            seen.add(r.url)
            out.append(r)
    return prov.name, out[:limit]
