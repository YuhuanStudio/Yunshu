"""Pluggable web search backends for the server-side ``web_search`` tool.

Auto tries configured SearXNG, keyed providers, then best-effort DuckDuckGo HTML and
Wikipedia. Queries leave the machine; ``none`` disables all search.

Error codes follow Anthropic's ``web_search_tool_result_error``: ``too_many_requests``,
``invalid_input``, ``max_uses_exceeded``, ``query_too_long``, ``unavailable``.
"""

from __future__ import annotations

import contextvars
import copy
import html
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote, urljoin, urlsplit

import httpx

from yunshu_engine import settings
from yunshu_engine.netguard import domain_matches

_query_log_context = contextvars.ContextVar("yunshu_search_query_log", default=False)


class _QueryLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # HTTPX logs full GET URLs at INFO. Keep query text out of ordinary logs,
        # scoped to the current async search task so other HTTP traffic is untouched.
        if (
            _query_log_context.get()
            and record.levelno >= logging.INFO
            and isinstance(record.args, tuple)
        ):
            record.args = tuple(
                arg.copy_with(query=b"") if isinstance(arg, httpx.URL) else arg
                for arg in record.args
            )
        return True


logging.getLogger("httpx").addFilter(_QueryLogFilter())


MAX_QUERY_LEN = 400
PROVIDER_ORDER = (
    "brave",
    "searxng",
    "tavily",
    "exa",
    "serper",
    "perplexity",
    "ddg_html",
    "wikipedia",
    "mwmbl",
)
logger = logging.getLogger(__name__)
PRIVACY_NOTICE = "Web search sends query text off-device. Keyless mode uses DuckDuckGo (best effort) and Wikipedia; fetched pages contact their origins without cookies. Set YUNSHU_WEB_SEARCH_PROVIDER=none to disable."

SETUP_HINT = (
    "Server-side web search is off. Enable built-in metasearch with "
    "YUNSHU_WEB_SEARCH_PROVIDER=auto, or provide a Brave, Serper or Exa key. "
    "SearXNG is an optional explicit backend. `yunshu config` shows provider health."
)


class SearchError(Exception):
    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code
        self.message = message or code


@dataclass
class Passage:
    text: str
    score: float = 0.0
    heading: str = ""
    start: int = 0
    end: int = 0


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    page_age: str | None = None
    passages: list[Passage] = field(default_factory=list)
    content_hash: str = ""
    fetched: bool = False

    @property
    def domain(self) -> str:
        return (urlsplit(self.url).hostname or "").lower()


class SearchProvider:
    name = "base"
    capabilities = {"content": False, "freshness": False, "domain_filter": False}

    async def search(
        self,
        query: str,
        *,
        limit: int,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        user_location: dict | None = None,
        client: httpx.AsyncClient,
        options: dict | None = None,
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
    capabilities = {"content": False, "freshness": True, "domain_filter": False}

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
        options=None,
    ):
        params = {
            "q": query,
            "format": "json",
            "categories": "general",
            "safesearch": "0",
        }
        options = options or {}
        params["categories"] = (
            "news" if options.get("topic") in ("news", "finance") else "general"
        )
        params["safesearch"] = "2" if options.get("safe_search") else "0"
        if options.get("time_range"):
            params["time_range"] = options["time_range"]
        if options.get("language"):
            params["language"] = options["language"]
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
    capabilities = {"content": False, "freshness": True, "domain_filter": False}

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
        options=None,
    ):
        params = {"q": query, "count": min(20, max(limit * 2, limit))}
        if user_location and user_location.get("country"):
            params["country"] = user_location["country"]
        options = options or {}
        if options.get("language"):
            params["search_lang"] = options["language"]
        if options.get("time_range"):
            params["freshness"] = {
                "day": "pd",
                "week": "pw",
                "month": "pm",
                "year": "py",
            }.get(options["time_range"], options["time_range"])
        elif options.get("start_date") or options.get("end_date"):
            params["freshness"] = (
                f"{options.get('start_date') or '1900-01-01'}to{options.get('end_date') or time.strftime('%Y-%m-%d')}"
            )
        params["safesearch"] = "strict" if options.get("safe_search") else "off"
        endpoint = "news" if options.get("topic") in ("news", "finance") else "web"
        r = await client.get(
            f"https://api.search.brave.com/res/v1/{endpoint}/search",
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
            for x in (
                (r.json().get("web") or {}).get("results", [])
                if endpoint == "web"
                else r.json().get("results", [])
            )
            if x.get("url")
        ]


class Tavily(SearchProvider):
    name = "tavily"
    capabilities = {"content": False, "freshness": True, "domain_filter": True}

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
        options=None,
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
    capabilities = {"content": False, "freshness": True, "domain_filter": True}

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
        options=None,
    ):
        body: dict = {
            "query": query,
            "numResults": min(25, limit * 2),
            "contents": {"text": {"maxCharacters": 800}},
        }
        options = options or {}
        if options.get("start_date"):
            body["startPublishedDate"] = options["start_date"]
        if options.get("end_date"):
            body["endPublishedDate"] = options["end_date"]
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


class _DDGParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: list[SearchResult] = []
        self.mode = ""
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = (attrs.get("class") or "").split()
        if tag == "a" and "result__a" in classes:
            url = urljoin("https://duckduckgo.com", attrs.get("href") or "")
            url = parse_qs(urlsplit(url).query).get("uddg", [url])[0]
            if urlsplit(url).scheme in ("http", "https"):
                self.rows.append(SearchResult("", url))
                self.mode, self.depth = "title", 1
        elif "result__snippet" in classes and self.rows:
            self.mode, self.depth = "snippet", 1
        elif self.mode:
            self.depth += 1

    def handle_endtag(self, tag):
        if self.mode:
            self.depth -= 1
            if self.depth <= 0:
                self.mode = ""

    def handle_data(self, data):
        if self.mode and self.rows:
            row = self.rows[-1]
            if self.mode == "title":
                row.title += data
            else:
                row.snippet += data


class DuckDuckGo(SearchProvider):
    name = "ddg_html"
    # Shared admission across requests: no retry loops, at most one request/second.
    _next = 0.0
    _blocked_until = 0.0

    async def search(self, query, *, limit, client, **kwargs):
        now = time.monotonic()
        if now < self._blocked_until or now < self._next:
            raise SearchError("too_many_requests", "DuckDuckGo cooling down")
        type(self)._next = now + 1.0
        r = await client.get(
            "https://html.duckduckgo.com/html/",
            params={
                "q": query,
                **(
                    {"df": (kwargs.get("options") or {})["time_range"][0]}
                    if (kwargs.get("options") or {}).get("time_range")
                    else {}
                ),
            },
            headers={
                "User-Agent": "YunshuSearch/1.0 (+https://github.com/YuhuanStudio/Yunshu)"
            },
        )
        if r.status_code in (202, 403, 429) or "anomaly.js" in r.text:
            type(self)._blocked_until = now + 300
            raise SearchError(
                "too_many_requests", "DuckDuckGo blocked automated search"
            )
        _raise_http(r, "DuckDuckGo")
        parser = _DDGParser()
        parser.feed(r.text)
        for row in parser.rows:
            row.title, row.snippet = _clean(row.title, 300), _clean(row.snippet)
        return parser.rows


class Wikipedia(SearchProvider):
    name = "wikipedia"

    async def search(self, query, *, limit, client, **kwargs):
        r = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": min(20, limit * 2),
                "format": "json",
                "utf8": 1,
            },
            headers={
                "User-Agent": "YunshuSearch/1.0 (https://github.com/YuhuanStudio/Yunshu; local search)"
            },
        )
        _raise_http(r, "Wikipedia")
        return [
            SearchResult(
                x["title"],
                "https://en.wikipedia.org/wiki/" + quote(x["title"].replace(" ", "_")),
                _clean(html.unescape(x.get("snippet", ""))),
                x.get("timestamp"),
            )
            for x in r.json().get("query", {}).get("search", [])
            if x.get("title")
        ]


class Serper(SearchProvider):
    name = "serper"
    capabilities = {"content": False, "freshness": True, "domain_filter": False}

    def __init__(self, key):
        self.key = key

    async def search(self, query, *, limit, client, **kwargs):
        options = kwargs.get("options") or {}
        body = {"q": query, "num": min(100, limit * 2)}
        if options.get("language"):
            body["hl"] = options["language"]
        location = kwargs.get("user_location") or {}
        if location.get("country"):
            body["gl"] = location["country"]
        if options.get("time_range"):
            body["tbs"] = "qdr:" + options["time_range"][0]
        if options.get("safe_search"):
            body["safe"] = "active"
        endpoint = "news" if options.get("topic") in ("news", "finance") else "search"
        r = await client.post(
            f"https://google.serper.dev/{endpoint}",
            json=body,
            headers={"X-API-KEY": self.key},
        )
        _raise_http(r, "Serper")
        return [
            SearchResult(
                _clean(x.get("title"), 300),
                x["link"],
                _clean(x.get("snippet")),
                x.get("date"),
            )
            for x in r.json().get("news" if endpoint == "news" else "organic", [])
            if x.get("link")
        ]


class Perplexity(SearchProvider):
    name = "perplexity"
    capabilities = {"content": False, "freshness": True, "domain_filter": True}

    def __init__(self, key):
        self.key = key

    async def search(
        self,
        query,
        *,
        limit,
        client,
        allowed_domains=None,
        blocked_domains=None,
        **kwargs,
    ):
        body = {
            "query": query,
            "max_results": min(20, limit * 2),
            "max_tokens_per_page": 256,
        }
        if allowed_domains or blocked_domains:
            body["search_domain_filter"] = (allowed_domains or []) + [
                "-" + x for x in blocked_domains or []
            ]
        r = await client.post(
            "https://api.perplexity.ai/search",
            json=body,
            headers={"Authorization": f"Bearer {self.key}"},
        )
        _raise_http(r, "Perplexity")
        return [
            SearchResult(
                _clean(x.get("title"), 300),
                x["url"],
                _clean(x.get("snippet")),
                x.get("date"),
            )
            for x in r.json().get("results", [])
            if x.get("url")
        ]


class FallbackChain(SearchProvider):
    def __init__(self, providers):
        self.providers = providers
        self.name = providers[0].name

    async def search(self, query, *, limit, client, **kwargs):
        last = None
        for provider in self.providers:
            try:
                rows = await provider.search(
                    query, limit=limit, client=client, **kwargs
                )
                rows = _filter(
                    rows, kwargs.get("allowed_domains"), kwargs.get("blocked_domains")
                )
                if rows:
                    self.name = provider.name
                    return rows
            except (
                SearchError,
                httpx.HTTPError,
                ValueError,
                KeyError,
                TypeError,
            ) as exc:
                last = exc
        if last:
            if isinstance(last, SearchError):
                raise last
            raise SearchError("unavailable", "All search providers failed") from last
        return []


def _filter(rows, allowed, blocked):
    return [
        r
        for r in rows
        if urlsplit(r.url).scheme in ("http", "https")
        and (not allowed or domain_matches(r.domain, allowed))
        and (not blocked or not domain_matches(r.domain, blocked))
    ]


_search_cache: OrderedDict[tuple, tuple[float, str, list[SearchResult]]] = OrderedDict()

_override: SearchProvider | None = None  # tests inject a fake provider here


def set_provider_for_tests(p: SearchProvider | None) -> None:
    global _override
    _override = p


def get_provider() -> SearchProvider | None:
    if _override is not None:
        return _override
    want = settings.get("YUNSHU_WEB_SEARCH_PROVIDER")
    if want == "none":
        return None
    from .metasearch import Metasearch, Mwmbl

    providers: list[SearchProvider] = []
    for name in PROVIDER_ORDER if want == "auto" else (want,):
        if name == "searxng":
            if url := settings.get("YUNSHU_SEARXNG_URL"):
                providers.append(SearXNG(url))
        elif name in ("ddg_html", "wikipedia", "mwmbl"):
            if settings.get("YUNSHU_WEB_KEYLESS"):
                providers.append(
                    {"ddg_html": DuckDuckGo, "wikipedia": Wikipedia, "mwmbl": Mwmbl}[
                        name
                    ]()
                )
        elif key := settings.get(f"YUNSHU_{name.upper()}_API_KEY"):
            providers.append(
                {
                    "brave": Brave,
                    "tavily": Tavily,
                    "exa": Exa,
                    "serper": Serper,
                    "perplexity": Perplexity,
                }[name](key)
            )
    if not providers:
        return None
    return Metasearch(providers) if want == "auto" else providers[0]


async def run_search(
    query: str,
    *,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    user_location: dict | None = None,
    client: httpx.AsyncClient | None = None,
    limit: int | None = None,
    options: dict | None = None,
    enrich_results: bool | None = None,
    max_query_len: int = MAX_QUERY_LEN,
) -> tuple[str, list[SearchResult]]:
    """Run one search. Returns (provider name, filtered results) or raises :class:`SearchError`."""
    if not isinstance(query, str) or not query.strip():
        raise SearchError("invalid_input", "query is required")
    if len(query) > max_query_len:
        raise SearchError(
            "query_too_long", f"query longer than {max_query_len} characters"
        )
    prov = get_provider()
    if prov is None:
        raise SearchError("unavailable", SETUP_HINT)
    limit = int(settings.get("YUNSHU_WEB_SEARCH_RESULTS")) if limit is None else limit
    if limit == 0:
        return prov.name, []
    do_enrich = (
        settings.get("YUNSHU_WEB_RESEARCH")
        if enrich_results is None
        else enrich_results
    )
    candidates = prov.providers if isinstance(prov, FallbackChain) else [prov]
    cache_key = (
        query.strip(),
        limit,
        tuple(allowed_domains or []),
        tuple(blocked_domains or []),
        str(user_location),
        repr(sorted((options or {}).items())),
        tuple(
            (p.name, getattr(p, "base", None), getattr(p, "key", None))
            for p in candidates
        ),
    )
    cached = _search_cache.get(cache_key) if _override is None else None
    if cached and cached[0] > time.monotonic():
        name, out = cached[1], copy.deepcopy(cached[2])
        _search_cache.move_to_end(cache_key)
        if do_enrich:
            from .research.pipeline import enrich

            out = await enrich(
                query,
                out,
                allowed_domains=allowed_domains,
                blocked_domains=blocked_domains,
            )
        return name, out
    own = client is None
    client = client or httpx.AsyncClient(
        headers={"Cookie": ""},
        trust_env=False,
        follow_redirects=False,
        timeout=float(settings.get("YUNSHU_WEB_FETCH_TIMEOUT")),
    )
    log_token = _query_log_context.set(True)
    try:
        try:
            rows = await prov.search(
                query.strip(),
                limit=limit,
                allowed_domains=allowed_domains,
                blocked_domains=blocked_domains,
                user_location=user_location,
                client=client,
                **({"options": options} if options is not None else {}),
            )
        except SearchError:
            raise
        except httpx.HTTPError as e:
            raise SearchError("unavailable", f"{prov.name}: {type(e).__name__}") from e
    finally:
        _query_log_context.reset(log_token)
        if own:
            await client.aclose()
    rows = _filter(rows, allowed_domains, blocked_domains)
    seen, out = set(), []
    for r in rows:
        if r.url not in seen:
            seen.add(r.url)
            out.append(r)
    out = out[:limit]
    if _override is None and out:
        _search_cache[cache_key] = (
            time.monotonic() + 300,
            prov.name,
            copy.deepcopy(out),
        )
        _search_cache.move_to_end(cache_key)
        while len(_search_cache) > 128:
            _search_cache.popitem(last=False)
    if do_enrich and out:
        from .research.pipeline import enrich

        out = await enrich(
            query, out, allowed_domains=allowed_domains, blocked_domains=blocked_domains
        )
    return prov.name, out
