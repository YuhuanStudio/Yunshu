"""Optional image SERPs from configured providers; origins are not fetched here."""

import asyncio

import httpx

from yunshu_engine.netguard import UrlNotAllowedError, domain_matches, parse_url
from yunshu_gateway.server_tools.search import (
    SearchError,
    _query_log_context,
    _raise_http,
    get_provider,
)


def candidates(rows, provider, *, allowed_domains=None, blocked_domains=None):
    out, seen = [], set()
    for row in rows[:100]:
        if not isinstance(row, dict):
            continue
        properties = row.get("properties") or {}
        if not isinstance(properties, dict):
            continue
        image = row.get("img_src") if provider == "searxng" else properties.get("url")
        source = row.get("url")
        try:
            image_host = parse_url(image).hostname or ""
            source_host = parse_url(source).hostname or ""
        except (UrlNotAllowedError, TypeError):
            continue
        if allowed_domains and not domain_matches(source_host, allowed_domains):
            continue
        if blocked_domains and (
            domain_matches(source_host, blocked_domains)
            or domain_matches(image_host, blocked_domains)
        ):
            continue
        if image not in seen:
            seen.add(image)
            out.append({"url": image})
    return out


async def run_images(
    query,
    *,
    limit=20,
    allowed_domains=None,
    blocked_domains=None,
    options=None,
    user_location=None,
    client=None,
):
    provider = get_provider()
    providers = getattr(provider, "providers", [provider])
    providers = [p for p in providers if p and p.name in ("searxng", "brave")]
    if not providers or limit <= 0:
        return []
    options = options or {}
    timeout = min(3.0, float(options.get("provider_timeout", 2.5)))

    async def one(p):
        try:
            async with asyncio.timeout(timeout):
                params = {"q": query}
                headers = {"Accept": "application/json"}
                if p.name == "searxng":
                    endpoint = p.base + "/search"
                    params.update(
                        format="json",
                        categories="images",
                        safesearch="2" if options.get("safe_search") else "0",
                    )
                    if options.get("language"):
                        params["language"] = options["language"]
                else:
                    endpoint = "https://api.search.brave.com/res/v1/images/search"
                    headers["X-Subscription-Token"] = p.key
                    params.update(
                        count=min(200, limit * 2),
                        safesearch="strict" if options.get("safe_search") else "off",
                    )
                    if options.get("language"):
                        params["search_lang"] = options["language"]
                    if user_location and user_location.get("country"):
                        params["country"] = user_location["country"]
                response = await client.get(endpoint, params=params, headers=headers)
                _raise_http(response, p.name)
                rows = response.json().get("results", [])
                if not isinstance(rows, list):
                    return []
                return candidates(
                    rows,
                    p.name,
                    allowed_domains=allowed_domains,
                    blocked_domains=blocked_domains,
                )
        except (SearchError, httpx.HTTPError, TimeoutError, ValueError, TypeError):
            return []  # Images are optional; a failed vertical must not fail web search.

    own = client is None
    client = client or httpx.AsyncClient(
        trust_env=False, follow_redirects=False, timeout=timeout, headers={"Cookie": ""}
    )
    token = _query_log_context.set(True)
    try:
        groups = await asyncio.gather(*(one(p) for p in providers))
        return list({item["url"]: item for group in groups for item in group}.values())[
            :limit
        ]
    finally:
        _query_log_context.reset(token)
        if own:
            await client.aclose()
