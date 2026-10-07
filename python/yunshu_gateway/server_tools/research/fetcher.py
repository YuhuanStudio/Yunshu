"""Parallel, guarded fetch and conditional cache revalidation."""

import asyncio
from urllib.parse import urlsplit

from yunshu_engine.netguard import UrlNotAllowedError, domain_matches, parse_url

from ..webfetch import FetchError, _fetch_url
from .cache import pages
from .extract import clean_text, extract_with_metadata
from .politeness import politeness

_global = asyncio.Semaphore(4)


async def page(url: str, *, automated: bool = True, **kwargs):
    try:
        host = parse_url(url).hostname or ""
    except UrlNotAllowedError as exc:
        raise FetchError("url_not_allowed", str(exc)) from exc
    allowed, blocked = kwargs.get("allowed_domains"), kwargs.get("blocked_domains")
    if (allowed and not domain_matches(host, allowed)) or (
        blocked and domain_matches(host, blocked)
    ):
        raise FetchError("url_not_allowed", "Domain filter rejected page")
    if cached := pages.get(url):
        # Recheck final redirect domain for each request's filters.
        final = parse_url(cached.url).hostname or ""
        if (allowed and not domain_matches(final, allowed)) or (
            blocked and domain_matches(final, blocked)
        ):
            raise FetchError(
                "url_not_allowed", "Domain filter rejected cached redirect"
            )
        return cached
    state = politeness.host(url)
    async with _global, state.slots, asyncio.timeout(3):
        if automated and not await politeness.allowed(url, state, **kwargs):
            raise FetchError(
                "url_not_allowed", "robots.txt disallows or is unavailable"
            )
        await politeness.admit(state)
        stale = pages.get(url, stale=True)
        headers = {}
        if stale:
            if stale.etag:
                headers["If-None-Match"] = stale.etag
            if stale.last_modified:
                headers["If-Modified-Since"] = stale.last_modified

        async def before_redirect(target):
            # Automated cross-origin redirects require a new admission/robots budget.
            # Conservatively keep the provider snippet instead of bypassing that policy.
            if urlsplit(target).netloc.lower() != urlsplit(url).netloc.lower():
                raise FetchError("url_not_allowed", "Cross-origin research redirect")
            if automated and not await politeness.allowed(target, state, **kwargs):
                raise FetchError("url_not_allowed", "Redirect disallowed by robots.txt")
            await politeness.admit(state)

        try:
            result = await _fetch_url(
                url,
                extractor=extract_with_metadata,
                request_headers=headers,
                before_redirect=before_redirect,
                **kwargs,
            )
            if result.not_modified:
                if not stale:
                    raise FetchError(
                        "url_not_accessible", "Unexpected 304 without cache"
                    )
                result = stale
            result.text = clean_text(result.text)
            state.failures = 0
            docs_hosts = (
                "docs.python.org",
                "developer.mozilla.org",
                "docs.rs",
                "doc.rust-lang.org",
            )
            ttl = (
                86400
                if (urlsplit(result.url).hostname or "").lower() in docs_hosts
                else 900
            )
            pages.put(url, result, ttl=ttl)
            return result
        except TimeoutError:
            politeness.failed(state, FetchError("url_not_accessible", "timed out"))
            raise
        except FetchError as exc:
            politeness.failed(state, exc)
            raise
