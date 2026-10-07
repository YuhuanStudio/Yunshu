"""Parallel, guarded fetch and conditional cache revalidation."""

import asyncio
from urllib.parse import urlsplit

from yunshu_engine.netguard import UrlNotAllowedError, domain_matches, parse_url

from ..webfetch import FetchError, _fetch_url
from .cache import pages
from .extract import clean_text, extract_with_metadata
from .politeness import politeness

_global = asyncio.Semaphore(4)


async def page(
    url: str,
    *,
    automated: bool = True,
    timeout: float = 3,
    extractor=None,
    cache_namespace: str = "",
    **kwargs,
):
    try:
        host = parse_url(url).hostname or ""
    except UrlNotAllowedError as exc:
        raise FetchError("url_not_allowed", str(exc)) from exc
    allowed, blocked = kwargs.get("allowed_domains"), kwargs.get("blocked_domains")
    if (allowed and not domain_matches(host, allowed)) or (
        blocked and domain_matches(host, blocked)
    ):
        raise FetchError("url_not_allowed", "Domain filter rejected page")
    cache_key = (cache_namespace + "::" + url) if cache_namespace else url
    state = politeness.host(url)
    if cached := pages.get(cache_key):
        # Recheck final redirect domain for each request's filters.
        final = parse_url(cached.url).hostname or ""
        if (allowed and not domain_matches(final, allowed)) or (
            blocked and domain_matches(final, blocked)
        ):
            raise FetchError(
                "url_not_allowed", "Domain filter rejected cached redirect"
            )
        if automated:
            source, destination = urlsplit(url), urlsplit(cached.url)
            if (source.scheme.lower(), source.netloc.lower()) != (
                destination.scheme.lower(),
                destination.netloc.lower(),
            ):
                raise FetchError(
                    "url_not_allowed", "Cross-origin cached research redirect"
                )
            async with asyncio.timeout(timeout):
                if not await politeness.allowed(cached.url, state, **kwargs):
                    raise FetchError(
                        "url_not_allowed", "robots.txt disallows cached page"
                    )
        return cached
    # Deadline includes admission/semaphore waiting, not just the network transfer.
    async with asyncio.timeout(timeout), _global, state.slots:
        if automated and not await politeness.allowed(url, state, **kwargs):
            raise FetchError(
                "url_not_allowed", "robots.txt disallows or is unavailable"
            )
        await politeness.admit(state)
        stale = pages.get(cache_key, stale=True)
        headers = {}
        if stale:
            if stale.etag:
                headers["If-None-Match"] = stale.etag
            if stale.last_modified:
                headers["If-Modified-Since"] = stale.last_modified

        async def before_redirect(target):
            # Automated cross-origin redirects require a new admission/robots budget.
            # Conservatively keep the provider snippet instead of bypassing that policy.
            source, destination = urlsplit(url), urlsplit(target)
            if destination.scheme.lower() != source.scheme.lower():
                raise FetchError(
                    "url_not_allowed", "Cross-origin scheme-changing redirect"
                )
            if automated and (
                destination.scheme.lower(),
                destination.netloc.lower(),
            ) != (
                source.scheme.lower(),
                source.netloc.lower(),
            ):
                raise FetchError("url_not_allowed", "Cross-origin research redirect")
            if automated and not await politeness.allowed(target, state, **kwargs):
                raise FetchError("url_not_allowed", "Redirect disallowed by robots.txt")
            await politeness.admit(state)

        try:
            result = await _fetch_url(
                url,
                extractor=extractor or extract_with_metadata,
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
            pages.put(cache_key, result, ttl=ttl)
            return result
        except TimeoutError:
            politeness.failed(state, FetchError("url_not_accessible", "timed out"))
            raise
        except FetchError as exc:
            politeness.failed(state, exc)
            raise
