"""Parallel providers, bounded deadlines, RRF and a query-free health snapshot."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from yunshu_engine import settings

from .search import (
    FallbackChain,
    SearchError,
    SearchProvider,
    SearchResult,
    _clean,
    _filter,
    _raise_http,
)

logger = logging.getLogger(__name__)


class Mwmbl(SearchProvider):
    name = "mwmbl"

    async def search(self, query, *, limit, client, **kwargs):
        r = await client.get(
            "https://api.mwmbl.org/api/v2/search/", params={"q": query}
        )
        _raise_http(r, "Mwmbl")
        return [
            SearchResult(
                _clean(x.get("title"), 300), x["url"], _clean(x.get("content"))
            )
            for x in r.json().get("results", [])
            if x.get("url")
        ][: limit * 2]


@dataclass
class Health:
    requests: int = 0
    successes: int = 0
    consecutive_failures: int = 0
    latency_ms: float = 0
    disabled_until: float = 0
    last_error: str = ""
    probing: bool = False


_health: dict[str, Health] = {}


def canonical_url(url):
    parts = urlsplit(url)
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in ("fbclid", "gclid")
    ]
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path or "/",
            urlencode(query),
            "",
        )
    )


def health_snapshot():
    now = time.time()
    return {
        name: {
            **asdict(value),
            "success_rate": value.successes / value.requests
            if value.requests
            else None,
            "backoff_seconds": max(0, value.disabled_until - now),
        }
        for name, value in _health.items()
    }


def snapshot_path():
    return Path(settings.get("YUNSHU_WEB_SEARCH_HEALTH_FILE")).expanduser()


def read_health():
    try:
        path = snapshot_path()
        if path.stat().st_size > 65536:
            return {}
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def save_health():
    # Contains provider names, health and timings only; no queries, URLs or keys.
    try:
        path = snapshot_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"updated_at": time.time(), "providers": health_snapshot()})
        )
        tmp.replace(path)
    except OSError:
        logger.debug("Could not write provider health", exc_info=True)


async def _first_then_grace(tasks, grace):
    """Return as soon as one provider has rows, giving the rest `grace` more seconds.

    Slow providers are cancelled (their circuit is not charged); with no rows yet we
    keep waiting, so the per-provider timeout stays the only hard deadline.
    """
    pending = set(tasks)
    done_batches = []
    deadline = None
    while pending:
        timeout = None if deadline is None else max(0, deadline - time.monotonic())
        done, pending = await asyncio.wait(
            pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        if not done:
            break
        for task in done:
            done_batches.append(task.result())
            if deadline is None and task.result()[1]:
                deadline = time.monotonic() + grace
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    return done_batches


class Metasearch(FallbackChain):
    """Same provider interface as the older chain; all independent sources run together."""

    def __init__(self, providers):
        super().__init__(providers)
        self.name = "metasearch"

    async def search(self, query, *, limit, client, **kwargs):
        async def one(provider):
            # Credential/config changes get a new circuit without disclosing the key.
            identity = (
                provider.name
                + ":"
                + hashlib.sha256(
                    str(
                        (getattr(provider, "base", ""), getattr(provider, "key", ""))
                    ).encode()
                ).hexdigest()[:12]
            )
            state = _health.setdefault(identity, Health())
            now = time.time()
            if state.disabled_until > now or state.probing:
                return provider.name, [], None
            half_open = state.consecutive_failures >= 3
            if half_open:
                state.probing = True
            start = time.perf_counter()
            state.requests += 1
            error = None
            try:
                timeout = float(settings.get("YUNSHU_WEB_SEARCH_PROVIDER_TIMEOUT"))
                options = kwargs.get("options") or {}
                timeout = min(timeout, float(options.get("provider_timeout", timeout)))
                provider_query = query
                allowed = kwargs.get("allowed_domains")
                if allowed and not provider.capabilities["domain_filter"]:
                    provider_query += (
                        " ("
                        + " OR ".join("site:" + domain for domain in allowed[:10])
                        + ")"
                    )
                async with asyncio.timeout(timeout):
                    rows = await provider.search(
                        provider_query, limit=limit, client=client, **kwargs
                    )
                state.successes += 1
                state.consecutive_failures = 0
                state.disabled_until = 0
                state.last_error = ""
                rows = _filter(
                    rows, kwargs.get("allowed_domains"), kwargs.get("blocked_domains")
                )
            except (
                SearchError,
                httpx.HTTPError,
                TimeoutError,
                ValueError,
                KeyError,
                TypeError,
            ) as exc:
                state.consecutive_failures += 1
                state.last_error = (
                    exc.code if isinstance(exc, SearchError) else type(exc).__name__
                )
                if (
                    state.consecutive_failures >= 3
                    or isinstance(exc, SearchError)
                    and exc.code == "too_many_requests"
                ):
                    state.disabled_until = time.time() + min(
                        3600, 30 * 2 ** min(7, state.consecutive_failures)
                    )
                rows, error = [], exc
            finally:
                state.latency_ms = (time.perf_counter() - start) * 1000
                state.probing = False
            return provider.name, rows, error

        grace = (kwargs.get("options") or {}).get("serp_grace")
        if grace is None:
            batches = await asyncio.gather(*(one(p) for p in self.providers))
        else:
            batches = await _first_then_grace(
                [asyncio.ensure_future(one(p)) for p in self.providers], grace
            )
        await asyncio.to_thread(save_health)
        scores, results, contributors = {}, {}, []
        errors = []
        for name, rows, error in batches:
            if error is not None:
                errors.append(error)
            if rows:
                contributors.append(name)
            seen = set()
            for pos, row in enumerate(rows):
                url = canonical_url(row.url)
                if url in seen:
                    continue
                seen.add(url)
                scores[url] = scores.get(url, 0) + 1 / (61 + pos)
                if url not in results or len(row.snippet) > len(results[url].snippet):
                    results[url] = row
        if not results and errors and len(errors) == len(batches):
            raise SearchError("unavailable", "All metasearch providers failed")
        self.name = contributors[0] if len(contributors) == 1 else "metasearch"
        return [
            results[url] for url in sorted(results, key=lambda url: (-scores[url], url))
        ]
