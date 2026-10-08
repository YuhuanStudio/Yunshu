"""Four-second enrichment deadline; one global dense batch and independent fallbacks."""

import asyncio
import hashlib
import logging
from dataclasses import replace

from yunshu_engine import settings

from ..search import Passage, SearchResult
from ..webfetch import FetchResult
from .chunk import chunks
from .fetcher import page
from .rank import bm25, rank, tokens

logger = logging.getLogger(__name__)


def excerpt(p: Passage, query: str, cap: int) -> Passage:
    """Crop around query hits while preserving an exact contiguous source span."""
    import re

    hits = [
        m
        for term in set(tokens(query))
        for m in re.finditer(
            (r"(?<!\w)" + re.escape(term) + r"(?!\w)")
            if term.isascii()
            else re.escape(term),
            p.text,
            re.I,
        )
    ]
    starts = [0] + [max(0, m.start() - cap // 3) for m in hits]
    start = max(
        starts,
        key=lambda s: (
            len({m.group().casefold() for m in hits if s <= m.start() < s + cap}),
            -s,
        ),
    )
    text = p.text[start : start + cap]
    return replace(p, text=text, start=p.start + start, end=p.start + start + len(text))


def select(
    query: str, passages: list[Passage], order: list[int], cap: int = 1200
) -> list[Passage]:
    selected: list[Passage] = []
    for i in order:
        if cap <= 0 or len(selected) >= 3:
            break
        p = excerpt(passages[i], query, min(400, cap))
        if not p.text or any(
            min(p.end, old.end) - max(p.start, old.start) > len(p.text) // 2
            for old in selected
        ):
            continue
        selected.append(p)
        cap -= len(p.text)
    return selected


def result_from_page(
    result: SearchResult, fetched: FetchResult, selected: list[Passage]
) -> SearchResult:
    return replace(
        result,
        title=(fetched.title or result.title)[:300],
        snippet="\n\n".join(p.text for p in selected),
        passages=selected,
        content_hash=hashlib.sha256(fetched.text.encode()).hexdigest(),
        fetched=True,
        page_age=fetched.published_at or result.page_age,
    )


async def enrich(
    query: str, results: list[SearchResult], *, budget: float | None = None, **kwargs
) -> list[SearchResult]:
    out = list(results)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + min(
        4,
        budget
        if budget is not None
        else float(settings.get("YUNSHU_WEB_RESEARCH_BUDGET")),
    )
    ready = {}
    cap = min(1200, 6000 // max(1, len(results)))

    async def one(index, result):
        try:
            fetched = await page(result.url, **kwargs)
            passages = chunks(fetched.text)
            selected = select(query, passages, bm25(query, passages), cap)
            if selected:
                ready[index] = (fetched, passages)
                # Install CPU-ranked passages as soon as available, even if dense times out.
                out[index] = result_from_page(result, fetched, selected)
        except Exception:
            logger.debug("Enrichment kept provider snippet", exc_info=True)

    tasks = [
        asyncio.create_task(one(i, r))
        for i, r in enumerate(
            results[: min(6, int(settings.get("YUNSHU_WEB_RESEARCH_PAGES")))]
        )
    ]
    try:
        if tasks:
            await asyncio.wait(tasks, timeout=min(3, max(0, deadline - loop.time())))
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    flat, owners = [], []
    for index, result in enumerate(out):
        spans = (
            ready[index][1]
            if index in ready
            else (
                [Passage(result.snippet, heading=result.title)]
                if result.snippet
                else []
            )
        )
        for passage in spans:
            flat.append(passage)
            owners.append(index)
    if ready and flat and deadline > loop.time():
        try:
            async with asyncio.timeout(deadline - loop.time()):
                order = await rank(query, flat)
            first_rank = {
                index: min(pos for pos, i in enumerate(order) if owners[i] == index)
                for index in set(owners)
            }
            for index, (fetched, _) in ready.items():
                indices = [i for i in order if owners[i] == index]
                selected = select(query, flat, indices, cap)
                out[index] = result_from_page(results[index], fetched, selected)
            # Dense/BM25 fusion curates source order as well as passages.
            return [out[i] for i in sorted(first_rank, key=lambda i: first_rank[i])] + [
                r for i, r in enumerate(out) if i not in first_rank
            ]
        except Exception:
            logger.debug("Global ranking kept ready BM25 passages", exc_info=True)
    return out


async def open_page(url: str, pattern: str = "", **kwargs) -> SearchResult:
    async with asyncio.timeout(4):
        fetched = await page(url, **kwargs)
        passages = chunks(fetched.text)
        sparse = bm25(pattern, passages) if pattern else list(range(len(passages)))
        selected = select(pattern, passages, sparse)
        # Return ready BM25 excerpts if the optional resident model misses its deadline.
        if pattern:
            try:
                async with asyncio.timeout(1):
                    order = await rank(pattern, passages)
                selected = select(pattern, passages, order)
            except TimeoutError:
                pass
        return result_from_page(
            SearchResult(fetched.title, fetched.url), fetched, selected
        )
