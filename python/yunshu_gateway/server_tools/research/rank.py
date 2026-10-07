"""BM25 first; RRF with a resident local embedder, using the scoring route's backend."""

import asyncio
import logging
import math
import re
from collections import Counter

from yunshu_engine import settings

from ..search import Passage

logger = logging.getLogger(__name__)


def tokens(text: str) -> list[str]:
    # CJK individual characters plus Latin words: useful without an extra tokenizer.
    return re.findall(r"[\u3400-\u9fff]|[\w]+", text.lower())


def bm25(query: str, passages: list[Passage]) -> list[int]:
    docs = [Counter(tokens(p.heading + " " + p.text)) for p in passages]
    lengths = [sum(d.values()) for d in docs]
    avg = sum(lengths) / max(1, len(docs)) or 1
    terms = set(tokens(query))
    df = {t: sum(t in d for d in docs) for t in terms}
    for i, doc in enumerate(docs):
        passages[i].score = sum(
            math.log(1 + (len(docs) - df[t] + 0.5) / (df[t] + 0.5))
            * (doc[t] * 2.5)
            / (doc[t] + 1.5 * (0.25 + 0.75 * lengths[i] / avg))
            for t in terms
            if doc[t]
        )
    return sorted(range(len(docs)), key=lambda i: (-passages[i].score, i))


def fuse(sparse: list[int], dense: list[int]) -> list[int]:
    scores: dict[int, float] = {}
    for order in (sparse, dense):
        for pos, index in enumerate(order):
            scores[index] = scores.get(index, 0) + 1 / (60 + pos + 1)
    return sorted(scores, key=lambda i: (-scores[i], i))


def embedding_inputs(model: str, texts: list[str]) -> list[str]:
    texts = list(texts)
    if texts and "qwen3-embedding" in model.lower():
        texts[0] = (
            "Instruct: Retrieve web excerpts relevant to this question.\nQuery: "
            + texts[0]
        )
    return texts


async def local_vectors(texts: list[str]):
    model = settings.get("YUNSHU_WEB_RESEARCH_MODEL")
    if not model:
        return None
    from yunshu_gateway.engine import get_model_manager
    from yunshu_gateway.routers.scoring import _get_embeddings

    manager = get_model_manager()
    entry = manager.get_entry(model) if manager is not None else None
    if (
        entry is None
        or not entry.is_loaded
        or entry.engine is None
        or not hasattr(entry.engine, "embed")
    ):
        return None  # never resolve/load a model or fall back to a generation engine
    # Pin residency for this use just like an API request, so unload cannot race inference.
    from yunshu_engine.model_manager import LeaseScope

    scope = LeaseScope()
    scope.add(entry)
    task = asyncio.create_task(
        _get_embeddings(entry.engine, embedding_inputs(model, texts))
    )

    def finished(future):
        scope.release()
        if not future.cancelled():
            future.exception()  # consume failures even when the caller's deadline expired

    task.add_done_callback(finished)
    vectors = await asyncio.shield(task)
    logger.info("Web research dense ranking: model=%s chunks=%d", model, len(texts) - 1)
    return vectors


async def rank(query: str, passages: list[Passage], embedder=None) -> list[int]:
    sparse = bm25(query, passages)
    top = sparse[:40]
    if not top:
        return []
    try:
        vectors = await (embedder or local_vectors)(
            [query] + [passages[i].text for i in top]
        )
        if vectors is None or len(vectors) != len(top) + 1:
            return sparse
        q = vectors[0]
        if not q or any(
            len(v) != len(q) or not all(math.isfinite(x) for x in v) for v in vectors
        ):
            return sparse

        def cosine(v):
            return sum(a * b for a, b in zip(q, v, strict=True)) / (
                math.sqrt(sum(x * x for x in q) * sum(x * x for x in v)) or 1
            )

        dense = sorted(top, key=lambda i: -cosine(vectors[top.index(i) + 1]))
        return fuse(top, dense) + sparse[40:]
    except Exception:
        logger.debug("Local research ranking unavailable; using BM25", exc_info=True)
        return sparse
