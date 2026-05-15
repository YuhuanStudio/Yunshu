from __future__ import annotations
"""Scoring endpoints: /v1/pooling, /v1/score, /v1/rerank.

Implements vLLM-compatible scoring endpoints built on the existing
embeddings infrastructure. All three endpoints leverage the same
underlying engine resolution and embedding generation.

- /v1/pooling  — raw hidden state pooling (CLS, mean, last)
- /v1/score    — single/pair similarity scoring
- /v1/rerank   — cross-encoder reranking with relevance scores
"""
import logging
import math
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter(tags=["scoring"])


# ── /v1/pooling ──────────────────────────────────────────────────────────────

class PoolingRequest(BaseModel):
    model: str
    input: str | list[str]
    pooling_type: str = "CLS"  # CLS, MEAN, LAST
    encoding_format: str = "float"


@router.post("/pooling", response_model=None)
async def create_pooling(req: PoolingRequest):
    texts = req.input if isinstance(req.input, list) else [req.input]
    if not texts:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    engine = await _resolve_engine(req.model)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found")

    try:
        raw = await _get_hidden_states(engine, texts, req.pooling_type)
    except MemoryError:
        logger.error("Pooling OOM", exc_info=True)
        raise HTTPException(status_code=507, detail="Out of GPU memory")
    except Exception as e:
        logger.error(f"Pooling error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Pooling failed")

    data = []
    total_tokens = 0
    tok = getattr(engine, '_tokenizer', None)
    for i, vec in enumerate(raw):
        if req.encoding_format == "base64":
            import base64, struct
            packed = struct.pack(f"{len(vec)}f", *vec)
            val = base64.b64encode(packed).decode("ascii")
        else:
            val = vec
        data.append({"object": "pooling", "index": i, "data": val})
        total_tokens += len(tok.encode(texts[i])) if tok else max(1, len(texts[i]) // 4)

    return JSONResponse({
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    })


# ── /v1/score ────────────────────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    model: str
    text_1: str | list[str]
    text_2: str | list[str]
    scoring_type: str = "cosine"  # cosine, dot, euclidean


@router.post("/score", response_model=None)
async def create_score(req: ScoreRequest):
    texts_a = req.text_1 if isinstance(req.text_1, list) else [req.text_1]
    texts_b = req.text_2 if isinstance(req.text_2, list) else [req.text_2]

    if len(texts_a) != len(texts_b):
        if len(texts_a) == 1:
            texts_a = texts_a * len(texts_b)
        elif len(texts_b) == 1:
            texts_b = texts_b * len(texts_a)
        else:
            raise HTTPException(status_code=400, detail="text_1 and text_2 must have same length or be broadcastable (length 1)")

    engine = await _resolve_engine(req.model)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found")

    try:
        emb_a = await _get_embeddings(engine, texts_a)
        emb_b = await _get_embeddings(engine, texts_b)
    except MemoryError:
        logger.error("Score OOM", exc_info=True)
        raise HTTPException(status_code=507, detail="Out of GPU memory")
    except Exception as e:
        logger.error(f"Score error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Scoring failed")

    data = []
    total_tokens = 0
    tok = getattr(engine, '_tokenizer', None)
    for i, (a, b) in enumerate(zip(emb_a, emb_b)):
        score = _compute_similarity(a, b, req.scoring_type)
        data.append({"object": "score", "index": i, "score": score})
        total_tokens += (len(tok.encode(texts_a[i])) if tok else max(1, len(texts_a[i]) // 4))
        total_tokens += (len(tok.encode(texts_b[i])) if tok else max(1, len(texts_b[i]) // 4))

    return JSONResponse({
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    })


# ── /v1/rerank ───────────────────────────────────────────────────────────────

class RerankRequest(BaseModel):
    model: str
    query: str
    documents: list[str]
    top_n: Optional[int] = None
    return_documents: bool = True


@router.post("/rerank", response_model=None)
async def create_rerank(req: RerankRequest):
    if not req.documents:
        raise HTTPException(status_code=400, detail="Documents cannot be empty")

    engine = await _resolve_engine(req.model)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found")

    try:
        # Build query+doc pairs and compute embeddings
        query_emb = (await _get_embeddings(engine, [req.query]))[0]
        doc_embs = await _get_embeddings(engine, req.documents)
    except MemoryError:
        logger.error("Rerank OOM", exc_info=True)
        raise HTTPException(status_code=507, detail="Out of GPU memory")
    except Exception as e:
        logger.error(f"Rerank error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Reranking failed")

    # Score each document against the query
    scored = []
    for i, doc_emb in enumerate(doc_embs):
        cos_sim = _compute_similarity(query_emb, doc_emb, "cosine")
        # Scale to [0, 1] relevance range
        relevance = (cos_sim + 1.0) / 2.0
        scored.append((i, relevance))

    # Sort by relevance descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Apply top_n
    if req.top_n is not None and req.top_n > 0:
        scored = scored[:req.top_n]

    tok = getattr(engine, '_tokenizer', None)
    total_tokens = len(tok.encode(req.query)) if tok else max(1, len(req.query) // 4)
    for doc in req.documents:
        total_tokens += len(tok.encode(doc)) if tok else max(1, len(doc) // 4)

    results = []
    for rank, (idx, score) in enumerate(scored):
        item = {
            "index": idx,
            "relevance_score": round(score, 6),
        }
        if req.return_documents:
            item["document"] = {"text": req.documents[idx]}
        results.append(item)

    return JSONResponse({
        "id": f"rerank-{int(time.time())}",
        "object": "list",
        "model": req.model,
        "results": results,
        "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    })


# ── Shared helpers ───────────────────────────────────────────────────────────

@router.post("/classify", response_model=None)
async def classify_input(req: "ClassifyRequest"):
    """Classify input text using a model's hidden states.

    Returns class probabilities computed from the model's pooled
    hidden representation using a softmax over label embeddings.
    """
    if not req.input:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    engine = await _resolve_engine(req.model)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found")

    try:
        input_emb = (await _get_embeddings(engine, [req.input]))[0]
    except MemoryError:
        logger.error("Classify OOM", exc_info=True)
        raise HTTPException(status_code=507, detail="Out of GPU memory")
    except Exception as e:
        logger.error(f"Classify error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Classification failed")

    if req.labels:
        label_embs = await _get_embeddings(engine, req.labels)
        scores = []
        for label_emb in label_embs:
            sim = _compute_similarity(input_emb, label_emb, "cosine")
            scores.append(sim)
        # Softmax normalization
        import math
        max_score = max(scores) if scores else 0
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores]

        results = []
        for i, (label, prob) in enumerate(zip(req.labels, probs)):
            results.append({"label": label, "score": round(prob, 6), "index": i})

        results.sort(key=lambda x: x["score"], reverse=True)
    else:
        results = []

    return JSONResponse({
        "model": req.model,
        "results": results,
    })


class ClassifyRequest(BaseModel):
    model: str
    input: str
    labels: list[str] = Field(default_factory=list)


async def _resolve_engine(model_id: str):
    from ..engine import get_engine, get_model_manager

    manager = get_model_manager()
    if manager is not None:
        entry = manager.get_entry(model_id)
        if entry is not None and entry.is_loaded and entry.engine is not None:
            return entry.engine
        try:
            return await manager.get_engine(model_id)
        except (KeyError, Exception):
            pass

    engine = get_engine()
    if engine and engine.is_loaded:
        return engine
    return None


async def _get_embeddings(engine, texts: list[str]) -> list[list[float]]:
    if hasattr(engine, 'embed'):
        return engine.embed(texts)
    return await _fallback_embeddings(engine, texts, "MEAN")


async def _get_hidden_states(engine, texts: list[str], pooling_type: str) -> list[list[float]]:
    if hasattr(engine, 'embed'):
        embs = engine.embed(texts)
        if pooling_type == "CLS" and hasattr(engine, 'embed_cls'):
            return engine.embed_cls(texts)
        return embs
    return await _fallback_embeddings(engine, texts, pooling_type)


async def _fallback_embeddings(engine, texts: list[str], pooling_type: str) -> list[list[float]]:
    import mlx.core as mx
    from yunshu_engine.mlx_executor import get_mlx_executor
    import asyncio

    tokenizer = getattr(engine, '_tokenizer', None)
    model = getattr(engine, '_model', None)
    if tokenizer is None or model is None:
        raise RuntimeError("Engine does not support embedding generation")

    loop = asyncio.get_running_loop()
    results = []

    for text in texts:
        tokens = tokenizer.encode(text)
        if not tokens:
            results.append([])
            continue

        input_ids = mx.array([tokens])

        def _forward():
            out = model(input_ids)
            if isinstance(out, mx.array):
                return out
            if isinstance(out, (tuple, list)):
                return out[0]
            if hasattr(out, 'last_hidden_state'):
                return out.last_hidden_state
            return out[0] if isinstance(out, (tuple, list)) else out

        hidden = await loop.run_in_executor(get_mlx_executor(), _forward)

        if pooling_type == "CLS":
            pooled = hidden[:, 0, :]
        elif pooling_type == "LAST":
            pooled = hidden[:, -1, :]
        else:  # MEAN
            pooled = mx.mean(hidden, axis=1)

        pooled_list = pooled.tolist()
        results.append(pooled_list[0] if isinstance(pooled_list, list) and len(pooled_list) == 1 else pooled_list)

    return results


def _compute_similarity(a: list[float], b: list[float], method: str) -> float:
    if method == "cosine":
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    elif method == "dot":
        return sum(x * y for x, y in zip(a, b))
    elif method == "euclidean":
        return -math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    else:
        raise ValueError(f"Unknown scoring method: {method}")
