from __future__ import annotations
"""Scoring endpoints: /v1/pooling, /v1/score, /v1/rerank, /v1/classify.

Implements vLLM-compatible scoring endpoints built on the existing
embeddings infrastructure. All endpoints leverage the same
underlying engine resolution and embedding generation.

- /v1/pooling  — raw hidden state pooling (CLS, MEAN, LAST)
- /v1/score    — single/pair similarity scoring (cosine, dot, euclidean)
- /v1/rerank   — cross-encoder reranking with relevance scores
- /v1/classify — zero-shot classification via label similarity
"""
import logging
import math
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter(tags=["scoring"])

_VALID_POOLING_TYPES = {"CLS", "MEAN", "LAST"}
_VALID_SCORING_TYPES = {"cosine", "dot", "euclidean"}
_MAX_DOCUMENT_LENGTH = 8192  # max characters per document for rerank


# ── Request models ───────────────────────────────────────────────────────────

class PoolingRequest(BaseModel):
    model: str
    input: str | list[str]
    pooling_type: str = "CLS"  # CLS, MEAN, LAST
    encoding_format: str = "float"

    @model_validator(mode="after")
    def validate_request(self):
        if not self.model or not self.model.strip():
            raise ValueError("model: field is required and cannot be empty")
        if self.encoding_format not in ("float", "base64"):
            raise ValueError(f"encoding_format: must be 'float' or 'base64', got '{self.encoding_format}'")
        return self


class ScoreRequest(BaseModel):
    model: str
    text_1: str | list[str]
    text_2: str | list[str]
    scoring_type: str = "cosine"  # cosine, dot, euclidean

    @model_validator(mode="after")
    def validate_request(self):
        if not self.model or not self.model.strip():
            raise ValueError("model: field is required and cannot be empty")
        if self.scoring_type not in _VALID_SCORING_TYPES:
            raise ValueError(f"scoring_type: must be one of {', '.join(sorted(_VALID_SCORING_TYPES))}, got '{self.scoring_type}'")
        texts_a = self.text_1 if isinstance(self.text_1, list) else [self.text_1]
        texts_b = self.text_2 if isinstance(self.text_2, list) else [self.text_2]
        if not texts_a or all(not t.strip() for t in texts_a):
            raise ValueError("text_1: field is required and cannot be empty")
        if not texts_b or all(not t.strip() for t in texts_b):
            raise ValueError("text_2: field is required and cannot be empty")
        return self


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: list[str]
    top_n: Optional[int] = None
    return_documents: bool = True

    @model_validator(mode="after")
    def validate_request(self):
        if not self.model or not self.model.strip():
            raise ValueError("model: field is required and cannot be empty")
        if not self.query or not self.query.strip():
            raise ValueError("query: field is required and cannot be empty")
        if not self.documents:
            raise ValueError("documents: field is required and cannot be empty")
        if self.top_n is not None and self.top_n <= 0:
            raise ValueError("top_n: must be a positive integer")
        return self


class ClassifyRequest(BaseModel):
    model: str
    input: str
    labels: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_request(self):
        if not self.model or not self.model.strip():
            raise ValueError("model: field is required and cannot be empty")
        if not self.input or not self.input.strip():
            raise ValueError("input: field is required and cannot be empty")
        if len(self.labels) < 2:
            raise ValueError("labels: at least 2 labels required for classification")
        return self


# ── /v1/pooling ──────────────────────────────────────────────────────────────

@router.post("/pooling", response_model=None)
async def create_pooling(req: PoolingRequest):
    texts = req.input if isinstance(req.input, list) else [req.input]
    if not texts:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    # Validate pooling type
    if req.pooling_type.upper() not in _VALID_POOLING_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid pooling_type '{req.pooling_type}'. "
                   f"Must be one of: {', '.join(sorted(_VALID_POOLING_TYPES))}",
        )

    engine = await _resolve_engine(req.model)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found")

    try:
        raw = await _get_hidden_states(engine, texts, req.pooling_type.upper())
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
        if not vec:
            logger.warning(
                "Empty pooled vector at index %d — engine returned no vector. "
                "Skipping entry (index gap in response).",
                i,
            )
            continue
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

@router.post("/score", response_model=None)
async def create_score(req: ScoreRequest):
    texts_a = req.text_1 if isinstance(req.text_1, list) else [req.text_1]
    texts_b = req.text_2 if isinstance(req.text_2, list) else [req.text_2]

    # Validate scoring type
    if req.scoring_type not in _VALID_SCORING_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scoring_type '{req.scoring_type}'. "
                   f"Must be one of: {', '.join(sorted(_VALID_SCORING_TYPES))}",
        )

    if not texts_a or not texts_b:
        raise HTTPException(status_code=400, detail="text_1 and text_2 cannot be empty")

    # Broadcast single-element lists
    if len(texts_a) != len(texts_b):
        if len(texts_a) == 1:
            texts_a = texts_a * len(texts_b)
        elif len(texts_b) == 1:
            texts_b = texts_b * len(texts_a)
        else:
            raise HTTPException(
                status_code=400,
                detail="text_1 and text_2 must have same length or be broadcastable (length 1)",
            )

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
    try:
        for i, (a, b) in enumerate(zip(emb_a, emb_b)):
            score = _compute_similarity(a, b, req.scoring_type)
            data.append({"object": "score", "index": i, "score": score})
            total_tokens += (len(tok.encode(texts_a[i])) if tok else max(1, len(texts_a[i]) // 4))
            total_tokens += (len(tok.encode(texts_b[i])) if tok else max(1, len(texts_b[i]) // 4))
    except ValueError as e:
        logger.error(f"Score computation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    })


# ── /v1/rerank ───────────────────────────────────────────────────────────────

@router.post("/rerank", response_model=None)
async def create_rerank(req: RerankRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if not req.documents:
        raise HTTPException(status_code=400, detail="Documents cannot be empty")
    if len(req.documents) > 2048:
        raise HTTPException(
            status_code=400,
            detail=f"Too many documents: {len(req.documents)} > 2048",
        )

    # Truncate long documents
    truncated_docs = []
    for doc in req.documents:
        if len(doc) > _MAX_DOCUMENT_LENGTH:
            logger.debug(
                "Truncating document from %d to %d chars",
                len(doc), _MAX_DOCUMENT_LENGTH,
            )
            truncated_docs.append(doc[:_MAX_DOCUMENT_LENGTH])
        else:
            truncated_docs.append(doc)

    engine = await _resolve_engine(req.model)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found")

    try:
        # Compute normalized embeddings for query and documents
        query_emb = (await _get_embeddings(engine, [req.query]))[0]
        doc_embs = await _get_embeddings(engine, truncated_docs)
    except MemoryError:
        logger.error("Rerank OOM", exc_info=True)
        raise HTTPException(status_code=507, detail="Out of GPU memory")
    except Exception as e:
        logger.error(f"Rerank error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Reranking failed")

    # Score each document against the query using cosine similarity
    # Since embeddings are L2-normalized, cosine similarity = dot product
    scored = []
    try:
        for i, doc_emb in enumerate(doc_embs):
            if not doc_emb:
                scored.append((i, 0.0))
                continue
            cos_sim = _compute_similarity(query_emb, doc_emb, "cosine")
            # Scale cosine [-1, 1] to relevance [0, 1]
            relevance = (cos_sim + 1.0) / 2.0
            scored.append((i, relevance))
    except ValueError as e:
        logger.error(f"Rerank scoring error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    # Sort by relevance descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Apply top_n
    if req.top_n is not None and req.top_n > 0:
        scored = scored[:req.top_n]

    tok = getattr(engine, '_tokenizer', None)
    total_tokens = len(tok.encode(req.query)) if tok else max(1, len(req.query) // 4)
    for doc in truncated_docs:
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


# ── /v1/classify ─────────────────────────────────────────────────────────────

@router.post("/classify", response_model=None)
async def classify_input(req: ClassifyRequest):
    """Classify input text using a model's hidden states.

    Returns class probabilities computed from the model's pooled
    hidden representation using a softmax over label embeddings.
    Uses temperature scaling (temperature=0.07) on cosine similarities
    to produce well-separated probability distributions.
    """
    if not req.input or not req.input.strip():
        raise HTTPException(status_code=400, detail="Input cannot be empty")
    if not req.labels:
        raise HTTPException(status_code=400, detail="Labels cannot be empty")
    if len(req.labels) < 2:
        raise HTTPException(status_code=400, detail="At least 2 labels required for classification")

    engine = await _resolve_engine(req.model)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found")

    try:
        input_emb = (await _get_embeddings(engine, [req.input]))[0]
        label_embs = await _get_embeddings(engine, req.labels)
    except MemoryError:
        logger.error("Classify OOM", exc_info=True)
        raise HTTPException(status_code=507, detail="Out of GPU memory")
    except Exception as e:
        logger.error(f"Classify error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Classification failed")

    # Compute cosine similarity between input and each label embedding.
    # Temperature scaling (0.07) sharpens the distribution so the top label
    # gets a meaningful probability rather than a near-uniform spread.
    temperature = 0.07
    scores = []
    try:
        for label_emb in label_embs:
            sim = _compute_similarity(input_emb, label_emb, "cosine")
            scores.append(sim / temperature)
    except ValueError as e:
        logger.error(f"Classify scoring error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    # Stable softmax
    max_score = max(scores) if scores else 0
    exp_scores = [math.exp(s - max_score) for s in scores]
    total = sum(exp_scores)
    if total == 0:
        # All exponentials underflowed to zero — fall back to uniform distribution
        n = len(exp_scores)
        probs = [1.0 / n] * n if n > 0 else []
    else:
        probs = [e / total for e in exp_scores]

    results = []
    for i, (label, prob) in enumerate(zip(req.labels, probs)):
        results.append({"label": label, "score": round(prob, 6), "index": i})

    results.sort(key=lambda x: x["score"], reverse=True)

    return JSONResponse({
        "model": req.model,
        "results": results,
    })


# ── Shared helpers ───────────────────────────────────────────────────────────

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
    """Get L2-normalized embeddings for scoring tasks.

    Uses engine.embed() when available (already normalized).
    Falls back to raw hidden-state extraction + mean pooling + L2 normalization.
    """
    if hasattr(engine, 'embed'):
        import asyncio
        from yunshu_engine.mlx_executor import get_mlx_executor
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(get_mlx_executor(), engine.embed, texts)

    return await _fallback_embeddings(engine, texts, "MEAN", normalize=True)


async def _get_hidden_states(engine, texts: list[str], pooling_type: str) -> list[list[float]]:
    """Get pooled hidden states (NOT normalized) for the pooling endpoint.

    Uses engine.pool() when available for proper pooling support.
    Falls back to engine.embed() only when no pooling-specific method exists.
    """
    if hasattr(engine, 'pool'):
        import asyncio
        from yunshu_engine.mlx_executor import get_mlx_executor
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(get_mlx_executor(), engine.pool, texts, pooling_type)

    # Fallback: raw hidden state extraction with pooling
    return await _fallback_embeddings(engine, texts, pooling_type, normalize=False)


async def _fallback_embeddings(
    engine,
    texts: list[str],
    pooling_type: str,
    normalize: bool = False,
) -> list[list[float]]:
    """Fallback embedding extraction using raw model forward pass.

    Args:
        engine: Engine with _tokenizer and _model attributes.
        texts: Input strings.
        pooling_type: "CLS", "MEAN", or "LAST".
        normalize: If True, L2-normalize the output vectors.
    """
    import mlx.core as mx
    from yunshu_engine.mlx_executor import get_mlx_executor
    import asyncio

    tokenizer = getattr(engine, '_tokenizer', None)
    model = getattr(engine, '_model', None)
    if tokenizer is None or model is None:
        raise RuntimeError("Engine does not support embedding generation")

    loop = asyncio.get_running_loop()

    def _compute_all():
        results = []
        for text in texts:
            tokens = tokenizer.encode(text)
            if not tokens:
                results.append([])
                continue

            input_ids = mx.array([tokens])

            out = model(input_ids)

            # Extract hidden states
            if isinstance(out, mx.array):
                hidden = out
            elif isinstance(out, (tuple, list)):
                hidden = out[0]
            elif hasattr(out, 'last_hidden_state'):
                hidden = out.last_hidden_state
            else:
                hidden = out[0] if isinstance(out, (tuple, list)) else out

            # Pooling
            if pooling_type == "CLS":
                pooled = hidden[:, 0, :]
            elif pooling_type == "LAST":
                pooled = hidden[:, -1, :]
            else:  # MEAN
                pooled = mx.mean(hidden, axis=1)

            if normalize:
                # L2 normalize
                pooled_flat = pooled.reshape(-1)
                norm = mx.sqrt(mx.sum(pooled_flat * pooled_flat) + 1e-12)
                pooled = pooled / norm

            pooled_list = pooled.tolist()
            results.append(
                pooled_list[0] if isinstance(pooled_list, list) and len(pooled_list) == 1
                else pooled_list
            )
        return results

    return await loop.run_in_executor(get_mlx_executor(), _compute_all)


def _compute_similarity(a: list[float], b: list[float], method: str) -> float:
    """Compute similarity between two vectors.

    Raises ValueError if vectors have different dimensions — a dimension
    mismatch indicates a bug upstream (e.g. different models or broken
    truncation) and must not be silently ignored.
    """
    if not a or not b:
        return 0.0

    if len(a) != len(b):
        raise ValueError(
            f"Vector dimension mismatch: {len(a)} != {len(b)}. "
            "Both vectors must come from the same model."
        )

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
