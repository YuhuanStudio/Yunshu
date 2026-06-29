from __future__ import annotations

"""Scoring endpoints: /v1/pooling, /v1/score, /v1/rerank, /v1/classify.

Implements vLLM-compatible scoring endpoints built on the existing
embeddings infrastructure. All endpoints leverage the same
underlying engine resolution and embedding generation.

- /v1/pooling  — raw hidden state pooling (CLS, MEAN, LAST)
- /v1/score    — single/pair similarity scoring (cosine, dot, euclidean)
- /v1/rerank   — BI-ENCODER reranking: embeds the query and each document
                 separately and ranks by cosine similarity scaled to a [0,1]
                 relevance score. This is NOT a true cross-encoder (no joint
                 (query, document) forward pass), so ranking quality is lower
                 than a dedicated cross-encoder reranker — a true cross-encoder
                 model + joint scoring is a deferred enhancement.
- /v1/classify — zero-shot classification via label similarity
"""
import logging
import math
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from .models import _check_model_access, _check_permission

logger = logging.getLogger(__name__)

router = APIRouter(tags=["scoring"])

_VALID_POOLING_TYPES = {"CLS", "MEAN", "LAST"}
_VALID_SCORING_TYPES = {"cosine", "dot", "euclidean"}
_MAX_DOCUMENT_LENGTH = 8192  # max characters per document for rerank
# /v1/embeddings caps every input string (8192 chars) and the list (2048
# elements); the scoring endpoints only capped rerank *documents*, leaving the rerank
# query, score text_1/text_2, pooling input and classify input/labels uncapped per-text —
# a single ~10MB string (under the global 10MB body limit) or a 2048-list of multi-KB
# strings drives one giant uncapped forward pass (context-length / memory-pressure DoS on
# a 36GB Mac). Apply the same caps as embeddings at the schema boundary.
_MAX_INPUT_TEXT_LENGTH = 8192  # chars per scoring text input
_MAX_INPUT_TEXTS = 2048  # elements per scoring list input


def _reject_overlong_texts(field: str, texts: list[str]) -> None:
    """Raise (→ 422) if any text exceeds the per-text char cap. Count caps are enforced
    by each validator (some lists, like score's, broadcast)."""
    for i, t in enumerate(texts):
        if isinstance(t, str) and len(t) > _MAX_INPUT_TEXT_LENGTH:
            raise ValueError(
                f"{field}: item at index {i} exceeds {_MAX_INPUT_TEXT_LENGTH} characters "
                f"({len(t)}). Split or truncate the input."
            )


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
        if self.pooling_type.upper() not in _VALID_POOLING_TYPES:
            raise ValueError(
                f"pooling_type: must be one of {', '.join(sorted(_VALID_POOLING_TYPES))}, got '{self.pooling_type}'"
            )
        if self.encoding_format not in ("float", "base64"):
            raise ValueError(
                f"encoding_format: must be 'float' or 'base64', got '{self.encoding_format}'"
            )
        # Validate input: string must be non-empty, list must have elements
        if isinstance(self.input, str) and not self.input.strip():
            raise ValueError("input: cannot be empty or whitespace-only")
        if isinstance(self.input, list) and not self.input:
            raise ValueError("input: cannot be an empty list")
        if isinstance(self.input, list):
            if len(self.input) > _MAX_INPUT_TEXTS:
                raise ValueError(f"input: maximum {_MAX_INPUT_TEXTS} items per request")
            for i, t in enumerate(self.input):
                if not isinstance(t, str) or not t.strip():
                    raise ValueError(
                        f"input: item at index {i} is empty or whitespace-only"
                    )
        _reject_overlong_texts(
            "input", self.input if isinstance(self.input, list) else [self.input]
        )
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
            raise ValueError(
                f"scoring_type: must be one of {', '.join(sorted(_VALID_SCORING_TYPES))}, got '{self.scoring_type}'"
            )
        texts_a = self.text_1 if isinstance(self.text_1, list) else [self.text_1]
        texts_b = self.text_2 if isinstance(self.text_2, list) else [self.text_2]
        if not texts_a or all(not t.strip() for t in texts_a):
            raise ValueError("text_1: field is required and cannot be empty")
        if not texts_b or all(not t.strip() for t in texts_b):
            raise ValueError("text_2: field is required and cannot be empty")
        # Reject individual empty items
        for i, t in enumerate(texts_a):
            if not isinstance(t, str) or not t.strip():
                raise ValueError(
                    f"text_1: item at index {i} is empty or whitespace-only"
                )
        for i, t in enumerate(texts_b):
            if not isinstance(t, str) or not t.strip():
                raise ValueError(
                    f"text_2: item at index {i} is empty or whitespace-only"
                )
        # when both inputs are LISTS of length > 1,
        # they must match. List-of-1 broadcasts to any length (existing
        # behavior tested in test_score_request_broadcast). Equal-length is
        # always OK. Mismatched non-broadcastable rejected to prevent the
        # silent zip-shortest behavior that returned fewer scores than caller
        # expected.
        if isinstance(self.text_1, list) and isinstance(self.text_2, list):
            la, lb = len(texts_a), len(texts_b)
            if la != lb and la != 1 and lb != 1:
                raise ValueError(
                    f"text_1 / text_2 length mismatch: {la} vs {lb}. "
                    f"Lengths must match or one side must be length=1 for broadcast."
                )
        if len(texts_a) > _MAX_INPUT_TEXTS:
            raise ValueError(f"text_1: maximum {_MAX_INPUT_TEXTS} items per request")
        if len(texts_b) > _MAX_INPUT_TEXTS:
            raise ValueError(f"text_2: maximum {_MAX_INPUT_TEXTS} items per request")
        _reject_overlong_texts("text_1", texts_a)
        _reject_overlong_texts("text_2", texts_b)
        return self


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: list[str]
    top_n: int | None = None
    return_documents: bool = True

    @model_validator(mode="after")
    def validate_request(self):
        if not self.model or not self.model.strip():
            raise ValueError("model: field is required and cannot be empty")
        if not self.query or not self.query.strip():
            raise ValueError("query: field is required and cannot be empty")
        _reject_overlong_texts("query", [self.query])
        if not self.documents:
            raise ValueError("documents: field is required and cannot be empty")
        if self.top_n is not None:
            if self.top_n <= 0:
                raise ValueError("top_n: must be a positive integer")
            if self.top_n > 2048:
                raise ValueError("top_n: maximum 2048")
        # Validate individual documents are not empty
        for i, doc in enumerate(self.documents):
            if not isinstance(doc, str) or not doc.strip():
                raise ValueError(
                    f"documents: item at index {i} is empty or whitespace-only"
                )
        if len(self.documents) > 2048:
            raise ValueError("documents: maximum 2048 documents per request")
        return self


class ClassifyRequest(BaseModel):
    model: str
    input: str
    labels: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_candidate_labels_alias(cls, data):
        """Accept HuggingFace zero-shot 'candidate_labels' as an alias for 'labels'.

        If both are provided, 'labels' wins (explicit canonical name).
        """
        if isinstance(data, dict):
            if not data.get("labels") and data.get("candidate_labels"):
                data = dict(data)
                data["labels"] = data["candidate_labels"]
        return data

    @model_validator(mode="after")
    def validate_request(self):
        if not self.model or not self.model.strip():
            raise ValueError("model: field is required and cannot be empty")
        if not self.input or not self.input.strip():
            raise ValueError("input: field is required and cannot be empty")
        if len(self.labels) < 2:
            raise ValueError("labels: at least 2 labels required for classification")
        if len(self.labels) > _MAX_INPUT_TEXTS:
            raise ValueError(f"labels: maximum {_MAX_INPUT_TEXTS} labels per request")
        # Validate individual labels are not empty
        for i, label in enumerate(self.labels):
            if not isinstance(label, str) or not label.strip():
                raise ValueError(
                    f"labels: item at index {i} is empty or whitespace-only"
                )
        _reject_overlong_texts("input", [self.input])
        _reject_overlong_texts("labels", self.labels)
        return self


# ── /v1/pooling ──────────────────────────────────────────────────────────────


@router.post("/pooling", response_model=None)
async def create_pooling(req: PoolingRequest, request: Request):
    _check_permission(request, "can_infer")
    _check_model_access(request, req.model)
    texts = req.input if isinstance(req.input, list) else [req.input]

    engine = await _resolve_engine(req.model)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found")

    try:
        raw = await _get_hidden_states(engine, texts, req.pooling_type.upper())
    except MemoryError:
        logger.error("Pooling OOM", exc_info=True)
        raise HTTPException(status_code=507, detail="Out of GPU memory") from None
    except ValueError as e:
        # bad-input errors should be 400 not 500
        logger.warning(f"Pooling validation: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Pooling error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Pooling failed") from None

    data = []
    total_tokens = 0
    tok = getattr(engine, "_tokenizer", None)
    for i, vec in enumerate(raw):
        # preserve 1:1 input↔output index alignment. The old code SKIPPED
        # empty vectors and re-indexed the survivors with a contiguous counter, so a
        # single empty result silently shifted every later entry's index and returned
        # fewer rows than inputs — the caller could no longer map results back to
        # inputs. Mirror /v1/embeddings: substitute a zero vector and keep index `i`.
        # (Rarely hit — pool() returns a zero vector, not [] — but the contract
        # requires alignment.)
        if not vec:
            logger.warning(
                "Empty pooled vector at input index %d — engine returned no vector. "
                "Substituting zero vector to preserve index alignment.",
                i,
            )
            _native_dim = None
            for _v in raw:
                if _v:
                    _native_dim = len(_v)
                    break
            vec = [0.0] * (_native_dim or 768)
        # pool() returns RAW, un-normalized hidden states (no +1e-12 norm
        # protection like embed()), so a degenerate/overflowing state can yield NaN/Inf
        # components. The other three scoring endpoints sanitize their scalar
        # scores, but /v1/pooling shipped its vectors un-checked — and JSONResponse renders
        # with allow_nan=False, so a single non-finite component raised an uncaught
        # ValueError at the return (a 500), and in base64 mode struct.pack packed garbage.
        # Mirror /v1/embeddings: coerce non-finite components to 0.0.
        if any(not math.isfinite(x) for x in vec):
            vec = [x if math.isfinite(x) else 0.0 for x in vec]
        val: str | list[float]
        if req.encoding_format == "base64":
            import base64
            import struct

            packed = struct.pack(f"{len(vec)}f", *vec)
            val = base64.b64encode(packed).decode("ascii")
        else:
            val = vec
        data.append({"object": "pooling", "index": i, "data": val})
        total_tokens += len(tok.encode(texts[i])) if tok else max(1, len(texts[i]) // 4)

    return JSONResponse(
        {
            "object": "list",
            "data": data,
            "model": req.model,
            "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
        }
    )


# ── /v1/score ────────────────────────────────────────────────────────────────


@router.post("/score", response_model=None)
async def create_score(req: ScoreRequest, request: Request):
    _check_permission(request, "can_infer")
    _check_model_access(request, req.model)
    texts_a = req.text_1 if isinstance(req.text_1, list) else [req.text_1]
    texts_b = req.text_2 if isinstance(req.text_2, list) else [req.text_2]

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

    # dot/euclidean need RAW (un-normalized) vectors; cosine needs unit vectors.
    _norm_for_score = req.scoring_type == "cosine"
    try:
        emb_a = await _get_embeddings(engine, texts_a, normalize=_norm_for_score)
        emb_b = await _get_embeddings(engine, texts_b, normalize=_norm_for_score)
    except MemoryError:
        logger.error("Score OOM", exc_info=True)
        raise HTTPException(status_code=507, detail="Out of GPU memory") from None
    except ValueError as e:
        # bad-input errors should be 400 not 500
        logger.warning(f"Score validation: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Score error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Scoring failed") from None

    data = []
    total_tokens = 0
    tok = getattr(engine, "_tokenizer", None)
    try:
        for i, (a, b) in enumerate(zip(emb_a, emb_b, strict=False)):
            score = _compute_similarity(a, b, req.scoring_type)
            if not math.isfinite(score):  # avoid a bare NaN JSON literal (invalid)
                score = 0.0
            data.append({"object": "score", "index": i, "score": score})
            total_tokens += (
                len(tok.encode(texts_a[i])) if tok else max(1, len(texts_a[i]) // 4)
            )
            total_tokens += (
                len(tok.encode(texts_b[i])) if tok else max(1, len(texts_b[i]) // 4)
            )
    except ValueError as e:
        logger.error(f"Score computation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from None

    return JSONResponse(
        {
            "object": "list",
            "data": data,
            "model": req.model,
            "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
        }
    )


# ── /v1/rerank ───────────────────────────────────────────────────────────────


@router.post("/rerank", response_model=None)
async def create_rerank(req: RerankRequest, request: Request):
    _check_permission(request, "can_infer")
    _check_model_access(request, req.model)
    # Truncate long documents
    truncated_docs = []
    for doc in req.documents:
        if len(doc) > _MAX_DOCUMENT_LENGTH:
            logger.debug(
                "Truncating document from %d to %d chars",
                len(doc),
                _MAX_DOCUMENT_LENGTH,
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
        raise HTTPException(status_code=507, detail="Out of GPU memory") from None
    except ValueError as e:
        logger.warning(f"Rerank validation: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Rerank error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Reranking failed") from None

    # Score each document against the query using cosine similarity
    # Since embeddings are L2-normalized, cosine similarity = dot product
    scored = []
    try:
        if not query_emb:
            # If query embedding is empty, all documents get relevance 0.0
            scored = [(i, 0.0) for i in range(len(doc_embs))]
        else:
            for i, doc_emb in enumerate(doc_embs):
                if not doc_emb:
                    scored.append((i, 0.0))
                    continue
                cos_sim = _compute_similarity(query_emb, doc_emb, "cosine")
                # Scale cosine [-1, 1] to relevance [0, 1]
                relevance = (cos_sim + 1.0) / 2.0
                # a NaN/Inf component in an embedding (overflow/degenerate
                # pooled state — the cosine norm==0 guard doesn't catch Inf) makes
                # relevance non-finite, which (a) CORRUPTS the sort below — every
                # comparison against NaN is False so Timsort can't place it, leaving a
                # position-dependent broken order, and the top_n slice then drops
                # genuinely high-scoring docs — and (b) serializes as a bare `NaN` JSON
                # literal (invalid per RFC 8259). Coerce to 0.0 (least-relevant, sorts
                # last) so a bad embedding can't mis-rank the whole result set.
                if not math.isfinite(relevance):
                    relevance = 0.0
                scored.append((i, relevance))
    except ValueError as e:
        logger.error(f"Rerank scoring error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from None

    # Sort by relevance descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Apply top_n
    if req.top_n is not None and req.top_n > 0:
        scored = scored[: req.top_n]

    tok = getattr(engine, "_tokenizer", None)
    total_tokens = len(tok.encode(req.query)) if tok else max(1, len(req.query) // 4)
    for doc in truncated_docs:
        total_tokens += len(tok.encode(doc)) if tok else max(1, len(doc) // 4)

    results: list[dict[str, Any]] = []
    for _rank, (idx, score) in enumerate(scored):
        item: dict[str, Any] = {
            "index": idx,
            "relevance_score": round(score, 6),
        }
        if req.return_documents:
            item["document"] = {"text": req.documents[idx]}
        results.append(item)

    return JSONResponse(
        {
            "id": f"rerank-{int(time.time())}",
            "object": "list",
            "model": req.model,
            "results": results,
            "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
        }
    )


# ── /v1/classify ─────────────────────────────────────────────────────────────


@router.post("/classify", response_model=None)
async def classify_input(req: ClassifyRequest, request: Request):
    _check_permission(request, "can_infer")
    _check_model_access(request, req.model)
    """Classify input text using a model's hidden states.

    Returns class probabilities computed from the model's pooled
    hidden representation using a softmax over label embeddings.
    Uses temperature scaling (temperature=0.07) on cosine similarities
    to produce well-separated probability distributions.
    """
    engine = await _resolve_engine(req.model)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found")

    try:
        input_emb = (await _get_embeddings(engine, [req.input]))[0]
        label_embs = await _get_embeddings(engine, req.labels)
    except MemoryError:
        logger.error("Classify OOM", exc_info=True)
        raise HTTPException(status_code=507, detail="Out of GPU memory") from None
    except Exception as e:
        logger.error(f"Classify error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Classification failed") from None

    # Compute cosine similarity between input and each label embedding.
    # Temperature scaling (0.07) sharpens the distribution so the top label
    # gets a meaningful probability rather than a near-uniform spread.
    temperature = 0.07
    scores = []
    try:
        for label_emb in label_embs:
            sim = _compute_similarity(input_emb, label_emb, "cosine")
            _logit = sim / temperature
            # a non-finite sim would make the softmax below all-NaN, corrupting
            # the ENTIRE classification distribution (exp(NaN)=NaN). Coerce to a very
            # negative logit (≈ prob 0 for that label) so one bad label can't poison all.
            if not math.isfinite(_logit):
                _logit = -1e30
            scores.append(_logit)
    except ValueError as e:
        logger.error(f"Classify scoring error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from None

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

    results: list[dict[str, Any]] = []
    for i, (label, prob) in enumerate(zip(req.labels, probs, strict=False)):
        results.append({"label": label, "score": round(prob, 6), "index": i})

    results.sort(key=lambda x: x["score"], reverse=True)

    tok = getattr(engine, "_tokenizer", None)
    total_tokens = len(tok.encode(req.input)) if tok else max(1, len(req.input) // 4)
    for label in req.labels:
        total_tokens += len(tok.encode(label)) if tok else max(1, len(label) // 4)

    return JSONResponse(
        {
            "model": req.model,
            "results": results,
            "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
        }
    )


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
        except KeyError:
            # Not registered — fall through to the single-engine path below
            # (single-engine/stub mode). Mirrors embeddings._resolve_embedding_engine.
            pass
        except Exception:
            # The model IS registered but failed to load. Do NOT fall through to
            # the default engine — serving pooling/score/rerank/classify from a
            # DIFFERENT model than the caller asked for is silently incorrect
            # (the wrong-model-serving class hardened in embeddings.py).
            logger.warning(
                "Failed to load engine for scoring model %r", model_id, exc_info=True
            )
            return None

    engine = get_engine()
    if engine and engine.is_loaded:
        return engine
    return None


async def _get_embeddings(
    engine, texts: list[str], normalize: bool = True
) -> list[list[float]]:
    """Get embeddings for scoring tasks.

    ``normalize=True`` (default) L2-normalizes — correct for cosine similarity.
    ``normalize=False`` returns raw pooled vectors — REQUIRED for ``dot`` and
    ``euclidean`` scoring: on unit vectors a dot product collapses to cosine and
    euclidean distance becomes a monotone re-encoding of cosine, so both would
    silently return cosine-equivalent scores instead of the requested metric.

    Uses engine.embed() when available; falls back to raw hidden-state extraction
    + mean pooling.
    """
    if hasattr(engine, "embed"):
        import asyncio
        import functools

        from yunshu_engine.mlx_executor import get_mlx_executor

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            get_mlx_executor(),
            functools.partial(engine.embed, texts, normalize=normalize),
        )

    return await _fallback_embeddings(engine, texts, "MEAN", normalize=normalize)


async def _get_hidden_states(
    engine, texts: list[str], pooling_type: str
) -> list[list[float]]:
    """Get pooled hidden states (NOT normalized) for the pooling endpoint.

    Uses engine.pool() when available for proper pooling support.
    Falls back to engine.embed() only when no pooling-specific method exists.
    """
    if hasattr(engine, "pool"):
        import asyncio

        from yunshu_engine.mlx_executor import get_mlx_executor

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            get_mlx_executor(), engine.pool, texts, pooling_type
        )

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
    import asyncio

    import mlx.core as mx

    from yunshu_engine.mlx_executor import get_mlx_executor

    tokenizer = getattr(engine, "_tokenizer", None)
    model = getattr(engine, "_model", None)
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

            # resolve the TRUE hidden-state backbone (mirrors the engine's
            # _get_backbone). For mlx-vlm models, model.language_model is a LanguageModel
            # that applies lm_head → returns vocab-space LOGITS, not hidden states; its
            # inner `.model` is the real text backbone. The old resolution returned the
            # wrapper (or even `model.model`) and pooled LOGITS silently → wrong-dimensioned,
            # semantically meaningless embeddings (the logits→backbone bug, never
            # propagated into this scoring fallback sibling). Descend to language_model.model
            # first, then the standard model.model / body / transformer / backbone.
            backbone = None
            _lm = getattr(model, "language_model", None)
            if _lm is not None:
                _inner = getattr(_lm, "model", None)
                if _inner is not None and callable(_inner):
                    backbone = _inner
            if backbone is None:
                for _attr in ("model", "body", "transformer", "backbone"):
                    _cand = getattr(model, _attr, None)
                    if _cand is not None and callable(_cand):
                        backbone = _cand
                        break
            try:
                out = backbone(input_ids) if backbone is not None else model(input_ids)
            except TypeError:
                out = model(input_ids)

            # Extract hidden states (handles mx.array, tuple, LanguageModelOutput,
            # and — last resort — CausalLM logits).
            if isinstance(out, mx.array):
                hidden = out
            elif isinstance(out, (tuple, list)):
                hidden = out[0]
            elif hasattr(out, "last_hidden_state"):
                hidden = out.last_hidden_state
            elif hasattr(out, "hidden_states") and out.hidden_states:
                hidden = out.hidden_states[-1]
            elif hasattr(out, "logits"):
                # the backbone resolution above should prevent reaching here.
                # Pooling vocab-space logits yields a meaningless embedding — warn loudly
                # (was SILENT) so a mis-resolved model surfaces instead of returning a
                # confidently-wrong vector.
                logger.warning(
                    "embedding fallback got logits (no hidden-state backbone for %s) — "
                    "pooled vector is approximate/meaningless",
                    type(model).__name__,
                )
                hidden = out.logits
            else:
                hidden = out

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
                pooled_list[0]
                if isinstance(pooled_list, list) and len(pooled_list) == 1
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
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    elif method == "dot":
        return sum(x * y for x, y in zip(a, b, strict=False))
    elif method == "euclidean":
        return -math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=False)))
    else:
        raise ValueError(f"Unknown scoring method: {method}")
