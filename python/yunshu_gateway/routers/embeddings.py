from __future__ import annotations
"""OpenAI Embeddings API compatible router.

Supports text embedding generation for semantic search, clustering, etc.
Uses MLX-native model inference (BGE, E5, Nomic, etc.).
When YUNSHU_ANE_EMBEDDINGS=1 is set and ANE is available, embeddings are
computed on the Apple Neural Engine via CoreML for lower latency and
reduced GPU contention.

All embeddings are L2-normalized to unit vectors by default, matching the
OpenAI API contract. Matryoshka dimension truncation is supported via the
``dimensions`` parameter.
"""

import logging
import math
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator

from ..engine import get_engine, get_model_manager
from .models import _check_permission

logger = logging.getLogger(__name__)

router = APIRouter(tags=["embeddings"])

_VALID_ENCODING_FORMATS = {"float", "base64"}


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: str = "float"  # float, base64
    dimensions: Optional[int] = None

    @model_validator(mode="after")
    def validate_request(self):
        if not self.model or not self.model.strip():
            raise ValueError("model: field is required and cannot be empty")
        if self.encoding_format not in _VALID_ENCODING_FORMATS:
            raise ValueError(
                f"encoding_format: must be one of {', '.join(sorted(_VALID_ENCODING_FORMATS))}, "
                f"got '{self.encoding_format}'"
            )
        # Validate input: string must be non-empty, list must have elements
        if isinstance(self.input, str) and not self.input.strip():
            raise ValueError("input: cannot be empty or whitespace-only")
        if isinstance(self.input, list) and not self.input:
            raise ValueError("input: cannot be an empty list")
        # Validate dimensions
        if self.dimensions is not None:
            if self.dimensions <= 0:
                raise ValueError("dimensions must be a positive integer")
            if self.dimensions > 8192:
                raise ValueError("dimensions must not exceed 8192")
        return self


@router.post("/embeddings", response_model=None)
async def create_embedding(req: EmbeddingRequest, request: Request):
    _check_permission(request, "can_infer")
    """Generate embeddings for the given input text(s)."""
    texts = req.input if isinstance(req.input, list) else [req.input]

    if not texts:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    # Reject empty strings in the list
    for i, t in enumerate(texts):
        if not t.strip():
            raise HTTPException(
                status_code=400,
                detail=f"Input at index {i} is empty or whitespace-only",
            )

    if len(texts) > 2048:
        raise HTTPException(
            status_code=400,
            detail=f"Too many inputs: {len(texts)} > 2048",
        )

    # Resolve embedding engine
    engine = await _resolve_embedding_engine(req.model)
    if engine is None:
        raise HTTPException(
            status_code=404,
            detail=f"Embedding model '{req.model}' not found",
        )

    try:
        embeddings = await _generate_embeddings(engine, texts, model_id=req.model)
    except MemoryError:
        logger.error("Embedding generation OOM", exc_info=True)
        raise HTTPException(status_code=507, detail="Out of GPU memory")
    except Exception as e:
        logger.error(f"Embedding error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Embedding generation failed")

    # Resolve tokenizer for token counting (shared across iterations)
    tokenizer = getattr(engine, '_tokenizer', None) if engine else None

    # Format response
    data = []
    total_tokens = 0
    for i, (emb, text) in enumerate(zip(embeddings, texts)):
        # Engine.embed() always returns a vector (zero vector for empty input),
        # so empty embeddings should not occur.  Warn and substitute a zero
        # vector to preserve the 1:1 index correspondence required by the
        # OpenAI API.  Skipping entries would create index gaps that break
        # client expectations.
        if not emb:
            logger.warning(
                "Empty embedding at index %d — engine returned no vector. "
                "Substituting zero vector to preserve index alignment.",
                i,
            )
            # Determine a reasonable dimension for the zero vector fallback
            _native_dim = None
            for _e in embeddings:
                if _e:
                    _native_dim = len(_e)
                    break
            _dim = req.dimensions or _native_dim or 768
            emb = [0.0] * _dim

        # Truncate to requested dimensions (Matryoshka embedding support)
        if req.dimensions is not None and req.dimensions > 0:
            if len(emb) < req.dimensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Requested dimensions {req.dimensions} exceeds model's "
                           f"native embedding dimension {len(emb)}",
                )
            emb = emb[:req.dimensions]
            # Re-normalize after truncation to maintain unit vector property
            norm = math.sqrt(sum(x * x for x in emb))
            if norm > 0:
                emb = [x / norm for x in emb]

        if req.encoding_format == "base64":
            import base64
            import struct
            packed = struct.pack(f"{len(emb)}f", *emb)
            emb_value = base64.b64encode(packed).decode("ascii")
        else:
            emb_value = emb

        data.append({
            "object": "embedding",
            "index": i,
            "embedding": emb_value,
        })
        # Use tokenizer for accurate token count, fallback to word count.
        # Exclude special tokens (BOS/EOS) from the count to match OpenAI's
        # behavior where prompt_tokens only includes content tokens.
        if tokenizer:
            try:
                total_tokens += len(tokenizer.encode(text, add_special_tokens=False))
            except TypeError:
                # Tokenizer doesn't support add_special_tokens — some MLX
                # tokenizers only accept a single positional argument.
                # The encode() result typically includes BOS (+1) and may
                # include EOS (+1), so we subtract up to 2 as a heuristic.
                # Guard against negative count for very short texts.
                raw_count = len(tokenizer.encode(text))
                total_tokens += max(1, raw_count - 2)
        else:
            total_tokens += max(1, len(text) // 4)

    _record_embedding_metrics(total_tokens)

    return JSONResponse({
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    })


def _record_embedding_metrics(prompt_tokens: int) -> None:
    """Record embedding request metrics."""
    try:
        from ..middleware.metrics import get_metrics
        get_metrics().record_tokens(prompt_tokens, 0)
        get_metrics().record_inference()
    except Exception:
        pass


async def _resolve_embedding_engine(model_id: str):
    """Find an embedding engine for the given model."""

    # Try model manager first
    manager = get_model_manager()
    if manager is not None:
        # Direct lookup
        entry = manager.get_entry(model_id)
        if entry is not None and entry.is_loaded and entry.engine is not None:
            return entry.engine

        # Try loading
        try:
            engine = await manager.get_engine(model_id)
            return engine
        except KeyError:
            logger.debug(f"Model not registered: {model_id}")
            # Not registered — fall through to single-engine only if manager
            # has no entries at all (single-engine mode with manager stub)
        except Exception:
            logger.warning(f"Failed to load engine for {model_id}", exc_info=True)
            # Do NOT fall through to single-engine — the user asked for a
            # specific model via the manager, and serving embeddings from a
            # different model would be silently incorrect.
            return None

    # Single engine
    engine = get_engine()
    if engine and engine.is_loaded:
        return engine

    return None


async def _generate_embeddings(engine, texts: list[str], model_id: str = "") -> list[list[float]]:
    """Generate embeddings using the engine.

    Supports:
    1. ANE path: when YUNSHU_ANE_EMBEDDINGS=1 and ANE is available
    2. Engine with embed() method (native embedding model, already normalized)
    3. Fallback: use hidden states from the model with L2 normalization
    """
    # ── ANE path: offload to Apple Neural Engine via CoreML ──
    if os.environ.get("YUNSHU_ANE_EMBEDDINGS", "").strip() in ("1", "true", "yes"):
        try:
            from yunshu_engine.ane_embedding import get_ane_processor
            proc = get_ane_processor()
            if proc is not None:
                logger.debug("Using ANE for embedding inference (model_id=%s)", model_id)
                import asyncio
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, proc.embed, texts)
        except Exception as exc:
            logger.warning(
                "ANE embedding failed, falling back to GPU: %s", exc, exc_info=True,
            )

    # ── Engine with embed() method (BatchedEngine.embed normalizes by default) ──
    if hasattr(engine, 'embed'):
        import asyncio
        from yunshu_engine.mlx_executor import get_mlx_executor
        loop = asyncio.get_running_loop()
        # embed() is synchronous and does GPU work — run on MLX executor thread
        return await loop.run_in_executor(get_mlx_executor(), engine.embed, texts)

    # Fallback: use the model's tokenizer + forward pass for last hidden state
    tokenizer = getattr(engine, '_tokenizer', None)
    model = getattr(engine, '_model', None)
    if tokenizer is None or model is None:
        raise RuntimeError("Engine does not support embedding generation")

    import mlx.core as mx
    from yunshu_engine.mlx_executor import get_mlx_executor
    import asyncio

    loop = asyncio.get_running_loop()

    def _generate_all():
        embeddings = []
        for text in texts:
            tokens = tokenizer.encode(text)
            if not tokens:
                # Return zero vector — dimension unknown without a forward pass,
                # so we use a common default. Callers should check for all-zeros.
                embeddings.append([0.0] * 768)
                continue

            input_ids = mx.array([tokens])
            output = model(input_ids)

            # Extract hidden states
            hidden = _extract_hidden(output)

            # Mean pooling over sequence dimension
            pooled = mx.mean(hidden, axis=1).squeeze(0)

            # L2 normalize (OpenAI API contract)
            norm = mx.sqrt(mx.sum(pooled * pooled) + 1e-12)
            pooled = pooled / norm

            embeddings.append(pooled.tolist())
        return embeddings

    return await loop.run_in_executor(get_mlx_executor(), _generate_all)


def _extract_hidden(output) -> "mx.array":
    """Extract hidden states from various model output formats."""
    import mlx.core as mx

    if isinstance(output, mx.array):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, 'last_hidden_state'):
        return output.last_hidden_state
    if hasattr(output, 'hidden_states') and output.hidden_states:
        return output.hidden_states[-1]
    if hasattr(output, 'logits'):
        # For causal LMs, the logits tensor IS the last layer output (before
        # softmax).  This is a reasonable approximation for models that don't
        # expose explicit hidden states, but it is not optimal — the logits
        # have been projected through the vocabulary head which distorts the
        # embedding space.
        logger.warning(
            "Using logits as embedding — model may not produce optimal embeddings. "
            "Consider using a model with explicit hidden state outputs."
        )
        return output.logits
    raise ValueError(
        "Model output has no hidden states or logits — cannot generate embeddings"
    )
