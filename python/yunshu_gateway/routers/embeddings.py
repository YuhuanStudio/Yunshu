"""OpenAI Embeddings API compatible router.

Supports text embedding generation for semantic search, clustering, etc.
Uses MLX-native model inference (BGE, E5, Nomic, etc.).
"""
from __future__ import annotations

import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..engine import get_engine, get_model_manager

router = APIRouter(tags=["embeddings"])


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: str = "float"  # float, base64
    dimensions: Optional[int] = None


@router.post("/embeddings", response_model=None)
async def create_embedding(req: EmbeddingRequest):
    """Generate embeddings for the given input text(s)."""
    texts = req.input if isinstance(req.input, list) else [req.input]

    if not texts:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

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
        embeddings = await _generate_embeddings(engine, texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Format response
    data = []
    total_tokens = 0
    for i, (emb, text) in enumerate(zip(embeddings, texts)):
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
        # Use tokenizer for accurate token count, fallback to word count
        tok = getattr(engine, '_tokenizer', None) if engine else None
        if tok:
            total_tokens += len(tok.encode(text))
        else:
            total_tokens += max(1, len(text) // 4)

    return JSONResponse({
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    })


async def _resolve_embedding_engine(model_id: str):
    """Find an embedding engine for the given model."""
    from yunshu_engine.batched_engine import BatchedEngine

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
        except (KeyError, Exception):
            pass

    # Single engine
    engine = get_engine()
    if engine and engine.is_loaded:
        return engine

    return None


async def _generate_embeddings(engine, texts: list[str]) -> list[list[float]]:
    """Generate embeddings using the engine.

    Supports:
    1. Engine with embed() method (native embedding model)
    2. BatchedEngine with embed() method
    3. Fallback: use hidden states from the model
    """
    if hasattr(engine, 'embed'):
        return engine.embed(texts)

    # Fallback: use the model's tokenizer + forward pass for last hidden state
    tokenizer = getattr(engine, '_tokenizer', None)
    model = getattr(engine, '_model', None)
    if tokenizer is None or model is None:
        raise RuntimeError("Engine does not support embedding generation")

    import mlx.core as mx
    from ...yunshu_engine.mlx_executor import get_mlx_executor
    import asyncio

    async def _generate_embeddings_async():
        embeddings = []
        for text in texts:
            tokens = tokenizer.encode(text)
            if not tokens:
                embeddings.append([])
                continue

            input_ids = mx.array([tokens])

            # Run on MLX executor thread (not blocking event loop)
            def _forward():
                if hasattr(model, '__call__'):
                    return model(input_ids)
                else:
                    return model(input_ids)

            loop = asyncio.get_running_loop()
            output = await loop.run_in_executor(get_mlx_executor(), _forward)

            # Get last hidden state
            if isinstance(output, mx.array):
                hidden = output
            elif isinstance(output, (tuple, list)):
                hidden = output[0]
            elif hasattr(output, 'last_hidden_state'):
                hidden = output.last_hidden_state
            else:
                hidden = output[0] if isinstance(output, (tuple, list)) else output

            # Mean pooling over sequence length
            pooled = mx.mean(hidden, axis=1)
            pooled_np = pooled.tolist()
            embeddings.append(pooled_np[0] if isinstance(pooled_np, list) and len(pooled_np) == 1 else pooled_np)

        return embeddings

    return await _generate_embeddings_async()
