"""Disaggregated serving endpoints — separate prefill and decode phases.

Implements the P/D disaggregation pattern from vLLM/SGLang:
- POST /v1/prefill: Run prompt prefill only, return a cache handle
- POST /v1/decode: Run decode with a prefill cache handle

This enables:
1. Distributed prefill: prefill on a separate node, decode locally
2. Memory management: prefill-heavy workloads can be isolated
3. Pipeline parallelism: overlap prefill of next request with decode of current

Architecture:
  Client → /v1/prefill → Engine.prefill_only() → cache_handle
  Client → /v1/decode → Engine.decode_with_handle(cache_handle) → tokens
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["disaggregate"])

# In-memory cache handle store
_cache_handles: dict[str, dict[str, Any]] = {}


class PrefillRequest(BaseModel):
    model: str = ""
    prompt: str | list[dict] = ""
    max_prefill_tokens: int | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    stop: list[str] | None = None
    seed: int | None = None
    json_schema: dict | str | None = None
    enable_thinking: bool | None = None
    chunk_size: int | None = None


class PrefillResponse(BaseModel):
    id: str = ""
    cache_handle: str = ""
    prompt_tokens: int = 0
    cached_tokens: int = 0
    duration_s: float = 0.0


class DecodeRequest(BaseModel):
    cache_handle: str = ""
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    stop: list[str] | None = None
    seed: int | None = None
    stream: bool = False


class DecodeResponse(BaseModel):
    id: str = ""
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None


def _resolve_engine(request: Request):
    """Resolve the engine from the gateway's engine registry.

    Tries app.state.engine first (for backwards compatibility), then
    falls back to the gateway engine module's get_engine().
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is not None:
        return engine
    try:
        from ..engine import get_engine
        engine = get_engine()
    except Exception:
        pass
    return engine


@router.post("/prefill", response_model=PrefillResponse)
async def prefill(req: PrefillRequest, request: Request):
    """Run prefill only — tokenize and build KV cache, return cache handle."""
    engine = _resolve_engine(request)
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")

    await engine.start()
    tokenizer = engine._tokenizer

    # Tokenize
    if isinstance(req.prompt, list) and req.prompt and isinstance(req.prompt[0], dict):
        tpl_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if req.enable_thinking is not None:
            tpl_kwargs["enable_thinking"] = req.enable_thinking
        prompt_text = tokenizer.apply_chat_template(req.prompt, **tpl_kwargs)
    else:
        prompt_text = req.prompt

    token_ids = tokenizer.encode(prompt_text)
    prompt_tokens = len(token_ids)

    # Determine chunk size: request override > engine config > default 2048
    chunk_size = req.chunk_size or 2048
    # Check if engine's scheduler has external prefill configured (for chunk size)
    try:
        scheduler = getattr(engine._engine_core, "scheduler", None)
        if scheduler is not None:
            chunk_size = req.chunk_size or scheduler.config.prefill_chunk_size
    except Exception:
        logger.debug("failed to read scheduler config for chunk size", exc_info=True)

    # Run prefill
    from yunshu_engine.external_prefill import ExternalPrefiller

    prefiller = ExternalPrefiller(engine._model, tokenizer)

    def _prefill():
        return prefiller.prefill_chunked(
            token_ids=token_ids,
            chunk_size=chunk_size,
        )

    from yunshu_engine.mlx_executor import get_mlx_executor
    executor = get_mlx_executor()
    loop = asyncio.get_running_loop()
    t0 = time.monotonic()
    result = await loop.run_in_executor(executor, _prefill)
    duration_s = time.monotonic() - t0

    # Store cache handle
    handle_id = f"pf-{uuid.uuid4().hex[:12]}"
    _cache_handles[handle_id] = {
        "cache": result.kv_cache,
        "token_ids": token_ids,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": result.cached_tokens,
        "created_at": time.monotonic(),
    }

    return PrefillResponse(
        id=handle_id,
        cache_handle=handle_id,
        prompt_tokens=prompt_tokens,
        cached_tokens=result.cached_tokens,
        duration_s=duration_s,
    )


@router.post("/decode", response_model=DecodeResponse)
async def decode(req: DecodeRequest, request: Request):
    """Run decode with a prefill cache handle."""
    engine = _resolve_engine(request)
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")

    await engine.start()

    # Look up cache handle
    handle_data = _cache_handles.get(req.cache_handle)
    if handle_data is None:
        raise HTTPException(status_code=404, detail=f"Cache handle {req.cache_handle} not found")

    prompt_tokens = handle_data["prompt_tokens"]

    # Use engine's generate with the prefilled prompt
    gen_output = await engine.generate(
        prompt=handle_data["token_ids"],
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        stop=req.stop,
        seed=req.seed,
    )

    # Remove handle after use
    _cache_handles.pop(req.cache_handle, None)

    return DecodeResponse(
        id=f"dec-{uuid.uuid4().hex[:8]}",
        text=gen_output.text,
        prompt_tokens=prompt_tokens,
        completion_tokens=gen_output.completion_tokens,
        finish_reason=gen_output.finish_reason,
    )


@router.get("/cache-handles")
async def list_cache_handles():
    """List active cache handles (debug/monitoring)."""
    now = time.monotonic()
    return {
        "handles": [
            {
                "id": hid,
                "prompt_tokens": data["prompt_tokens"],
                "cached_tokens": data["cached_tokens"],
                "age_s": round(now - data["created_at"], 2),
            }
            for hid, data in _cache_handles.items()
        ],
        "total": len(_cache_handles),
    }


@router.delete("/cache-handles/{handle_id}")
async def delete_cache_handle(handle_id: str):
    """Delete a cache handle to free GPU memory."""
    if handle_id not in _cache_handles:
        raise HTTPException(status_code=404, detail=f"Cache handle {handle_id} not found")
    del _cache_handles[handle_id]
    return {"deleted": handle_id}
