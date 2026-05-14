"""Batch inference endpoint — process multiple requests efficiently.

Leverages Yunshu's continuous batching to handle multiple prompts
in a single batch with optimal GPU utilization.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["batch"])


class BatchItem(BaseModel):
    custom_id: str
    method: str = "POST"
    url: str = "/v1/chat/completions"
    body: dict


class BatchRequest(BaseModel):
    requests: list[BatchItem]
    max_concurrent: int = 4


class BatchResponse(BaseModel):
    id: str
    object: str = "batch"
    status: str
    results: list[dict]


@router.post("/batch", response_model=None)
async def create_batch(req: BatchRequest):
    """Execute a batch of inference requests concurrently."""
    if not req.requests:
        raise HTTPException(status_code=400, detail="Batch cannot be empty")

    if len(req.requests) > 100:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(req.requests)} exceeds limit of 100",
        )

    batch_id = f"batch_{uuid.uuid4().hex[:24]}"
    semaphore = asyncio.Semaphore(req.max_concurrent)

    async def _process_item(item: BatchItem) -> dict:
        async with semaphore:
            return await _execute_batch_item(item)

    tasks = [_process_item(item) for item in req.requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed = []
    errors = 0
    for item, result in zip(req.requests, results):
        if isinstance(result, Exception):
            processed.append({
                "custom_id": item.custom_id,
                "status": "error",
                "error": str(result),
            })
            errors += 1
        else:
            processed.append({
                "custom_id": item.custom_id,
                "status": "success",
                "response": result,
            })

    return JSONResponse({
        "id": batch_id,
        "object": "batch",
        "status": "completed",
        "total": len(processed),
        "succeeded": len(processed) - errors,
        "failed": errors,
        "results": processed,
    })


async def _execute_batch_item(item: BatchItem) -> dict:
    """Execute a single batch item."""
    body = item.body
    url = item.url

    if url == "/v1/chat/completions":
        return await _execute_chat_completion(body)
    elif url == "/v1/completions":
        return await _execute_completion(body)
    elif url == "/v1/embeddings":
        return await _execute_embedding(body)
    else:
        raise ValueError(f"Unsupported batch URL: {url}")


async def _execute_chat_completion(body: dict) -> dict:
    """Execute a chat completion request."""
    from yunshu_engine.batched_engine import BatchedEngine
    from ..engine import get_engine, get_engine_for_model

    model = body.get("model", "")
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 512)
    temperature = body.get("temperature", 0.7)
    top_p = body.get("top_p", 1.0)
    stop = body.get("stop")

    engine = get_engine()
    is_batched = isinstance(engine, BatchedEngine) if engine else False

    if engine is None or not engine.is_loaded or not engine.resolve_model_id(model):
        try:
            engine = await get_engine_for_model(model)
            from yunshu_engine.batched_engine import BatchedEngine
            is_batched = isinstance(engine, BatchedEngine)
        except (KeyError, Exception):
            raise ValueError(f"Model '{model}' not available")

    if is_batched:
        result = await engine.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top,
            top_k=body.get("top_k", 0),
            seed=body.get("seed"),
            stop=stop,
            enable_thinking=body.get("enable_thinking"),
            thinking_budget=body.get("thinking_budget"),
            repetition_penalty=body.get("repetition_penalty", 1.0),
            frequency_penalty=body.get("frequency_penalty", 0.0),
            presence_penalty=body.get("presence_penalty", 0.0),
        )
        text = result.text
        prompt_tokens = result.prompt_tokens
        completion_tokens = result.completion_tokens
        finish_reason = result.finish_reason or "stop"
    else:
        state = await engine.generate(
            prompt=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=body.get("top_k", 0),
            seed=body.get("seed"),
            stop=stop,
            enable_thinking=body.get("enable_thinking"),
            thinking_budget=body.get("thinking_budget"),
            repetition_penalty=body.get("repetition_penalty", 1.0),
        )
        text = state.generated_text
        prompt_tokens = state.prompt_token_count
        completion_tokens = state.completion_token_count
        finish_reason = state.finish_reason or "stop"

    import time as _time
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(_time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def _execute_completion(body: dict) -> dict:
    """Execute a text completion request."""
    from ..engine import get_engine, get_engine_for_model
    from yunshu_engine.batched_engine import BatchedEngine

    model = body.get("model", "")
    prompt = body.get("prompt", "")
    max_tokens = body.get("max_tokens", 128)
    temperature = body.get("temperature", 0.7)

    engine = get_engine()
    if engine is None or not engine.is_loaded or not engine.resolve_model_id(model):
        try:
            engine = await get_engine_for_model(model)
        except (KeyError, Exception):
            raise ValueError(f"Model '{model}' not available")

    is_batched = isinstance(engine, BatchedEngine)
    if is_batched:
        result = await engine.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=body.get("top_p", 1.0),
            top_k=body.get("top_k", 0),
            seed=body.get("seed"),
            repetition_penalty=body.get("repetition_penalty", 1.0),
            stop=body.get("stop"),
        )
        text = result.text
        prompt_tokens = result.prompt_tokens
        completion_tokens = result.completion_tokens
        finish_reason = result.finish_reason
    else:
        state = await engine.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = state.generated_text
        prompt_tokens = state.prompt_token_count
        completion_tokens = state.completion_token_count
        finish_reason = state.finish_reason or "stop"

    import time as _time
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:24]}",
        "object": "text_completion",
        "created": int(_time.time()),
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def _execute_embedding(body: dict) -> dict:
    """Execute an embedding request."""
    raise ValueError("Batch embedding not yet supported")
