"""OpenAI Completions API compatible router (text completions, not chat).

Supports:
- Text completions (non-chat)
- Streaming and non-streaming
- Logprobs
- Echo mode
"""
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..engine import get_engine, get_engine_for_model, get_model_manager

logger = logging.getLogger(__name__)
from ..streaming import format_openai_chunk, format_openai_done, format_openai_usage_chunk

router = APIRouter(tags=["completions"])


class StreamOptions(BaseModel):
    """OpenAI stream_options parameter."""
    include_usage: bool = False


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[int]
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logit_bias: Optional[dict[int, float]] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    stop: Optional[list[str]] = None
    echo: bool = False
    logprobs: int = 0
    seed: Optional[int] = None


@router.post("/completions", response_model=None)
async def create_completion(req: CompletionRequest, request: Request):
    """OpenAI-compatible text completion endpoint."""
    engine = get_engine()

    if engine is None or not engine.is_loaded or not engine.resolve_model_id(req.model):
        try:
            engine = await get_engine_for_model(req.model)
        except (KeyError, Exception) as e:
            if engine is None:
                raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found: {e}")
            raise HTTPException(
                status_code=404,
                detail=f"Model '{req.model}' not loaded. Loaded: {engine.model_name}",
            )

    # Convert prompt to text if it's token IDs
    if isinstance(req.prompt, list):
        tokenizer = getattr(engine, '_tokenizer', None)
        if tokenizer:
            prompt = tokenizer.decode(req.prompt)
        else:
            prompt = " ".join(str(t) for t in req.prompt)
    else:
        prompt = req.prompt

    completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"

    if req.stream:
        return StreamingResponse(
            _stream_completion(engine, prompt, req, completion_id, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming
    from yunshu_engine.batched_engine import BatchedEngine
    is_batched = isinstance(engine, BatchedEngine)

    if is_batched:
        result = await engine.generate(
            prompt=prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            min_p=req.min_p,
            repetition_penalty=req.repetition_penalty,
            frequency_penalty=req.frequency_penalty,
            presence_penalty=req.presence_penalty,
            logit_bias=req.logit_bias,
            stop=req.stop,
            seed=req.seed,
        )
        text = result.text
        prompt_tokens = result.prompt_tokens
        completion_tokens = result.completion_tokens
        finish_reason = result.finish_reason
        logprobs_data = None
    else:
        state = await engine.generate(
            prompt=prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            min_p=req.min_p,
            repetition_penalty=req.repetition_penalty,
            frequency_penalty=req.frequency_penalty,
            presence_penalty=req.presence_penalty,
            logit_bias=req.logit_bias,
            stop=req.stop,
            seed=req.seed,
        )
        text = state.generated_text
        prompt_tokens = state.prompt_token_count
        completion_tokens = state.completion_token_count
        finish_reason = state.finish_reason or "stop"
        logprobs_data = None
        if req.logprobs > 0:
            logprobs_data = _format_logprobs(
                state, getattr(engine, '_tokenizer', None), req.logprobs
            )

    if req.echo:
        text = prompt + text

    return JSONResponse({
        "id": completion_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "text": text,
            "finish_reason": finish_reason,
            **({"logprobs": logprobs_data} if logprobs_data else {}),
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })


async def _stream_completion(
    engine, prompt, req, completion_id, request
) -> AsyncIterator[bytes]:
    """SSE streaming for text completions with keepalive and disconnect detection."""
    from ..streaming import with_sse_keepalive
    from yunshu_engine.batched_engine import BatchedEngine
    is_batched = isinstance(engine, BatchedEngine)
    include_usage = (
        req.stream_options is not None and req.stream_options.include_usage
    )
    prompt_tok = 0
    completion_tok = 0

    async def _token_source():
        if req.echo:
            yield format_openai_chunk(
                completion_id=completion_id,
                model=req.model,
                delta_content=prompt,
            )

        if is_batched:
            async for output in engine.stream_generate(
                prompt=prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                min_p=req.min_p,
                repetition_penalty=req.repetition_penalty,
                frequency_penalty=req.frequency_penalty,
                presence_penalty=req.presence_penalty,
                logit_bias=req.logit_bias,
                stop=req.stop,
                seed=req.seed,
            ):
                if hasattr(output, 'prompt_tokens') and output.prompt_tokens:
                    prompt_tok = output.prompt_tokens
                if output.new_text:
                    completion_tok += 1
                yield format_openai_chunk(
                    completion_id=completion_id,
                    model=req.model,
                    delta_content=output.new_text,
                    finish_reason=output.finish_reason,
                )
        else:
            async for output in engine.generate_stream(
                prompt=prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                min_p=req.min_p,
                repetition_penalty=req.repetition_penalty,
                frequency_penalty=req.frequency_penalty,
                presence_penalty=req.presence_penalty,
                logit_bias=req.logit_bias,
                stop=req.stop,
                seed=req.seed,
            ):
                if hasattr(output, 'prompt_token_count') and output.prompt_token_count:
                    prompt_tok = output.prompt_token_count
                if hasattr(output, 'token_text') and output.token_text:
                    completion_tok += 1
                yield format_openai_chunk(
                    completion_id=completion_id,
                    model=req.model,
                    delta_content=output.token_text,
                    finish_reason=output.finish_reason,
                )

        if include_usage:
            yield format_openai_usage_chunk(
                completion_id=completion_id,
                model=req.model,
                prompt_tokens=prompt_tok,
                completion_tokens=completion_tok,
            )

        yield format_openai_done()

    async for event in with_sse_keepalive(
        _token_source(),
        http_request=request,
    ):
        yield event.encode("utf-8")


def _format_logprobs(state, tokenizer, top_logprobs: int) -> dict | None:
    """Format logprobs from request state into OpenAI Completions format."""
    raw_logprobs = getattr(state, 'logprobs', None)
    if not raw_logprobs:
        return None

    token_logprobs = []
    if isinstance(raw_logprobs, (list, tuple)):
        for lp_entry in raw_logprobs:
            if isinstance(lp_entry, dict):
                token_str = lp_entry.get("token", "")
                if not token_str and tokenizer and "token_id" in lp_entry:
                    try:
                        token_str = tokenizer.decode([lp_entry["token_id"]])
                    except Exception:
                        logger.debug("tokenizer decode failed", exc_info=True)
                token_logprobs.append({
                    "token": token_str,
                    "logprob": lp_entry.get("logprob", 0.0),
                    "top_logprobs": lp_entry.get("top_logprobs", []),
                })
            elif isinstance(lp_entry, (int, float)):
                token_logprobs.append({
                    "token": "",
                    "logprob": float(lp_entry),
                    "top_logprobs": [],
                })

    if not token_logprobs:
        return None

    return {
        "tokens": [e["token"] for e in token_logprobs],
        "token_logprobs": [e["logprob"] for e in token_logprobs],
        "top_logprobs": [e["top_logprobs"] for e in token_logprobs],
    }
