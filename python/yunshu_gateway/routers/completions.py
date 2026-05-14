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
from .chat import _apply_lora_adapter, _release_lora_adapter

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
    stop_token_ids: Optional[list[int]] = None
    echo: bool = False
    logprobs: int = 0
    top_logprobs: Optional[int] = None
    seed: Optional[int] = None
    spec_decode: bool = False
    enable_thinking: Optional[bool] = None
    thinking_budget: Optional[int] = None
    response_format: Optional[dict] = None
    reasoning_effort: Optional[str] = None
    xtc_probability: float = 0.0
    xtc_threshold: float = 0.0
    lora_adapter: Optional[str] = None
    grammar: Optional[dict] = None  # {"type": "regex", "pattern": "..."} etc.
    user: Optional[str] = None
    n: int = 1


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

    # Extract JSON schema from response_format or grammar
    json_schema = None
    if req.grammar:
        gtype = req.grammar.get("type")
        if gtype == "json":
            schema = req.grammar.get("schema")
            json_schema = schema if schema else {}
        elif gtype in ("regex", "choice", "cfg"):
            json_schema = req.grammar  # Pass through for ConstraintFactory
    if json_schema is None and req.response_format:
        rf = req.response_format
        if rf.get("type") == "json_schema":
            js = rf.get("json_schema")
            if js:
                json_schema = js.get("schema", js)
        elif rf.get("type") == "json_object":
            json_schema = {}

    completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"

    if req.stream:
        return StreamingResponse(
            _stream_completion(engine, prompt, req, completion_id, request, json_schema=json_schema),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming
    from yunshu_engine.batched_engine import BatchedEngine
    is_batched = isinstance(engine, BatchedEngine)

    loaded_adapter = _apply_lora_adapter(engine, req.lora_adapter)
    try:
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
                stop_token_ids=req.stop_token_ids,
                seed=req.seed,
                spec_decode=req.spec_decode,
                enable_thinking=req.enable_thinking,
                thinking_budget=req.thinking_budget,
                json_schema=json_schema,
                reasoning_effort=req.reasoning_effort,
                xtc_probability=req.xtc_probability,
                xtc_threshold=req.xtc_threshold,
                logprobs=req.logprobs,
                top_logprobs=req.top_logprobs,
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
                stop_token_ids=req.stop_token_ids,
                seed=req.seed,
                enable_thinking=req.enable_thinking,
                thinking_budget=req.thinking_budget,
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
    finally:
        _release_lora_adapter(engine, loaded_adapter)


async def _stream_completion(
    engine, prompt, req, completion_id, request, json_schema=None
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
                stop_token_ids=req.stop_token_ids,
                seed=req.seed,
                enable_thinking=req.enable_thinking,
                thinking_budget=req.thinking_budget,
                json_schema=json_schema,
                reasoning_effort=req.reasoning_effort,
                xtc_probability=req.xtc_probability,
                xtc_threshold=req.xtc_threshold,
                spec_decode=req.spec_decode,
                logprobs=req.logprobs,
                top_logprobs=req.top_logprobs,
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
                thinking_budget=req.thinking_budget,
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

    loaded_adapter = _apply_lora_adapter(engine, req.lora_adapter)
    try:
        async for event in with_sse_keepalive(
            _token_source(),
            http_request=request,
        ):
            yield event.encode("utf-8")
    finally:
        _release_lora_adapter(engine, loaded_adapter)


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
