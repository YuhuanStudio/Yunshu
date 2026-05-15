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

from yunshu_engine.tracing import get_inference_tracer, get_structured_logger

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

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
    priority: int = Field(default=0, ge=0, le=100)
    n: int = 1
    logits_processors: Optional[list] = None  # SAMP-2: User-provided custom logits processors


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

    # Structured tracing + logging
    tracer = get_inference_tracer()
    slog = get_structured_logger()
    trace_id = f"cmpl-{uuid.uuid4().hex[:16]}"
    tracer.start_trace(trace_id, metadata={
        "model": req.model,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": req.stream,
        "endpoint": "/completions",
    })
    tracer.span(trace_id, "prefill", {"model": req.model})
    slog.info("inference_request", model=req.model, trace_id=trace_id,
              max_tokens=req.max_tokens, stream=req.stream)

    if req.stream:
        return StreamingResponse(
            _stream_completion(engine, prompt, req, completion_id, request, json_schema=json_schema, trace_id=trace_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming
    from yunshu_engine.batched_engine import BatchedEngine
    is_batched = isinstance(engine, BatchedEngine)

    loaded_adapter = _apply_lora_adapter(engine, req.lora_adapter)
    try:
        async def _gen_one(idx: int):
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
                    priority=req.priority,
                    logits_processors=req.logits_processors,
                )
                text = result.text
                pt = result.prompt_tokens
                ct = result.completion_tokens
                fr = result.finish_reason
                rt = getattr(result, 'reasoning_tokens', 0)
                lp = None
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
                    reasoning_effort=req.reasoning_effort,
                    xtc_probability=req.xtc_probability,
                    xtc_threshold=req.xtc_threshold,
                    spec_decode=req.spec_decode,
                    json_schema=json_schema,
                    logprobs=req.logprobs > 0,
                    top_logprobs=req.top_logprobs,
                    priority=req.priority,
                    logits_processors=req.logits_processors,
                )
                text = state.generated_text
                pt = state.prompt_token_count
                ct = state.completion_token_count
                fr = state.finish_reason or "stop"
                rt = getattr(state, 'reasoning_tokens', 0)
                lp = None
                if req.logprobs > 0:
                    lp = _format_logprobs(
                        state, getattr(engine, '_tokenizer', None), req.logprobs
                    )

            if req.echo:
                text = prompt + text
            _gen_result = result if is_batched else state
            _cached = getattr(_gen_result, 'cached_tokens', 0)
            return idx, pt, ct, fr, rt, lp, text, _cached

        n = max(req.n, 1)
        if n == 1:
            results = [await _gen_one(0)]
        else:
            import asyncio
            results = await asyncio.gather(*[_gen_one(i) for i in range(n)])
            results.sort(key=lambda x: x[0])

        prompt_tokens = results[0][1]
        total_completion_tokens = sum(r[2] for r in results)
        total_reasoning_tokens = sum(r[4] for r in results)
        max_cached_tokens = max(r[7] for r in results)
        max_finish_reason = results[0][3]

        choices = []
        for idx, pt, ct, fr, rt, lp, text, _cached in results:
            choices.append({
                "index": idx,
                "text": text,
                "finish_reason": fr,
                **({"logprobs": lp} if lp else {}),
            })

        # End tracing
        tracer.end_span(trace_id, "prefill")
        tracer.end_trace(trace_id, result={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "finish_reason": max_finish_reason,
        })
        slog.info("inference_complete", model=req.model, trace_id=trace_id,
                  prompt_tokens=prompt_tokens, completion_tokens=total_completion_tokens)

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": prompt_tokens + total_completion_tokens,
        }
        if total_reasoning_tokens:
            usage["completion_tokens_details"] = {"reasoning_tokens": total_reasoning_tokens}
        if max_cached_tokens > 0:
            usage["prompt_tokens_details"] = {"cached_tokens": max_cached_tokens}

        return JSONResponse({
            "id": completion_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": choices,
            "usage": usage,
        })
    except MemoryError:
        raise HTTPException(status_code=507, detail="Insufficient GPU memory")
    except Exception as e:
        logger.error(f"Completions generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        _release_lora_adapter(engine, loaded_adapter)


async def _stream_completion(
    engine, prompt, req, completion_id, request, json_schema=None, trace_id=None
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
    # Track reasoning tokens per-choice to avoid overwrite across n>1 choices
    reasoning_tok_per_choice: dict[int, int] = {}
    cached_tok = 0
    n = max(req.n, 1)

    async def _stream_choice(choice_idx: int):
        nonlocal prompt_tok, completion_tok, cached_tok
        if req.echo:
            yield format_openai_chunk(
                completion_id=completion_id,
                model=req.model,
                delta_content=prompt,
                choice_index=choice_idx,
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
                priority=req.priority,
                logits_processors=req.logits_processors,
            ):
                if hasattr(output, 'prompt_tokens') and output.prompt_tokens:
                    prompt_tok = output.prompt_tokens
                if output.new_text:
                    completion_tok += 1
                _choice_reasoning = getattr(output, 'reasoning_tokens', 0)
                reasoning_tok_per_choice[choice_idx] = _choice_reasoning
                if hasattr(output, 'cached_tokens') and output.cached_tokens:
                    cached_tok = max(cached_tok, output.cached_tokens)
                # Format logprobs for this token if present
                _chunk_logprobs = None
                if output.logprobs:
                    _chunk_logprobs = _format_streaming_logprobs(output.logprobs)
                yield format_openai_chunk(
                    completion_id=completion_id,
                    model=req.model,
                    delta_content=output.new_text,
                    finish_reason=output.finish_reason,
                    choice_index=choice_idx,
                    logprobs=_chunk_logprobs,
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
                enable_thinking=req.enable_thinking,
                thinking_budget=req.thinking_budget,
                reasoning_effort=req.reasoning_effort,
                stop_token_ids=req.stop_token_ids,
                xtc_probability=req.xtc_probability,
                xtc_threshold=req.xtc_threshold,
                spec_decode=req.spec_decode,
                json_schema=json_schema,
                priority=req.priority,
                logprobs=req.logprobs,
                top_logprobs=req.top_logprobs,
                logits_processors=req.logits_processors,
            ):
                if hasattr(output, 'prompt_token_count') and output.prompt_token_count:
                    prompt_tok = output.prompt_token_count
                if hasattr(output, 'token_text') and output.token_text:
                    completion_tok += 1
                _chunk_lp = _format_streaming_logprobs(output.logprobs) if req.logprobs and hasattr(output, 'logprobs') else None
                yield format_openai_chunk(
                    completion_id=completion_id,
                    model=req.model,
                    delta_content=output.token_text,
                    finish_reason=output.finish_reason,
                    choice_index=choice_idx,
                    logprobs=_chunk_lp,
                )

    async def _token_source():
        # Stream each choice sequentially (matches OpenAI spec behavior)
        for choice_idx in range(n):
            async for chunk in _stream_choice(choice_idx):
                yield chunk

        if include_usage:
            # Sum reasoning tokens across all choices for total usage
            _total_reasoning = sum(reasoning_tok_per_choice.values())
            yield format_openai_usage_chunk(
                completion_id=completion_id,
                model=req.model,
                prompt_tokens=prompt_tok,
                completion_tokens=completion_tok,
                reasoning_tokens=_total_reasoning,
                cached_tokens=cached_tok,
            )

        yield format_openai_done()

    loaded_adapter = _apply_lora_adapter(engine, req.lora_adapter)
    # Register with request tracker for cancellation support
    from yunshu_engine.request_tracker import get_request_tracker
    _tracker = get_request_tracker()
    _tracker_gen = _tracker.register(completion_id, req.model)
    try:
        async for event in with_sse_keepalive(
            _token_source(),
            http_request=request,
            cancel_event=_tracker_gen.cancel_event,
        ):
            yield event.encode("utf-8")
    except MemoryError:
        yield f"data: {{\"error\": {{\"message\": \"Insufficient GPU memory\", \"type\": \"server_error\"}}}}\n\n".encode()
        yield b"data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Completions streaming error: {e}", exc_info=True)
        yield f"data: {{\"error\": {{\"message\": \"Internal server error\", \"type\": \"server_error\"}}}}\n\n".encode()
        yield b"data: [DONE]\n\n"
    finally:
        _release_lora_adapter(engine, loaded_adapter)
        _tracker.unregister(completion_id)


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


def _format_streaming_logprobs(logprobs_list: list[dict]) -> dict | None:
    """Format per-token logprobs from streaming GenerationOutput into OpenAI Completions format.

    In streaming mode, each GenerationOutput has at most 1 logprob entry.
    Returns the single-token logprobs dict in OpenAI completions format, or None.
    """
    if not logprobs_list:
        return None
    entries = []
    for lp_entry in logprobs_list:
        if not isinstance(lp_entry, dict):
            continue
        token_str = lp_entry.get("token", "")
        top_lps = lp_entry.get("top_logprobs", [])
        # Decode top_logprobs token_ids to strings
        decoded_top = []
        for tlp in top_lps:
            decoded_top.append({
                "token": tlp.get("token", ""),
                "logprob": tlp.get("logprob", 0.0),
            })
        entries.append({
            "token": token_str,
            "logprob": lp_entry.get("logprob", 0.0),
            "top_logprobs": decoded_top,
        })
    if not entries:
        return None
    return {
        "tokens": [e["token"] for e in entries],
        "token_logprobs": [e["logprob"] for e in entries],
        "top_logprobs": [e["top_logprobs"] for e in entries],
    }
