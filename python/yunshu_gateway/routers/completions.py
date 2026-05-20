from __future__ import annotations
"""OpenAI Completions API compatible router (text completions, not chat).

Supports:
- Text completions (non-chat)
- Streaming and non-streaming
- Logprobs
- Echo mode
"""
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Optional

from yunshu_engine.tracing import get_inference_tracer, get_structured_logger

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from ..engine import get_engine, get_engine_for_model
from .chat import _apply_lora_adapter, _release_lora_adapter, _normalize_finish_reason, _parse_response_format

logger = logging.getLogger(__name__)

_MAX_STREAMING_TEXT_BUFFER = 1 * 1024 * 1024
_TRUNCATE_KEEP = 512 * 1024
from ..streaming import format_openai_completion_chunk, format_openai_done, format_openai_completion_usage_chunk
from .models import _check_permission

router = APIRouter(tags=["completions"])


def _record_metrics(prompt_tokens: int, completion_tokens: int) -> None:
    """Record token counts to metrics middleware and server stats for completions endpoint."""
    try:
        from ..middleware.metrics import get_metrics
        get_metrics().record_tokens(prompt_tokens, completion_tokens)
        get_metrics().record_inference()
    except Exception:
        logger.debug("metrics recording failed", exc_info=True)
    try:
        from yunshu_engine.server_metrics import get_server_metrics
        get_server_metrics().record_request_complete(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except Exception:
        logger.debug("server_metrics recording failed", exc_info=True)
    try:
        from yunshu_engine.tracing import get_metrics_v2
        get_metrics_v2().counter("yunshu_tokens_total", {"type": "prompt"}, prompt_tokens)
        get_metrics_v2().counter("yunshu_tokens_total", {"type": "completion"}, completion_tokens)
    except Exception:
        logger.debug("metrics recording failed", exc_info=True)


class StreamOptions(BaseModel):
    """OpenAI stream_options parameter."""
    include_usage: bool = False


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[int]
    max_tokens: int = Field(default=128, ge=1, le=131072)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1, le=131072)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=0.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[dict[int, float]] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    stop: Optional[list[str]] = None
    stop_token_ids: Optional[list[int]] = None
    echo: bool = False
    logprobs: int = Field(default=0, ge=0, le=5)
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=5)
    seed: Optional[int] = None
    spec_decode: bool = False
    enable_thinking: Optional[bool] = None
    thinking_budget: Optional[int] = Field(default=None, ge=1, le=32768)
    response_format: Optional[dict] = None
    reasoning_effort: Optional[str] = None
    xtc_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    xtc_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    lora_adapter: Optional[str] = None
    grammar: Optional[dict] = None  # {"type": "regex", "pattern": "..."} etc.
    user: Optional[str] = None
    suffix: Optional[str] = None  # OpenAI: suffix after inserted text completion
    best_of: Optional[int] = Field(default=None, ge=1, le=128)  # OpenAI: server-side best-of selection
    priority: int = Field(default=0, ge=0, le=100)
    n: int = Field(default=1, ge=1, le=128)
    logits_processors: Optional[list] = None  # SAMP-2: User-provided custom logits processors
    timeout: Optional[float] = Field(default=None, ge=1.0, le=600.0)  # Request timeout in seconds

    def effective_max_tokens(self) -> int:
        """Return max_completion_tokens if set, else max_tokens (OpenAI SDK compat)."""
        return self.max_completion_tokens if self.max_completion_tokens is not None else self.max_tokens

    @model_validator(mode="after")
    def validate_request(self):
        if not self.model or not self.model.strip():
            raise ValueError("model: field is required and cannot be empty")
        # Validate prompt: string must be non-empty, list must have elements
        if isinstance(self.prompt, str) and not self.prompt.strip():
            raise ValueError("prompt: cannot be empty or whitespace-only")
        if isinstance(self.prompt, list) and not self.prompt:
            raise ValueError("prompt: cannot be an empty list")
        if self.top_logprobs is not None and self.logprobs <= 0:
            raise ValueError("top_logprobs: can only be set when logprobs > 0")
        if self.stop and len(self.stop) > 16:
            raise ValueError("stop: maximum 16 stop sequences")
        if self.stop_token_ids and len(self.stop_token_ids) > 16:
            raise ValueError("stop_token_ids: maximum 16 stop token IDs")
        # Validate response_format type if provided
        if self.response_format is not None:
            rf_type = self.response_format.get("type") if isinstance(self.response_format, dict) else None
            if rf_type not in ("json_object", "json_schema", "text", None):
                raise ValueError(f"response_format.type: must be 'json_object', 'json_schema', or 'text', got '{rf_type}'")
        # Validate grammar type if provided
        if self.grammar is not None:
            gtype = self.grammar.get("type") if isinstance(self.grammar, dict) else None
            if gtype not in ("json", "regex", "choice", "cfg", None):
                raise ValueError(f"grammar.type: must be one of 'json', 'regex', 'choice', 'cfg', got '{gtype}'")
        # Validate best_of: must be >= n, and not used with streaming
        if self.best_of is not None:
            if self.best_of < self.n:
                raise ValueError(f"best_of ({self.best_of}) must be >= n ({self.n})")
            if self.stream:
                raise ValueError("best_of is not supported when stream is True")
        return self


@router.post("/completions", response_model=None)
async def create_completion(req: CompletionRequest, request: Request):
    _check_permission(request, "can_infer")
    """OpenAI-compatible text completion endpoint."""
    # Validate stop strings: reject empty strings (would match immediately)
    if req.stop:
        req.stop = [s for s in req.stop if s]
        if not req.stop:
            req.stop = None
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

    # Extract JSON schema from response_format or grammar (shared with chat router)
    json_schema = _parse_response_format(req.response_format, req.grammar)

    completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"

    # Structured tracing + logging
    tracer = get_inference_tracer()
    slog = get_structured_logger()
    trace_id = f"cmpl-{uuid.uuid4().hex[:16]}"
    tracer.start_trace(trace_id, metadata={
        "model": req.model,
        "max_tokens": req.effective_max_tokens(),
        "temperature": req.temperature,
        "stream": req.stream,
        "endpoint": "/completions",
    })
    tracer.span(trace_id, "prefill", {"model": req.model})
    slog.info("inference_request", model=req.model, trace_id=trace_id,
              max_tokens=req.effective_max_tokens(), stream=req.stream)

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

    # Register with request tracker for cancellation support in non-streaming path
    _ns_tracker = None
    _ns_gen = None
    _ns_cancel_event = None
    try:
        from yunshu_engine.request_tracker import get_request_tracker
        _ns_tracker = get_request_tracker()
        _ns_gen = _ns_tracker.register(completion_id, req.model)
        _ns_cancel_event = _ns_gen.cancel_event
    except Exception:
        _ns_tracker = None

    try:
        async def _gen_one(idx: int):
            if is_batched:
                result = await engine.generate(
                    prompt=prompt,
                    max_tokens=req.effective_max_tokens(),
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
                    seed=(req.seed + idx) if req.seed is not None else None,
                    spec_decode=req.spec_decode,
                    enable_thinking=req.enable_thinking,
                    thinking_budget=req.thinking_budget,
                    json_schema=json_schema,
                    grammar=req.grammar,
                    reasoning_effort=req.reasoning_effort,
                    xtc_probability=req.xtc_probability,
                    xtc_threshold=req.xtc_threshold,
                    logprobs=req.logprobs > 0,
                    top_logprobs=req.top_logprobs,
                    priority=req.priority,
                    logits_processors=req.logits_processors,
                    cancel_event=_ns_cancel_event,
                    timeout_seconds=req.timeout,
                    lora_adapter=loaded_adapter,
                )
                text = result.text
                pt = result.prompt_tokens
                ct = result.completion_tokens
                fr = _normalize_finish_reason(result.finish_reason)
                rt = getattr(result, 'reasoning_tokens', 0)
                lp = None
                if req.logprobs > 0:
                    lp = _format_logprobs(
                        result, getattr(engine, '_tokenizer', None),
                        req.top_logprobs if req.top_logprobs is not None else req.logprobs,
                        echo=req.echo, prompt=prompt,
                    )
            else:
                state = await engine.generate(
                    prompt=prompt,
                    max_tokens=req.effective_max_tokens(),
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
                    seed=(req.seed + idx) if req.seed is not None else None,
                    enable_thinking=req.enable_thinking,
                    thinking_budget=req.thinking_budget,
                    reasoning_effort=req.reasoning_effort,
                    xtc_probability=req.xtc_probability,
                    xtc_threshold=req.xtc_threshold,
                    spec_decode=req.spec_decode,
                    json_schema=json_schema,
                    grammar=req.grammar,
                    logprobs=req.logprobs > 0,
                    top_logprobs=req.top_logprobs,
                    priority=req.priority,
                    logits_processors=req.logits_processors,
                    cancel_event=_ns_cancel_event,
                    timeout_seconds=req.timeout,
                    lora_adapter=loaded_adapter,
                )
                text = state.generated_text
                pt = state.prompt_token_count
                ct = state.completion_token_count
                fr = _normalize_finish_reason(state.finish_reason)
                rt = getattr(state, 'reasoning_tokens', 0)
                lp = None
                if req.logprobs > 0:
                    lp = _format_logprobs(
                        state, getattr(engine, '_tokenizer', None),
                        req.top_logprobs if req.top_logprobs is not None else req.logprobs,
                        echo=req.echo, prompt=prompt,
                    )

            if req.echo:
                text = prompt + text
            if req.suffix:
                text = text + req.suffix
            _gen_result = result if is_batched else state
            _cached = getattr(_gen_result, 'cached_tokens', 0)
            return idx, pt, ct, fr, rt, lp, text, _cached

        n = max(req.n, 1)
        # best_of: generate more completions than returned, keep best by logprob
        _generate_count = max(req.best_of, n) if req.best_of is not None else n
        if _generate_count == 1:
            results = [await _gen_one(0)]
        else:
            import asyncio
            results = await asyncio.gather(
                *[_gen_one(i) for i in range(_generate_count)], return_exceptions=True,
            )
            # Filter out exceptions, log them
            valid_results = []
            for r in results:
                if isinstance(r, BaseException):
                    logger.error(f"Choice generation failed: {r}", exc_info=r)
                else:
                    valid_results.append(r)
            results = valid_results
            results.sort(key=lambda x: x[0])

        if not results:
            raise HTTPException(status_code=500, detail="All choices failed to generate")

        # best_of: select top n results by average log probability per token
        if req.best_of is not None and len(results) > n:
            def _avg_logprob(r):
                """Compute average log probability for a result tuple."""
                _, pt, ct, fr, rt, lp, text, _cached = r
                if lp and "token_logprobs" in lp:
                    probs = lp["token_logprobs"]
                    # Filter out None values (some tokens may have null logprobs)
                    valid_probs = [p for p in probs if p is not None]
                    if valid_probs:
                        return sum(valid_probs) / len(valid_probs)
                return float('-inf')  # no logprobs -> lowest priority
            results.sort(key=_avg_logprob, reverse=True)
            results = results[:n]
            # Re-index choices after best_of selection
            results = [(i, *r[1:]) for i, r in enumerate(results)]

        prompt_tokens = results[0][1]
        total_completion_tokens = sum(r[2] for r in results)
        total_reasoning_tokens = sum(r[4] for r in results)
        max_cached_tokens = max(r[7] for r in results) if results else 0
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

        # Record metrics for completions endpoint
        _record_metrics(prompt_tokens, total_completion_tokens)

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
        return JSONResponse(
            status_code=507,
            content={"error": {"message": "Out of GPU memory", "type": "memory_error"}},
        )
    except Exception as e:
        logger.error(f"Completions generation error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Internal server error", "type": "internal_error"}},
        )
    finally:
        _release_lora_adapter(engine, loaded_adapter)
        if _ns_tracker is not None:
            try:
                _ns_tracker.unregister(completion_id)
            except Exception:
                pass


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
    completion_tok_per_choice: dict[int, int] = {}
    cached_tok = 0
    n = max(req.n, 1)

    async def _stream_choice(choice_idx: int):
        nonlocal prompt_tok, cached_tok
        choice_finish_reason = None
        _choice_streamed_text = ""  # track emitted text for stop-sequence correction
        # Per-choice text offset tracker for logprobs text_offset field.
        # When echo=True, the prompt text is emitted first, so the completion
        # text offsets must account for the prompt length.
        _choice_text_offset = len(prompt) if req.echo else 0
        if req.echo:
            yield format_openai_completion_chunk(
                completion_id=completion_id,
                model=req.model,
                text=prompt,
                choice_index=choice_idx,
            )

        if is_batched:
            async for output in engine.stream_generate(
                prompt=prompt,
                max_tokens=req.effective_max_tokens(),
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
                seed=(req.seed + choice_idx) if req.seed is not None else None,
                enable_thinking=req.enable_thinking,
                thinking_budget=req.thinking_budget,
                json_schema=json_schema,
                grammar=req.grammar,
                reasoning_effort=req.reasoning_effort,
                xtc_probability=req.xtc_probability,
                xtc_threshold=req.xtc_threshold,
                spec_decode=req.spec_decode,
                logprobs=req.logprobs > 0,
                top_logprobs=req.top_logprobs,
                priority=req.priority,
                logits_processors=req.logits_processors,
                cancel_event=_comp_cancel_evt,
                timeout_seconds=req.timeout,
                lora_adapter=loaded_adapter,
            ):
                if hasattr(output, 'prompt_tokens') and output.prompt_tokens:
                    prompt_tok = output.prompt_tokens
                if hasattr(output, 'completion_tokens') and output.completion_tokens is not None and output.completion_tokens > 0:
                    completion_tok_per_choice[choice_idx] = output.completion_tokens
                elif output.new_text:
                    completion_tok_per_choice[choice_idx] = completion_tok_per_choice.get(choice_idx, 0) + 1
                _choice_reasoning = getattr(output, 'reasoning_tokens', 0)
                reasoning_tok_per_choice[choice_idx] = _choice_reasoning
                if hasattr(output, 'cached_tokens') and output.cached_tokens:
                    cached_tok = max(cached_tok, output.cached_tokens)
                # Track finish_reason from engine; only emit on final chunk
                if output.finish_reason is not None:
                    choice_finish_reason = output.finish_reason
                # vLLM pattern: emit prefill progress as SSE comment for
                # client-side progress bars during long chunked prefills.
                _pf_prog = getattr(output, 'prefill_progress', None)
                if _pf_prog is not None:
                    yield f": prefill-progress {_pf_prog[0]}/{_pf_prog[1]}\n\n"
                    continue  # progress outputs carry no text
                # Track emitted text for stop-sequence overcount correction
                if output.new_text:
                    _choice_streamed_text += output.new_text
                if len(_choice_streamed_text) > _MAX_STREAMING_TEXT_BUFFER:
                    logger.error("Choice streaming text buffer exceeded 1MB — truncating")
                    _choice_streamed_text = _choice_streamed_text[-_TRUNCATE_KEEP:]
                # Detect stop-sequence overcount on final output
                if req.stop and choice_finish_reason == "stop" and getattr(output, 'finished', False):
                    for _seq in req.stop:
                        if _seq and _seq in _choice_streamed_text:
                            _idx = _choice_streamed_text.find(_seq)
                            _choice_streamed_text = _choice_streamed_text[:_idx]
                            _tok = getattr(engine, '_tokenizer', None)
                            if _tok:
                                try:
                                    _correct_count = len(_tok.encode(_choice_streamed_text))
                                    if _correct_count < completion_tok_per_choice.get(choice_idx, 0):
                                        completion_tok_per_choice[choice_idx] = _correct_count
                                except Exception:
                                    pass
                            break
                # Format logprobs for this token if present
                _chunk_logprobs = None
                if output.logprobs:
                    _chunk_logprobs, _choice_text_offset = _format_streaming_logprobs(
                        output.logprobs,
                        text_offset_start=_choice_text_offset,
                        top_logprobs=req.top_logprobs if req.top_logprobs is not None else req.logprobs,
                        tokenizer=getattr(engine, '_tokenizer', None),
                    )
                yield format_openai_completion_chunk(
                    completion_id=completion_id,
                    model=req.model,
                    text=output.new_text,
                    finish_reason=None,  # intermediate: always None
                    choice_index=choice_idx,
                    logprobs=_chunk_logprobs,
                )
        else:
            async for output in engine.generate_stream(
                prompt=prompt,
                max_tokens=req.effective_max_tokens(),
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                min_p=req.min_p,
                repetition_penalty=req.repetition_penalty,
                frequency_penalty=req.frequency_penalty,
                presence_penalty=req.presence_penalty,
                logit_bias=req.logit_bias,
                stop=req.stop,
                seed=(req.seed + choice_idx) if req.seed is not None else None,
                enable_thinking=req.enable_thinking,
                thinking_budget=req.thinking_budget,
                reasoning_effort=req.reasoning_effort,
                stop_token_ids=req.stop_token_ids,
                xtc_probability=req.xtc_probability,
                xtc_threshold=req.xtc_threshold,
                spec_decode=req.spec_decode,
                json_schema=json_schema,
                grammar=req.grammar,
                priority=req.priority,
                logprobs=req.logprobs > 0,
                top_logprobs=req.top_logprobs,
                logits_processors=req.logits_processors,
                cancel_event=_comp_cancel_evt,
                timeout_seconds=req.timeout,
                lora_adapter=loaded_adapter,
            ):
                if hasattr(output, 'prompt_tokens') and output.prompt_tokens:
                    prompt_tok = output.prompt_tokens
                if hasattr(output, 'completion_tokens') and output.completion_tokens is not None and output.completion_tokens > 0:
                    completion_tok_per_choice[choice_idx] = output.completion_tokens
                elif output.token_text:
                    completion_tok_per_choice[choice_idx] = completion_tok_per_choice.get(choice_idx, 0) + 1
                if output.finish_reason is not None:
                    choice_finish_reason = output.finish_reason
                _chunk_lp = None
                if req.logprobs and hasattr(output, 'logprobs'):
                    _chunk_lp, _choice_text_offset = _format_streaming_logprobs(
                        output.logprobs,
                        text_offset_start=_choice_text_offset,
                        top_logprobs=req.top_logprobs if req.top_logprobs is not None else req.logprobs,
                        tokenizer=getattr(engine, '_tokenizer', None),
                    )
                # Track emitted text for stop-sequence overcount correction
                if output.token_text:
                    _choice_streamed_text += output.token_text
                if len(_choice_streamed_text) > _MAX_STREAMING_TEXT_BUFFER:
                    logger.error("Choice streaming text buffer exceeded 1MB — truncating")
                    _choice_streamed_text = _choice_streamed_text[-_TRUNCATE_KEEP:]
                # Detect stop-sequence overcount on final output
                if req.stop and choice_finish_reason == "stop" and getattr(output, 'finished', False):
                    for _seq in req.stop:
                        if _seq and _seq in _choice_streamed_text:
                            _idx = _choice_streamed_text.find(_seq)
                            _choice_streamed_text = _choice_streamed_text[:_idx]
                            _tok = getattr(engine, '_tokenizer', None)
                            if _tok:
                                try:
                                    _correct_count = len(_tok.encode(_choice_streamed_text))
                                    if _correct_count < completion_tok_per_choice.get(choice_idx, 0):
                                        completion_tok_per_choice[choice_idx] = _correct_count
                                except Exception:
                                    pass
                            break
                yield format_openai_completion_chunk(
                    completion_id=completion_id,
                    model=req.model,
                    text=output.token_text,
                    finish_reason=None,  # intermediate: always None
                    choice_index=choice_idx,
                    logprobs=_chunk_lp,
                )

        # Emit suffix text after completion if requested
        if req.suffix:
            yield format_openai_completion_chunk(
                completion_id=completion_id,
                model=req.model,
                text=req.suffix,
                choice_index=choice_idx,
            )

        # Emit final chunk with finish_reason for this choice (even if zero tokens)
        yield format_openai_completion_chunk(
            completion_id=completion_id,
            model=req.model,
            text="",
            finish_reason=_normalize_finish_reason(choice_finish_reason),
            choice_index=choice_idx,
        )

    async def _token_source():
        nonlocal _done_emitted, metrics_recorded
        # Stream each choice sequentially (matches OpenAI spec behavior)
        for choice_idx in range(n):
            async for chunk in _stream_choice(choice_idx):
                yield chunk

        if include_usage:
            # Sum reasoning tokens across all choices for total usage
            _total_reasoning = sum(reasoning_tok_per_choice.values())
            _total_completion = sum(completion_tok_per_choice.values()) if completion_tok_per_choice else completion_tok
            yield format_openai_completion_usage_chunk(
                completion_id=completion_id,
                model=req.model,
                prompt_tokens=prompt_tok,
                completion_tokens=_total_completion,
                reasoning_tokens=_total_reasoning,
                cached_tokens=cached_tok,
            )

        # Record metrics for completions streaming path
        _total_completion = sum(completion_tok_per_choice.values()) if completion_tok_per_choice else completion_tok
        if prompt_tok > 0 or _total_completion > 0:
            _record_metrics(prompt_tok, _total_completion)

        metrics_recorded = True
        _done_emitted = True
        yield format_openai_done()
    metrics_recorded = False
    _done_emitted = False
    loaded_adapter = _apply_lora_adapter(engine, req.lora_adapter)
    # Register with request tracker for cancellation support
    _tracker = None
    _tracker_gen = None
    try:
        from yunshu_engine.request_tracker import get_request_tracker
        _tracker = get_request_tracker()
        _tracker_gen = _tracker.register(completion_id, req.model)
    except Exception:
        _tracker = None
    _comp_cancel_evt = _tracker_gen.cancel_event if _tracker_gen is not None else None

    # SSE line-buffer: accumulate partial data and only yield complete
    # \n\n-terminated SSE events.  Prevents clients from receiving partial
    # SSE lines when TCP chunk boundaries split an event mid-way.
    _sse_buffer = ""

    def _drain_sse_buffer():
        """Return list of complete SSE events from the buffer, keeping any trailing partial line."""
        nonlocal _sse_buffer
        chunks = []
        while "\n\n" in _sse_buffer:
            event, _sse_buffer = _sse_buffer.split("\n\n", 1)
            chunks.append((event + "\n\n").encode("utf-8"))
        return chunks

    try:
        async for event in with_sse_keepalive(
            _token_source(),
            http_request=request,
            cancel_event=_comp_cancel_evt,
        ):
            _sse_buffer += event
            for chunk in _drain_sse_buffer():
                yield chunk
        # Flush any remaining complete event in buffer
        for chunk in _drain_sse_buffer():
            yield chunk
        # If buffer still has residual content without \n\n terminator,
        # append terminator and flush
        if _sse_buffer.strip():
            _sse_buffer += "\n\n"
            for chunk in _drain_sse_buffer():
                yield chunk
    except MemoryError:
        if _comp_cancel_evt is not None:
            _comp_cancel_evt.set()
        # Flush any buffered partial data before error
        if _sse_buffer.strip():
            _sse_buffer += "\n\n"
            for chunk in _drain_sse_buffer():
                yield chunk
        yield b'data: {"error": {"message": "Insufficient GPU memory", "type": "memory_error", "code": "oom"}}\n\n'
        if not _done_emitted:
            yield b"data: [DONE]\n\n"
    except Exception as e:
        if _comp_cancel_evt is not None:
            _comp_cancel_evt.set()
        # Flush any buffered partial data before error
        if _sse_buffer.strip():
            _sse_buffer += "\n\n"
            for chunk in _drain_sse_buffer():
                yield chunk
        logger.error(f"Completions streaming error: {e}", exc_info=True)
        err_payload = {"error": {"message": str(e)[:200], "type": "internal_error"}}
        yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n".encode("utf-8")
        if not _done_emitted:
            yield b"data: [DONE]\n\n"
    finally:
        _release_lora_adapter(engine, loaded_adapter)
        if _tracker is not None:
            try:
                _tracker.unregister(completion_id)
            except Exception:
                pass
        # Fallback metrics recording if generator raised before completing
        if not metrics_recorded:
            _total = sum(completion_tok_per_choice.values()) if completion_tok_per_choice else completion_tok
            if prompt_tok > 0 or _total > 0:
                try:
                    _record_metrics(prompt_tok, _total)
                except Exception:
                    pass


def _format_logprobs(state, tokenizer, top_logprobs: int, echo: bool = False, prompt: str = "") -> dict | None:
    """Format logprobs from request state into OpenAI Completions format.

    OpenAI Completions API returns logprobs as a flat structure:
      {
        "tokens": ["tok1", "tok2", ...],
        "token_logprobs": [-0.5, -1.2, ...],
        "top_logprobs": [{"tok_a": -0.5, "tok_b": -1.0}, ...],  # Dict[str, float], NOT chat-style
        "text_offset": [0, 3, ...]
      }

    Note: top_logprobs entries are Dict[str, float] (token -> logprob),
    NOT the Chat Completions format with token/logprob/bytes keys.

    Args:
        state: Generation result with logprobs attribute.
        tokenizer: Tokenizer for decoding token IDs.
        top_logprobs: Maximum number of top logprobs to return per token.
        echo: Whether echo mode is enabled (shifts text_offset by prompt length).
        prompt: The prompt text, used for text_offset shift when echo=True.
    """
    raw_logprobs = getattr(state, 'logprobs', None)
    if not raw_logprobs:
        return None

    token_logprobs = []
    text_offsets = []
    _offset = len(prompt) if echo else 0
    if isinstance(raw_logprobs, (list, tuple)):
        for lp_entry in raw_logprobs:
            if isinstance(lp_entry, dict):
                token_str = lp_entry.get("token", "")
                if not token_str and tokenizer and "token_id" in lp_entry:
                    try:
                        token_str = tokenizer.decode([lp_entry["token_id"]])
                    except Exception:
                        logger.debug("tokenizer decode failed", exc_info=True)
                top_lps = lp_entry.get("top_logprobs", [])
                # OpenAI Completions API: top_logprobs is List[Dict[str, float]]
                # Each dict maps token string -> logprob float value.
                decoded_top = {}
                for tlp in top_lps[:top_logprobs] if top_logprobs else []:
                    tlp_token = tlp.get("token", "")
                    if not tlp_token and tokenizer and "token_id" in tlp:
                        try:
                            tlp_token = tokenizer.decode([tlp["token_id"]])
                        except Exception:
                            pass
                    if tlp_token:
                        decoded_top[tlp_token] = tlp.get("logprob", 0.0)
                token_logprobs.append({
                    "token": token_str,
                    "logprob": lp_entry.get("logprob", 0.0),
                    "top_logprobs": decoded_top,
                })
                text_offsets.append(_offset)
                _offset += len(token_str)
            elif isinstance(lp_entry, (int, float)):
                token_logprobs.append({
                    "token": "",
                    "logprob": float(lp_entry),
                    "top_logprobs": {},
                })
                text_offsets.append(_offset)

    if not token_logprobs:
        return None

    return {
        "tokens": [e["token"] for e in token_logprobs],
        "token_logprobs": [e["logprob"] for e in token_logprobs],
        "top_logprobs": [e["top_logprobs"] for e in token_logprobs],
        "text_offset": text_offsets,
    }


def _format_streaming_logprobs(
    logprobs_list: list[dict],
    *,
    text_offset_start: int = 0,
    top_logprobs: int | None = None,
    tokenizer: object | None = None,
) -> tuple[dict | None, int]:
    """Format per-token logprobs from streaming GenerationOutput into OpenAI Completions format.

    In streaming mode, each GenerationOutput has at most 1 logprob entry.
    Returns (logprobs_dict, new_text_offset) where new_text_offset is the
    running offset to pass into the next call.

    Per the OpenAI Completions API, logprobs must include ``text_offset``
    (character offset of each token in the output text).

    OpenAI Completions top_logprobs format: Dict[str, float] per token,
    NOT the Chat Completions style with token/logprob/bytes keys.
    """
    if not logprobs_list:
        return None, text_offset_start
    entries = []
    offsets = []
    _offset = text_offset_start
    # When top_logprobs is not specified, include all available top logprobs.
    # Use a large default to avoid truncation (engine already limits this).
    _max_top = top_logprobs if top_logprobs is not None else 999
    for lp_entry in logprobs_list:
        if not isinstance(lp_entry, dict):
            continue
        token_str = lp_entry.get("token", "")
        if not token_str and "token_id" in lp_entry:
            if tokenizer is not None:
                try:
                    token_str = tokenizer.decode([lp_entry["token_id"]])
                except Exception:
                    token_str = str(lp_entry["token_id"])
            else:
                token_str = str(lp_entry["token_id"])
        top_lps = lp_entry.get("top_logprobs", [])
        # OpenAI Completions API: top_logprobs is Dict[str, float]
        decoded_top = {}
        for tlp in top_lps[:_max_top]:
            if not isinstance(tlp, dict):
                continue
            tlp_token = tlp.get("token", "")
            if not tlp_token and "token_id" in tlp:
                if tokenizer is not None:
                    try:
                        tlp_token = tokenizer.decode([tlp["token_id"]])
                    except Exception:
                        tlp_token = str(tlp["token_id"])
                else:
                    tlp_token = str(tlp["token_id"])
            if tlp_token:
                decoded_top[tlp_token] = tlp.get("logprob", 0.0)
        entries.append({
            "token": token_str,
            "logprob": lp_entry.get("logprob", 0.0),
            "top_logprobs": decoded_top,
        })
        offsets.append(_offset)
        _offset += len(token_str)
    if not entries:
        return None, text_offset_start
    return {
        "tokens": [e["token"] for e in entries],
        "token_logprobs": [e["logprob"] for e in entries],
        "top_logprobs": [e["top_logprobs"] for e in entries],
        "text_offset": offsets,
    }, _offset
