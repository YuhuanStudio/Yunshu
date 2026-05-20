from __future__ import annotations
"""OpenAI Responses API compatible router.

The Responses API is OpenAI's newest API format that combines
chat completions with tool use, structured output, and streaming
into a unified interface.

Supports:
- Text and chat responses
- Streaming via SSE
- Tool use (function calling)
- Structured output (response_format)
- Image and audio input (routed to VLM/Omni)
"""
import json
import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from ..engine import get_engine, get_engine_for_model

logger = logging.getLogger(__name__)
from ..streaming import (
    format_responses_created,
    format_responses_in_progress,
    format_responses_output_item_added,
    format_responses_content_part_added,
    format_responses_text_delta,
    format_responses_text_done,
    format_responses_content_part_done,
    format_responses_output_item_done,
    format_responses_completed,
    format_responses_failed,
    format_responses_incomplete,
)
from .chat import _format_chat_logprobs, _normalize_finish_reason, _record_metrics

router = APIRouter(tags=["responses"])


class ResponseInputText(BaseModel):
    type: str = "message"
    role: str = "user"
    content: str | list[dict]


class ResponseTool(BaseModel):
    type: str = "function"
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None


class StreamOptions(BaseModel):
    """OpenAI stream_options parameter."""
    include_usage: bool = False


class ResponsesRequest(BaseModel):
    model: str
    input: str | list[ResponseInputText]
    instructions: Optional[str] = None
    max_output_tokens: int = Field(default=2048, ge=1, le=131072)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    stream: bool = False
    n: int = Field(default=1, ge=1, le=128)
    tools: Optional[list[ResponseTool]] = None
    response_format: Optional[dict] = None
    seed: Optional[int] = None
    enable_thinking: Optional[bool] = None
    thinking_budget: Optional[int] = Field(default=None, ge=1, le=32768)
    reasoning_effort: Optional[str] = None
    repetition_penalty: float = Field(default=1.0, ge=0.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[dict[str, float]] = None
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    stop: Optional[list[str]] = None
    stop_token_ids: Optional[list[int]] = None
    logprobs: bool = False
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=20)
    spec_decode: bool = False
    xtc_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    xtc_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    grammar: Optional[dict] = None
    lora_adapter: Optional[str] = None
    stream_options: Optional[StreamOptions] = None  # {"include_usage": true}
    user: Optional[str] = None
    priority: int = Field(default=0, ge=0, le=100)
    logits_processors: Optional[list] = None  # User-provided custom logits processors
    timeout: Optional[float] = Field(default=None, ge=1.0, le=600.0)  # Request timeout in seconds

    @model_validator(mode="after")
    def validate_request(self):
        if not self.model or not self.model.strip():
            raise ValueError("model: field is required and cannot be empty")
        # Validate input: string must be non-empty, list must have elements
        if isinstance(self.input, str) and not self.input.strip():
            raise ValueError("input: cannot be empty or whitespace-only")
        if isinstance(self.input, list) and not self.input:
            raise ValueError("input: cannot be an empty list")
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
        return self


def _convert_to_messages(req: ResponsesRequest) -> list[dict]:
    """Convert Responses API input to OpenAI chat messages."""
    messages = []

    if req.instructions:
        messages.append({"role": "system", "content": req.instructions})

    if isinstance(req.input, str):
        messages.append({"role": "user", "content": req.input})
    elif isinstance(req.input, list):
        for item in req.input:
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                messages.append({"role": role, "content": content})
            elif hasattr(item, "content"):
                role = getattr(item, "role", "user")
                content = item.content
                if isinstance(content, str):
                    messages.append({"role": role, "content": content})
                elif isinstance(content, list):
                    messages.append({"role": role, "content": content})

    return messages


def _parse_response_format(rf: dict | None, grammar: dict | None = None) -> dict | None:
    if grammar is not None:
        gtype = grammar.get("type")
        if gtype == "json":
            schema = grammar.get("schema")
            if schema:
                return schema
            return {}
        return grammar
    if rf is None:
        return None
    rf_type = rf.get("type")
    if rf_type == "json_schema":
        js = rf.get("json_schema", {})
        return js.get("schema", js)
    elif rf_type == "json_object":
        return {}
    return None


@router.post("/responses", response_model=None)
async def create_response(req: ResponsesRequest, request: Request):
    """OpenAI Responses API endpoint."""
    messages = _convert_to_messages(req)
    json_schema = _parse_response_format(req.response_format, req.grammar)

    # Convert logit_bias keys from str to int (API sends string keys,
    # engine expects int keys for tensor indexing)
    _logit_bias = req.logit_bias
    if _logit_bias:
        _logit_bias = {int(k): v for k, v in _logit_bias.items()}

    # Structured tracing
    from yunshu_engine.tracing import get_inference_tracer, get_structured_logger
    tracer = get_inference_tracer()
    slog = get_structured_logger()
    trace_id = f"resp-{uuid.uuid4().hex[:16]}"
    trace = tracer.start_trace(trace_id, metadata={
        "model": req.model, "stream": req.stream,
        "endpoint": "/responses",
    })
    slog.info("inference_request", model=req.model, trace_id=trace_id, stream=req.stream)

    engine = get_engine()
    if engine is None or not engine.is_loaded or not engine.resolve_model_id(req.model):
        try:
            engine = await get_engine_for_model(req.model)
        except (KeyError, Exception) as e:
            raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found: {e}")

    # Check for VLM/audio routing
    from .chat import _has_images, _has_audio
    has_media = _has_images(messages) or _has_audio(messages)
    if has_media:
        from .chat import _handle_vlm_chat
        from .chat import ChatCompletionRequest, ChatMessage
        # Build ChatMessage list from dict messages
        chat_messages = []
        for m in messages:
            chat_messages.append(ChatMessage(
                role=m.get("role", "user"),
                content=m.get("content", ""),
            ))
        # Convert response_format to the format chat.py expects
        chat_response_format = None
        if req.response_format:
            chat_response_format = req.response_format
        chat_req = ChatCompletionRequest(
            model=req.model,
            messages=chat_messages,
            max_tokens=req.max_output_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            min_p=req.min_p,
            repetition_penalty=req.repetition_penalty,
            frequency_penalty=req.frequency_penalty,
            presence_penalty=req.presence_penalty,
            logit_bias=_logit_bias,
            seed=req.seed,
            enable_thinking=req.enable_thinking,
            thinking_budget=req.thinking_budget,
            reasoning_effort=req.reasoning_effort,
            stop=req.stop,
            stop_token_ids=req.stop_token_ids,
            logprobs=req.logprobs,
            top_logprobs=req.top_logprobs,
            spec_decode=req.spec_decode,
            n=req.n,
            stream=req.stream,
            stream_options=req.stream_options,
            logits_processors=req.logits_processors,
            response_format=chat_response_format,
            grammar=req.grammar,
            xtc_probability=req.xtc_probability,
            xtc_threshold=req.xtc_threshold,
            lora_adapter=req.lora_adapter,
            priority=req.priority,
            user=req.user,
            timeout=req.timeout,
        )
        vlm_json_schema = _parse_response_format(chat_response_format)
        return await _handle_vlm_chat(chat_req, messages, request, json_schema=vlm_json_schema)

    # Inject tool definitions
    if req.tools:
        from .chat import _inject_tool_system_prompt
        from .chat import ToolDefinition, ToolFunction
        tools = [
            ToolDefinition(function=ToolFunction(
                name=t.name, description=t.description, parameters=t.parameters,
            ))
            for t in req.tools
        ]
        messages = _inject_tool_system_prompt(messages, tools)

    response_id = f"resp-{uuid.uuid4().hex[:24]}"

    # LoRA adapter lifecycle
    from .chat import _apply_lora_adapter, _release_lora_adapter
    loaded_adapter = _apply_lora_adapter(engine, req.lora_adapter)

    # Streaming: must return before the try/finally releases LoRA.
    # LoRA lifecycle is managed inside _stream_response's finally block.
    if req.stream:
        if req.n > 1:
            _release_lora_adapter(engine, loaded_adapter)
            raise HTTPException(
                status_code=400,
                detail="n>1 is not supported with stream=True for the Responses API. "
                       "Use stream=False for multiple completions, or stream=True with n=1.",
            )
        return StreamingResponse(
            _stream_response(engine, req, messages, response_id, json_schema, loaded_adapter, request=request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming: register with request tracker for cancellation support
    _ns_tracker = None
    _ns_gen = None
    _ns_cancel_event = None
    try:
        from yunshu_engine.request_tracker import get_request_tracker
        _ns_tracker = get_request_tracker()
        _ns_gen = _ns_tracker.register(response_id, req.model)
        _ns_cancel_event = _ns_gen.cancel_event
    except Exception:
        _ns_tracker = None

    # Non-streaming: LoRA is released in the finally block below.
    try:
        from yunshu_engine.batched_engine import BatchedEngine
        is_batched = isinstance(engine, BatchedEngine)

        # Non-batched Engine path: apply chat template ourselves before
        # passing to engine.generate().  The legacy Engine / EngineCore
        # _messages_to_text() does not call adapt_messages() and may strip
        # tool-related fields, producing garbage for tool-use conversations.
        _non_batched_prompt: str | list[dict] = messages
        if not is_batched and messages:
            _tokenizer = getattr(engine, '_tokenizer', None)
            if _tokenizer is not None and hasattr(_tokenizer, 'apply_chat_template'):
                try:
                    from yunshu_engine.message_adapter import adapt_messages
                    _adapted = adapt_messages(messages, getattr(engine, 'model_name', '') or '')
                except Exception:
                    _adapted = messages
                try:
                    _tpl_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
                    if req.enable_thinking is not None:
                        _tpl_kwargs["enable_thinking"] = req.enable_thinking
                    _rendered = _tokenizer.apply_chat_template(_adapted, **_tpl_kwargs)
                    if _rendered:
                        _non_batched_prompt = _rendered
                except TypeError as _te:
                    if 'enable_thinking' in str(_te):
                        _tpl_kwargs.pop('enable_thinking', None)
                        _rendered = _tokenizer.apply_chat_template(_adapted, **_tpl_kwargs)
                        if _rendered:
                            _non_batched_prompt = _rendered
                    else:
                        logger.debug("chat template failed for non-batched Responses path", exc_info=True)
                except Exception:
                    logger.debug("chat template failed for non-batched Responses path", exc_info=True)

        # ── n>1 support: generate n responses sequentially ──
        # Single GPU cannot parallelize multiple generations; they run
        # sequentially.  Each choice gets its own message output item.
        all_output_items: list[dict] = []
        total_pt = 0
        total_ct = 0
        total_reasoning_tokens = 0
        max_cached_tokens = 0

        for choice_idx in range(req.n):
            result = None
            state = None

            if is_batched:
                result = await engine.chat(
                    messages=messages,
                    max_tokens=req.max_output_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    seed=req.seed,
                    enable_thinking=req.enable_thinking,
                    thinking_budget=req.thinking_budget,
                    reasoning_effort=req.reasoning_effort,
                    repetition_penalty=req.repetition_penalty,
                    frequency_penalty=req.frequency_penalty,
                    presence_penalty=req.presence_penalty,
                    logit_bias=_logit_bias,
                    min_p=req.min_p,
                    json_schema=json_schema,
                    stop=req.stop,
                    stop_token_ids=req.stop_token_ids,
                    spec_decode=req.spec_decode,
                    xtc_probability=req.xtc_probability,
                    xtc_threshold=req.xtc_threshold,
                    priority=req.priority,
                    logprobs=req.logprobs,
                    top_logprobs=req.top_logprobs,
                    logits_processors=req.logits_processors,
                    cancel_event=_ns_cancel_event,
                    timeout_seconds=req.timeout,
                )
                text = result.text
                pt = result.prompt_tokens
                ct = result.completion_tokens
                finish_reason = _normalize_finish_reason(result.finish_reason)
                _reasoning_tokens = getattr(result, 'reasoning_tokens', 0)
                _cached_tokens = getattr(result, 'cached_tokens', 0)
            else:
                state = await engine.generate(
                    prompt=_non_batched_prompt,
                    max_tokens=req.max_output_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    seed=req.seed,
                    enable_thinking=req.enable_thinking,
                    thinking_budget=req.thinking_budget,
                    reasoning_effort=req.reasoning_effort,
                    repetition_penalty=req.repetition_penalty,
                    frequency_penalty=req.frequency_penalty,
                    presence_penalty=req.presence_penalty,
                    logit_bias=_logit_bias,
                    min_p=req.min_p,
                    json_schema=json_schema,
                    stop=req.stop,
                    stop_token_ids=req.stop_token_ids,
                    spec_decode=req.spec_decode,
                    xtc_probability=req.xtc_probability,
                    xtc_threshold=req.xtc_threshold,
                    priority=req.priority,
                    logprobs=req.logprobs,
                    top_logprobs=req.top_logprobs,
                    logits_processors=req.logits_processors,
                    cancel_event=_ns_cancel_event,
                    timeout_seconds=req.timeout,
                )
                text = state.generated_text
                pt = state.prompt_token_count
                ct = state.completion_token_count
                finish_reason = _normalize_finish_reason(state.finish_reason)
                _reasoning_tokens = getattr(state, 'reasoning_tokens', 0)
                _cached_tokens = getattr(state, 'cached_tokens', 0)

            # Extract thinking content for reasoning models
            from ..streaming import extract_thinking
            _thinking, text = extract_thinking(text, req.model)
            if _thinking and _reasoning_tokens == 0:
                _reasoning_tokens = len(engine._tokenizer.encode(_thinking)) if hasattr(engine, '_tokenizer') and engine._tokenizer else 0

            # Accumulate usage across all choices
            total_pt = pt  # prompt tokens are the same for every choice
            total_ct += ct
            total_reasoning_tokens += _reasoning_tokens
            max_cached_tokens = max(max_cached_tokens, _cached_tokens)

            # Extract tool calls for this choice
            tool_calls = None
            if req.tools:
                from .chat import extract_tool_calls_model_aware, clean_tool_call_markup
                tool_calls = extract_tool_calls_model_aware(text, req.model)
                if tool_calls:
                    text = clean_tool_call_markup(text)
                    finish_reason = "tool_calls"

            # Build output item for this choice
            text_part = {"type": "output_text", "text": text.strip()}
            # Include logprobs if requested
            if req.logprobs:
                _result_lp = getattr(result, 'logprobs', None) if is_batched else getattr(state, 'logprobs', None)
                if _result_lp:
                    _chunk_lp = _format_chat_logprobs(_result_lp)
                    if _chunk_lp:
                        text_part["logprobs"] = _chunk_lp
            content_parts = [text_part]
            choice_item = {
                "type": "message",
                "id": f"msg-{uuid.uuid4().hex[:24]}",
                "role": "assistant",
                "content": content_parts,
                "status": "completed",
            }
            # For n>1, include a choice_index so clients can distinguish
            if req.n > 1:
                choice_item["index"] = choice_idx

            all_output_items.append(choice_item)

            if tool_calls:
                for tc in tool_calls:
                    all_output_items.append({
                        "type": "function_call",
                        "id": f"fc-{uuid.uuid4().hex[:24]}",
                        "call_id": f"call_{uuid.uuid4().hex[:8]}",
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    })

        _record_metrics(total_pt, total_ct)

        # End tracing
        tracer.end_trace(trace_id, result={
            "prompt_tokens": total_pt,
            "completion_tokens": total_ct,
            "choices": req.n,
        })
        slog.info("inference_complete", model=req.model, trace_id=trace_id,
                  prompt_tokens=total_pt, completion_tokens=total_ct, choices=req.n)

        return JSONResponse({
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "model": req.model,
            "status": "completed",
            "output": all_output_items,
            "usage": {
                "input_tokens": total_pt,
                "output_tokens": total_ct,
                "total_tokens": total_pt + total_ct,
                **({"output_tokens_details": {"reasoning_tokens": total_reasoning_tokens}} if total_reasoning_tokens else {}),
                **({"input_tokens_details": {"cached_tokens": max_cached_tokens}} if max_cached_tokens else {}),
            },
        })
    except MemoryError:
        return JSONResponse(
            status_code=507,
            content={"error": {"message": "Insufficient GPU memory", "type": "server_error"}},
        )
    except Exception as e:
        logger.error(f"Responses API generation error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Internal server error", "type": "server_error"}},
        )
    finally:
        _release_lora_adapter(engine, loaded_adapter)
        if _ns_tracker is not None:
            try:
                _ns_tracker.unregister(response_id)
            except Exception:
                pass


async def _stream_response(engine, req, messages, response_id, json_schema, loaded_adapter=None, request=None):
    """SSE streaming for Responses API using proper event types.

    Emits the correct Responses API SSE events:
      response.created → response.in_progress → response.output_item.added
      → response.content_part.added → response.output_text.delta (per token)
      → response.output_text.done → response.content_part.done
      → response.output_item.done → response.completed

    Usage is reported via the response.completed event (not a separate chunk),
    regardless of include_usage — the flag is kept for backward compatibility
    but usage is always included in response.completed.

    n>1 is rejected at the router level (see create_response) before
    reaching this function.
    """
    from ..streaming import with_sse_keepalive
    from yunshu_engine.batched_engine import BatchedEngine
    from .chat import _release_lora_adapter
    is_batched = isinstance(engine, BatchedEngine)

    # Non-batched Engine path: apply chat template ourselves before passing
    # to engine.generate_stream().  The legacy Engine / EngineCore
    # _messages_to_text() does not call adapt_messages() and may strip
    # tool-related fields, producing garbage for tool-use conversations.
    _stream_prompt: str | list[dict] = messages
    if not is_batched and messages:
        _tokenizer = getattr(engine, '_tokenizer', None)
        if _tokenizer is not None and hasattr(_tokenizer, 'apply_chat_template'):
            try:
                from yunshu_engine.message_adapter import adapt_messages
                _adapted = adapt_messages(messages, getattr(engine, 'model_name', '') or '')
            except Exception:
                _adapted = messages
            try:
                _tpl_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
                if req.enable_thinking is not None:
                    _tpl_kwargs["enable_thinking"] = req.enable_thinking
                _rendered = _tokenizer.apply_chat_template(_adapted, **_tpl_kwargs)
                if _rendered:
                    _stream_prompt = _rendered
            except TypeError as _te:
                if 'enable_thinking' in str(_te):
                    _tpl_kwargs.pop('enable_thinking', None)
                    _rendered = _tokenizer.apply_chat_template(_adapted, **_tpl_kwargs)
                    if _rendered:
                        _stream_prompt = _rendered
                else:
                    logger.debug("chat template failed for non-batched streaming Responses path", exc_info=True)
            except Exception:
                logger.debug("chat template failed for non-batched streaming Responses path", exc_info=True)

    _logit_bias = req.logit_bias
    if _logit_bias:
        _logit_bias = {int(k): v for k, v in _logit_bias.items()}
    prompt_tok = 0
    completion_tok = 0
    reasoning_tok = 0
    cached_tok = 0

    # Register with request tracker for cancellation support
    import uuid as _uuid
    _stream_id = f"resp-{_uuid.uuid4().hex[:8]}"
    _tracker = None
    _tracker_gen = None
    try:
        from yunshu_engine.request_tracker import get_request_tracker
        _tracker = get_request_tracker()
        _tracker_gen = _tracker.register(_stream_id, req.model or "")
    except Exception:
        _tracker = None

    _cancel_evt = _tracker_gen.cancel_event if _tracker_gen is not None else None

    # IDs for the output message and sequence numbering
    msg_id = f"msg-{uuid.uuid4().hex[:24]}"
    _seq = 0
    accumulated_text = ""

    def _next_seq():
        nonlocal _seq
        _seq += 1
        return _seq

    _metrics_recorded = False
    try:
      async def _token_source():
        nonlocal prompt_tok, completion_tok, reasoning_tok, cached_tok, accumulated_text, _metrics_recorded
        last_finish_reason = None

        # ── Lifecycle: response.created ──
        yield format_responses_created(response_id, req.model, seq=_next_seq())

        # ── Lifecycle: response.in_progress ──
        yield format_responses_in_progress(response_id, req.model, seq=_next_seq())

        # ── Lifecycle: response.output_item.added ──
        yield format_responses_output_item_added(
            response_id, req.model, item_id=msg_id, output_index=0, seq=_next_seq(),
        )

        # ── Lifecycle: response.content_part.added ──
        yield format_responses_content_part_added(
            item_id=msg_id, output_index=0, content_index=0, seq=_next_seq(),
        )

        if is_batched:
            async for output in engine.stream_chat(
                messages=messages,
                max_tokens=req.max_output_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                seed=req.seed,
                enable_thinking=req.enable_thinking,
                thinking_budget=req.thinking_budget,
                reasoning_effort=req.reasoning_effort,
                repetition_penalty=req.repetition_penalty,
                frequency_penalty=req.frequency_penalty,
                presence_penalty=req.presence_penalty,
                logit_bias=_logit_bias,
                min_p=req.min_p,
                json_schema=json_schema,
                stop=req.stop,
                stop_token_ids=req.stop_token_ids,
                spec_decode=req.spec_decode,
                xtc_probability=req.xtc_probability,
                xtc_threshold=req.xtc_threshold,
                priority=req.priority,
                logprobs=req.logprobs,
                top_logprobs=req.top_logprobs,
                logits_processors=req.logits_processors,
                cancel_event=_cancel_evt,
                timeout_seconds=req.timeout,
            ):
                if hasattr(output, 'prompt_tokens') and output.prompt_tokens:
                    prompt_tok = output.prompt_tokens
                if hasattr(output, 'reasoning_tokens') and output.reasoning_tokens:
                    reasoning_tok = output.reasoning_tokens
                if hasattr(output, 'cached_tokens') and output.cached_tokens:
                    cached_tok = max(cached_tok, output.cached_tokens)
                if hasattr(output, 'completion_tokens') and output.completion_tokens:
                    completion_tok = output.completion_tokens
                elif output.new_text:
                    completion_tok += 1
                if output.new_text:
                    _is_reasoning = getattr(output, 'current_state', None) == "reasoning"
                    if not _is_reasoning:
                        accumulated_text += output.new_text
                if output.finish_reason is not None:
                    last_finish_reason = output.finish_reason

                # ── Per-token: response.output_text.delta ──
                if output.new_text and not _is_reasoning:
                    yield format_responses_text_delta(
                        delta=output.new_text,
                        item_id=msg_id,
                        output_index=0,
                        content_index=0,
                        seq=_next_seq(),
                    )
        else:
            async for output in engine.generate_stream(
                prompt=_stream_prompt,
                max_tokens=req.max_output_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                seed=req.seed,
                enable_thinking=req.enable_thinking,
                thinking_budget=req.thinking_budget,
                reasoning_effort=req.reasoning_effort,
                repetition_penalty=req.repetition_penalty,
                frequency_penalty=req.frequency_penalty,
                presence_penalty=req.presence_penalty,
                logit_bias=_logit_bias,
                min_p=req.min_p,
                json_schema=json_schema,
                stop=req.stop,
                stop_token_ids=req.stop_token_ids,
                spec_decode=req.spec_decode,
                xtc_probability=req.xtc_probability,
                xtc_threshold=req.xtc_threshold,
                priority=req.priority,
                logprobs=req.logprobs,
                top_logprobs=req.top_logprobs,
                logits_processors=req.logits_processors,
                cancel_event=_cancel_evt,
                timeout_seconds=req.timeout,
            ):
                if hasattr(output, 'prompt_tokens') and output.prompt_tokens:
                    prompt_tok = output.prompt_tokens
                if hasattr(output, 'reasoning_tokens') and output.reasoning_tokens:
                    reasoning_tok = output.reasoning_tokens
                if hasattr(output, 'cached_tokens') and output.cached_tokens:
                    cached_tok = max(cached_tok, output.cached_tokens)
                token_text = getattr(output, 'token_text', '')
                _is_reasoning = getattr(output, 'current_state', None) == "reasoning"
                if hasattr(output, 'completion_token_count') and output.completion_token_count:
                    completion_tok = output.completion_token_count
                elif token_text and not _is_reasoning:
                    completion_tok += 1
                if token_text:
                    if not _is_reasoning:
                        accumulated_text += token_text
                if getattr(output, 'finish_reason', None) is not None:
                    last_finish_reason = output.finish_reason

                # ── Per-token: response.output_text.delta ──
                if token_text and not _is_reasoning:
                    yield format_responses_text_delta(
                        delta=token_text,
                        item_id=msg_id,
                        output_index=0,
                        content_index=0,
                        seq=_next_seq(),
                    )

        # ── Lifecycle: response.output_text.done ──
        yield format_responses_text_done(
            text=accumulated_text,
            item_id=msg_id,
            output_index=0,
            content_index=0,
            seq=_next_seq(),
        )

        # ── Lifecycle: response.content_part.done ──
        yield format_responses_content_part_done(
            item_id=msg_id,
            text=accumulated_text,
            output_index=0,
            content_index=0,
            seq=_next_seq(),
        )

        # ── Lifecycle: response.output_item.done ──
        yield format_responses_output_item_done(
            item_id=msg_id,
            text=accumulated_text,
            output_index=0,
            seq=_next_seq(),
        )

        # ── Check for tool calls in the accumulated text ──
        tool_calls = None
        clean_text = accumulated_text
        if req.tools:
            from .chat import extract_tool_calls_model_aware, clean_tool_call_markup
            tool_calls = extract_tool_calls_model_aware(accumulated_text, req.model)
            if tool_calls:
                clean_text = clean_tool_call_markup(accumulated_text)

        # ── Build final output for response.completed ──
        content_parts = [
            {
                "type": "output_text",
                "text": clean_text,
                "annotations": [],
            }
        ]
        final_output = [
            {
                "type": "message",
                "id": msg_id,
                "role": "assistant",
                "content": content_parts,
                "status": "completed",
            }
        ]

        # Append function_call items for detected tool calls
        if tool_calls:
            for tc in tool_calls:
                final_output.append({
                    "type": "function_call",
                    "id": f"fc-{uuid.uuid4().hex[:24]}",
                    "call_id": f"call_{uuid.uuid4().hex[:8]}",
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                })

        # ── Lifecycle: response.completed (includes usage) ──
        yield format_responses_completed(
            response_id=response_id,
            model=req.model,
            output=final_output,
            input_tokens=prompt_tok,
            output_tokens=completion_tok,
            total_tokens=prompt_tok + completion_tok,
            reasoning_tokens=reasoning_tok,
            cached_tokens=cached_tok,
            seq=_next_seq(),
        )

        _record_metrics(prompt_tok, completion_tok)
        _metrics_recorded = True

      async for chunk in with_sse_keepalive(_token_source(), http_request=request, cancel_event=_cancel_evt):
        yield chunk.encode("utf-8") if isinstance(chunk, str) else chunk
    except MemoryError:
        if _cancel_evt is not None:
            _cancel_evt.set()
        # Close open lifecycle items before reporting failure
        yield format_responses_content_part_done(
            msg_id, text=accumulated_text,
            output_index=0, content_index=0, seq=_next_seq(),
        ).encode("utf-8")
        yield format_responses_output_item_done(
            msg_id, text=accumulated_text,
            output_index=0, seq=_next_seq(),
        ).encode("utf-8")
        yield format_responses_failed(
            response_id, req.model,
            error_code="server_error",
            error_message="Insufficient GPU memory",
            seq=_next_seq(),
        ).encode("utf-8")
        return
    except Exception as e:
        if _cancel_evt is not None:
            _cancel_evt.set()
        logger.error(f"Responses API streaming error: {e}", exc_info=True)
        # Close open lifecycle items before reporting failure
        yield format_responses_content_part_done(
            msg_id, text=accumulated_text,
            output_index=0, content_index=0, seq=_next_seq(),
        ).encode("utf-8")
        yield format_responses_output_item_done(
            msg_id, text=accumulated_text,
            output_index=0, seq=_next_seq(),
        ).encode("utf-8")
        yield format_responses_failed(
            response_id, req.model,
            error_code="server_error",
            error_message=str(e)[:200],
            seq=_next_seq(),
        ).encode("utf-8")
        return
    finally:
        if _tracker is not None:
            try:
                _tracker.unregister(_stream_id)
            except Exception:
                pass
        if loaded_adapter is not None:
            _release_lora_adapter(engine, loaded_adapter)
        if not _metrics_recorded and (prompt_tok > 0 or completion_tok > 0):
            try:
                _record_metrics(prompt_tok, completion_tok)
            except Exception:
                pass
