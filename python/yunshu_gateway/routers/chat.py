"""OpenAI Chat Completions compatible router.

Supports:
- Text chat completions (LLM mode)
- Vision chat completions with image input (VLM mode)
- Streaming and non-streaming
- enable_thinking parameter for reasoning models
- Tool calling with extraction from model output
- Context window validation (oMLX pattern)
- SSE keepalive + disconnect guard (oMLX pattern)
- Full OpenAI message format (text, image_url, content arrays)
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Optional, Union

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from ..engine import get_engine, get_model_manager
from ..streaming import (
    extract_thinking,
    extract_tool_calls_v2 as extract_tool_calls,
    extract_tool_calls_model_aware,
    clean_tool_call_markup,
    format_openai_chunk,
    format_openai_done,
    format_openai_usage_chunk,
    format_openai_non_stream,
    validate_context_window,
    with_sse_keepalive,
    run_with_disconnect_guard,
)
from yunshu_engine.tool_call_streamer import ToolCallStreamer
from yunshu_engine.gateway_optimizer import get_streaming_buffer

logger = logging.getLogger(__name__)


async def _try_execute_mcp_tools(
    tool_calls: list[dict],
    request: Request,
) -> list[dict]:
    """Execute MCP tool calls and return results list.

    For each extracted tool call, checks if it matches an MCP tool
    (via the MCPClientManager). If so, executes the tool call and
    appends the result. Non-MCP tools are skipped.

    Returns a list of {"tool_call_id": str, "output": str} dicts
    for each executed MCP tool call.
    """
    results = []
    mcp_mgr = getattr(request.app.state, "mcp_client", None)
    if mcp_mgr is None or not tool_calls:
        return results

    for tc in tool_calls:
        name = tc.get("name", "")
        arguments = tc.get("arguments", {})
        if isinstance(arguments, str):
            try:
                import json as _json
                arguments = _json.loads(arguments)
            except Exception:
                logger.debug("operation failed", exc_info=True)
                arguments = {}

        try:
            result = await mcp_mgr.call_tool(name, arguments)
            results.append({
                "tool_call_id": tc.get("id", ""),
                "output": json.dumps(result) if not isinstance(result, str) else result,
            })
            logger.info("MCP tool executed: %s", name)
        except KeyError:
            # Not an MCP tool — skip (client-side tool)
            pass
        except Exception as e:
            logger.warning("MCP tool execution failed for %s: %s", name, e)
            results.append({
                "tool_call_id": tc.get("id", ""),
                "output": json.dumps({"error": str(e)}),
            })

    return results

router = APIRouter(tags=["chat"])


def _record_metrics(prompt_tokens: int, completion_tokens: int) -> None:
    """Record token counts to metrics middleware and server stats."""
    try:
        from ..middleware.metrics import get_metrics
        get_metrics().record_tokens(prompt_tokens, completion_tokens)
        get_metrics().record_inference()
    except Exception:
        logger.debug("metrics recording failed", exc_info=True)
    try:
        from yunshu_engine.server_metrics import get_server_metrics
        get_server_metrics().record(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    except Exception:
        logger.debug("server_metrics recording failed", exc_info=True)
    try:
        from yunshu_engine.tracing import get_metrics_v2
        get_metrics_v2().counter("yunshu_tokens_total", {"type": "prompt"}, prompt_tokens)
        get_metrics_v2().counter("yunshu_tokens_total", {"type": "completion"}, completion_tokens)
    except Exception:
        logger.debug("metrics recording failed", exc_info=True)


def _apply_lora_adapter(engine, adapter_id: str | None) -> str | None:
    """Apply a LoRA adapter to the engine for this request.

    Returns the adapter_id if loaded, None if not applicable.
    The caller must call _release_lora_adapter() after generation.
    """
    if not adapter_id:
        return None
    lora_mgr = getattr(engine, 'get_lora_manager', lambda: None)()
    if lora_mgr is None:
        logger.warning(f"LoRA adapter '{adapter_id}' requested but engine has no LoRA manager")
        return None
    if lora_mgr.load_adapter(adapter_id):
        return adapter_id
    logger.warning(f"Failed to load LoRA adapter '{adapter_id}'")
    return None


def _release_lora_adapter(engine, adapter_id: str | None) -> None:
    """Unload a LoRA adapter after generation completes."""
    if not adapter_id:
        return
    lora_mgr = getattr(engine, 'get_lora_manager', lambda: None)()
    if lora_mgr is not None:
        lora_mgr.unload_adapter(adapter_id)


# ── Request / Response schemas (OpenAI-compatible) ──


class TextContent(BaseModel):
    type: str = "text"
    text: str


class ImageURL(BaseModel):
    url: str


class ImageContent(BaseModel):
    type: str = "image_url"
    image_url: ImageURL


ContentPart = Union[TextContent, ImageContent, dict]


class ChatMessage(BaseModel):
    role: str
    content: Union[str, list[ContentPart], None] = None


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None


class ToolDefinition(BaseModel):
    type: str = "function"
    function: ToolFunction


class ToolChoiceString(BaseModel):
    """tool_choice = 'auto' | 'none'"""
    pass


class ToolChoiceFunction(BaseModel):
    """tool_choice = {"type": "function", "function": {"name": "..."}}"."""
    type: str = "function"
    function: ToolFunction


class StreamOptions(BaseModel):
    """OpenAI stream_options parameter."""
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=0.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[dict[int, float]] = None
    max_tokens: int = Field(default=512, ge=1, le=131072)
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    stop: Optional[list[str]] = None
    enable_thinking: Optional[bool] = None
    tools: Optional[list[ToolDefinition]] = None
    tool_choice: Optional[Union[str, ToolChoiceFunction]] = None
    parallel_tool_calls: bool = True
    response_format: Optional[dict] = None
    seed: Optional[int] = None
    logprobs: bool = False
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=20)
    n: int = Field(default=1, ge=1, le=128)
    user: Optional[str] = None
    # Advanced engine parameters
    spec_decode: bool = False
    thinking_budget: Optional[int] = Field(default=None, ge=1, le=32768)
    reasoning_effort: Optional[str] = None
    stop_token_ids: Optional[list[int]] = None
    priority: int = Field(default=0, ge=0, le=100)
    xtc_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    xtc_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    grammar: Optional[dict] = None  # {"type": "json", "schema": {...}} or {"type": "regex", "pattern": "..."}
    lora_adapter: Optional[str] = None  # LoRA adapter ID to apply for this request

    @model_validator(mode="after")
    def validate_request(self):
        if self.stop and len(self.stop) > 16:
            raise ValueError("stop: maximum 16 stop sequences")
        return self


def _parse_response_format(response_format: dict | None, grammar: dict | None = None) -> dict | str | None:
    """Parse OpenAI response_format and grammar parameters into json_schema.

    Supports:
    - {"type": "json_object"} → generic object schema
    - {"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}} → specific schema
    - grammar: {"type": "json", "schema": {...}} → specific schema
    - grammar: {"type": "json"} → generic JSON constraint
    - grammar: {"type": "regex", "pattern": "..."} → regex constraint
    - grammar: {"type": "choice", "choices": [...]} → enumeration constraint
    - grammar: {"type": "cfg", "grammar": "...", "start": "start"} → context-free grammar
    - None → no constraint

    For non-JSON grammar types, returns the grammar dict as-is for the engine
    to create the appropriate constraint via ConstraintFactory.
    """
    # grammar takes priority when it specifies a schema
    if grammar is not None:
        gtype = grammar.get("type")
        if gtype == "json":
            schema = grammar.get("schema")
            if schema:
                return schema
            return "json_object"
        if gtype == "regex":
            return grammar  # Pass through for ConstraintFactory
        if gtype == "choice":
            return grammar
        if gtype == "cfg":
            return grammar

    if response_format is None:
        return None

    rf_type = response_format.get("type")
    if rf_type == "json_object":
        return "json_object"
    if rf_type == "json_schema":
        js = response_format.get("json_schema", {})
        schema = js.get("schema")
        if schema:
            return schema
        return "json_object"

    return None


def _extract_messages(msgs: list[ChatMessage]) -> list[dict]:
    """Convert ChatMessage objects to dicts, preserving multimodal content.

    Follows oMLX's extract_multimodal_content pattern:
    - String content -> {"role": ..., "content": "..."}
    - List content with image_url -> {"role": ..., "content": [{type: "text", ...}, {type: "image_url", ...}]}
    """
    result = []
    for msg in msgs:
        d: dict[str, Any] = {"role": msg.role}
        if isinstance(msg.content, str):
            d["content"] = msg.content
        elif msg.content is None:
            d["content"] = ""
        elif isinstance(msg.content, list):
            parts = []
            for part in msg.content:
                if isinstance(part, dict):
                    parts.append(part)
                elif hasattr(part, "model_dump"):
                    parts.append(part.model_dump())
                elif isinstance(part, TextContent):
                    parts.append({"type": "text", "text": part.text})
                elif isinstance(part, ImageContent):
                    parts.append({"type": "image_url", "image_url": {"url": part.image_url.url}})
                else:
                    parts.append({"type": "text", "text": str(part)})
            d["content"] = parts
        else:
            d["content"] = str(msg.content)
        result.append(d)
    return result


def _has_images(messages: list[dict]) -> bool:
    """Check if any message contains image content."""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def _has_audio(messages: list[dict]) -> bool:
    """Check if any message contains audio content."""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("input_audio", "audio_url"):
                    return True
    return False


def _has_video(messages: list[dict]) -> bool:
    """Check if any message contains video content."""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("video_url", "video_file"):
                    return True
    return False


def _inject_tool_system_prompt(
    messages: list[dict],
    tools: list[ToolDefinition],
    tool_choice: Optional[Union[str, ToolChoiceFunction]] = None,
    parallel_tool_calls: bool = True,
) -> list[dict]:
    """Inject tool definitions into the system prompt (oMLX pattern).

    For models without native tool calling, we inject tool descriptions
    into the system message so the model can generate tool calls.

    Supports tool_choice:
    - "auto" (default): model decides whether to call tools
    - "none": tools provided but model must NOT call them
    - {"type": "function", "function": {"name": "..."}}: force specific tool
    """
    if not tools:
        return messages

    # tool_choice="none": inject minimal info so model knows tools exist
    # but is instructed NOT to call them
    if tool_choice == "none":
        tool_prompt = (
            "You have access to tools, but you must NOT call any tools in this response. "
            "Respond to the user directly using your own knowledge.\n"
        )
        messages = list(messages)
        system_idx = None
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                system_idx = i
                break
        if system_idx is not None:
            existing = messages[system_idx].get("content", "")
            messages[system_idx]["content"] = f"{existing}\n\n{tool_prompt}"
        else:
            messages.insert(0, {"role": "system", "content": tool_prompt})
        return messages

    tool_descriptions = []
    for tool in tools:
        func = tool.function
        desc = {
            "name": func.name,
            "description": func.description or "",
        }
        if func.parameters:
            desc["parameters"] = func.parameters
        tool_descriptions.append(desc)

    tool_prompt = (
        "You have access to the following tools. When you need to call a tool, "
        "output a tool call in the following format:\n"
        '<tool_call\\>{"name": "function_name", "arguments": {...}}</tool_call\\>\n\n'
    )
    if not parallel_tool_calls:
        tool_prompt += "You MUST make only ONE tool call per response.\n\n"
    tool_prompt += "Available tools:\n"
    for td in tool_descriptions:
        tool_prompt += f"- {td['name']}: {td['description']}\n"
        if 'parameters' in td:
            tool_prompt += f"  Parameters: {td['parameters']}\n"

    # tool_choice = specific function: instruct model to call that tool
    if isinstance(tool_choice, ToolChoiceFunction):
        forced_name = tool_choice.function.name
        tool_prompt += (
            f"\nYou MUST call the tool '{forced_name}'. "
            f"Do not respond with text — only output a tool call.\n"
        )
    elif tool_choice == "auto" or tool_choice is None:
        tool_prompt += (
            "\nDecide whether to call a tool based on the user's request. "
            "If you can answer directly, do so. If you need a tool, use it.\n"
        )

    messages = list(messages)  # copy

    # Find existing system message and append
    system_idx = None
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            system_idx = i
            break

    if system_idx is not None:
        existing = messages[system_idx].get("content", "")
        messages[system_idx]["content"] = f"{existing}\n\n{tool_prompt}"
    else:
        messages.insert(0, {"role": "system", "content": tool_prompt})

    return messages


def _format_logprobs(
    raw_logprobs: Any,
    tokenizer: Any,
    top_logprobs: int | None = None,
) -> dict | None:
    """Format logprobs from engine output into OpenAI Chat Completions format.

    OpenAI returns logprobs as:
    {"content": [{"token": "...", "logprob": -1.23, "top_logprobs": [{"token": "...", "logprob": -0.5}, ...]}]}
    """
    if not raw_logprobs:
        return None

    entries = []
    if isinstance(raw_logprobs, (list, tuple)):
        for lp in raw_logprobs:
            if isinstance(lp, dict):
                token_str = lp.get("token", "")
                if not token_str and tokenizer and "token_id" in lp:
                    try:
                        token_str = tokenizer.decode([lp["token_id"]])
                    except Exception:
                        logger.debug("tokenizer decode failed for logprobs", exc_info=True)
                        token_str = ""
                entries.append({
                    "token": token_str,
                    "logprob": lp.get("logprob", 0.0),
                    "bytes": list(token_str.encode("utf-8")) if token_str else [],
                    "top_logprobs": lp.get("top_logprobs", []),
                })
            elif isinstance(lp, (int, float)):
                entries.append({
                    "token": "",
                    "logprob": float(lp),
                    "bytes": [],
                    "top_logprobs": [],
                })

    if not entries:
        return None

    return {"content": entries}


async def _build_multi_choice(
    engine, req, messages, completion_id, is_batched, json_schema,
):
    """Build n > 1 completions by running parallel generation calls."""
    import asyncio

    prompt_tok = 0
    completion_tok = 0
    reasoning_tok = 0
    cached_tok = 0
    choices = []

    async def _gen_one(idx: int):
        nonlocal prompt_tok, completion_tok, reasoning_tok
        if is_batched:
            result = await engine.chat(
                messages=messages,
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
                json_schema=json_schema,
                spec_decode=req.spec_decode,
                thinking_budget=req.thinking_budget,
                stop_token_ids=req.stop_token_ids,
                reasoning_effort=req.reasoning_effort,
                xtc_probability=req.xtc_probability,
                xtc_threshold=req.xtc_threshold,
                priority=req.priority,
            )
            text = result.text
            pt = result.prompt_tokens
            ct = result.completion_tokens
            fr = result.finish_reason or "stop"
        else:
            state = await engine.generate(
                prompt=messages,
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
                stop_token_ids=req.stop_token_ids,
                thinking_budget=req.thinking_budget,
                reasoning_effort=req.reasoning_effort,
                xtc_probability=req.xtc_probability,
                xtc_threshold=req.xtc_threshold,
                spec_decode=req.spec_decode,
                json_schema=json_schema,
                priority=req.priority,
            )
            text = state.generated_text
            pt = state.prompt_token_count
            ct = state.completion_token_count
            fr = state.finish_reason or "stop"

        thinking_content, regular_content = extract_thinking(text, req.model)
        cleaned = regular_content.strip()

        tool_calls = []
        if req.tools:
            tool_calls = extract_tool_calls_model_aware(regular_content, req.model)
            if tool_calls:
                cleaned = clean_tool_call_markup(regular_content)
                fr = "tool_calls"

        message = {"role": "assistant", "content": cleaned}
        if thinking_content:
            message["reasoning_content"] = thinking_content
        if tool_calls:
            message["tool_calls"] = [
                {"id": f"call_{idx}:{i:x}", "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for i, tc in enumerate(tool_calls)
            ]

        _record_metrics(pt, ct)
        _rt = getattr(result, 'reasoning_tokens', 0) if is_batched else 0
        _ct_cached = getattr(result, 'cached_tokens', 0) if is_batched else 0
        return idx, pt, ct, _rt, _ct_cached, {"index": idx, "message": message, "finish_reason": fr}

    results = await asyncio.gather(
        *[_gen_one(i) for i in range(req.n)], return_exceptions=True,
    )

    errors = []
    for r in results:
        if isinstance(r, BaseException):
            idx = results.index(r)
            errors.append((idx, r))
            logger.error(f"choice {idx} failed: {r}", exc_info=r)
            continue
        idx, pt, ct, _rt, _ct_cached, choice = r
        choices.append(choice)
        prompt_tok = pt
        completion_tok += ct
        reasoning_tok += _rt
        cached_tok = max(cached_tok, _ct_cached)

    if not choices and errors:
        exc = errors[0][1]
        if isinstance(exc, MemoryError):
            return JSONResponse(
                status_code=507,
                content={"error": {"message": "Out of GPU memory", "type": "memory_error"}},
            )
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(exc), "type": type(exc).__name__}},
        )

    usage = {
        "prompt_tokens": prompt_tok,
        "completion_tokens": completion_tok,
        "total_tokens": prompt_tok + completion_tok,
    }
    if reasoning_tok > 0:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tok}
    if cached_tok > 0:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tok}

    return JSONResponse({
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": sorted(choices, key=lambda c: c["index"]),
        "usage": usage,
    })


# ── Endpoints ──


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(req: ChatCompletionRequest, request: Request):
    # Audit: log user field if provided (OpenAI spec: end-user tracking)
    if req.user:
        logger.info(f"[{getattr(request.state, 'request_id', '-')}] user={req.user}")

    # Structured tracing + logging
    from yunshu_engine.tracing import get_inference_tracer, get_structured_logger
    tracer = get_inference_tracer()
    slog = get_structured_logger()

    trace_id = f"chat-{uuid.uuid4().hex[:16]}"
    trace = tracer.start_trace(trace_id, metadata={
        "model": req.model,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": req.stream,
        "endpoint": "/chat/completions",
    })
    tracer.span(trace_id, "prefill", {"model": req.model})
    slog.info("inference_request", model=req.model, trace_id=trace_id,
              max_tokens=req.max_tokens, stream=req.stream)

    messages = _extract_messages(req.messages)
    has_images = _has_images(messages)
    has_audio = _has_audio(messages)
    has_video = _has_video(messages)

    # Route to VLM/Omni engine if images, audio, or video are present
    if has_images or has_audio or has_video:
        json_schema = _parse_response_format(req.response_format, req.grammar)
        return await _handle_vlm_chat(req, messages, request, json_schema=json_schema)

    # Check if the target model is a VLM/Omni (route through VLM handler)
    from yunshu_engine.vlm_engine import VLMEngine
    manager = get_model_manager()
    if manager is not None:
        entry = manager.get_entry(req.model)
        if entry is not None and entry.model_type.name == "VLM":
            json_schema = _parse_response_format(req.response_format, req.grammar)
            return await _handle_vlm_chat(req, messages, request, json_schema=json_schema)

    # Standard LLM chat
    engine = get_engine()

    # Multi-model mode: resolve through model manager
    if engine is None or not engine.is_loaded or not engine.resolve_model_id(req.model):
        from ..engine import get_engine_for_model
        try:
            engine = await get_engine_for_model(req.model)
        except (KeyError, Exception) as e:
            if engine is None:
                raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found: {e}")
            # Single engine mode but model doesn't match
            raise HTTPException(
                status_code=404,
                detail=f"Model '{req.model}' not loaded. Loaded: {engine.model_name}",
            )

    # Inject tool definitions if provided
    if req.tools:
        messages = _inject_tool_system_prompt(messages, req.tools, req.tool_choice, req.parallel_tool_calls)

    # Parse response_format for structured output (JSON schema)
    json_schema = _parse_response_format(req.response_format, req.grammar)

    # Context window validation (oMLX pattern)
    # Estimate prompt tokens for validation before generation
    try:
        tokenizer = getattr(engine, '_tokenizer', None)
        if tokenizer is not None:
            text_parts = []
            image_count = 0
            for m in messages:
                content = m.get("content", "")
                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                            elif block.get("type") == "image_url":
                                image_count += 1
            from yunshu_control.token_counter import count_message_tokens
            est_tokens = count_message_tokens(messages, tokenizer)
            est_tokens += image_count * 576
            validate_context_window(est_tokens, req.model, engine)
    except HTTPException:
        raise
    except Exception:
        logger.debug("context window validation failed", exc_info=True)

    # Check if this is a BatchedEngine (oMLX pattern)
    from yunshu_engine.batched_engine import BatchedEngine
    is_batched = isinstance(engine, BatchedEngine)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if req.stream:
        if req.n > 1:
            return StreamingResponse(
                _stream_response_multi(
                    engine, messages, req, completion_id, request,
                    is_batched=is_batched, json_schema=json_schema,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        return StreamingResponse(
            _stream_response(
                engine, messages, req, completion_id, request,
                is_batched=is_batched, json_schema=json_schema,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming with disconnect guard (oMLX pattern)
    async def _build_response():
        loaded_adapter = _apply_lora_adapter(engine, req.lora_adapter)
        try:
            if req.n > 1:
                return await _build_multi_choice(
                    engine, req, messages, completion_id, is_batched, json_schema,
                )

            try:
                if is_batched:
                    result = await engine.chat(
                        messages=messages,
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
                        json_schema=json_schema,
                        logprobs=req.logprobs,
                        spec_decode=req.spec_decode,
                        thinking_budget=req.thinking_budget,
                        stop_token_ids=req.stop_token_ids,
                        reasoning_effort=req.reasoning_effort,
                        xtc_probability=req.xtc_probability,
                        xtc_threshold=req.xtc_threshold,
                        priority=req.priority,
                    )
                    raw_text = result.text
                    prompt_tok = result.prompt_tokens
                    completion_tok = result.completion_tokens
                    finish = result.finish_reason or "stop"
                    logprobs_data = _format_logprobs(
                        getattr(result, 'logprobs', None),
                        getattr(engine, '_tokenizer', None),
                        req.top_logprobs,
                    )
                else:
                    state = await engine.generate(
                        prompt=messages,
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
                        stop_token_ids=req.stop_token_ids,
                        thinking_budget=req.thinking_budget,
                        reasoning_effort=req.reasoning_effort,
                        xtc_probability=req.xtc_probability,
                        xtc_threshold=req.xtc_threshold,
                        spec_decode=req.spec_decode,
                        json_schema=json_schema,
                        logprobs=req.logprobs,
                        top_logprobs=req.top_logprobs,
                        priority=req.priority,
                    )
                    raw_text = state.generated_text
                    prompt_tok = state.prompt_token_count
                    completion_tok = state.completion_token_count
                    finish = state.finish_reason or "stop"
                    logprobs_data = _format_logprobs(
                        getattr(state, 'logprobs', None),
                        getattr(engine, '_tokenizer', None),
                        req.top_logprobs,
                    )
            except MemoryError:
                return JSONResponse(
                    status_code=507,
                    content={"error": {"message": "Out of GPU memory", "type": "memory_error"}},
                )
            except Exception as e:
                logger.error("engine inference failed", exc_info=True)
                return JSONResponse(
                    status_code=500,
                    content={"error": {"message": str(e), "type": type(e).__name__}},
                )

            # Extract thinking (oMLX pattern)
            thinking_content, regular_content = extract_thinking(raw_text, req.model)

            # Extract tool calls using model-aware format detection (C15)
            tool_calls = []
            cleaned_content = regular_content
            if req.tools:
                tool_calls = extract_tool_calls_model_aware(regular_content, req.model)
                if tool_calls:
                    cleaned_content = clean_tool_call_markup(regular_content)

            finish_reason = "tool_calls" if tool_calls else finish

            # Execute MCP tool calls if any (server-side tool execution)
            mcp_results = []
            if tool_calls:
                try:
                    mcp_results = await _try_execute_mcp_tools(tool_calls, request)
                except Exception:
                    logger.debug("MCP tool execution failed", exc_info=True)

            _record_metrics(prompt_tok, completion_tok)

            # End tracing
            tracer.end_span(trace_id, "prefill")
            tracer.end_trace(trace_id, result={
                "prompt_tokens": prompt_tok,
                "completion_tokens": completion_tok,
                "finish_reason": finish_reason,
            })
            slog.info("inference_complete", model=req.model, trace_id=trace_id,
                      prompt_tokens=prompt_tok, completion_tokens=completion_tok)

            response_body = format_openai_non_stream(
                completion_id=completion_id,
                model=req.model,
                content=cleaned_content.strip(),
                prompt_tokens=prompt_tok,
                completion_tokens=completion_tok,
                finish_reason=finish_reason,
                thinking_content=thinking_content if thinking_content else None,
                tool_calls=tool_calls if tool_calls else None,
                logprobs=logprobs_data,
            )

            # Attach MCP tool execution results (if any were executed)
            if mcp_results:
                response_body["mcp_tool_results"] = mcp_results

            return JSONResponse(response_body)
        finally:
            _release_lora_adapter(engine, loaded_adapter)

    return await run_with_disconnect_guard(request, _build_response())


async def _handle_vlm_chat(
    req: ChatCompletionRequest,
    messages: list[dict],
    request: Request,
    json_schema: dict | str | None = None,
) -> StreamingResponse | JSONResponse:
    """Handle chat completion via VLM engine (streaming + non-streaming)."""
    from yunshu_engine.vlm_engine import VLMEngine

    manager = get_model_manager()
    vlm_engine = None

    if manager is not None:
        from yunshu_engine.model_manager import ModelType

        # Try to match req.model first
        if req.model:
            for entry in manager.list_entries():
                if (entry.is_loaded and entry.model_id == req.model
                        and isinstance(getattr(entry, 'engine', None), VLMEngine)):
                    vlm_engine = entry.engine
                    break
            if vlm_engine is None:
                for entry in manager.list_entries():
                    if entry.model_id == req.model and entry.model_type == ModelType.VLM:
                        try:
                            vlm_engine = await manager.get_engine(entry.model_id)
                            break
                        except Exception:
                            logger.debug(f"VLM engine load failed for {entry.model_id}", exc_info=True)

        # Fallback: first available VLM engine
        if vlm_engine is None:
            for entry in manager.list_entries():
                if entry.is_loaded and isinstance(getattr(entry, 'engine', None), VLMEngine):
                    vlm_engine = entry.engine
                    break

            if vlm_engine is None:
                for entry in manager.list_entries():
                    if entry.model_type == ModelType.VLM:
                        try:
                            vlm_engine = await manager.get_engine(entry.model_id)
                            break
                        except Exception:
                            logger.debug(f"VLM engine load failed for {entry.model_id}", exc_info=True)

    if vlm_engine is None:
        raise HTTPException(
            status_code=404,
            detail="No VLM engine available for image input",
        )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    # Inject tool definitions if provided
    if req.tools:
        messages = _inject_tool_system_prompt(messages, req.tools, req.tool_choice, req.parallel_tool_calls)

    if req.stream:
        return StreamingResponse(
            _stream_vlm_response(vlm_engine, messages, req, completion_id, request, json_schema=json_schema),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    gen_kwargs: dict[str, Any] = dict(
        messages=messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        seed=req.seed,
        repetition_penalty=req.repetition_penalty,
        stop=req.stop,
        stop_token_ids=req.stop_token_ids,
        enable_thinking=req.enable_thinking,
        thinking_budget=req.thinking_budget,
        reasoning_effort=req.reasoning_effort,
        frequency_penalty=req.frequency_penalty,
        presence_penalty=req.presence_penalty,
        logit_bias=req.logit_bias,
        xtc_probability=req.xtc_probability,
        xtc_threshold=req.xtc_threshold,
    )
    if json_schema:
        gen_kwargs["json_schema"] = json_schema

    tok = getattr(vlm_engine, '_tokenizer', None)
    prompt_tok = 0
    if tok:
        try:
            prompt_text = vlm_engine._format_prompt(messages)
            prompt_tok = len(tok.encode(prompt_text))
        except Exception:
            logger.debug("prompt token count failed", exc_info=True)

    async def _vlm_gen_one(idx: int):
        try:
            r = await vlm_engine.generate(**gen_kwargs)
        except MemoryError:
            return idx, None, "memory_error"
        except Exception as e:
            logger.error("VLM engine inference failed", exc_info=True)
            return idx, None, str(e)

        content = r.get("text", "")
        rt = r.get("reasoning_tokens", 0)
        ct = len(tok.encode(content)) if tok else max(1, len(content) // 4)
        finish_reason = r.get("finish_reason", "stop")
        tool_calls = None
        if req.tools:
            tool_calls = extract_tool_calls_model_aware(content, req.model)
            if tool_calls:
                content = clean_tool_call_markup(content)
                finish_reason = "tool_calls"
        return idx, {
            "content": content.strip(),
            "reasoning_tokens": rt,
            "completion_tokens": ct,
            "finish_reason": finish_reason,
            "tool_calls": tool_calls,
        }, None

    n = max(req.n, 1)
    if n == 1:
        results = [await _vlm_gen_one(0)]
    else:
        import asyncio
        results = await asyncio.gather(*[_vlm_gen_one(i) for i in range(n)])
        results.sort(key=lambda x: x[0])

    # Check for errors
    for idx, data, err in results:
        if err == "memory_error":
            return JSONResponse(
                status_code=507,
                content={"error": {"message": "Out of GPU memory", "type": "memory_error"}},
            )
        if err is not None:
            return JSONResponse(
                status_code=500,
                content={"error": {"message": err, "type": "inference_error"}},
            )

    total_completion_tok = 0
    total_reasoning_tok = 0
    choices = []
    for idx, data, _ in results:
        total_completion_tok += data["completion_tokens"]
        total_reasoning_tok += data["reasoning_tokens"]
        message = {"role": "assistant", "content": data["content"]}
        if data["tool_calls"]:
            message["tool_calls"] = [
                {"id": f"call_vlm:{idx}:{i:x}", "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for i, tc in enumerate(data["tool_calls"])
            ]
        choices.append({
            "index": idx,
            "message": message,
            "finish_reason": data["finish_reason"],
        })

    vlm_usage = {
        "prompt_tokens": prompt_tok,
        "completion_tokens": total_completion_tok,
        "total_tokens": prompt_tok + total_completion_tok,
    }
    if total_reasoning_tok > 0:
        vlm_usage["completion_tokens_details"] = {"reasoning_tokens": total_reasoning_tok}

    return JSONResponse({
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": choices,
        "usage": vlm_usage,
    })


async def _stream_vlm_response(
    vlm_engine,
    messages: list[dict],
    req: ChatCompletionRequest,
    completion_id: str,
    request: Request,
    json_schema: dict | str | None = None,
) -> AsyncIterator[bytes]:
    """SSE streaming for VLM engine (oMLX with_sse_keepalive pattern)."""
    loaded_adapter = _apply_lora_adapter(vlm_engine, req.lora_adapter)

    async def _token_source():
        nonlocal loaded_adapter
        first_chunk = True
        vlm_prompt_tok = 0
        vlm_completion_tok = 0
        vlm_reasoning_tok = 0
        vlm_last_finish_reason = None
        include_usage = (
            req.stream_options is not None and req.stream_options.include_usage
        )
        stream_kwargs: dict[str, Any] = dict(
            messages=messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            seed=req.seed,
            repetition_penalty=req.repetition_penalty,
            stop=req.stop,
            stop_token_ids=req.stop_token_ids,
            enable_thinking=req.enable_thinking,
            thinking_budget=req.thinking_budget,
            reasoning_effort=req.reasoning_effort,
            frequency_penalty=req.frequency_penalty,
            presence_penalty=req.presence_penalty,
            logit_bias=req.logit_bias,
            xtc_probability=req.xtc_probability,
            xtc_threshold=req.xtc_threshold,
        )
        if json_schema:
            stream_kwargs["json_schema"] = json_schema
        async for output in vlm_engine.generate_stream(**stream_kwargs):
            if hasattr(output, 'token_text') and output.token_text:
                vlm_completion_tok += 1
            if hasattr(output, 'reasoning_tokens') and output.reasoning_tokens:
                vlm_reasoning_tok = output.reasoning_tokens
            if output.finish_reason is not None:
                vlm_last_finish_reason = output.finish_reason
            yield format_openai_chunk(
                completion_id=completion_id,
                model=req.model,
                delta_content=output.token_text,
                finish_reason=None,  # intermediate: always None
                include_role=first_chunk,
            )
            first_chunk = False

        # Final chunk with finish_reason
        yield format_openai_chunk(
            completion_id=completion_id,
            model=req.model,
            delta_content="",
            finish_reason=vlm_last_finish_reason or "stop",
        )

        if include_usage:
            tok = getattr(vlm_engine, '_tokenizer', None)
            if tok and not vlm_prompt_tok:
                try:
                    vlm_prompt_tok = len(tok.encode(vlm_engine._format_prompt(messages)))
                except Exception:
                    logger.debug("operation failed", exc_info=True)
                    pass
            yield format_openai_usage_chunk(
                completion_id=completion_id,
                model=req.model,
                prompt_tokens=vlm_prompt_tok,
                completion_tokens=vlm_completion_tok,
                reasoning_tokens=vlm_reasoning_tok,
            )

        yield format_openai_done()

    try:
      async for event in with_sse_keepalive(
          _token_source(),
          http_request=request,
      ):
          yield event.encode("utf-8")
    finally:
      _release_lora_adapter(vlm_engine, loaded_adapter)


async def _stream_response_multi(
    engine,
    messages: list[dict],
    req: ChatCompletionRequest,
    completion_id: str,
    request: Request,
    is_batched: bool = False,
    json_schema: dict | str | None = None,
) -> AsyncIterator[bytes]:
    """n>1 streaming: generate each choice sequentially, emit with correct index.

    On single-GPU systems parallel generation would serialize anyway, so we
    run choices one after another and interleave their SSE events.
    Each choice gets its own streaming loop with its `index` set correctly.
    """
    from yunshu_engine.request_tracker import get_request_tracker
    tracker = get_request_tracker()
    gen = tracker.register(completion_id, req.model)
    include_usage = (
        req.stream_options is not None and req.stream_options.include_usage
    )
    total_prompt_tok = 0
    total_completion_tok = 0
    total_reasoning_tok = 0

    async def _token_source():
        nonlocal total_prompt_tok, total_completion_tok, total_reasoning_tok
        for choice_idx in range(req.n):
            if gen.cancel_event.is_set():
                yield _format_choice_chunk(
                    completion_id, req.model, choice_idx, "", "cancelled",
                )
                break
            first_chunk_for_choice = True
            choice_completion_tok = 0
            choice_finish_reason = None  # track actual finish_reason from engine
            choice_reasoning_tok = 0  # track per-choice reasoning tokens

            if is_batched:
                stream = engine.stream_chat(
                    messages=messages,
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
                    seed=(req.seed + choice_idx) if req.seed is not None else None,
                    enable_thinking=req.enable_thinking,
                    json_schema=json_schema,
                    thinking_budget=req.thinking_budget,
                    reasoning_effort=req.reasoning_effort,
                    xtc_probability=req.xtc_probability,
                    xtc_threshold=req.xtc_threshold,
                    spec_decode=req.spec_decode,
                    priority=req.priority,
                )
                async for output in stream:
                    if gen.cancel_event.is_set():
                        yield _format_choice_chunk(
                            completion_id, req.model, choice_idx, "", "cancelled",
                        )
                        return
                    if hasattr(output, 'prompt_tokens') and output.prompt_tokens:
                        total_prompt_tok = output.prompt_tokens
                    if hasattr(output, 'reasoning_tokens') and output.reasoning_tokens:
                        choice_reasoning_tok = output.reasoning_tokens
                    token_text = output.new_text
                    if token_text:
                        choice_completion_tok += 1
                    # Only set finish_reason on the final token from engine
                    fr = output.finish_reason
                    if fr is not None:
                        choice_finish_reason = fr
                    yield _format_choice_chunk(
                        completion_id, req.model, choice_idx,
                        token_text, None,  # intermediate: always None
                        include_role=first_chunk_for_choice,
                    )
                    first_chunk_for_choice = False
            else:
                stream = engine.generate_stream(
                    prompt=messages,
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
                    seed=(req.seed + choice_idx) if req.seed is not None else None,
                    enable_thinking=req.enable_thinking,
                    stop_token_ids=req.stop_token_ids,
                    thinking_budget=req.thinking_budget,
                    reasoning_effort=req.reasoning_effort,
                    xtc_probability=req.xtc_probability,
                    xtc_threshold=req.xtc_threshold,
                    spec_decode=req.spec_decode,
                    json_schema=json_schema,
                    priority=req.priority,
                )
                async for output in stream:
                    if gen.cancel_event.is_set():
                        yield _format_choice_chunk(
                            completion_id, req.model, choice_idx, "", "cancelled",
                        )
                        return
                    if hasattr(output, 'prompt_token_count') and output.prompt_token_count:
                        total_prompt_tok = output.prompt_token_count
                    if hasattr(output, 'reasoning_tokens') and output.reasoning_tokens:
                        choice_reasoning_tok = output.reasoning_tokens
                    token_text = getattr(output, 'token_text', '')
                    if token_text:
                        choice_completion_tok += 1
                    # Only set finish_reason on the final token from engine
                    fr = getattr(output, 'finish_reason', None)
                    if fr is not None:
                        choice_finish_reason = fr
                    yield _format_choice_chunk(
                        completion_id, req.model, choice_idx,
                        token_text, None,  # intermediate: always None
                        include_role=first_chunk_for_choice,
                    )
                    first_chunk_for_choice = False

            total_completion_tok += choice_completion_tok
            total_reasoning_tok += choice_reasoning_tok

            # Emit final chunk with actual finish_reason for this choice
            yield _format_choice_chunk(
                completion_id, req.model, choice_idx,
                "", choice_finish_reason or "stop",
            )

        if include_usage:
            yield format_openai_usage_chunk(
                completion_id=completion_id,
                model=req.model,
                prompt_tokens=total_prompt_tok,
                completion_tokens=total_completion_tok,
                reasoning_tokens=total_reasoning_tok,
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
    tracker.unregister(completion_id)


def _format_choice_chunk(
    completion_id: str,
    model: str,
    index: int,
    delta_content: str,
    finish_reason: Optional[str],
    include_role: bool = False,
) -> str:
    """Format an SSE chunk for a specific choice index."""
    delta: dict[str, Any] = {}
    if include_role:
        delta["role"] = "assistant"
    delta["content"] = delta_content
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": index,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


async def _stream_response(
    engine,
    messages: list[dict],
    req: ChatCompletionRequest,
    completion_id: str,
    request: Request,
    is_batched: bool = False,
    json_schema: dict | str | None = None,
) -> AsyncIterator[bytes]:
    """SSE streaming response with keepalive and disconnect detection.

    Uses oMLX's with_sse_keepalive pattern for robust streaming.
    Supports both Engine (legacy) and BatchedEngine (oMLX pattern).
    Uses engine's current_state (from SequenceStateMachine) to route
    reasoning vs visible content, matching mlx-lm's server.py pattern.

    When tools are provided, uses ToolCallStreamer for incremental
    tool call detection — buffers tokens and emits tool_calls in
    OpenAI streaming delta format when detected.
    """
    loaded_adapter = _apply_lora_adapter(engine, req.lora_adapter)
    use_tool_streamer = req.tools is not None and len(req.tools) > 0
    tool_streamer = ToolCallStreamer() if use_tool_streamer else None
    tool_call_index = 0  # Track index for streaming tool_calls delta
    has_emitted_tool_call = False
    include_usage = (
        req.stream_options is not None and req.stream_options.include_usage
    )
    prompt_tok = 0
    completion_tok = 0
    reasoning_tok = 0

    def _format_tool_call_chunk(tc, idx: int) -> str:
        """Format a tool_call as an OpenAI streaming chunk with delta."""
        delta: dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "index": idx,
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments,
                },
            }],
        }
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": None,
            }],
        }
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    # Per-request streaming buffer for zero-alloc SSE ring buffering
    _stream_buf = None
    try:
        _stream_buf = get_streaming_buffer()
    except Exception:
        logger.debug("StreamingResponseBuffer creation failed", exc_info=True)

    async def _token_source():
        nonlocal tool_call_index, has_emitted_tool_call, prompt_tok, completion_tok
        first_chunk = True
        last_finish_reason = None  # track actual finish_reason from engine

        if is_batched:
            async for output in engine.stream_chat(
                messages=messages,
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
                json_schema=json_schema,
                thinking_budget=req.thinking_budget,
                reasoning_effort=req.reasoning_effort,
                xtc_probability=req.xtc_probability,
                xtc_threshold=req.xtc_threshold,
                priority=req.priority,
            ):
                token_text = output.new_text
                if output.finish_reason is not None:
                    last_finish_reason = output.finish_reason
                if hasattr(output, 'prompt_tokens') and output.prompt_tokens:
                    prompt_tok = output.prompt_tokens
                if hasattr(output, 'reasoning_tokens') and output.reasoning_tokens:
                    reasoning_tok = output.reasoning_tokens
                if token_text:
                    completion_tok += 1

                if use_tool_streamer and tool_streamer and token_text:
                    # Run through tool call streamer
                    outputs = tool_streamer.process_token(token_text)
                    for out in outputs:
                        if out.text:
                            yield format_openai_chunk(
                                completion_id=completion_id,
                                model=req.model,
                                delta_content=out.text,
                                include_role=first_chunk,
                            )
                            first_chunk = False
                        elif out.tool_call:
                            yield _format_tool_call_chunk(out.tool_call, tool_call_index)
                            tool_call_index += 1
                            has_emitted_tool_call = True
                else:
                    yield format_openai_chunk(
                        completion_id=completion_id,
                        model=req.model,
                        delta_content=token_text,
                        finish_reason=None,  # intermediate: always None
                        include_role=first_chunk,
                    )
                    first_chunk = False
        else:
            async for output in engine.generate_stream(
                prompt=messages,
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
            ):
                # Track token counts for usage reporting
                if hasattr(output, 'prompt_token_count') and output.prompt_token_count:
                    prompt_tok = output.prompt_token_count
                if hasattr(output, 'token_text') and output.token_text:
                    completion_tok += 1
                if hasattr(output, 'reasoning_tokens') and output.reasoning_tokens:
                    reasoning_tok = output.reasoning_tokens
                if output.finish_reason is not None:
                    last_finish_reason = output.finish_reason
                # Route based on SequenceStateMachine state (mlx-lm pattern)
                if output.current_state == "reasoning":
                    yield format_openai_chunk(
                        completion_id=completion_id,
                        model=req.model,
                        delta_content="",
                        thinking_content=output.token_text,
                        finish_reason=None,  # intermediate: always None
                        include_role=first_chunk,
                    )
                    first_chunk = False
                else:
                    token_text = output.token_text
                    if use_tool_streamer and tool_streamer and token_text:
                        outputs = tool_streamer.process_token(token_text)
                        for out in outputs:
                            if out.text:
                                yield format_openai_chunk(
                                    completion_id=completion_id,
                                    model=req.model,
                                    delta_content=out.text,
                                    include_role=first_chunk,
                                )
                                first_chunk = False
                            elif out.tool_call:
                                yield _format_tool_call_chunk(out.tool_call, tool_call_index)
                                tool_call_index += 1
                                has_emitted_tool_call = True
                    else:
                        yield format_openai_chunk(
                            completion_id=completion_id,
                            model=req.model,
                            delta_content=token_text,
                            finish_reason=None,  # intermediate: always None
                            include_role=first_chunk,
                        )
                        first_chunk = False

        # Flush any remaining content from tool streamer
        if use_tool_streamer and tool_streamer:
            for out in tool_streamer.flush():
                if out.text:
                    yield format_openai_chunk(
                        completion_id=completion_id,
                        model=req.model,
                        delta_content=out.text,
                    )
                elif out.tool_call:
                    yield _format_tool_call_chunk(out.tool_call, tool_call_index)
                    tool_call_index += 1
                    has_emitted_tool_call = True

        # Final chunk with finish_reason from engine
        if has_emitted_tool_call:
            final_reason = "tool_calls"
        else:
            final_reason = last_finish_reason or "stop"
        yield format_openai_chunk(
            completion_id=completion_id,
            model=req.model,
            delta_content="",
            finish_reason=final_reason,
        )

        # Emit usage stats if stream_options.include_usage is true
        if include_usage:
            yield format_openai_usage_chunk(
                completion_id=completion_id,
                model=req.model,
                prompt_tokens=prompt_tok,
                completion_tokens=completion_tok,
                reasoning_tokens=reasoning_tok,
            )

        yield format_openai_done()

    try:
      async for event in with_sse_keepalive(
          _token_source(),
          http_request=request,
      ):
          encoded = event.encode("utf-8")
          # Best-effort write to streaming buffer
          if _stream_buf is not None:
              try:
                  _stream_buf.write(encoded)
              except Exception:
                  logger.debug("StreamingResponseBuffer write failed", exc_info=True)
          yield encoded
    finally:
      _release_lora_adapter(engine, loaded_adapter)
      # Log buffer stats at debug level
      if _stream_buf is not None:
          try:
              stats = _stream_buf.get_stats()
              logger.debug(
                  "StreamingResponseBuffer stats: writes=%d, bytes=%d, flushes=%d, "
                  "utilization=%.1f%%",
                  stats["write_count"],
                  stats["bytes_written"],
                  stats["flush_count"],
                  stats["utilization_pct"],
              )
          except Exception:
              logger.debug("operation failed", exc_info=True)
              logger.debug("StreamingResponseBuffer stats logging failed", exc_info=True)
