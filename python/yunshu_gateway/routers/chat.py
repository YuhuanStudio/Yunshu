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
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Optional, Union

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..engine import get_engine, get_model_manager
from ..streaming import (
    extract_thinking,
    extract_tool_calls,
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

router = APIRouter(tags=["chat"])


def _record_metrics(prompt_tokens: int, completion_tokens: int) -> None:
    """Record token counts to the metrics middleware."""
    try:
        from ..middleware.metrics import get_metrics
        get_metrics().record_tokens(prompt_tokens, completion_tokens)
        get_metrics().record_inference()
    except Exception:
        pass


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
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logit_bias: Optional[dict[int, float]] = None
    max_tokens: int = 512
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
    top_logprobs: Optional[int] = None
    n: int = 1
    user: Optional[str] = None


def _parse_response_format(response_format: dict | None) -> dict | str | None:
    """Parse OpenAI response_format parameter into a json_schema for SamplingParams.

    Supports:
    - {"type": "json_object"} → generic object schema
    - {"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}} → specific schema
    - None → no constraint
    """
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


def _inject_tool_system_prompt(
    messages: list[dict],
    tools: list[ToolDefinition],
    tool_choice: Optional[Union[str, ToolChoiceFunction]] = None,
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
        "Available tools:\n"
    )
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
    choices = []

    async def _gen_one(idx: int):
        nonlocal prompt_tok, completion_tok
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
                enable_thinking=req.enable_thinking,
                json_schema=json_schema,
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
                enable_thinking=req.enable_thinking,
            )
            text = state.generated_text
            pt = state.prompt_token_count
            ct = state.completion_token_count
            fr = state.finish_reason or "stop"

        thinking_content, regular_content = extract_thinking(text)
        cleaned = regular_content.strip()

        tool_calls = []
        if req.tools:
            tool_calls = extract_tool_calls(regular_content)
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
        return idx, pt, ct, {"index": idx, "message": message, "finish_reason": fr}

    results = await asyncio.gather(*[_gen_one(i) for i in range(req.n)])
    for idx, pt, ct, choice in results:
        choices.append(choice)
        prompt_tok = pt
        completion_tok += ct

    return JSONResponse({
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": sorted(choices, key=lambda c: c["index"]),
        "usage": {
            "prompt_tokens": prompt_tok,
            "completion_tokens": completion_tok,
            "total_tokens": prompt_tok + completion_tok,
        },
    })


# ── Endpoints ──


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(req: ChatCompletionRequest, request: Request):
    messages = _extract_messages(req.messages)
    has_images = _has_images(messages)

    # Route to VLM engine if images are present
    if has_images:
        return await _handle_vlm_chat(req, messages, request)

    # Check if the target model is a VLM (route through VLM handler)
    from yunshu_engine.vlm_engine import VLMEngine
    manager = get_model_manager()
    if manager is not None:
        entry = manager.get_entry(req.model)
        if entry is not None and entry.model_type.name == "VLM":
            return await _handle_vlm_chat(req, messages, request)

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
        messages = _inject_tool_system_prompt(messages, req.tools, req.tool_choice)

    # Parse response_format for structured output (JSON schema)
    json_schema = _parse_response_format(req.response_format)

    # Context window validation (oMLX pattern)
    # Estimate prompt tokens for validation before generation
    try:
        tokenizer = getattr(engine, '_tokenizer', None)
        if tokenizer is not None:
            test_text = " ".join(
                m.get("content", "") if isinstance(m.get("content"), str) else ""
                for m in messages
            )
            est_tokens = len(tokenizer.encode(test_text))
            validate_context_window(est_tokens, req.model, engine)
    except HTTPException:
        raise
    except Exception:
        pass

    # Check if this is a BatchedEngine (oMLX pattern)
    from yunshu_engine.batched_engine import BatchedEngine
    is_batched = isinstance(engine, BatchedEngine)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if req.stream:
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
        if req.n > 1:
            return await _build_multi_choice(
                engine, req, messages, completion_id, is_batched, json_schema,
            )

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
                enable_thinking=req.enable_thinking,
                json_schema=json_schema,
                logprobs=req.logprobs,
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
                enable_thinking=req.enable_thinking,
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

        # Extract thinking (oMLX pattern)
        thinking_content, regular_content = extract_thinking(raw_text)

        # Extract tool calls (oMLX pattern)
        tool_calls = []
        cleaned_content = regular_content
        if req.tools:
            tool_calls = extract_tool_calls(regular_content)
            if tool_calls:
                cleaned_content = clean_tool_call_markup(regular_content)

        finish_reason = "tool_calls" if tool_calls else finish

        _record_metrics(prompt_tok, completion_tok)

        return JSONResponse(format_openai_non_stream(
            completion_id=completion_id,
            model=req.model,
            content=cleaned_content.strip(),
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            finish_reason=finish_reason,
            thinking_content=thinking_content if thinking_content else None,
            tool_calls=tool_calls if tool_calls else None,
            logprobs=logprobs_data,
        ))

    return await run_with_disconnect_guard(request, _build_response())


async def _handle_vlm_chat(
    req: ChatCompletionRequest,
    messages: list[dict],
    request: Request,
) -> StreamingResponse | JSONResponse:
    """Handle chat completion via VLM engine (streaming + non-streaming)."""
    from yunshu_engine.vlm_engine import VLMEngine

    manager = get_model_manager()
    vlm_engine = None

    if manager is not None:
        from yunshu_engine.model_manager import ModelType
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
                        pass

    if vlm_engine is None:
        raise HTTPException(
            status_code=404,
            detail="No VLM engine available for image input",
        )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if req.stream:
        return StreamingResponse(
            _stream_vlm_response(vlm_engine, messages, req, completion_id, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    result = await vlm_engine.generate(
        messages=messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )

    content = result.get("text", "")
    tok = getattr(vlm_engine, '_tokenizer', None)
    prompt_tok = 0
    completion_tok = 0
    if tok:
        try:
            prompt_text = vlm_engine._format_prompt(messages)
            prompt_tok = len(tok.encode(prompt_text))
        except Exception:
            pass
        completion_tok = len(tok.encode(content))
    else:
        completion_tok = max(1, len(content) // 4)

    return JSONResponse(format_openai_non_stream(
        completion_id=completion_id,
        model=req.model,
        content=content,
        prompt_tokens=prompt_tok,
        completion_tokens=completion_tok,
        finish_reason=result.get("finish_reason", "stop"),
    ))


async def _stream_vlm_response(
    vlm_engine,
    messages: list[dict],
    req: ChatCompletionRequest,
    completion_id: str,
    request: Request,
) -> AsyncIterator[bytes]:
    """SSE streaming for VLM engine (oMLX with_sse_keepalive pattern)."""

    async def _token_source():
        first_chunk = True
        async for output in vlm_engine.generate_stream(
            messages=messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        ):
            yield format_openai_chunk(
                completion_id=completion_id,
                model=req.model,
                delta_content=output.token_text,
                finish_reason=output.finish_reason,
                include_role=first_chunk,
            )
            first_chunk = False

        yield format_openai_done()

    async for event in with_sse_keepalive(
        _token_source(),
        http_request=request,
    ):
        yield event.encode("utf-8")


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
    use_tool_streamer = req.tools is not None and len(req.tools) > 0
    tool_streamer = ToolCallStreamer() if use_tool_streamer else None
    tool_call_index = 0  # Track index for streaming tool_calls delta
    has_emitted_tool_call = False
    include_usage = (
        req.stream_options is not None and req.stream_options.include_usage
    )
    prompt_tok = 0
    completion_tok = 0

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

    async def _token_source():
        nonlocal tool_call_index, has_emitted_tool_call, prompt_tok, completion_tok
        first_chunk = True

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
                enable_thinking=req.enable_thinking,
                json_schema=json_schema,
            ):
                token_text = output.new_text
                finish_reason = output.finish_reason
                if hasattr(output, 'prompt_tokens') and output.prompt_tokens:
                    prompt_tok = output.prompt_tokens
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
                        finish_reason=finish_reason,
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
                enable_thinking=req.enable_thinking,
            ):
                # Track token counts for usage reporting
                if hasattr(output, 'prompt_token_count') and output.prompt_token_count:
                    prompt_tok = output.prompt_token_count
                if hasattr(output, 'token_text') and output.token_text:
                    completion_tok += 1
                # Route based on SequenceStateMachine state (mlx-lm pattern)
                if output.current_state == "reasoning":
                    yield format_openai_chunk(
                        completion_id=completion_id,
                        model=req.model,
                        delta_content="",
                        thinking_content=output.token_text,
                        finish_reason=output.finish_reason,
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
                            finish_reason=output.finish_reason,
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

        # Final chunk with finish_reason
        if has_emitted_tool_call:
            yield format_openai_chunk(
                completion_id=completion_id,
                model=req.model,
                delta_content="",
                finish_reason="tool_calls",
            )
        else:
            yield format_openai_chunk(
                completion_id=completion_id,
                model=req.model,
                delta_content="",
                finish_reason="stop",
            )

        # Emit usage stats if stream_options.include_usage is true
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
