"""Anthropic Messages API compatible router.

Supports:
- Messages API (streaming + non-streaming)
- Token counting
- Thinking/reasoning mode
- Image blocks (base64)
- Cache control hints (ephemeral)
- Streaming tool-use deltas (input_json_delta)
- stop_sequences parameter handling
- Both Engine (legacy) and BatchedEngine backends
"""
import json
import re
import uuid
from collections.abc import AsyncIterator
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from ..engine import get_engine, get_model_manager
from ..streaming import (
    ThinkingParser,
    format_anthropic_chunk,
    with_sse_keepalive,
)

router = APIRouter(tags=["anthropic"])


def _record_metrics(prompt_tokens: int, completion_tokens: int) -> None:
    """Record token counts to the metrics middleware."""
    try:
        from ..middleware.metrics import get_metrics
        get_metrics().record_tokens(prompt_tokens, completion_tokens)
        get_metrics().record_inference()
    except Exception:
        pass


# ── Request / Response schemas ──


class AnthropicMessage(BaseModel):
    role: str
    content: Optional[str | list[dict]] = None


class AnthropicToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Optional[dict] = None


class AnthropicTool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Optional[dict] = None
    type: str = "custom"


class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage]
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    stream: bool = False
    stop_sequences: Optional[list[str]] = None
    system: Optional[str | list[dict]] = None
    thinking: Optional[dict] = None
    metadata: Optional[dict] = None
    tools: Optional[list[AnthropicTool]] = None
    tool_choice: Optional[dict | str] = None


# ── Content block helpers ──


def _extract_text_from_content(content: str | list[dict] | None) -> str:
    """Convert Anthropic content (string or list of blocks) to plain text.

    Handles:
    - Plain string content
    - List of content blocks (text, image, tool_result, etc.)
    - Image blocks are converted to a placeholder description
    - Cache control hints are stripped (routing hints, not content)
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        block_type = block.get("type", "")

        if block_type == "text":
            parts.append(block.get("text", ""))

        elif block_type == "image":
            source = block.get("source", {})
            media_type = source.get("media_type", "unknown")
            parts.append(f"[Image: {media_type}]")

        elif block_type == "tool_use":
            tool_name = block.get("name", "unknown")
            tool_input = block.get("input", {})
            parts.append(f"[Tool use: {tool_name}({json.dumps(tool_input)})]")

        elif block_type == "tool_result":
            tool_use_id = block.get("tool_use_id", "unknown")
            inner = block.get("content", "")
            if isinstance(inner, list):
                inner_text = _extract_text_from_content(inner)
            else:
                inner_text = str(inner) if inner is not None else ""
            parts.append(f"[Tool result {tool_use_id}: {inner_text}]")

        elif block_type == "thinking":
            parts.append(block.get("thinking", ""))

        else:
            parts.append(str(block))

    return "\n".join(parts)


def _has_image_blocks(content: str | list[dict] | None) -> bool:
    """Check whether a message content contains any image blocks."""
    if not isinstance(content, list):
        return False
    return any(
        isinstance(b, dict) and b.get("type") == "image"
        for b in content
    )


def _extract_cache_control_hints(system: str | list[dict] | None) -> list[dict]:
    """Extract cache_control hints from system messages.

    Anthropic uses cache_control to signal prompt-caching breakpoints.
    We return these as structured hints for downstream routing but do not
    alter the generation logic.
    """
    if system is None:
        return []
    hints: list[dict] = []
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and "cache_control" in block:
                hints.append(block["cache_control"])
    return hints


# ── Streaming tool-use helpers ──

# Regex to detect tool-call JSON inside model output
_TOOL_CALL_JSON_RE = re.compile(
    r'\{[\s\n]*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}',
)
_TOOL_CALL_XML_RE = re.compile(
    r'<tool_call\s*/?\s*>\s*(.*?)\s*</tool_call\s*/?\s*>',
    re.DOTALL,
)


def _try_parse_tool_call_delta(text: str) -> list[dict] | None:
    """Try to parse partial or complete tool-call JSON from streaming text.

    Returns a list of {"name": str, "arguments": str} dicts if a tool call
    is detected, or None if the text doesn't contain a recognisable tool call.
    """
    # Try XML-wrapped tool calls first
    for m in _TOOL_CALL_XML_RE.finditer(text):
        inner = m.group(1).strip()
        try:
            data = json.loads(inner)
            if "name" in data:
                return [{"name": data["name"], "arguments": json.dumps(data.get("arguments", {}))}]
        except json.JSONDecodeError:
            pass

    # Try bare JSON
    calls: list[dict] = []
    for m in _TOOL_CALL_JSON_RE.finditer(text):
        name = m.group(1)
        args_str = m.group(2)
        calls.append({"name": name, "arguments": args_str})
    return calls or None


# ── Endpoint ──


@router.post("/messages", response_model=None)
async def create_message(req: AnthropicMessagesRequest, request: Request):
    """Anthropic Messages API endpoint."""
    # Build messages list (prepend system if present)
    messages = []
    if req.system:
        system_text = _extract_text_from_content(req.system) if isinstance(req.system, list) else req.system
        messages.append({"role": "system", "content": system_text})

    for m in req.messages:
        content = _extract_text_from_content(m.content)
        messages.append({"role": m.role, "content": content})

    stop = req.stop_sequences or []

    # Inject tool definitions into system prompt if provided
    if req.tools:
        tool_prompt = "\n\nYou have access to the following tools. When you need to call a tool, "
        tool_prompt += 'output a tool call in the following format:\n<tool_call\\>{"name": "...", "arguments": {...}}</tool_call\\>\n\n'
        tool_prompt += "Available tools:\n"
        for tool in req.tools:
            tool_prompt += f"- {tool.name}"
            if tool.description:
                tool_prompt += f": {tool.description}"
            if tool.input_schema:
                tool_prompt += f"\n  Parameters: {tool.input_schema}"
            tool_prompt += "\n"

        if req.tool_choice and isinstance(req.tool_choice, dict):
            forced = req.tool_choice.get("name")
            if forced:
                tool_prompt += f"\nYou MUST call the tool '{forced}'.\n"

        if messages and messages[0].get("role") == "system":
            messages[0]["content"] += tool_prompt
        else:
            messages.insert(0, {"role": "system", "content": tool_prompt.strip()})

    # Resolve engine
    engine, is_batched = await _resolve_engine(req.model)

    if req.stream:
        return StreamingResponse(
            _stream_anthropic(engine, messages, req, stop, request, is_batched=is_batched),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    # Non-streaming
    if is_batched:
        return await _non_stream_batched(engine, messages, req, stop)

    return await _non_stream_legacy(engine, messages, req, stop)


async def _resolve_engine(model_id: str):
    """Resolve engine, returning (engine, is_batched) tuple."""
    engine = get_engine()

    if engine is not None and engine.is_loaded and engine.resolve_model_id(model_id):
        return engine, False

    # Multi-model mode
    from ..engine import get_engine_for_model
    try:
        engine = await get_engine_for_model(model_id)
    except (KeyError, Exception):
        engine = None

    if engine is None:
        if get_engine() is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    from yunshu_engine.batched_engine import BatchedEngine
    is_batched = isinstance(engine, BatchedEngine)
    return engine, is_batched


async def _non_stream_batched(engine, messages, req, stop):
    """Non-streaming response via BatchedEngine."""
    enable_thinking = req.thinking and req.thinking.get("type") == "enabled"
    budget_tokens = req.thinking.get("budget_tokens") if req.thinking else None

    result = await engine.chat(
        messages=messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop=stop,
        enable_thinking=enable_thinking,
    )
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    _record_metrics(result.prompt_tokens, result.completion_tokens)

    content = []
    thinking_text = ""
    visible_text = result.text

    if enable_thinking:
        from ..streaming import extract_thinking
        thinking_text, visible_text = extract_thinking(result.text)
        if thinking_text:
            content.append({"type": "thinking", "thinking": thinking_text})

    content.append({"type": "text", "text": visible_text})

    return JSONResponse({
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": req.model,
        "stop_reason": "end_turn" if result.finish_reason == "stop" else (result.finish_reason or "end_turn"),
        "usage": {
            "input_tokens": result.prompt_tokens,
            "output_tokens": result.completion_tokens,
        },
    })


async def _non_stream_legacy(engine, messages, req, stop):
    """Non-streaming response via legacy Engine."""
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    state = await engine.generate(
        prompt=messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop=stop,
    )
    _record_metrics(state.prompt_token_count, state.completion_token_count)
    return JSONResponse({
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": state.generated_text}],
        "model": req.model,
        "stop_reason": "end_turn" if state.finish_reason == "stop" else (state.finish_reason or "end_turn"),
        "usage": {
            "input_tokens": state.prompt_token_count,
            "output_tokens": state.completion_token_count,
        },
    })


async def _stream_anthropic(
    engine, messages, req, stop, request, is_batched=False
) -> AsyncIterator[bytes]:
    """Anthropic SSE streaming with keepalive, disconnect detection, and tool-use deltas."""
    parser = ThinkingParser()
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    input_tokens = 0
    output_tokens = 0
    enable_thinking = req.thinking and req.thinking.get("type") == "enabled"
    has_tools = req.tools is not None and len(req.tools) > 0
    block_index = 0
    thinking_block_started = False
    text_block_started = False
    tool_use_block_started = False
    accumulated_text = ""  # for tool-call detection
    matched_stop: str | None = None

    # message_start event
    msg_start = {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": req.model,
            "stop_reason": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }
    yield f"event: message_start\ndata: {json.dumps(msg_start)}\n\n".encode("utf-8")

    # content_block_start for thinking (if enabled)
    if enable_thinking:
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'thinking', 'thinking': ''}})}\n\n".encode("utf-8")
        thinking_block_started = True

    async def _token_source():
        nonlocal input_tokens, output_tokens, block_index
        nonlocal thinking_block_started, text_block_started, tool_use_block_started
        nonlocal accumulated_text, matched_stop

        if is_batched:
            async for output in engine.stream_chat(
                messages=messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                stop=stop,
                enable_thinking=enable_thinking,
            ):
                parsed = parser.process_chunk(output.new_text)

                # Thinking content
                if enable_thinking and parsed["thinking"]:
                    if not thinking_block_started:
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'thinking', 'thinking': ''}})}\n\n"
                        thinking_block_started = True
                    output_tokens += 1
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'thinking_delta', 'thinking': parsed['thinking']}})}\n\n"

                # Visible text content
                if parsed["visible"]:
                    if thinking_block_started and not text_block_started:
                        # Close thinking block, open text block
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                        block_index += 1
                        thinking_block_started = False
                        text_block_started = True
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    elif not text_block_started:
                        text_block_started = True
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

                    output_tokens += 1
                    accumulated_text += parsed["visible"]

                    # Check for stop sequences
                    if stop:
                        for seq in stop:
                            if seq in accumulated_text:
                                idx = accumulated_text.find(seq)
                                safe_text = accumulated_text[:idx]
                                matched_stop = seq
                                accumulated_text = safe_text
                                break

                    # If tools are defined, try to detect and emit tool-use deltas
                    if has_tools and not matched_stop:
                        tool_calls = _try_parse_tool_call_delta(accumulated_text)
                        if tool_calls:
                            if not tool_use_block_started:
                                # Close the text block, open a tool_use block
                                if text_block_started:
                                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                                    block_index += 1
                                tool_use_block_started = True
                                for tc in tool_calls:
                                    tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': tc['name'], 'input': {}}})}\n\n"
                                    # Emit input_json_delta for streaming the arguments
                                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'input_json_delta', 'partial_json': tc['arguments']}})}\n\n"
                                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                                    block_index += 1
                            return  # tool calls emitted; stop normal text streaming

                    # Normal text delta
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': parsed['visible']}})}\n\n"

                if output.prompt_tokens and not input_tokens:
                    input_tokens = output.prompt_tokens
        else:
            async for output in engine.generate_stream(
                prompt=messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                stop=stop,
            ):
                parsed = parser.process_chunk(output.token_text)
                if parsed["visible"]:
                    if not text_block_started:
                        text_block_started = True
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    output_tokens += 1
                    accumulated_text += parsed["visible"]

                    # Check for stop sequences
                    if stop:
                        for seq in stop:
                            if seq in accumulated_text:
                                idx = accumulated_text.find(seq)
                                safe_text = accumulated_text[:idx]
                                matched_stop = seq
                                accumulated_text = safe_text
                                break

                    # Tool-use delta detection for legacy engine
                    if has_tools and not matched_stop:
                        tool_calls = _try_parse_tool_call_delta(accumulated_text)
                        if tool_calls:
                            if text_block_started:
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                                block_index += 1
                            tool_use_block_started = True
                            for tc in tool_calls:
                                tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': tc['name'], 'input': {}}})}\n\n"
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'input_json_delta', 'partial_json': tc['arguments']}})}\n\n"
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                                block_index += 1
                            return

                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': parsed['visible']}})}\n\n"

        # Flush final
        final = parser.finalize()
        if final.get("visible"):
            if not text_block_started:
                text_block_started = True
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': final['visible']}})}\n\n"

        # Close last content block
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"

        # message_delta (stop + usage)
        stop_reason = "end_turn"
        if matched_stop:
            stop_reason = "stop_sequence"
        delta_data = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason},
            "usage": {"output_tokens": output_tokens},
        }
        yield f"event: message_delta\ndata: {json.dumps(delta_data)}\n\n"

    async for event in with_sse_keepalive(
        _token_source(),
        http_request=request,
    ):
        yield event.encode("utf-8") if isinstance(event, str) else event

    yield "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".encode("utf-8")

    _record_metrics(input_tokens, output_tokens)


@router.post("/messages/count_tokens")
async def count_tokens(req: AnthropicMessagesRequest) -> dict:
    """Token counting endpoint (Anthropic compatible)."""
    engine, _ = await _resolve_engine(req.model)

    tokenizer = getattr(engine, '_tokenizer', None)
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="No tokenizer available")

    text_parts = []
    if req.system:
        text_parts.append(
            _extract_text_from_content(req.system) if isinstance(req.system, list) else req.system
        )
    for m in req.messages:
        content = _extract_text_from_content(m.content)
        text_parts.append(content)
    full_text = "\n".join(text_parts)
    tokens = tokenizer.encode(full_text)

    return {"input_tokens": len(tokens)}
