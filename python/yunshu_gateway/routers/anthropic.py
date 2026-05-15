from __future__ import annotations
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
import logging
import re
import uuid
from collections.abc import AsyncIterator
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse

logger = logging.getLogger(__name__)
from pydantic import BaseModel

from ..engine import get_engine
from .chat import _apply_lora_adapter, _release_lora_adapter
from ..streaming import (
    ThinkingParser,
    format_anthropic_chunk,
    with_sse_keepalive,
)

router = APIRouter(tags=["anthropic"])


# ── Anthropic stop_reason mapping ──
# Internal: "stop", "length", "tool_calls"
# Anthropic: "end_turn", "max_tokens", "stop_sequence", "tool_use"

_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


def _map_stop_reason(
    finish_reason: str | None,
    matched_stop: str | None = None,
    has_tool_calls: bool = False,
) -> str:
    """Map internal finish_reason to Anthropic stop_reason."""
    if has_tool_calls:
        return "tool_use"
    if matched_stop:
        return "stop_sequence"
    if finish_reason in _FINISH_REASON_MAP:
        return _FINISH_REASON_MAP[finish_reason]
    return "end_turn"


def _record_metrics(prompt_tokens: int, completion_tokens: int) -> None:
    """Record token counts to the metrics middleware."""
    try:
        from ..middleware.metrics import get_metrics
        get_metrics().record_tokens(prompt_tokens, completion_tokens)
        get_metrics().record_inference()
    except Exception:
        logger.debug("metrics recording failed", exc_info=True)


# ── Request / Response schemas ──


class AnthropicMessage(BaseModel):
    role: str
    content: Optional[str | list[dict]] = None


class AnthropicTool(BaseModel):
    """Anthropic tool definition.

    Per the Anthropic Messages API spec, tools have: name, description,
    input_schema. The ``type`` field is NOT part of the Anthropic spec —
    Anthropic server-side tools (web_search, computer, etc.) carry versioned
    types like ``web_search_20250305`` but user-defined tools have no type.
    """
    name: str
    description: Optional[str] = None
    input_schema: Optional[dict] = None
    type: Optional[str] = None  # Server-side tools set this; user tools omit it


class AnthropicMessagesRequest(BaseModel):
    """Anthropic Messages API request model.

    Native Anthropic fields: model, messages, max_tokens, temperature, top_p,
    top_k, stream, stop_sequences, system, thinking, metadata, tools,
    tool_choice.

    Extended fields (Yunshu-specific, not in the Anthropic spec) are marked
    with comments and forwarded to the engine for additional functionality.
    """
    # ── Anthropic-native fields ──
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

    # ── Yunshu-extended fields (forwarded to engine) ──
    lora_adapter: Optional[str] = None
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logit_bias: Optional[dict[str, float]] = None
    seed: Optional[int] = None
    reasoning_effort: Optional[str] = None
    stop_token_ids: Optional[list[int]] = None
    spec_decode: bool = False
    xtc_probability: float = 0.0
    xtc_threshold: float = 0.0
    priority: int = 0
    json_schema: Optional[dict] = None
    logprobs: bool = False
    top_logprobs: Optional[int] = None
    logits_processors: Optional[list] = None
    # Client-forwarded field (not Anthropic spec, but commonly sent by SDKs)
    response_format: Optional[dict] = None
    chat_template_kwargs: Optional[dict] = None


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
    r'\{[\s\n]*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*',
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

    # Try bare JSON — find the opening brace after name and parse from there
    calls: list[dict] = []
    for m in _TOOL_CALL_JSON_RE.finditer(text):
        name = m.group(1)
        rest = text[m.end():]
        brace_depth = 0
        end = -1
        for i, ch in enumerate(rest):
            if ch == '{':
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
                if brace_depth == 0:
                    end = i
                    break
        if end >= 0:
            args_str = rest[:end + 1]
            try:
                json.loads(args_str)
                calls.append({"name": name, "arguments": args_str})
            except json.JSONDecodeError:
                pass
    return calls or None


# ── Endpoint ──


@router.post("/messages", response_model=None)
async def create_message(req: AnthropicMessagesRequest, request: Request):
    """Anthropic Messages API endpoint."""
    # Build messages list (prepend system if present)
    messages = []
    _temp_files: list[str] = []  # track temp files for cleanup
    if req.system:
        system_text = _extract_text_from_content(req.system) if isinstance(req.system, list) else req.system
        messages.append({"role": "system", "content": system_text})

    has_images = any(_has_image_blocks(m.content) for m in req.messages)

    for m in req.messages:
        if has_images and isinstance(m.content, list):
            # VLM path: preserve image blocks as OpenAI-style content parts
            converted_parts = []
            for block in m.content:
                if not isinstance(block, dict):
                    converted_parts.append({"type": "text", "text": str(block)})
                    continue
                bt = block.get("type", "")
                if bt == "text":
                    converted_parts.append({"type": "text", "text": block.get("text", "")})
                elif bt == "image":
                    source = block.get("source", {})
                    media_type = source.get("media_type", "unknown")
                    data = source.get("data")
                    if data and source.get("type") == "base64":
                        import base64 as _b64
                        import tempfile as _tf
                        try:
                            raw = _b64.b64decode(data, validate=False)
                        except Exception:
                            logger.debug("base64 decode failed, trying with padding", exc_info=True)
                            raw = _b64.b64decode(data + "==", validate=False)
                        ext_map = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp"}
                        ext = ext_map.get(media_type, "png")
                        tmp = _tf.NamedTemporaryFile(suffix=f".{ext}", delete=False)
                        tmp.write(raw)
                        tmp.close()
                        _temp_files.append(tmp.name)
                        converted_parts.append({"type": "image_url", "image_url": {"url": f"file://{tmp.name}"}})
                    else:
                        converted_parts.append({"type": "text", "text": f"[Image: {media_type}]"})
                else:
                    text = _extract_text_from_content([block])
                    converted_parts.append({"type": "text", "text": text})
            messages.append({"role": m.role, "content": converted_parts})
        else:
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
    try:
        engine, is_batched = await _resolve_engine(req.model)
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={
                "type": "error",
                "error": {
                    "type": "not_found_error" if e.status_code == 404 else "overloaded_error",
                    "message": e.detail,
                },
            },
        )

    if req.stream:
        # Note: temp files for streaming are cleaned up by the caller
        # (HTTP response completion). For long-running streams, this is
        # acceptable since the files are small.
        return StreamingResponse(
            _stream_anthropic(engine, messages, req, stop, request, is_batched=is_batched),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    # Non-streaming
    loaded_adapter = _apply_lora_adapter(engine, req.lora_adapter)
    try:
        if is_batched:
            return await _non_stream_batched(engine, messages, req, stop)
        return await _non_stream_legacy(engine, messages, req, stop)
    finally:
        _release_lora_adapter(engine, loaded_adapter)
        # Clean up temp files created for image blocks
        import os as _os
        for _tf_path in _temp_files:
            try:
                _os.unlink(_tf_path)
            except OSError:
                pass


async def _resolve_engine(model_id: str):
    """Resolve engine, returning (engine, is_batched) tuple."""
    from yunshu_engine.batched_engine import BatchedEngine
    engine = get_engine()

    if engine is not None and engine.is_loaded and engine.resolve_model_id(model_id):
        return engine, isinstance(engine, BatchedEngine)

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
    from fastapi.responses import JSONResponse
    enable_thinking = req.thinking and req.thinking.get("type") == "enabled"
    budget_tokens = req.thinking.get("budget_tokens") if req.thinking else None
    effective_max_tokens = min(req.max_tokens, budget_tokens) if budget_tokens else req.max_tokens

    try:
        result = await engine.chat(
        messages=messages,
        max_tokens=effective_max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        min_p=getattr(req, 'min_p', 0.0),
        repetition_penalty=getattr(req, 'repetition_penalty', 1.0),
        frequency_penalty=getattr(req, 'frequency_penalty', 0.0),
        presence_penalty=getattr(req, 'presence_penalty', 0.0),
        logit_bias=getattr(req, 'logit_bias', None),
        stop=stop,
        seed=getattr(req, 'seed', None),
        enable_thinking=enable_thinking,
        thinking_budget=budget_tokens,
        reasoning_effort=getattr(req, 'reasoning_effort', None),
        stop_token_ids=getattr(req, 'stop_token_ids', None),
        spec_decode=getattr(req, 'spec_decode', False),
        xtc_probability=getattr(req, 'xtc_probability', 0.0),
        xtc_threshold=getattr(req, 'xtc_threshold', 0.0),
        priority=getattr(req, 'priority', 0),
        json_schema=getattr(req, 'json_schema', None),
        logprobs=getattr(req, 'logprobs', False),
        top_logprobs=getattr(req, 'top_logprobs', None),
        logits_processors=getattr(req, 'logits_processors', None),
    )
    except MemoryError:
        return JSONResponse(
            status_code=507,
            content={"type": "error", "error": {"type": "overloaded_error", "message": "Insufficient GPU memory"}},
        )
    except Exception as e:
        logger.error(f"Anthropic batched generation error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"type": "error", "error": {"type": "api_error", "message": str(e)}},
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
            content.append({"type": "thinking", "thinking": thinking_text, "signature": "yunshu-reasoning"})

    text_block: dict = {"type": "text", "text": visible_text}

    # Include logprobs in the text content block if requested
    if req.logprobs:
        _result_lp = getattr(result, 'logprobs', None)
        if _result_lp:
            # Anthropic format: logprobs array in the content block
            # Each entry: {"token": str, "logprob": float, "top_logprobs": [...]}
            formatted_lp = _format_anthropic_logprobs(_result_lp)
            if formatted_lp:
                text_block["logprobs"] = formatted_lp

    content.append(text_block)

    matched_stop = None
    if stop and visible_text:
        for seq in stop:
            if visible_text.rstrip().endswith(seq.rstrip()):
                matched_stop = seq
                break

    stop_reason = _map_stop_reason(result.finish_reason, matched_stop)

    cache_creation = getattr(result, 'prompt_tokens', 0) - getattr(result, 'cached_tokens', 0)
    cache_read = getattr(result, 'cached_tokens', 0)

    resp = {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": req.model,
        "stop_reason": stop_reason,
        "stop_sequence": matched_stop,
        "usage": {
            "input_tokens": result.prompt_tokens,
            "output_tokens": result.completion_tokens,
            "cache_creation_input_tokens": max(0, cache_creation),
            "cache_read_input_tokens": max(0, cache_read),
        },
    }
    return JSONResponse(resp)


async def _non_stream_legacy(engine, messages, req, stop):
    """Non-streaming response via Engine or BatchedEngine."""
    from fastapi.responses import JSONResponse
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    enable_thinking = req.thinking and req.thinking.get("type") == "enabled"
    budget_tokens = req.thinking.get("budget_tokens") if req.thinking else None
    effective_max_tokens = min(req.max_tokens, budget_tokens) if budget_tokens else req.max_tokens
    try:
        result = await engine.generate(
            prompt=messages,
            max_tokens=effective_max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            min_p=getattr(req, 'min_p', 0.0),
            repetition_penalty=getattr(req, 'repetition_penalty', 1.0),
            frequency_penalty=getattr(req, 'frequency_penalty', 0.0),
            presence_penalty=getattr(req, 'presence_penalty', 0.0),
            logit_bias=getattr(req, 'logit_bias', None),
            stop=stop,
            seed=getattr(req, 'seed', None),
            enable_thinking=enable_thinking,
            thinking_budget=budget_tokens,
            reasoning_effort=getattr(req, 'reasoning_effort', None),
            stop_token_ids=getattr(req, 'stop_token_ids', None),
            spec_decode=getattr(req, 'spec_decode', False),
            xtc_probability=getattr(req, 'xtc_probability', 0.0),
            xtc_threshold=getattr(req, 'xtc_threshold', 0.0),
            priority=getattr(req, 'priority', 0),
            json_schema=getattr(req, 'json_schema', None),
            logprobs=getattr(req, 'logprobs', False),
            top_logprobs=getattr(req, 'top_logprobs', None),
            logits_processors=getattr(req, 'logits_processors', None),
        )
    except MemoryError:
        return JSONResponse(
            status_code=507,
            content={"type": "error", "error": {"type": "overloaded_error", "message": "Insufficient GPU memory"}},
        )
    except Exception as e:
        logger.error(f"Anthropic legacy generation error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"type": "error", "error": {"type": "api_error", "message": str(e)}},
        )
    # Handle both Engine (prompt_token_count) and BatchedEngine (prompt_tokens)
    prompt_toks = getattr(result, 'prompt_tokens', 0) or getattr(result, 'prompt_token_count', 0)
    completion_toks = getattr(result, 'completion_tokens', 0) or getattr(result, 'completion_token_count', 0)
    text = getattr(result, 'text', '') or getattr(result, 'generated_text', '')
    finish_reason = getattr(result, 'finish_reason', None) or getattr(result, 'finish_state', None)
    cached_toks = getattr(result, 'cached_tokens', 0) or 0
    _record_metrics(prompt_toks, completion_toks)

    # Extract thinking tokens if thinking mode enabled
    content = []
    visible_text = text
    if enable_thinking:
        from ..streaming import extract_thinking
        thinking_text, visible_text = extract_thinking(text)
        if thinking_text:
            content.append({"type": "thinking", "thinking": thinking_text, "signature": "yunshu-reasoning"})
    text_block: dict = {"type": "text", "text": visible_text}

    # Include logprobs in the text content block if requested
    if req.logprobs:
        _result_lp = getattr(result, 'logprobs', None)
        if _result_lp:
            formatted_lp = _format_anthropic_logprobs(_result_lp)
            if formatted_lp:
                text_block["logprobs"] = formatted_lp

    content.append(text_block)

    # Check for matched stop sequences
    matched_stop = None
    if stop and visible_text:
        for seq in stop:
            if visible_text.rstrip().endswith(seq.rstrip()):
                matched_stop = seq
                break

    stop_reason = _map_stop_reason(finish_reason, matched_stop)

    return JSONResponse({
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": req.model,
        "stop_reason": stop_reason,
        "stop_sequence": matched_stop,
        "usage": {
            "input_tokens": prompt_toks,
            "output_tokens": completion_toks,
            "cache_creation_input_tokens": max(0, prompt_toks - cached_toks),
            "cache_read_input_tokens": max(0, cached_toks),
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
    cached_tokens = 0
    enable_thinking = req.thinking and req.thinking.get("type") == "enabled"
    budget_tokens = req.thinking.get("budget_tokens") if req.thinking else None
    effective_max_tokens = min(req.max_tokens, budget_tokens) if budget_tokens else req.max_tokens
    has_tools = req.tools is not None and len(req.tools) > 0
    block_index = 0
    thinking_block_started = False
    text_block_started = False
    tool_use_block_started = False
    accumulated_text = ""  # for tool-call detection
    matched_stop: str | None = None

    # Register with request tracker for cancellation support
    from yunshu_engine.request_tracker import get_request_tracker
    _anth_tracker = get_request_tracker()
    _anth_gen = _anth_tracker.register(message_id, req.model)

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
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }
    yield f"event: message_start\ndata: {json.dumps(msg_start)}\n\n".encode("utf-8")

    # content_block_start for thinking (if enabled)
    if enable_thinking:
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'thinking', 'thinking': '', 'signature': 'yunshu-reasoning'}})}\n\n".encode("utf-8")
        thinking_block_started = True

    async def _token_source():
        nonlocal input_tokens, output_tokens, block_index, cached_tokens
        nonlocal thinking_block_started, text_block_started, tool_use_block_started
        nonlocal accumulated_text, matched_stop

        if is_batched:
            async for output in engine.stream_chat(
                messages=messages,
                max_tokens=effective_max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                min_p=getattr(req, 'min_p', 0.0),
                repetition_penalty=getattr(req, 'repetition_penalty', 1.0),
                frequency_penalty=getattr(req, 'frequency_penalty', 0.0),
                presence_penalty=getattr(req, 'presence_penalty', 0.0),
                logit_bias=getattr(req, 'logit_bias', None),
                stop=stop,
                seed=getattr(req, 'seed', None),
                enable_thinking=enable_thinking,
                thinking_budget=budget_tokens,
                reasoning_effort=getattr(req, 'reasoning_effort', None),
                stop_token_ids=getattr(req, 'stop_token_ids', None),
                spec_decode=getattr(req, 'spec_decode', False),
                xtc_probability=getattr(req, 'xtc_probability', 0.0),
                xtc_threshold=getattr(req, 'xtc_threshold', 0.0),
                priority=getattr(req, 'priority', 0),
                json_schema=getattr(req, 'json_schema', None),
                logprobs=getattr(req, 'logprobs', False),
                top_logprobs=getattr(req, 'top_logprobs', None),
                logits_processors=getattr(req, 'logits_processors', None),
                cancel_event=_anth_gen.cancel_event,
            ):
                parsed = parser.process_chunk(output.new_text)

                # Thinking content
                if enable_thinking and parsed["thinking"]:
                    if not thinking_block_started:
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'thinking', 'thinking': '', 'signature': 'yunshu-reasoning'}})}\n\n"
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
                if hasattr(output, 'cached_tokens') and output.cached_tokens:
                    cached_tokens = max(cached_tokens, output.cached_tokens)
        else:
            async for output in engine.generate_stream(
                prompt=messages,
                max_tokens=effective_max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                min_p=getattr(req, 'min_p', 0.0),
                repetition_penalty=getattr(req, 'repetition_penalty', 1.0),
                frequency_penalty=getattr(req, 'frequency_penalty', 0.0),
                presence_penalty=getattr(req, 'presence_penalty', 0.0),
                logit_bias=getattr(req, 'logit_bias', None),
                stop=stop,
                seed=getattr(req, 'seed', None),
                enable_thinking=enable_thinking,
                thinking_budget=budget_tokens,
                reasoning_effort=getattr(req, 'reasoning_effort', None),
                stop_token_ids=getattr(req, 'stop_token_ids', None),
                spec_decode=getattr(req, 'spec_decode', False),
                xtc_probability=getattr(req, 'xtc_probability', 0.0),
                xtc_threshold=getattr(req, 'xtc_threshold', 0.0),
                priority=getattr(req, 'priority', 0),
                json_schema=getattr(req, 'json_schema', None),
                logprobs=getattr(req, 'logprobs', False),
                top_logprobs=getattr(req, 'top_logprobs', None),
                logits_processors=getattr(req, 'logits_processors', None),
            ):
                if hasattr(output, 'prompt_token_count') and output.prompt_token_count and not input_tokens:
                    input_tokens = output.prompt_token_count
                if hasattr(output, 'cached_tokens') and output.cached_tokens:
                    cached_tokens = max(cached_tokens, output.cached_tokens)
                parsed = parser.process_chunk(output.token_text)

                # Thinking content (legacy engine path)
                if enable_thinking and parsed["thinking"]:
                    if not thinking_block_started:
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'thinking', 'thinking': '', 'signature': 'yunshu-reasoning'}})}\n\n"
                        thinking_block_started = True
                    output_tokens += 1
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'thinking_delta', 'thinking': parsed['thinking']}})}\n\n"

                if parsed["visible"]:
                    # Close thinking block if transitioning to text
                    if thinking_block_started and not text_block_started:
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                        block_index += 1
                        thinking_block_started = False
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

        # If no content blocks were started at all (zero tokens), emit an empty text block
        # so that the response always has at least one content block (Anthropic protocol requirement)
        if not text_block_started and not thinking_block_started:
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            text_block_started = True

        # Close last content block (only if one was actually started)
        if text_block_started or thinking_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"

        # message_delta (stop + usage)
        stop_reason = _map_stop_reason(None, matched_stop, has_tool_calls=tool_use_block_started)
        cache_creation = max(0, input_tokens - cached_tokens)
        delta_data = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": matched_stop},
            "usage": {
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": max(0, cached_tokens),
            },
        }
        yield f"event: message_delta\ndata: {json.dumps(delta_data)}\n\n"

    loaded_adapter = _apply_lora_adapter(engine, req.lora_adapter)
    error_occurred = False
    try:
        async for event in with_sse_keepalive(
            _token_source(),
            http_request=request,
            cancel_event=_anth_gen.cancel_event,
        ):
            yield event.encode("utf-8") if isinstance(event, str) else event

        # Only emit message_stop on normal completion, NOT after errors
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n".encode("utf-8")
    except MemoryError:
        error_occurred = True
        error_event = {"type": "error", "error": {"type": "overloaded_error", "message": "Out of GPU memory"}}
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode("utf-8")
    except Exception as e:
        error_occurred = True
        logger.error("Anthropic streaming error", exc_info=True)
        error_event = {"type": "error", "error": {"type": "api_error", "message": "Internal server error"}}
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode("utf-8")
    finally:
        _release_lora_adapter(engine, loaded_adapter)
        _anth_tracker.unregister(message_id)

    _record_metrics(input_tokens, output_tokens)


def _format_anthropic_logprobs(logprobs_list: list[dict] | None) -> list[dict] | None:
    """Format per-token logprobs from GenerationOutput into Anthropic Messages format.

    Anthropic returns logprobs as an array in each text content block:
    [{"token": str, "logprob": float, "top_logprobs": [{"token": str, "logprob": float}, ...]}]

    Returns None if no valid logprobs entries are found.
    """
    if not logprobs_list:
        return None
    entries = []
    for lp_entry in logprobs_list:
        if not isinstance(lp_entry, dict):
            continue
        top_lps = lp_entry.get("top_logprobs", [])
        decoded_top = [
            {"token": tlp.get("token", ""), "logprob": tlp.get("logprob", 0.0)}
            for tlp in top_lps
        ]
        entries.append({
            "token": lp_entry.get("token", ""),
            "logprob": lp_entry.get("logprob", 0.0),
            "top_logprobs": decoded_top,
        })
    return entries if entries else None


@router.post("/messages/count_tokens")
async def count_tokens(req: AnthropicMessagesRequest) -> dict:
    """Token counting endpoint (Anthropic compatible).

    Returns Anthropic-format errors on failure:
      {"type": "error", "error": {"type": "...", "message": "..."}}
    """
    try:
        engine, _ = await _resolve_engine(req.model)
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"type": "error", "error": {"type": "not_found_error" if e.status_code == 404 else "api_error", "message": e.detail}},
        )

    tokenizer = getattr(engine, '_tokenizer', None)
    if tokenizer is None:
        return JSONResponse(
            status_code=503,
            content={"type": "error", "error": {"type": "overloaded_error", "message": "No tokenizer available"}},
        )

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
