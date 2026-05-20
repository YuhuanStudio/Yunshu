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
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse

logger = logging.getLogger(__name__)

_MAX_STREAMING_TEXT_BUFFER = 1 * 1024 * 1024
_TRUNCATE_KEEP = 512 * 1024
from pydantic import BaseModel, Field, model_validator

from ..engine import get_engine
from .chat import _apply_lora_adapter, _release_lora_adapter
from .models import _check_permission
from ..streaming import (
    with_sse_keepalive,
)
from yunshu_engine.tool_call_streamer import ToolCallStreamer

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
    max_tokens: int = Field(default=1024, ge=1, le=131072)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    stream: bool = False
    stop_sequences: Optional[list[str]] = None
    system: Optional[str | list[dict]] = None
    thinking: Optional[dict] = None
    metadata: Optional[dict] = None
    tools: Optional[list[AnthropicTool]] = None
    tool_choice: Optional[dict | str] = None

    # ── Yunshu-extended fields (forwarded to engine) ──
    lora_adapter: Optional[str] = None
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=0.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[dict[str, float]] = None
    seed: Optional[int] = None
    reasoning_effort: Optional[str] = None
    stop_token_ids: Optional[list[int]] = None
    spec_decode: bool = False
    xtc_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    xtc_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    priority: int = Field(default=0, ge=0, le=100)
    json_schema: Optional[dict] = None
    logprobs: bool = False
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=20)
    logits_processors: Optional[list] = None
    # Client-forwarded field (not Anthropic spec, but commonly sent by SDKs)
    response_format: Optional[dict] = None
    timeout: Optional[float] = Field(default=None, ge=1.0, le=600.0)  # Request timeout in seconds
    grammar: Optional[dict] = None  # Grammar constraint (regex, choice, CFG)
    stream_options: Optional[dict] = None  # Anthropic stream_options (include_usage)

    @model_validator(mode="after")
    def validate_request(self):
        if not self.model or not self.model.strip():
            raise ValueError("model: field is required and cannot be empty")
        if not self.messages:
            raise ValueError("messages: field is required and cannot be empty")
        if self.stop_sequences and len(self.stop_sequences) > 16:
            raise ValueError("stop_sequences: maximum 16 stop sequences")
        if self.stop_sequences and any(not s for s in self.stop_sequences):
            raise ValueError("stop_sequences: individual sequences must be non-empty")
        if self.stop_token_ids and len(self.stop_token_ids) > 16:
            raise ValueError("stop_token_ids: maximum 16 stop token IDs")
        # Validate thinking configuration per Anthropic spec
        if self.thinking:
            thinking_type = self.thinking.get("type")
            if thinking_type == "enabled":
                budget = self.thinking.get("budget_tokens")
                if budget is not None:
                    if not isinstance(budget, int) or budget < 1:
                        raise ValueError("thinking: budget_tokens must be a positive integer")
            elif thinking_type == "disabled":
                pass  # Explicitly disabling thinking is valid
            elif thinking_type is not None:
                raise ValueError(f"thinking.type must be 'enabled' or 'disabled', got '{thinking_type}'")
        # Validate response_format type if provided
        if self.response_format is not None:
            rf_type = self.response_format.get("type") if isinstance(self.response_format, dict) else None
            if rf_type not in ("json_object", "json_schema", "text", None):
                raise ValueError(f"response_format.type: must be 'json_object', 'json_schema', or 'text', got '{rf_type}'")
        # Per Anthropic spec: top_logprobs requires logprobs=True
        if self.top_logprobs is not None and not self.logprobs:
            raise ValueError("top_logprobs requires logprobs to be true")
        return self


# ── Content block helpers ──


def _resolve_json_schema(req) -> dict | str | None:
    """Resolve json_schema from req.json_schema or req.response_format.

    The Anthropic API doesn't have a standard structured output mechanism,
    but clients may send response_format (OpenAI-style) or json_schema directly.
    """
    # Direct json_schema field takes priority
    js = getattr(req, 'json_schema', None)
    if js is not None:
        return js
    # Fall back to OpenAI-style response_format
    rf = getattr(req, 'response_format', None)
    if rf is not None:
        rf_type = rf.get("type")
        if rf_type == "json_object":
            return "json_object"
        if rf_type == "json_schema":
            js_obj = rf.get("json_schema", {})
            schema = js_obj.get("schema")
            if schema:
                return schema
            return "json_object"
    return None


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


def _convert_anthropic_messages(
    messages_input: list[AnthropicMessage],
    has_images: bool = False,
    temp_files: list[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """Convert Anthropic messages to OpenAI-compatible format.

    Properly handles:
    - tool_use blocks -> assistant messages with tool_calls
    - tool_result blocks -> tool role messages with tool_call_id
    - image blocks -> OpenAI image_url content parts (VLM path)
    - text blocks -> plain text content

    Returns (messages, temp_files) where temp_files tracks any image temp files
    created during conversion.
    """
    if temp_files is None:
        temp_files = []

    # Phase 1: Convert each Anthropic message to intermediate form
    # We need to split assistant messages with tool_use into separate messages:
    # the text part stays as assistant, and each tool_use gets its own entry.
    intermediate: list[dict] = []

    for m in messages_input:
        content = m.content
        role = m.role

        if not isinstance(content, list):
            # Simple string or None content — no block processing needed
            intermediate.append({"role": role, "content": content or ""})
            continue

        # Check for tool_use and tool_result blocks
        has_tool_use = any(
            isinstance(b, dict) and b.get("type") == "tool_use"
            for b in content
        )
        has_tool_result = any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        )

        if has_tool_use and role == "assistant":
            # Convert tool_use blocks to OpenAI tool_calls format
            text_parts = []
            tool_calls = []
            for block in content:
                if not isinstance(block, dict):
                    text_parts.append(str(block))
                    continue
                bt = block.get("type", "")
                if bt == "text":
                    text_parts.append(block.get("text", ""))
                elif bt == "tool_use":
                    tool_id = block.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                    tool_name = block.get("name", "unknown")
                    tool_input = block.get("input", {})
                    tool_calls.append({
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input),
                        },
                    })
                else:
                    text_parts.append(_extract_text_from_content([block]))

            msg: dict = {"role": "assistant", "content": "\n".join(text_parts).strip()}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            intermediate.append(msg)

        elif has_tool_result and role == "user":
            # Convert each tool_result block to a separate tool role message
            # This is how OpenAI format represents tool results
            for block in content:
                if not isinstance(block, dict):
                    continue
                bt = block.get("type", "")
                if bt == "tool_result":
                    tool_use_id = block.get("tool_use_id", f"toolu_unknown")
                    inner = block.get("content", "")
                    if isinstance(inner, list):
                        inner_text = _extract_text_from_content(inner)
                    else:
                        inner_text = str(inner) if inner is not None else ""
                    intermediate.append({
                        "role": "tool",
                        "content": inner_text,
                        "tool_call_id": tool_use_id,
                    })
                elif bt == "text":
                    intermediate.append({"role": "user", "content": block.get("text", "")})
                elif bt == "image" and has_images:
                    # Handle image blocks in VLM path
                    _convert_image_block(block, intermediate, temp_files)
                else:
                    text = _extract_text_from_content([block])
                    if text:
                        intermediate.append({"role": "user", "content": text})

        elif has_images:
            # VLM path: preserve image blocks as OpenAI-style content parts
            converted_parts = []
            for block in content:
                if not isinstance(block, dict):
                    converted_parts.append({"type": "text", "text": str(block)})
                    continue
                bt = block.get("type", "")
                if bt == "text":
                    converted_parts.append({"type": "text", "text": block.get("text", "")})
                elif bt == "image":
                    source = block.get("source", {})
                    media_type = source.get("media_type", "unknown")
                    source_type = source.get("type", "")
                    data = source.get("data")
                    if data and source_type == "base64":
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
                        temp_files.append(tmp.name)
                        converted_parts.append({"type": "image_url", "image_url": {"url": f"file://{tmp.name}"}})
                    elif source_type == "url" and source.get("url"):
                        # Anthropic url source: pass through the URL directly
                        converted_parts.append({"type": "image_url", "image_url": {"url": source["url"]}})
                    else:
                        converted_parts.append({"type": "text", "text": f"[Image: {media_type}]"})
                else:
                    text = _extract_text_from_content([block])
                    converted_parts.append({"type": "text", "text": text})
            intermediate.append({"role": role, "content": converted_parts})

        else:
            # Standard content: flatten to text
            content_text = _extract_text_from_content(content)
            intermediate.append({"role": role, "content": content_text})

    return intermediate, temp_files


def _convert_image_block(block: dict, intermediate: list[dict], temp_files: list[str]) -> None:
    """Convert an Anthropic image block and append to intermediate messages."""
    source = block.get("source", {})
    media_type = source.get("media_type", "unknown")
    source_type = source.get("type", "")
    data = source.get("data")
    if data and source_type == "base64":
        import base64 as _b64
        import tempfile as _tf
        try:
            raw = _b64.b64decode(data, validate=False)
        except Exception:
            raw = _b64.b64decode(data + "==", validate=False)
        ext_map = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp"}
        ext = ext_map.get(media_type, "png")
        tmp = _tf.NamedTemporaryFile(suffix=f".{ext}", delete=False)
        tmp.write(raw)
        tmp.close()
        temp_files.append(tmp.name)
        intermediate.append({
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": f"file://{tmp.name}"}}],
        })
    elif source_type == "url" and source.get("url"):
        # Anthropic url source: pass through the URL directly
        intermediate.append({
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": source["url"]}}],
        })
    else:
        intermediate.append({"role": "user", "content": f"[Image: {media_type}]"})


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
    # Fast rejection: skip regex entirely if no tool-call trigger chars present
    if '<' not in text and '{' not in text:
        return None
    # Try XML-wrapped tool calls first
    for m in _TOOL_CALL_XML_RE.finditer(text):
        inner = m.group(1).strip()
        try:
            data = json.loads(inner)
            if "name" in data:
                return [{"name": data["name"], "arguments": json.dumps(data.get("arguments", {}))}]
        except json.JSONDecodeError:
            pass

    # Try bare JSON — find the opening brace after name and parse from there.
    # Use json.JSONDecoder.raw_decode for robust parsing that correctly
    # handles nested braces inside string values (avoids false termination
    # on literal '}' characters within JSON string values).
    calls: list[dict] = []
    for m in _TOOL_CALL_JSON_RE.finditer(text):
        name = m.group(1)
        rest = text[m.end():]
        try:
            decoder = json.JSONDecoder()
            obj, end_idx = decoder.raw_decode(rest)
            args_str = rest[:end_idx]
            calls.append({"name": name, "arguments": args_str})
        except json.JSONDecodeError:
            pass
    return calls or None


# ── Endpoint ──


@router.post("/messages", response_model=None)
async def create_message(req: AnthropicMessagesRequest, request: Request):
    _check_permission(request, "can_infer")
    """Anthropic Messages API endpoint."""
    # Build messages list (prepend system if present)
    messages = []
    _temp_files: list[str] = []  # track temp files for cleanup
    if req.system:
        system_text = _extract_text_from_content(req.system) if isinstance(req.system, list) else req.system
        messages.append({"role": "system", "content": system_text})

    # Extract cache_control hints for future forwarding to the engine's
    # KV prefix cache (once the engine supports cache breakpoints).
    _cache_hints = _extract_cache_control_hints(req.system)
    if _cache_hints:
        logger.debug("Anthropic cache_control hints received: %s", _cache_hints)

    has_images = any(_has_image_blocks(m.content) for m in req.messages)

    # Convert Anthropic messages to OpenAI-compatible format
    # This properly handles tool_use/tool_result blocks instead of flattening them
    converted_msgs, _temp_files = _convert_anthropic_messages(
        req.messages, has_images=has_images, temp_files=_temp_files,
    )
    messages.extend(converted_msgs)

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

        if req.tool_choice:
            if isinstance(req.tool_choice, dict):
                tc_type = req.tool_choice.get("type", "")
                if tc_type == "none":
                    req._suppress_tools = True
                    tool_prompt = ""
                elif tc_type == "any":
                    tool_prompt += "\nYou MUST call at least one tool. Do NOT respond with only text.\n"
                elif tc_type == "tool":
                    forced = req.tool_choice.get("name")
                    if forced:
                        tool_prompt += f"\nYou MUST call the tool '{forced}'.\n"
                # {"type": "auto"} is the default — no additional prompt needed
            elif req.tool_choice == "any":
                tool_prompt += "\nYou MUST call at least one tool. Do NOT respond with only text.\n"
            elif req.tool_choice == "none":
                req._suppress_tools = True
                tool_prompt = ""

        if tool_prompt:
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
        return StreamingResponse(
            _stream_anthropic(engine, messages, req, stop, request, is_batched=is_batched, temp_files=_temp_files),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    # Non-streaming: register with request tracker for cancellation support
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    _ns_tracker = None
    _ns_gen = None
    _ns_cancel_event = None
    try:
        from yunshu_engine.request_tracker import get_request_tracker
        _ns_tracker = get_request_tracker()
        _ns_gen = _ns_tracker.register(message_id, req.model)
        _ns_cancel_event = _ns_gen.cancel_event
    except Exception:
        _ns_tracker = None

    loaded_adapter = _apply_lora_adapter(engine, req.lora_adapter)
    try:
        if is_batched:
            return await _non_stream_batched(engine, messages, req, stop, cancel_event=_ns_cancel_event)
        return await _non_stream_legacy(engine, messages, req, stop, cancel_event=_ns_cancel_event)
    finally:
        _release_lora_adapter(engine, loaded_adapter)
        if _ns_tracker is not None:
            try:
                _ns_tracker.unregister(message_id)
            except Exception:
                pass
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


def _convert_logit_bias(req):
    """Convert logit_bias keys from str to int for engine compatibility."""
    _lb = req.logit_bias
    if _lb:
        result = {}
        for k, v in _lb.items():
            try:
                result[int(k)] = v
            except (ValueError, TypeError):
                logger.warning(f"Skipping non-integer logit_bias key: {k!r}")
        return result if result else None
    return None


async def _non_stream_batched(engine, messages, req, stop, cancel_event=None):
    """Non-streaming response via BatchedEngine."""
    from fastapi.responses import JSONResponse
    enable_thinking = req.thinking and req.thinking.get("type") == "enabled"
    budget_tokens = req.thinking.get("budget_tokens") if req.thinking else None
    _logit_bias = _convert_logit_bias(req)

    try:
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
            logit_bias=_logit_bias,
            stop=stop,
            seed=req.seed,
            enable_thinking=enable_thinking,
            thinking_budget=budget_tokens,
            reasoning_effort=req.reasoning_effort,
            stop_token_ids=req.stop_token_ids,
            spec_decode=req.spec_decode,
            xtc_probability=req.xtc_probability,
            xtc_threshold=req.xtc_threshold,
            priority=req.priority,
            json_schema=_resolve_json_schema(req),
            logprobs=req.logprobs,
            top_logprobs=req.top_logprobs,
            logits_processors=req.logits_processors,
            cancel_event=cancel_event,
            timeout_seconds=req.timeout,
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
            content={"type": "error", "error": {"type": "api_error", "message": "Internal server error"}},
        )
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    _record_metrics(result.prompt_tokens, result.completion_tokens)

    content = []
    thinking_text = ""
    visible_text = result.text

    if enable_thinking:
        from ..streaming import extract_thinking
        thinking_text, visible_text = extract_thinking(result.text, req.model)
        if thinking_text:
            content.append({"type": "thinking", "thinking": thinking_text, "signature": ""})

    text_block: dict = {"type": "text", "text": visible_text}

    # Check for matched stop sequences — trim BEFORE tool call extraction
    matched_stop = None
    if stop and visible_text:
        for seq in stop:
            idx = visible_text.find(seq)
            if idx != -1:
                matched_stop = seq
                # Strip stop sequence text from visible_text per Anthropic spec
                visible_text = visible_text[:idx]
                text_block["text"] = visible_text
                break

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

    # Extract tool calls from model output if tools were provided
    # (but not when tool_choice is "none" — see create_message where tools are
    # suppressed from the prompt; skip extraction to avoid false stop_reason)
    _suppress_tool_extraction = getattr(req, '_suppress_tools', False)
    has_tool_calls = False
    if req.tools and not _suppress_tool_extraction:
        from ..streaming import extract_tool_calls_model_aware, clean_tool_call_markup
        tool_calls = extract_tool_calls_model_aware(visible_text, req.model)
        if tool_calls:
            has_tool_calls = True
            # Remove the text block and replace with cleaned version
            cleaned = clean_tool_call_markup(visible_text)
            text_block["text"] = cleaned
            # Per Anthropic spec: omit empty text blocks when tool_use is present
            if not cleaned.strip():
                content.remove(text_block)
            for tc in tool_calls:
                tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                try:
                    inp = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
                except (json.JSONDecodeError, TypeError):
                    inp = {}
                content.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tc["name"],
                    "input": inp,
                })

    stop_reason = _map_stop_reason(result.finish_reason, matched_stop, has_tool_calls=has_tool_calls)

    cache_creation = getattr(result, 'prompt_tokens', 0) - (getattr(result, 'cached_tokens', 0) or 0)
    cache_read = getattr(result, 'cached_tokens', 0) or 0
    reasoning_tok = getattr(result, 'reasoning_tokens', 0) or 0

    # Per Anthropic spec: output_tokens is the TOTAL (visible + reasoning).
    # When the engine reports reasoning_tokens separately, add them to the
    # completion_tokens count so the total is accurate.
    total_output_tokens = result.completion_tokens + reasoning_tok

    usage: dict[str, Any] = {
        "input_tokens": result.prompt_tokens,
        "output_tokens": total_output_tokens,
        "cache_creation_input_tokens": max(0, cache_creation),
        "cache_read_input_tokens": max(0, cache_read),
    }
    if reasoning_tok > 0:
        usage["output_tokens_details"] = {"reasoning_tokens": reasoning_tok}

    resp = {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": req.model,
        "stop_reason": stop_reason,
        "stop_sequence": matched_stop,
        "created_at": int(time.time()),
        "usage": usage,
        **({"metadata": req.metadata} if req.metadata else {}),
    }
    return JSONResponse(resp)


async def _non_stream_legacy(engine, messages, req, stop, cancel_event=None):
    """Non-streaming response via Engine or BatchedEngine."""
    from fastapi.responses import JSONResponse
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    enable_thinking = req.thinking and req.thinking.get("type") == "enabled"
    budget_tokens = req.thinking.get("budget_tokens") if req.thinking else None
    _logit_bias = _convert_logit_bias(req)
    try:
        result = await engine.generate(
            prompt=messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            min_p=req.min_p,
            repetition_penalty=req.repetition_penalty,
            frequency_penalty=req.frequency_penalty,
            presence_penalty=req.presence_penalty,
            logit_bias=_logit_bias,
            stop=stop,
            seed=req.seed,
            enable_thinking=enable_thinking,
            thinking_budget=budget_tokens,
            reasoning_effort=req.reasoning_effort,
            stop_token_ids=req.stop_token_ids,
            spec_decode=req.spec_decode,
            xtc_probability=req.xtc_probability,
            xtc_threshold=req.xtc_threshold,
            priority=req.priority,
            json_schema=_resolve_json_schema(req),
            logprobs=req.logprobs,
            top_logprobs=req.top_logprobs,
            logits_processors=req.logits_processors,
            cancel_event=cancel_event,
            timeout_seconds=req.timeout,
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
            content={"type": "error", "error": {"type": "api_error", "message": "Internal server error"}},
        )
    # Handle both Engine (prompt_token_count) and BatchedEngine (prompt_tokens)
    _pt = getattr(result, 'prompt_tokens', None)
    prompt_toks = _pt if _pt is not None else getattr(result, 'prompt_token_count', 0)
    _ct = getattr(result, 'completion_tokens', None)
    completion_toks = _ct if _ct is not None else getattr(result, 'completion_token_count', 0)
    _txt = getattr(result, 'text', None)
    text = _txt if _txt is not None else getattr(result, 'generated_text', '')
    _fr = getattr(result, 'finish_reason', None)
    finish_reason = _fr if _fr is not None else getattr(result, 'finish_state', None)
    cached_toks = getattr(result, 'cached_tokens', 0) or 0
    _record_metrics(prompt_toks, completion_toks)

    # Extract thinking tokens if thinking mode enabled
    content = []
    visible_text = text
    if enable_thinking:
        from ..streaming import extract_thinking
        thinking_text, visible_text = extract_thinking(text, req.model)
        if thinking_text:
            content.append({"type": "thinking", "thinking": thinking_text, "signature": ""})
    text_block: dict = {"type": "text", "text": visible_text}

    # Check for matched stop sequences — trim BEFORE tool call extraction
    matched_stop = None
    if stop and visible_text:
        for seq in stop:
            idx = visible_text.find(seq)
            if idx != -1:
                matched_stop = seq
                # Strip stop sequence text from visible_text per Anthropic spec
                visible_text = visible_text[:idx]
                text_block["text"] = visible_text
                break

    # Include logprobs in the text content block if requested
    if req.logprobs:
        _result_lp = getattr(result, 'logprobs', None)
        if _result_lp:
            formatted_lp = _format_anthropic_logprobs(_result_lp)
            if formatted_lp:
                text_block["logprobs"] = formatted_lp

    content.append(text_block)

    # Extract tool calls from model output if tools were provided
    _suppress_tool_extraction = getattr(req, '_suppress_tools', False)
    has_tool_calls = False
    if req.tools and not _suppress_tool_extraction:
        from ..streaming import extract_tool_calls_model_aware, clean_tool_call_markup
        tool_calls = extract_tool_calls_model_aware(visible_text, req.model)
        if tool_calls:
            has_tool_calls = True
            # Remove the text block and replace with cleaned version
            cleaned = clean_tool_call_markup(visible_text)
            text_block["text"] = cleaned
            # Per Anthropic spec: omit empty text blocks when tool_use is present
            if not cleaned.strip():
                content.remove(text_block)
            for tc in tool_calls:
                tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                try:
                    inp = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
                except (json.JSONDecodeError, TypeError):
                    inp = {}
                content.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tc["name"],
                    "input": inp,
                })

    stop_reason = _map_stop_reason(finish_reason, matched_stop, has_tool_calls=has_tool_calls)

    _reasoning_tok = getattr(result, 'reasoning_tokens', 0) or 0
    _legacy_total_output = completion_toks + _reasoning_tok
    _legacy_usage: dict[str, Any] = {
        "input_tokens": prompt_toks,
        "output_tokens": _legacy_total_output,
        "cache_creation_input_tokens": max(0, prompt_toks - cached_toks),
        "cache_read_input_tokens": max(0, cached_toks),
    }
    if _reasoning_tok > 0:
        _legacy_usage["output_tokens_details"] = {"reasoning_tokens": _reasoning_tok}

    return JSONResponse({
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": req.model,
        "stop_reason": stop_reason,
        "stop_sequence": matched_stop,
        "created_at": int(time.time()),
        "usage": _legacy_usage,
        **({"metadata": req.metadata} if req.metadata else {}),
    })


async def _stream_anthropic(
    engine, messages, req, stop, request, is_batched=False, temp_files=None
) -> AsyncIterator[bytes]:
    """Anthropic SSE streaming with keepalive, disconnect detection, and tool-use deltas."""
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    enable_thinking = req.thinking and req.thinking.get("type") == "enabled"
    budget_tokens = req.thinking.get("budget_tokens") if req.thinking else None
    _logit_bias = _convert_logit_bias(req)
    has_tools = req.tools is not None and len(req.tools) > 0
    _tool_streamer = ToolCallStreamer() if has_tools else None
    block_index = 0
    thinking_block_started = False
    text_block_started = False
    tool_use_block_started = False
    accumulated_text = ""  # for tool-call detection
    matched_stop: str | None = None
    _streaming_finish_reason: str | None = None
    reasoning_tok: int = 0  # reasoning tokens emitted as thinking_delta
    _token_boundaries: list[int] = []  # cumulative text length after each output token

    # Register with request tracker for cancellation support
    _anth_tracker = None
    try:
        from yunshu_engine.request_tracker import get_request_tracker
        _anth_tracker = get_request_tracker()
        _anth_gen = _anth_tracker.register(message_id, req.model)
    except Exception:
        _anth_gen = None

    # Track whether message_start has been emitted (deferred until first
    # engine output so we can report accurate cache token counts).
    _message_start_emitted = False

    def _emit_message_start(inp_tokens: int, cached_toks: int) -> bytes:
        """Build and return the message_start event bytes.

        Called once, when the first engine output arrives with prompt_tokens.
        cache_creation_input_tokens = prompt_tokens - cached_tokens
        cache_read_input_tokens = cached_tokens
        """
        cache_creation = max(0, inp_tokens - cached_toks)
        cache_read = max(0, cached_toks)
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
                "created_at": _start_ts,
                "usage": {
                    "input_tokens": inp_tokens,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": cache_creation,
                    "cache_read_input_tokens": cache_read,
                },
            },
        }
        return f"event: message_start\ndata: {json.dumps(msg_start)}\n\n".encode("utf-8")

    _start_ts = int(time.time())

    # content_block_start for thinking is NOT eagerly emitted — it opens
    # only when the first reasoning token arrives, avoiding empty thinking
    # blocks when the model decides not to think.

    async def _token_source():
        nonlocal input_tokens, output_tokens, block_index, cached_tokens
        nonlocal thinking_block_started, text_block_started, tool_use_block_started
        nonlocal accumulated_text, matched_stop, _message_start_emitted, _streaming_finish_reason, reasoning_tok

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
                logit_bias=_logit_bias,
                stop=stop,
                seed=req.seed,
                enable_thinking=enable_thinking,
                thinking_budget=budget_tokens,
                reasoning_effort=req.reasoning_effort,
                stop_token_ids=req.stop_token_ids,
                spec_decode=req.spec_decode,
                xtc_probability=req.xtc_probability,
                xtc_threshold=req.xtc_threshold,
                priority=req.priority,
                json_schema=_resolve_json_schema(req),
                logprobs=req.logprobs,
                top_logprobs=req.top_logprobs,
                logits_processors=req.logits_processors,
                cancel_event=_anth_gen.cancel_event if _anth_gen else None,
                timeout_seconds=req.timeout,
            ):
                # Use engine's current_state (token-level tracking) for
                # thinking routing — more accurate than text-level ThinkingParser
                # which may miss model-specific tags like Qwen3.5's special tokens.
                _is_reasoning = getattr(output, 'current_state', None) == "reasoning"
                _token_text = output.new_text

                if output.prompt_tokens and not input_tokens:
                    input_tokens = output.prompt_tokens
                if hasattr(output, 'cached_tokens') and output.cached_tokens:
                    cached_tokens = max(cached_tokens, output.cached_tokens)

                # Capture finish_reason from the last streaming output.
                # Use finish_reason whenever it's set (not just when finished=True)
                # because some engines set finish_reason without the finished flag.
                if output.finish_reason is not None:
                    _streaming_finish_reason = output.finish_reason

                # Emit message_start on first output with prompt_tokens.
                # Deferred from the initial yield so that cache token counts
                # are populated from the engine rather than reporting 0.
                # NOTE: Do NOT eagerly open the thinking block here — only
                # open it when the first reasoning token actually arrives.
                # Opening it eagerly causes an empty thinking block when the
                # model decides not to think (Anthropic spec: thinking blocks
                # appear only when the model produces reasoning output).
                if not _message_start_emitted:
                    _message_start_emitted = True
                    yield _emit_message_start(input_tokens, cached_tokens)

                # Thinking content
                if enable_thinking and _is_reasoning and _token_text:
                    if not thinking_block_started:
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'thinking', 'thinking': '', 'signature': ''}})}\n\n"
                        thinking_block_started = True
                    output_tokens += 1
                    reasoning_tok += 1
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'thinking_delta', 'thinking': _token_text}})}\n\n"
                elif _token_text:
                    # Visible text content
                    if thinking_block_started and not text_block_started:
                        # Close thinking block, but defer opening text block
                        # until we confirm there's actual text to emit (not just
                        # tool call markup that will be handled separately).
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                        block_index += 1
                        thinking_block_started = False

                    # Once a stop sequence was already matched, suppress all further text
                    if matched_stop:
                        continue
                    output_tokens += 1
                    _prev_len = len(accumulated_text)
                    if not (has_tools and _tool_streamer):
                        accumulated_text += _token_text
                        if len(accumulated_text) > _MAX_STREAMING_TEXT_BUFFER:
                            logger.error("Anthropic streaming text exceeded 1MB — truncating")
                            accumulated_text = accumulated_text[-_TRUNCATE_KEEP:]
                        _token_boundaries.append(len(accumulated_text))

                    # Check for stop sequences in the newly accumulated text
                    _stop_matched_this_token = False
                    if stop:
                        for seq in stop:
                            if seq in accumulated_text:
                                idx = accumulated_text.find(seq)
                                accumulated_text = accumulated_text[:idx]
                                matched_stop = seq
                                _stop_matched_this_token = True
                                break

                    if _stop_matched_this_token:
                        # Count how many tokens are entirely within the trimmed suffix.
                        # A token's text is entirely trimmed if its cumulative boundary
                        # exceeds the safe end (len of accumulated_text after trim).
                        _safe_end = len(accumulated_text)
                        _tokens_to_trim = 0
                        for _bi in range(len(_token_boundaries) - 1, -1, -1):
                            if _token_boundaries[_bi] > _safe_end:
                                _tokens_to_trim += 1
                            else:
                                break
                        # Emit only the safe portion of the current token
                        _safe_len = len(accumulated_text) - _prev_len
                        if _tokens_to_trim == 0 and _safe_len == 0:
                            _tokens_to_trim = 1
                        output_tokens = max(0, output_tokens - _tokens_to_trim)
                        if _safe_len > 0:
                            _safe_delta = _token_text[:_safe_len]
                            if not text_block_started:
                                text_block_started = True
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': _safe_delta}})}\n\n"
                    else:
                        # Route through ToolCallStreamer when tools are defined.
                        # The streamer buffers tokens and only emits confirmed
                        # text or complete tool calls, preventing partial tool
                        # call markup from being sent as visible text.
                        # NOTE: output_tokens was already incremented above for
                        # this token — do NOT increment again here.
                        if has_tools and _tool_streamer:
                            for _tc_out in _tool_streamer.process_token(_token_text):
                                if _tc_out.text:
                                    _prev_len = len(accumulated_text)
                                    accumulated_text += _tc_out.text
                                    if len(accumulated_text) > _MAX_STREAMING_TEXT_BUFFER:
                                        logger.error("Anthropic streaming text exceeded 1MB — truncating")
                                        accumulated_text = accumulated_text[-_TRUNCATE_KEEP:]
                                    _token_boundaries.append(len(accumulated_text))
                                    # Check for stop sequences after streamer text
                                    # is accumulated (the pre-streamer check at
                                    # line ~1139 operates on empty text when tools
                                    # are active, so we must check here instead).
                                    _stop_hit = False
                                    if stop:
                                        for seq in stop:
                                            if seq in accumulated_text:
                                                accumulated_text = accumulated_text[:accumulated_text.find(seq)]
                                                matched_stop = seq
                                                _stop_hit = True
                                                break
                                    if _stop_hit:
                                        output_tokens = max(1, output_tokens - 1)
                                        _safe_len = len(accumulated_text) - _prev_len
                                        if _safe_len > 0:
                                            if not text_block_started:
                                                text_block_started = True
                                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': _tc_out.text[:_safe_len]}})}\n\n"
                                    else:
                                        if not text_block_started:
                                            text_block_started = True
                                            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': _tc_out.text}})}\n\n"
                                elif _tc_out.tool_call:
                                    if not tool_use_block_started:
                                        if text_block_started:
                                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                                            block_index += 1
                                            text_block_started = False
                                        tool_use_block_started = True
                                    _tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'tool_use', 'id': _tool_id, 'name': _tc_out.tool_call.name, 'input': {}}})}\n\n"
                                    _args_str = _tc_out.tool_call.arguments or '{}'
                                    for _ci in range(0, len(_args_str), 8):
                                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'input_json_delta', 'partial_json': _args_str[_ci:_ci + 8]}})}\n\n"
                                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                                    block_index += 1
                                    break
                        else:
                            # No tools — emit text directly
                            if not text_block_started:
                                text_block_started = True
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': _token_text}})}\n\n"
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
                logit_bias=_logit_bias,
                stop=stop,
                seed=req.seed,
                enable_thinking=enable_thinking,
                thinking_budget=budget_tokens,
                reasoning_effort=req.reasoning_effort,
                stop_token_ids=req.stop_token_ids,
                spec_decode=req.spec_decode,
                xtc_probability=req.xtc_probability,
                xtc_threshold=req.xtc_threshold,
                priority=req.priority,
                json_schema=_resolve_json_schema(req),
                logprobs=req.logprobs,
                top_logprobs=req.top_logprobs,
                logits_processors=req.logits_processors,
                cancel_event=_anth_gen.cancel_event if _anth_gen else None,
                timeout_seconds=req.timeout,
            ):
                if hasattr(output, 'prompt_tokens') and output.prompt_tokens and not input_tokens:
                    input_tokens = output.prompt_tokens
                if hasattr(output, 'cached_tokens') and output.cached_tokens:
                    cached_tokens = max(cached_tokens, output.cached_tokens)

                # Capture finish_reason from the last streaming output.
                # Use finish_reason whenever it's set (not just when finished=True)
                # because some engines set finish_reason without the finished flag.
                if hasattr(output, 'finish_reason') and output.finish_reason is not None:
                    _streaming_finish_reason = output.finish_reason

                # Emit message_start on first output with prompt_tokens
                # (deferred from the initial yield for accurate cache tokens).
                # NOTE: Do NOT eagerly open the thinking block here — only
                # open it when the first reasoning token actually arrives.
                if not _message_start_emitted:
                    _message_start_emitted = True
                    yield _emit_message_start(input_tokens, cached_tokens)

                # Use engine's current_state (token-level tracking) when available,
                # fall back to ThinkingParser for engines that don't set current_state
                _is_reasoning = getattr(output, 'current_state', None) == "reasoning"
                _token_text = output.token_text

                if enable_thinking and _is_reasoning and _token_text:
                    # Thinking content via token-level state
                    if not thinking_block_started:
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'thinking', 'thinking': '', 'signature': ''}})}\n\n"
                        thinking_block_started = True
                    output_tokens += 1
                    reasoning_tok += 1
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'thinking_delta', 'thinking': _token_text}})}\n\n"
                elif _token_text:
                    # Visible text content
                    if thinking_block_started and not text_block_started:
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                        block_index += 1
                        thinking_block_started = False
                    # Once a stop sequence was already matched, suppress all further text
                    if matched_stop:
                        continue
                    if has_tools and _tool_streamer:
                        # Route through ToolCallStreamer for incremental detection.
                        # Only confirmed text is emitted; buffered tokens are held
                        # until the streamer can determine if they form a tool call tag.
                        output_tokens += 1
                        for _tc_out in _tool_streamer.process_token(_token_text):
                            if _tc_out.text:
                                if not text_block_started:
                                    text_block_started = True
                                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                                _prev_len = len(accumulated_text)
                                accumulated_text += _tc_out.text
                                if len(accumulated_text) > _MAX_STREAMING_TEXT_BUFFER:
                                    logger.error("Anthropic streaming text exceeded 1MB — truncating")
                                    accumulated_text = accumulated_text[-_TRUNCATE_KEEP:]
                                _stop_hit = False
                                if stop:
                                    for seq in stop:
                                        if seq in accumulated_text:
                                            accumulated_text = accumulated_text[:accumulated_text.find(seq)]
                                            matched_stop = seq
                                            _stop_hit = True
                                            break
                                if _stop_hit:
                                    output_tokens = max(1, output_tokens - 1)
                                    _safe_len = len(accumulated_text) - _prev_len
                                    if _safe_len > 0:
                                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': _tc_out.text[:_safe_len]}})}\n\n"
                                else:
                                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': _tc_out.text}})}\n\n"
                            elif _tc_out.tool_call:
                                if text_block_started:
                                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                                    block_index += 1
                                    text_block_started = False
                                tool_use_block_started = True
                                _tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'tool_use', 'id': _tool_id, 'name': _tc_out.tool_call.name, 'input': {}}})}\n\n"
                                _args_str = _tc_out.tool_call.arguments or '{}'
                                for _ci in range(0, len(_args_str), 8):
                                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'input_json_delta', 'partial_json': _args_str[_ci:_ci + 8]}})}\n\n"
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                                block_index += 1
                                break
                    else:
                        # No tool streamer — emit text directly
                        if not text_block_started:
                            text_block_started = True
                            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                        output_tokens += 1
                        _prev_len = len(accumulated_text)
                        accumulated_text += _token_text
                        if len(accumulated_text) > _MAX_STREAMING_TEXT_BUFFER:
                            logger.error("Anthropic streaming text exceeded 1MB — truncating")
                            accumulated_text = accumulated_text[-_TRUNCATE_KEEP:]
                        _token_boundaries.append(len(accumulated_text))

                        _stop_matched_this_token = False
                        if stop:
                            for seq in stop:
                                if seq in accumulated_text:
                                    accumulated_text = accumulated_text[:accumulated_text.find(seq)]
                                    matched_stop = seq
                                    _stop_matched_this_token = True
                                    break

                        if _stop_matched_this_token:
                            _safe_end = len(accumulated_text)
                            _tokens_to_trim = 0
                            for _bi in range(len(_token_boundaries) - 1, -1, -1):
                                if _token_boundaries[_bi] > _safe_end:
                                    _tokens_to_trim += 1
                                else:
                                    break
                            _safe_len = len(accumulated_text) - _prev_len
                            if _tokens_to_trim == 0 and _safe_len == 0:
                                _tokens_to_trim = 1
                            output_tokens = max(0, output_tokens - _tokens_to_trim)
                            if _safe_len > 0:
                                _safe_delta = _token_text[:_safe_len]
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': _safe_delta}})}\n\n"
                        else:
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': _token_text}})}\n\n"

            # Flush any remaining buffered content from the ToolCallStreamer.
            if has_tools and _tool_streamer:
                for _tc_out in _tool_streamer.flush():
                    if _tc_out.text:
                        if not text_block_started:
                            text_block_started = True
                            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': _tc_out.text}})}\n\n"
                    elif _tc_out.tool_call:
                        if text_block_started:
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                            block_index += 1
                            text_block_started = False
                        tool_use_block_started = True
                        _tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'tool_use', 'id': _tool_id, 'name': _tc_out.tool_call.name, 'input': {}}})}\n\n"
                        _args_str = _tc_out.tool_call.arguments or '{}'
                        for _ci in range(0, len(_args_str), 8):
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'input_json_delta', 'partial_json': _args_str[_ci:_ci + 8]}})}\n\n"
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                        block_index += 1

        # If message_start was never emitted (engine produced zero outputs or
        # only outputs without prompt_tokens), emit it now with whatever we have.
        if not _message_start_emitted:
            _message_start_emitted = True
            yield _emit_message_start(input_tokens, cached_tokens)

        # If no content blocks were started at all (zero tokens), emit an empty text block
        # so that the response always has at least one content block (Anthropic protocol requirement).
        # Always use text block, NOT thinking — thinking blocks should only appear when
        # the model actually produces reasoning output.
        if not text_block_started and not thinking_block_started and not tool_use_block_started:
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            text_block_started = True

        # Close last content block (only if one was actually started and not
        # already closed — tool_use blocks are closed inside the loop above).
        if text_block_started or thinking_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"

        # message_delta (stop + usage)
        # Per Anthropic streaming spec, message_delta usage contains output_tokens
        # and optionally output_tokens_details (reasoning_tokens).
        # cache_creation_input_tokens / cache_read_input_tokens are in message_start
        # (emitted deferred above when the first engine output arrives).
        stop_reason = _map_stop_reason(
            _streaming_finish_reason, matched_stop, has_tool_calls=tool_use_block_started
        )
        _delta_usage: dict = {"output_tokens": output_tokens}
        if reasoning_tok > 0:
            _delta_usage["output_tokens_details"] = {"reasoning_tokens": reasoning_tok}
        delta_data = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": matched_stop},
            "usage": _delta_usage,
        }
        yield f"event: message_delta\ndata: {json.dumps(delta_data)}\n\n"

    loaded_adapter = _apply_lora_adapter(engine, req.lora_adapter)
    try:
        async for event in with_sse_keepalive(
            _token_source(),
            http_request=request,
            cancel_event=_anth_gen.cancel_event if _anth_gen else None,
        ):
            yield event.encode("utf-8") if isinstance(event, str) else event

        # Only emit message_stop on normal completion, NOT after errors
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n".encode("utf-8")
    except MemoryError:
        # Emit message_start if it was never sent (error before first engine output)
        if not _message_start_emitted:
            _message_start_emitted = True
            yield _emit_message_start(input_tokens, cached_tokens)
        if text_block_started or thinking_block_started or tool_use_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n".encode("utf-8")
        error_event = {"type": "error", "error": {"type": "overloaded_error", "message": "Out of GPU memory"}}
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode("utf-8")
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n".encode("utf-8")
    except Exception as e:
        logger.error("Anthropic streaming error", exc_info=True)
        # Emit message_start if it was never sent (error before first engine output)
        if not _message_start_emitted:
            _message_start_emitted = True
            yield _emit_message_start(input_tokens, cached_tokens)
        if text_block_started or thinking_block_started or tool_use_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n".encode("utf-8")
        error_event = {"type": "error", "error": {"type": "api_error", "message": "Internal server error"}}
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode("utf-8")
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n".encode("utf-8")
    finally:
        _release_lora_adapter(engine, loaded_adapter)
        if _anth_tracker is not None:
            try:
                _anth_tracker.unregister(message_id)
            except Exception:
                logger.debug("tracker unregister failed", exc_info=True)
        # Clean up temp files created for image blocks during streaming
        if temp_files:
            import os as _os
            for _tf_path in temp_files:
                try:
                    _os.unlink(_tf_path)
                except OSError:
                    pass
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
async def count_tokens(req: AnthropicMessagesRequest, request: Request) -> dict:
    """Token counting endpoint (Anthropic compatible).

    Returns Anthropic-format errors on failure:
      {"type": "error", "error": {"type": "...", "message": "..."}}
    """
    _check_permission(request, "can_infer")
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

    # Apply chat template for accurate token counting (plain-text join undercounts
    # by missing role markers, special tokens, and generation prompt).
    messages = []
    if req.system:
        system_text = (
            _extract_text_from_content(req.system) if isinstance(req.system, list) else req.system
        )
        messages.append({"role": "system", "content": system_text})

    # Include tool definitions in the system prompt so that the token count
    # matches what the actual /messages endpoint would send.  Mirrors the
    # tool-prompt injection in create_message (lines 532-552).
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
        if req.tool_choice:
            if isinstance(req.tool_choice, dict):
                tc_type = req.tool_choice.get("type", "")
                if tc_type == "none":
                    tool_prompt = ""
                elif tc_type == "any":
                    tool_prompt += "\nYou MUST call at least one tool. Do NOT respond with only text.\n"
                elif tc_type == "tool":
                    forced = req.tool_choice.get("name")
                    if forced:
                        tool_prompt += f"\nYou MUST call the tool '{forced}'.\n"
            elif req.tool_choice == "any":
                tool_prompt += "\nYou MUST call at least one tool. Do NOT respond with only text.\n"
            elif req.tool_choice == "none":
                tool_prompt = ""
        if tool_prompt:
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] += tool_prompt
            else:
                messages.insert(0, {"role": "system", "content": tool_prompt.strip()})

    # Use the same conversion as /messages for accurate token counting.
    # _extract_text_from_content flattens tool_use/tool_result to text which
    # undercounts tokens compared to the actual OpenAI-format messages that
    # the /messages endpoint sends to the engine.
    has_images = any(_has_image_blocks(m.content) for m in req.messages)
    converted_msgs, _ = _convert_anthropic_messages(
        req.messages, has_images=has_images,
    )
    messages.extend(converted_msgs)
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        # Fallback: if chat template fails (e.g. missing template), use plain join
        text_parts = [m["content"] for m in messages]
        prompt = "\n".join(text_parts)
    tokens = tokenizer.encode(prompt)

    return {"type": "token_count", "input_tokens": len(tokens)}
