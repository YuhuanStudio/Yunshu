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
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)

_MAX_STREAMING_TEXT_BUFFER = 1 * 1024 * 1024
_TRUNCATE_KEEP = 512 * 1024
import contextlib

from pydantic import BaseModel, Field, model_validator

from yunshu_engine.tool_call_streamer import ToolCallStreamer

from ..engine import get_engine
from ..streaming import (
    run_with_disconnect_guard,
    with_sse_keepalive,
)
from .chat import _apply_lora_adapter, _release_lora_adapter
from .models import _check_permission

router = APIRouter(tags=["anthropic"])


# ── Anthropic stop_reason mapping ──
# Internal: "stop", "length", "tool_calls"
# Anthropic: "end_turn", "max_tokens", "stop_sequence", "tool_use"

_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


def _detect_matched_stop(
    text: str,
    stop_sequences: list[str] | None,
    finish_reason: str | None,
    stopped_by_stop_sequence: bool | None = None,
) -> str | None:
    """Identify which stop_sequence (if any) terminated the generation.

    Anthropic's protocol requires `stop_reason: "stop_sequence"` and a
    populated `stop_sequence` field whenever a user-supplied stop string
    fires. Cases:

    1) The engine returned text that still CONTAINS the stop sequence
       (no internal trimming) → we find and report it precisely.
    2) The engine trimmed the stop sequence and returned `finish_reason=="stop"`.
       ``stopped_by_stop_sequence`` (when the engine surfaces it) disambiguates a
       real user-stop hit from a natural EOS — BOTH return "stop", and fabricating
       a match on EOS wrongly reports stop_reason="stop_sequence" instead of
       "end_turn". So: if the flag is explicitly False, do NOT fabricate. Only when
       it's True (or unknown — legacy paths that don't set it) fall back to the lone
       provided stop_sequence (always correct), else a placeholder.
    """
    if not stop_sequences:
        return None
    if text:
        for seq in stop_sequences:
            if seq and seq in text:
                return seq
    if stopped_by_stop_sequence is False:
        return None  # natural EOS — not a stop_sequence hit
    if finish_reason == "stop":
        return stop_sequences[0] if len(stop_sequences) == 1 else _UNKNOWN_STOP_SENTINEL
    return None


# Internal marker: a stop sequence fired but we can't identify which (engine
# trimmed it and >1 sequence was supplied). Drives stop_reason="stop_sequence"
# while the public `stop_sequence` field is emitted as null (per Anthropic
# spec, which never fabricates a sequence the caller didn't submit).
_UNKNOWN_STOP_SENTINEL = "<stop_sequence>"


def _public_stop_sequence(matched_stop: str | None) -> str | None:
    """The value to emit in the response `stop_sequence` field."""
    return None if matched_stop == _UNKNOWN_STOP_SENTINEL else matched_stop


def _enforce_anthropic_tool_choice(tool_calls, tool_choice):
    """Filter parsed tool calls to honor Anthropic tool_choice post-generation.

    The OpenAI chat router runs _enforce_tool_choice on every path, but Anthropic only added
    an advisory system-prompt line and then emitted EVERY parsed tool_use as-is. So a forced
    {"type":"tool","name":X} that the model ignored (calling Y, or emitting multiple calls)
    surfaced the wrong/extra calls and a tool_use stop_reason. Mirror the chat enforcement:
    for a forced tool, drop calls whose name != the forced name; honor
    disable_parallel_tool_use by capping to one call."""
    if not tool_calls or not isinstance(tool_choice, dict):
        return tool_calls

    def _name(t):
        return getattr(t, "name", None) if not isinstance(t, dict) else t.get("name")

    out = tool_calls
    if tool_choice.get("type") == "tool":
        _forced = tool_choice.get("name")
        if _forced:
            out = [t for t in out if _name(t) == _forced]
    if tool_choice.get("disable_parallel_tool_use") and len(out) > 1:
        out = out[:1]
    return out


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
    # feed the per-request TPM box (see usage_context).
    try:
        from ..usage_context import record_billed_tokens

        record_billed_tokens((prompt_tokens or 0) + (completion_tokens or 0))
    except Exception:
        logger.debug("billed-token accounting failed", exc_info=True)


# ── Request / Response schemas ──


class AnthropicMessage(BaseModel):
    role: str
    content: str | list[dict] | None = None


class AnthropicTool(BaseModel):
    """Anthropic tool definition.

    Per the Anthropic Messages API spec, tools have: name, description,
    input_schema. The ``type`` field is NOT part of the Anthropic spec —
    Anthropic server-side tools (web_search, computer, etc.) carry versioned
    types like ``web_search_20250305`` but user-defined tools have no type.
    """

    name: str
    description: str | None = None
    input_schema: dict | None = None
    type: str | None = None  # Server-side tools set this; user tools omit it


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
    # Anthropic Messages API documents max_tokens as REQUIRED, but our
    # gateway is lenient and falls back to 1024 (the Claude default). All
    # official SDK clients send max_tokens explicitly, so the relaxed
    # behavior is harmless in practice. Keep an explicit default rather
    # than `...` (required) to preserve compatibility with internal tests
    # and lazy clients.
    max_tokens: int = Field(default=1024, ge=1, le=131072)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    stream: bool = False
    stop_sequences: list[str] | None = None
    system: str | list[dict] | None = None
    thinking: dict | None = None
    metadata: dict | None = None
    tools: list[AnthropicTool] | None = None
    tool_choice: dict | str | None = None

    # ── Yunshu-extended fields (forwarded to engine) ──
    lora_adapter: str | None = None
    cached_content: str | None = (
        None  # Gemini-style explicit context-cache handle to prepend
    )
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=0.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: dict[str, float] | None = None
    seed: int | None = None
    reasoning_effort: str | None = None
    stop_token_ids: list[int] | None = None
    spec_decode: bool = False
    xtc_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    xtc_threshold: float = Field(
        default=0.0, ge=0.0, le=0.5
    )  # engine requires [0,0.5]; le=1.0 made out-of-range 500 not 422
    # sampling controls, missing on /v1/messages —
    # silently ignored (see responses.py / chat.py:352-354).
    min_tokens: int = Field(default=0, ge=0)
    ignore_eos: bool = False
    suppress_tokens: list[int] | None = None
    priority: int = Field(default=0, ge=0, le=100)
    json_schema: dict | None = None
    logprobs: bool = False
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    logits_processors: list | None = None
    # Client-forwarded field (not Anthropic spec, but commonly sent by SDKs)
    response_format: dict | None = None
    timeout: float | None = Field(
        default=None, ge=1.0, le=600.0
    )  # Request timeout in seconds
    grammar: dict | None = None  # Grammar constraint (regex, choice, CFG)
    stream_options: dict | None = None  # Anthropic stream_options (include_usage)

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
                if budget is None:
                    raise ValueError(
                        "thinking: budget_tokens is required when thinking is enabled"
                    )
                if not isinstance(budget, int) or budget < 1:
                    raise ValueError(
                        "thinking: budget_tokens must be a positive integer"
                    )
                # Anthropic mandates max_tokens > thinking.budget_tokens (the budget
                # is the reasoning allowance, which must leave room for the visible answer);
                # the API returns 400 otherwise. Was accepted leniently before.
                if isinstance(self.max_tokens, int) and budget >= self.max_tokens:
                    raise ValueError(
                        f"thinking.budget_tokens ({budget}) must be less than max_tokens "
                        f"({self.max_tokens})"
                    )
            elif thinking_type == "disabled":
                pass  # Explicitly disabling thinking is valid
            elif thinking_type is not None:
                raise ValueError(
                    f"thinking.type must be 'enabled' or 'disabled', got '{thinking_type}'"
                )
        # Validate response_format type if provided
        if self.response_format is not None:
            rf_type = (
                self.response_format.get("type")
                if isinstance(self.response_format, dict)
                else None
            )
            if rf_type not in ("json_object", "json_schema", "text", None):
                raise ValueError(
                    f"response_format.type: must be 'json_object', 'json_schema', or 'text', got '{rf_type}'"
                )
        # Per Anthropic spec: top_logprobs requires logprobs=True
        if self.top_logprobs is not None and not self.logprobs:
            raise ValueError("top_logprobs requires logprobs to be true")
        return self


# ── Content block helpers ──


def _resolve_json_schema(req) -> dict | str | None:
    """Resolve json_schema from req.json_schema, req.grammar, or req.response_format.

    The Anthropic API doesn't have a standard structured output mechanism,
    but clients may send response_format (OpenAI-style), json_schema, or grammar directly.
    """
    # Forced-tool grammar from tool_choice="any"/"tool" takes highest
    # priority — clients enforcing a specific tool call should not have
    # their constraint silently overridden by a less specific schema.
    forced = getattr(req, "_forced_tool_grammar", None)
    if forced is not None:
        return forced
    # Forced-thinking grammar — set when thinking={enabled,...} on a model
    # that doesn't natively emit <think> tags. Forces the output to start
    # with <think>...</think> so downstream extract_thinking() can split.
    forced_think = getattr(req, "_forced_thinking_grammar", None)
    if forced_think is not None:
        return forced_think
    # Direct json_schema field takes priority
    js = getattr(req, "json_schema", None)
    if js is not None:
        return js
    # Grammar constraint (regex, choice, CFG) — only if actually provided
    grammar = getattr(req, "grammar", None)
    if grammar is not None and isinstance(grammar, (dict, str)):
        return grammar
    # Fall back to OpenAI-style response_format
    rf = getattr(req, "response_format", None)
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
    for b in content:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "image":
            return True
        # Anthropic allows image content INSIDE a tool_result block (a tool
        # returning a screenshot). Recurse so the VLM path is selected even when the
        # ONLY image lives there — else has_images=False and the image is dropped.
        if b.get("type") == "tool_result":
            inner = b.get("content")
            if isinstance(inner, list) and any(
                isinstance(x, dict) and x.get("type") == "image" for x in inner
            ):
                return True
    return False


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
    # map tool_use id → function name across turns, so a later tool_result message
    # can carry the originating tool's `name` (templates that key the tool turn on the
    # function name — Hermes/Qwen/Mistral — otherwise render a nameless tool response).
    _tool_id_to_name: dict[str, str] = {}

    for m in messages_input:
        content = m.content
        role = m.role

        if not isinstance(content, list):
            # Simple string or None content — no block processing needed
            intermediate.append({"role": role, "content": content or ""})
            continue

        # Check for tool_use and tool_result blocks
        has_tool_use = any(
            isinstance(b, dict) and b.get("type") == "tool_use" for b in content
        )
        has_tool_result = any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
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
                    _tool_id_to_name[tool_id] = tool_name  # for the tool_result name
                    # json.dumps handles any JSON-serializable input (Anthropic spec
                    # says object, but a list/scalar must still be valid JSON, not str()'s
                    # single-quoted Python repr which downstream arg-parsing can't json.loads).
                    try:
                        _args = json.dumps(tool_input)
                    except (TypeError, ValueError):
                        _args = json.dumps({"value": str(tool_input)})
                    tool_calls.append(
                        {
                            "id": tool_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": _args},
                        }
                    )
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
                    tool_use_id = block.get("tool_use_id", "toolu_unknown")
                    inner = block.get("content", "")
                    _img_blocks: list[dict] = []
                    if isinstance(inner, list):
                        # separate text from image sub-blocks. Text → the tool
                        # message; images → real image_url user messages (via
                        # _convert_image_block) so the VLM actually SEES a tool-returned
                        # screenshot (vision-tool / computer-use agents). Previously every
                        # image was flattened to the literal "[Image: …]" → model blind.
                        _img_blocks = [
                            x
                            for x in inner
                            if isinstance(x, dict) and x.get("type") == "image"
                        ]
                        _text_blocks = [x for x in inner if x not in _img_blocks]
                        inner_text = _extract_text_from_content(_text_blocks)
                    else:
                        inner_text = str(inner) if inner is not None else ""
                    _content = inner_text or ("[image result]" if _img_blocks else "")
                    # honor is_error. Anthropic sets it when the tool raised/
                    # failed; the old code dropped it → the model saw a FAILED result as a
                    # success and proceeded (skipped retries, hallucinated on garbage). Mark it.
                    if block.get("is_error"):
                        _content = (
                            f"[tool_error] {_content}"
                            if _content
                            else "[tool_error] (no detail)"
                        )
                    _tmsg = {
                        "role": "tool",
                        "content": _content,
                        "tool_call_id": tool_use_id,
                    }
                    # carry the originating tool's name (id→name map) for templates
                    # that key the tool turn on the function name.
                    _tn = _tool_id_to_name.get(tool_use_id)
                    if _tn:
                        _tmsg["name"] = _tn
                    intermediate.append(_tmsg)
                    # Emit each tool-result image as a real image_url user message so the
                    # VLM picks it up (only when the VLM path is active + temp_files sink).
                    if _img_blocks and has_images and temp_files is not None:
                        for _ib in _img_blocks:
                            try:
                                _convert_image_block(_ib, intermediate, temp_files)
                            except HTTPException:
                                raise
                            except Exception:
                                logger.debug(
                                    "tool_result image conversion failed", exc_info=True
                                )
                elif bt == "text":
                    intermediate.append(
                        {"role": "user", "content": block.get("text", "")}
                    )
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
                    converted_parts.append(
                        {"type": "text", "text": block.get("text", "")}
                    )
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
                            logger.debug(
                                "base64 decode failed, trying with padding",
                                exc_info=True,
                            )
                            raw = _b64.b64decode(data + "==", validate=False)
                        ext_map = {
                            "image/png": "png",
                            "image/jpeg": "jpg",
                            "image/gif": "gif",
                            "image/webp": "webp",
                        }
                        ext = ext_map.get(media_type, "png")
                        # delete=False is intentional — file must outlive function for
                        # downstream image inference; cleanup via temp_files registry.
                        tmp = _tf.NamedTemporaryFile(suffix=f".{ext}", delete=False)  # noqa: SIM115
                        try:
                            tmp.write(raw)
                            tmp.close()
                        except Exception:
                            tmp.close()
                            import os as _os

                            with contextlib.suppress(OSError):
                                _os.unlink(tmp.name)
                            raise
                        temp_files.append(tmp.name)
                        converted_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"file://{tmp.name}"},
                            }
                        )
                    elif source_type == "url" and source.get("url"):
                        # Anthropic url source: scheme-validate before pass-through
                        # to prevent SSRF (mirror the check in chat.py).
                        _src_url = source["url"]
                        from .chat import _is_safe_image_url

                        if not _is_safe_image_url(_src_url):
                            raise HTTPException(
                                status_code=400,
                                detail=f"Image URL blocked by SSRF protection: '{_src_url[:80]}'",
                            )
                        converted_parts.append(
                            {"type": "image_url", "image_url": {"url": _src_url}}
                        )
                    else:
                        converted_parts.append(
                            {"type": "text", "text": f"[Image: {media_type}]"}
                        )
                else:
                    text = _extract_text_from_content([block])
                    converted_parts.append({"type": "text", "text": text})
            intermediate.append({"role": role, "content": converted_parts})

        else:
            # Standard content: flatten to text
            content_text = _extract_text_from_content(content)
            intermediate.append({"role": role, "content": content_text})

    return intermediate, temp_files


def _convert_image_block(
    block: dict, intermediate: list[dict], temp_files: list[str]
) -> None:
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
        ext_map = {
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/gif": "gif",
            "image/webp": "webp",
        }
        ext = ext_map.get(media_type, "png")
        # delete=False is intentional — file must outlive function for downstream
        # image inference; cleanup via temp_files registry.
        tmp = _tf.NamedTemporaryFile(suffix=f".{ext}", delete=False)  # noqa: SIM115
        try:
            tmp.write(raw)
            tmp.close()
        except Exception:
            tmp.close()
            import os as _os

            with contextlib.suppress(OSError):
                _os.unlink(tmp.name)
            raise
        temp_files.append(tmp.name)
        intermediate.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"file://{tmp.name}"}}
                ],
            }
        )
    elif source_type == "url" and source.get("url"):
        # Anthropic url source: scheme-validate before pass-through (SSRF).
        _src_url = source["url"]
        from .chat import _is_safe_image_url

        if not _is_safe_image_url(_src_url):
            raise HTTPException(
                status_code=400,
                detail=f"Image URL blocked by SSRF protection: '{_src_url[:80]}'",
            )
        intermediate.append(
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": _src_url}}],
            }
        )
    else:
        intermediate.append({"role": "user", "content": f"[Image: {media_type}]"})


def _cacheable_prefix_token_count(system, char_offsets, tokenizer) -> int:
    """Tokens in the system text up to the LAST cache_control breakpoint — the portion
    Anthropic bills as cache_creation_input_tokens (everything after it is plain
    input_tokens). Best-effort: the offsets are character positions in the
    system-only assembled text, so this approximates the cacheable prefix in system-token
    space — but that is far better than the old behavior of billing the ENTIRE uncached
    prompt as cache_creation (which reported input_tokens=0 on a first cache_control call).
    """
    if not char_offsets or tokenizer is None or not isinstance(system, list):
        return 0
    # Reassemble exactly as _extract_cache_control_hints measured the offsets:
    # "\n".join of each dict block's text.
    assembled = "\n".join(b.get("text", "") for b in system if isinstance(b, dict))
    prefix = assembled[: max(char_offsets)]
    if not prefix:
        return 0
    try:
        return len(tokenizer.encode(prefix, add_special_tokens=False))
    except TypeError:
        with contextlib.suppress(Exception):
            return len(tokenizer.encode(prefix))
        return 0
    except Exception:
        return 0


def _anthropic_cache_usage(
    prompt_tokens: int, cached_tokens: int, cacheable_prefix_tokens: int = 0
) -> tuple[int, int, int]:
    """Map our single `cached_tokens` signal to Anthropic's three usage counters,
    respecting the invariant input + cache_creation + cache_read == prompt_tokens
    (the old code double-counted: input_tokens=prompt AND cache_read=cached → 2×).

    - cache_read = tokens served from cache (the reused prefix).
    - cache_creation = the cacheable prefix being WRITTEN this turn (the prefix up to the
      last cache_control breakpoint, minus whatever was already served from cache), capped
      at the uncached remainder.
    - input_tokens = the rest (content AFTER the last breakpoint = plain input).

    Previously `cache_creation = ALL uncached` whenever breakpoints existed,
    which reported input_tokens=0 on a first cache_control request (the post-breakpoint
    content was wrongly billed as cache write). `cacheable_prefix_tokens` (0 when no
    breakpoints) now bounds cache_creation so the remainder stays input_tokens.
    Returns (input_tokens, cache_creation_input_tokens, cache_read_input_tokens).
    """
    prompt = max(0, int(prompt_tokens or 0))
    cache_read = max(0, min(int(cached_tokens or 0), prompt))
    uncached = prompt - cache_read
    writable = max(0, int(cacheable_prefix_tokens or 0) - cache_read)
    cache_creation = min(writable, uncached)
    input_tokens = uncached - cache_creation
    return input_tokens, cache_creation, cache_read


def _extract_cache_control_hints(
    system: str | list[dict] | None,
) -> tuple[list[dict], list[int]]:
    """Extract cache_control hints and character offsets from system messages.

    Anthropic uses cache_control to signal prompt-caching breakpoints.
    Returns a tuple of (hints, char_offsets) where char_offsets are the
    cumulative character positions in the assembled system text at which
    KV cache breakpoints should be placed. These correspond to the end
    of each block that carries a cache_control marker.

    Offsets account for newlines between blocks (matching the \n.join
    in _extract_text_from_content).
    """
    # KNOWN LIMITATION: only `system` blocks are scanned for cache_control. The common
    # Anthropic pattern of marking cache_control on the LAST content block of a messages[]
    # entry (to cache a long document/conversation prefix) is not detected here, so such a
    # request reports cache_creation/cache_read = 0 and saves no KV breakpoint at that
    # position. This is a missing-feature/misreport, NOT corruption (usage stays
    # self-consistent: input+0+0==prompt; the fast-path KVPrefixCache still warms prefixes
    # automatically). A correct fix requires mapping message-block char offsets onto the
    # FULL chat-templated prompt the engine tokenizes (the offsets here are vs the
    # system-only text), which is a coordinated rework of the breakpoint plumbing.
    if system is None:
        return [], []
    hints: list[dict] = []
    char_offsets: list[int] = []
    if isinstance(system, list):
        cum_len = 0
        block_idx = 0
        for block in system:
            if isinstance(block, dict):
                text = block.get("text", "")
                if block_idx > 0:
                    cum_len += 1  # newline separator from _extract_text_from_content
                cum_len += len(text)
                block_idx += 1
                if "cache_control" in block:
                    hints.append(block["cache_control"])
                    char_offsets.append(cum_len)
    return hints, char_offsets


# ── Streaming tool-use helpers ──

# Regex to detect tool-call JSON inside model output
_TOOL_CALL_JSON_RE = re.compile(
    r'\{[\s\n]*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*',
)
_TOOL_CALL_XML_RE = re.compile(
    r"<tool_call\s*/?\s*>\s*(.*?)\s*</tool_call\s*/?\s*>",
    re.DOTALL,
)


def _try_parse_tool_call_delta(text: str) -> list[dict] | None:
    """Try to parse partial or complete tool-call JSON from streaming text.

    Returns a list of {"name": str, "arguments": str} dicts if a tool call
    is detected, or None if the text doesn't contain a recognisable tool call.
    """
    # Fast rejection: skip regex entirely if no tool-call trigger chars present
    if "<" not in text and "{" not in text:
        return None
    # Try XML-wrapped tool calls first
    for m in _TOOL_CALL_XML_RE.finditer(text):
        inner = m.group(1).strip()
        try:
            data = json.loads(inner)
            if "name" in data:
                return [
                    {
                        "name": data["name"],
                        "arguments": json.dumps(data.get("arguments", {})),
                    }
                ]
        except json.JSONDecodeError:
            pass

    # Try bare JSON — find the opening brace after name and parse from there.
    # Use json.JSONDecoder.raw_decode for robust parsing that correctly
    # handles nested braces inside string values (avoids false termination
    # on literal '}' characters within JSON string values).
    calls: list[dict] = []
    for m in _TOOL_CALL_JSON_RE.finditer(text):
        name = m.group(1)
        rest = text[m.end() :]
        try:
            decoder = json.JSONDecoder()
            obj, end_idx = decoder.raw_decode(rest)
            args_str = rest[:end_idx]
            calls.append({"name": name, "arguments": args_str})
        except json.JSONDecodeError:
            pass
    return calls or None


def _resolve_cached_content_text(
    cached_content, model, request, *, mutate: bool
) -> str:
    """Resolve a Gemini-style ``cached_content`` handle to its stored prefix text (or "").

    Shared by create_message (mutate=True — marks a real READ, bumps last_used/read_count)
    and count_tokens (mutate=False — a non-mutating estimation read). Enforces the
    cross-tenant ownership guard (a handle owned by another tenant is ignored, not leaked)
    and raises HTTPException(400) on a model mismatch (a handle is model-specific).

    count_tokens previously had ZERO cached_content handling, so its input_tokens
    omitted the entire cached prefix that generation prepends into the system prompt — a
    silent undercount of a (by-design large) reusable prefix. Reuse the SAME resolver as
    generation so the estimate matches the real prompt and the two can't drift.
    """
    if not cached_content:
        return ""
    from ..explicit_cache import get_store

    _cc = cached_content
    _key = _cc if _cc.startswith("cachedContents/") else f"cachedContents/{_cc}"
    store = get_store()
    entry = store.use(_key) if mutate else store.get(_key)
    _cc_owner = getattr(entry, "owner", None) if entry is not None else None
    if _cc_owner and _cc_owner != "anonymous":
        from yunshu_control.audit_log import resolve_actor

        if resolve_actor(request) != _cc_owner:
            logger.warning(
                "anthropic cached_content '%s' owned by another tenant — ignoring", _cc
            )
            entry = None
    if entry is not None and getattr(entry, "model", None) and entry.model != model:
        raise HTTPException(
            status_code=400,
            detail=(
                f"cached_content '{_cc}' was created for model '{entry.model}' "
                f"and cannot be used with '{model}'"
            ),
        )
    if entry is not None and entry.messages:

        def _cc_msg_text(_m):
            _c = _m.get("content")
            if isinstance(_c, str):
                return _c
            if isinstance(_c, list):
                return " ".join(
                    _b.get("text", "")
                    for _b in _c
                    if isinstance(_b, dict)
                    and _b.get("type") in ("text", "input_text", "output_text")
                )
            return ""

        return "\n".join(_cc_msg_text(m) for m in entry.messages).strip()
    return ""


# ── Endpoint ──


@router.post("/messages", response_model=None)
async def create_message(req: AnthropicMessagesRequest, request: Request):
    """Anthropic Messages API endpoint."""
    _check_permission(request, "can_infer")
    # Build messages list. The canonical system text (top-level `system` field +
    # any role="system" lifted from messages[]) is prepended once AFTER the lift
    # block below — do NOT add it here, or it gets lifted back out and the merge
    # duplicates it / leaves `messages` (what generation consumes) system-less.
    # Gemini-style explicit context-cache READ: prepend the handle's stored text
    # to the system prompt so the automatic KVPrefixCache serves the warmed prefix.
    if req.cached_content:
        try:
            # shared resolver (mutate=True marks a real READ). Enforces the
            # cross-tenant ownership guard + model-mismatch 400; flattens block content.
            # count_tokens reuses the SAME resolver so the estimate can't drift.
            _cc_text = _resolve_cached_content_text(
                req.cached_content, req.model, request, mutate=True
            )
            if _cc_text:
                if isinstance(req.system, str):
                    req.system = _cc_text + "\n" + req.system
                elif isinstance(req.system, list):
                    req.system = [{"type": "text", "text": _cc_text}] + req.system
                else:
                    req.system = _cc_text
        except HTTPException:
            raise  # model-mismatch 400 must propagate, not be swallowed
        except Exception:
            logger.debug("anthropic cached_content prepend failed", exc_info=True)

    messages = []
    _temp_files: list[str] = []  # track temp files for cleanup

    # Extract cache_control hints and compute KV cache breakpoint offsets.
    # char_offsets are cumulative character positions in the system text
    # where the engine should save KV prefix cache entries.
    _cache_hints, _kv_cache_breakpoints = _extract_cache_control_hints(req.system)
    if _kv_cache_breakpoints:
        logger.debug(
            "Anthropic cache_control breakpoints (char offsets): %s",
            _kv_cache_breakpoints,
        )

    has_images = any(_has_image_blocks(m.content) for m in req.messages)

    # Convert Anthropic messages to OpenAI-compatible format
    # This properly handles tool_use/tool_result blocks instead of flattening them
    converted_msgs, _temp_files = _convert_anthropic_messages(
        req.messages,
        has_images=has_images,
        temp_files=_temp_files,
    )
    messages.extend(converted_msgs)

    # Anthropic API semantics: role="system" entries in messages[] should be
    # lifted into the canonical system field, not left in the messages list.
    # This matches omlx behavior and ensures correct cache key computation.
    _system_parts: list[str] = []
    _filtered_messages: list[dict] = []
    for msg in messages:
        if msg.get("role") == "system":
            _system_parts.append(msg.get("content", ""))
        else:
            _filtered_messages.append(msg)
    messages = _filtered_messages
    # Canonical system text = top-level `system` field, then any lifted
    # role="system" entries from messages[].
    _all_system: list[str] = []
    if req.system:
        _all_system.append(
            _extract_text_from_content(req.system)
            if isinstance(req.system, list)
            else req.system
        )
    _all_system.extend(p for p in _system_parts if p)
    # snapshot the ORIGINAL system (the list-of-blocks carrying cache_control)
    # before collapsing it to a string below. _cacheable_prefix_token_count needs the LIST
    # form to locate the cache_control breakpoint, so the cache_creation_input_tokens
    # logic was DEAD once req.system became a string here (the helper returns 0 for a str) —
    # cache_creation_input_tokens was always 0 even with cache_control breakpoints.
    req._anthropic_orig_system = req.system
    if _all_system:
        req.system = "\n\n".join(_all_system)
        # Re-prepend ONE canonical system message so it actually reaches the
        # engine — `messages` is what generation consumes (the lift moved it out).
        messages.insert(0, {"role": "system", "content": req.system})

    stop = req.stop_sequences or []

    # Thinking-mode prompt injection — needed for models that don't
    # natively emit <think>...</think> via their chat template (e.g.,
    # Qwen2.5-*-Instruct). The Anthropic API contract requires returning
    # a "thinking" content block whenever thinking={enabled, budget_tokens}
    # is set, so we inject a system instruction telling the model to
    # wrap its reasoning. Native-thinking models (Qwen3 etc.) ignore the
    # added hint because their template emits <think> tokens regardless.
    if (
        req.thinking
        and isinstance(req.thinking, dict)
        and req.thinking.get("type") == "enabled"
    ):
        _budget = req.thinking.get("budget_tokens")
        _think_instruction = (
            "\n\nIMPORTANT FORMAT REQUIREMENT: You MUST begin every reply with "
            "an opening <think> tag, write your private step-by-step reasoning, "
            "then write a closing </think> tag, and only after that produce the "
            "final user-facing answer. Do not skip these tags. "
            "Example shape: <think>...reasoning...</think>final answer."
        )
        if isinstance(_budget, int) and _budget > 0:
            _think_instruction += f" Keep the reasoning inside <think>...</think> to roughly {_budget} tokens."
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = (
                messages[0].get("content") or ""
            ) + _think_instruction
        else:
            messages.insert(
                0, {"role": "system", "content": _think_instruction.strip()}
            )
        # Mark that we requested thinking via prompt injection — used later
        # to apply a heuristic-based <think> wrapper around the model's
        # natural reasoning prose if the model declined to emit literal tags.
        req._thinking_prompt_injected = True
        # When there are no tools and no explicit grammar already requested,
        # force the model to start with <think>...</think> via a regex
        # grammar. This is the most reliable way to guarantee the Anthropic
        # response contains a "thinking" content block.
        if (
            not req.tools
            and getattr(req, "json_schema", None) is None
            and getattr(req, "grammar", None) is None
            and getattr(req, "response_format", None) is None
        ):
            req._forced_thinking_grammar = {
                "type": "regex",
                # Require at least one non-trivial character inside <think>...</think>
                # so the model can't satisfy the grammar with an empty block.
                "pattern": r"<think>[\s\S]+?</think>[\s\S]*",
            }

    # Inject tool definitions into system prompt if provided
    if req.tools:
        tool_prompt = (
            "\n\nYou have access to the following tools. When you need to call a tool, "
        )
        tool_prompt += 'output a tool call in the following format:\n<tool_call\\>{"name": "...", "arguments": {...}}</tool_call\\>\n\n'
        tool_prompt += "Available tools:\n"
        for tool in req.tools:
            tool_prompt += f"- {tool.name}"
            if tool.description:
                tool_prompt += f": {tool.description}"
            if tool.input_schema:
                tool_prompt += f"\n  Parameters: {tool.input_schema}"
            tool_prompt += "\n"

        # Track grammar enforcement for "any" / "tool" tool_choice — pure
        # prompt-injection ("You MUST call …") is advisory; LLMs frequently
        # ignore it and emit prose. Attach a regex grammar that forces the
        # first emitted text to begin with <tool_call> markup.
        _forced_tool_grammar: dict | None = None
        if req.tool_choice:
            if isinstance(req.tool_choice, dict):
                tc_type = req.tool_choice.get("type", "")
                if tc_type == "none":
                    req._suppress_tools = True
                    tool_prompt = ""
                elif tc_type == "any":
                    # prompt-only enforcement (the regex DFA path
                    # returns empty allowed_tokens mid-generation → premature
                    # EOS). Strengthen the prompt with explicit
                    # listing of tool names and a forbidden-prefix rule, so
                    # the model is much less likely to ignore "any" even with
                    # benign prompts like "Hello there." (where the model
                    # would otherwise default to chitchat).
                    # req.tools is List[AnthropicTool] (Pydantic
                    # model), not List[dict] — use attribute access. Fall back
                    # to .get() for dict-shaped inputs (some callers pass raw
                    # dicts before Pydantic validation in custom flows).
                    def _tool_name(t: object) -> str:
                        name = getattr(t, "name", None)
                        if name is None and isinstance(t, dict):
                            name = t.get("name")
                        return str(name or "")

                    tool_names = ", ".join(
                        n for n in (_tool_name(t) for t in (req.tools or [])) if n
                    )
                    tool_prompt += (
                        f"\nCRITICAL: The caller set tool_choice=any. You MUST invoke "
                        f"EXACTLY ONE of the following tools regardless of what the user said: {tool_names}. "
                        "Do NOT respond with text-only content. The FIRST tokens of your reply MUST be `<tool_call>`. "
                        "Format exactly (no prose before or after):\n"
                        '<tool_call>{"name": "<tool>", "arguments": {<args>}}</tool_call>\n'
                    )
                elif tc_type == "tool":
                    forced = req.tool_choice.get("name")
                    if forced:
                        # the regex grammar for `<tool_call>{...}</tool_call>`
                        # with lazy `[\s\S]*?` was confusing the DFA's
                        # valid-next-chars probe — get_allowed_tokens() returned
                        # [] mid-generation, triggering the ConstrainedSampler's
                        # EOS-fallback at token 11 with stop_reason=end_turn.
                        # Result: forced tool stream emitted only
                        # `<tool_call>{"name": "X", "` and stopped.
                        # Switch to prompt-only enforcement: strongly request the
                        # `<tool_call>{...}</tool_call>` format in the system
                        # prompt + emit explicit JSON template, then rely on
                        # extract_tool_calls_v2 + ToolCallStreamer to parse the
                        # output. The "auto" path uses this exact strategy and
                        # produces valid tool_use blocks reliably.
                        tool_prompt += (
                            f"\nYou MUST call the tool '{forced}'. Emit ONLY a tool call, "
                            f"no prose. Format exactly:\n"
                            f'<tool_call>{{"name": "{forced}", "arguments": {{<args>}}}}</tool_call>\n'
                        )
                        # NOTE: _forced_tool_grammar deliberately NOT set — the
                        # prompt + parser is enough; the regex DFA path causes
                        # premature EOS for complex patterns.
                # {"type": "auto"} is the default — no additional prompt needed
            elif req.tool_choice == "any":
                # same prompt-only switch as the dict-form "any".
                tool_prompt += (
                    "\nYou MUST call at least one tool. Do NOT respond with only text. "
                    "Format exactly:\n"
                    '<tool_call>{"name": "<tool>", "arguments": {<args>}}</tool_call>\n'
                )
            elif req.tool_choice == "none":
                req._suppress_tools = True
                tool_prompt = ""

        # Stash the forced grammar so _non_stream_*/_stream_* can pass it
        # through to engine.generate via json_schema.
        if _forced_tool_grammar is not None:
            req._forced_tool_grammar = _forced_tool_grammar

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
                    "type": "not_found_error"
                    if e.status_code == 404
                    else "overloaded_error",
                    "message": e.detail,
                },
            },
        )

    # Reject prompts over the context window (400) or too large to prefill (413),
    # before we attempt generation (chat.py). Covers stream +
    # non-stream (before the stream branch below).
    try:
        from yunshu_control.token_counter import count_message_tokens

        from ..streaming import validate_context_window, validate_prefill_memory

        _pf_tok = getattr(engine, "_tokenizer", None) or getattr(
            engine, "tokenizer", None
        )
        _pf_est = count_message_tokens(messages, _pf_tok)
        validate_context_window(_pf_est, req.model, engine)
        validate_prefill_memory(_pf_est)
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": e.detail},
            },
        )
    except Exception:
        logger.debug("anthropic prefill validation skipped", exc_info=True)

    # validate logit_bias EAGERLY here, before the
    # stream branch. The streaming path otherwise only calls _convert_logit_bias
    # inside the _stream_anthropic generator body (after 200 + SSE headers are
    # already sent), so a NaN/Inf/out-of-range bias produced a broken/aborted
    # stream instead of a clean 422. Non-streaming was already fine (it validates
    # before returning), but doing it here once is idempotent and covers both.
    try:
        _convert_logit_bias(req)
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": e.detail},
            },
        )

    loaded_adapter = _apply_lora_adapter(engine, req.lora_adapter)

    if req.stream:
        return StreamingResponse(
            _stream_anthropic(
                engine,
                messages,
                req,
                stop,
                request,
                is_batched=is_batched,
                temp_files=_temp_files,
                lora_adapter=loaded_adapter,
                kv_cache_breakpoints=_kv_cache_breakpoints,
            ),
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

    try:
        if is_batched:
            return await _non_stream_batched(
                engine,
                messages,
                req,
                stop,
                cancel_event=_ns_cancel_event,
                lora_adapter=loaded_adapter,
                kv_cache_breakpoints=_kv_cache_breakpoints,
                request=request,
            )
        return await _non_stream_legacy(
            engine,
            messages,
            req,
            stop,
            cancel_event=_ns_cancel_event,
            lora_adapter=loaded_adapter,
            kv_cache_breakpoints=_kv_cache_breakpoints,
            request=request,
        )
    finally:
        _release_lora_adapter(engine, loaded_adapter)
        if _ns_tracker is not None:
            with contextlib.suppress(Exception):
                _ns_tracker.unregister(message_id)
        # Clean up temp files created for image blocks
        import os as _os

        for _tf_path in _temp_files:
            with contextlib.suppress(OSError):
                _os.unlink(_tf_path)


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
        import math

        result = {}
        for k, v in _lb.items():
            # validate the VALUE (chat/responses do this; anthropic was missing
            # it). A NaN/Inf bias → all-NaN softmax → garbage.
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or math.isnan(v)
                or math.isinf(v)
            ):
                raise HTTPException(
                    status_code=400, detail=f"logit_bias[{k}]: must be a finite number"
                )
            if v < -100.0 or v > 100.0:
                raise HTTPException(
                    status_code=400,
                    detail=f"logit_bias[{k}]={v}: must be between -100 and 100",
                )
            try:
                result[int(k)] = v
            except (ValueError, TypeError):
                logger.warning(f"Skipping non-integer logit_bias key: {k!r}")
        return result if result else None
    return None


async def _non_stream_batched(
    engine,
    messages,
    req,
    stop,
    cancel_event=None,
    lora_adapter=None,
    kv_cache_breakpoints=None,
    request=None,
):
    """Non-streaming response via BatchedEngine."""
    from fastapi.responses import JSONResponse

    enable_thinking = req.thinking and req.thinking.get("type") == "enabled"
    budget_tokens = req.thinking.get("budget_tokens") if req.thinking else None
    _logit_bias = _convert_logit_bias(req)

    try:
        _chat_coro = engine.chat(
            messages=messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            min_p=req.min_p,
            repetition_penalty=req.repetition_penalty,
            frequency_penalty=req.frequency_penalty,
            presence_penalty=req.presence_penalty,
            min_tokens=req.min_tokens,
            ignore_eos=req.ignore_eos,
            suppress_tokens=req.suppress_tokens,
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
            lora_adapter=lora_adapter,
            kv_cache_breakpoints=kv_cache_breakpoints,
        )
        # disconnect guard — set cancel_event on client disconnect so the engine
        # decode loop stops (was: registered cancel_event but never polled is_disconnected
        # → a disconnect ran to max_tokens/timeout, HOL-blocking the serial executor).
        if request is not None:
            result = await run_with_disconnect_guard(
                request, _chat_coro, cancel_event=cancel_event
            )
            if result is None:
                raise HTTPException(status_code=499, detail="Client disconnected")
        else:
            result = await _chat_coro
    except HTTPException:
        raise  # let the disconnect (499) propagate, not become a 500
    except MemoryError:
        return JSONResponse(
            status_code=507,
            content={
                "type": "error",
                "error": {
                    "type": "overloaded_error",
                    "message": "Insufficient GPU memory",
                },
            },
        )
    except Exception as e:
        logger.error(f"Anthropic batched generation error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": "Internal server error"},
            },
        )
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    # the engine's completion_tokens ALREADY includes reasoning tokens
    # (reasoning_tokens is a subset detail, not an addend). Adding it again
    # double-counted the server-wide completion metric for every thinking-model request —
    # the chat endpoint records it correctly (chat.py:1710 "already incl. reasoning").
    _record_metrics(result.prompt_tokens, result.completion_tokens)

    content = []
    thinking_text = ""
    visible_text = result.text

    # ALWAYS separate reasoning from visible text (engine emits CoT by default
    # for native-thinking models), so the raw <think>…</think> can't leak into the
    # text block when the client didn't pass thinking={type:enabled}. extract_thinking is a
    # no-op without think tags. Emit the thinking block when there IS reasoning OR thinking
    # was explicitly requested (the Anthropic-spec placeholder for an injected-but-silent CoT).
    from ..streaming import extract_thinking

    thinking_text, visible_text = extract_thinking(result.text, req.model)
    if thinking_text or getattr(req, "_thinking_prompt_injected", False):
        content.append(
            {
                "type": "thinking",
                "thinking": thinking_text,
                "signature": "",
            }
        )

    text_block: dict = {"type": "text", "text": visible_text}

    # Check for matched stop sequences — trim BEFORE tool call extraction.
    # Some engine paths trim the stop sequence themselves before returning
    # `finish_reason=="stop"`. In that case, recover by reporting the
    # caller's stop_sequences[0] (correct when only one stop was passed,
    # and "<stop_sequence>" as a placeholder otherwise) so `stop_reason`
    # is set truthfully per Anthropic's protocol.
    matched_stop = _detect_matched_stop(
        visible_text,
        stop,
        getattr(result, "finish_reason", None),
        stopped_by_stop_sequence=getattr(result, "stopped_by_stop_sequence", None),
    )
    if matched_stop and visible_text:
        idx = visible_text.find(matched_stop)
        if idx != -1:
            visible_text = visible_text[:idx]
            text_block["text"] = visible_text

    # Include logprobs in the text content block if requested
    if req.logprobs:
        _result_lp = getattr(result, "logprobs", None)
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
    _suppress_tool_extraction = getattr(req, "_suppress_tools", False)
    has_tool_calls = False
    if _suppress_tool_extraction and req.tools:
        # tool_choice="none" suppresses tool_use emission, but the model may still
        # emit <tool_call> markup — strip it from the visible text so it doesn't leak (the
        # chat router cleans even under "none"; the old Anthropic path skipped cleanup).
        from ..streaming import clean_tool_call_markup

        text_block["text"] = clean_tool_call_markup(visible_text)
    if req.tools and not _suppress_tool_extraction:
        from ..streaming import clean_tool_call_markup, extract_tool_calls_model_aware

        tool_calls = extract_tool_calls_model_aware(visible_text, req.model)
        # enforce a forced/none-parallel tool_choice post-generation (parity with
        # chat's _enforce_tool_choice) — drop wrong-named / surplus calls.
        tool_calls = _enforce_anthropic_tool_choice(tool_calls, req.tool_choice)
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
                    inp = (
                        json.loads(tc["arguments"])
                        if isinstance(tc["arguments"], str)
                        else tc["arguments"]
                    )
                except (json.JSONDecodeError, TypeError):
                    inp = {}
                content.append(
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tc["name"],
                        "input": inp,
                    }
                )

    stop_reason = _map_stop_reason(
        result.finish_reason, matched_stop, has_tool_calls=has_tool_calls
    )

    _cache_tok = getattr(engine, "_tokenizer", None) or getattr(
        engine, "tokenizer", None
    )
    _input_tok, cache_creation, cache_read = _anthropic_cache_usage(
        getattr(result, "prompt_tokens", 0),
        getattr(result, "cached_tokens", 0),
        _cacheable_prefix_token_count(
            getattr(req, "_anthropic_orig_system", req.system),
            kv_cache_breakpoints,
            _cache_tok,
        ),
    )
    reasoning_tok = getattr(result, "reasoning_tokens", 0) or 0

    # For models that don't natively report reasoning_tokens (e.g. Qwen2.5
    # with our prompt-injected <think> path), approximate by tokenizing
    # the extracted thinking text. completion_tokens in this path ALREADY
    # includes the thinking tokens (engine sees one undifferentiated stream),
    # so we only surface reasoning_tokens for client visibility — do NOT
    # double-add to output_tokens.
    _engine_reasoning_tok = getattr(result, "reasoning_tokens", 0) or 0
    if _engine_reasoning_tok == 0 and thinking_text:
        try:
            _tok = getattr(engine, "_tokenizer", None) or getattr(
                engine, "tokenizer", None
            )
            if _tok is not None and hasattr(_tok, "encode"):
                reasoning_tok = len(_tok.encode(thinking_text))
        except Exception:
            logger.debug("failed to count thinking tokens", exc_info=True)

    # Per Anthropic spec output_tokens is the TOTAL (visible + reasoning). The
    # engine's completion_tokens is len(generated tokens) and ALREADY includes
    # the reasoning tokens; reasoning_tokens is the detail SUBSET, not an addend.
    # Adding it double-counted (matches the chat.py fix and the Anthropic
    # streaming path, which both use completion_tokens unmodified).
    total_output_tokens = result.completion_tokens

    usage: dict[str, Any] = {
        "input_tokens": _input_tok,
        "output_tokens": total_output_tokens,
        "cache_creation_input_tokens": cache_creation,
        "cache_read_input_tokens": cache_read,
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
        "stop_sequence": _public_stop_sequence(matched_stop),
        "usage": usage,
        **({"metadata": req.metadata} if req.metadata else {}),
    }
    return JSONResponse(resp)


async def _non_stream_legacy(
    engine,
    messages,
    req,
    stop,
    cancel_event=None,
    lora_adapter=None,
    kv_cache_breakpoints=None,
    request=None,
):
    """Non-streaming response via Engine or BatchedEngine."""
    from fastapi.responses import JSONResponse

    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    enable_thinking = req.thinking and req.thinking.get("type") == "enabled"
    budget_tokens = req.thinking.get("budget_tokens") if req.thinking else None
    _logit_bias = _convert_logit_bias(req)
    try:
        _gen_coro = engine.generate(
            prompt=messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            min_p=req.min_p,
            repetition_penalty=req.repetition_penalty,
            frequency_penalty=req.frequency_penalty,
            presence_penalty=req.presence_penalty,
            min_tokens=req.min_tokens,
            ignore_eos=req.ignore_eos,
            suppress_tokens=req.suppress_tokens,
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
            lora_adapter=lora_adapter,
            kv_cache_breakpoints=kv_cache_breakpoints,
        )
        # disconnect guard (see _non_stream_batched).
        if request is not None:
            result = await run_with_disconnect_guard(
                request, _gen_coro, cancel_event=cancel_event
            )
            if result is None:
                raise HTTPException(status_code=499, detail="Client disconnected")
        else:
            result = await _gen_coro
    except HTTPException:
        raise  # let the disconnect (499) propagate, not become a 500
    except MemoryError:
        return JSONResponse(
            status_code=507,
            content={
                "type": "error",
                "error": {
                    "type": "overloaded_error",
                    "message": "Insufficient GPU memory",
                },
            },
        )
    except Exception as e:
        logger.error(f"Anthropic legacy generation error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": "Internal server error"},
            },
        )

    # Handle Engine (prompt_token_count attr), BatchedEngine (prompt_tokens
    # attr), and VLMEngine (returns a dict with prompt_tokens/text keys).
    def _g(name, fallback_name=None, default=0):
        if isinstance(result, dict):
            v = result.get(name)
            if v is None and fallback_name:
                v = result.get(fallback_name)
            return default if v is None else v
        v = getattr(result, name, None)
        if v is None and fallback_name:
            v = getattr(result, fallback_name, None)
        return default if v is None else v

    prompt_toks = _g("prompt_tokens", "prompt_token_count", 0)
    completion_toks = _g("completion_tokens", "completion_token_count", 0)
    text = _g("text", "generated_text", "")
    finish_reason = _g("finish_reason", "finish_state", None)
    cached_toks = _g("cached_tokens", None, 0) or 0
    # completion_toks already includes reasoning (subset, not addend) — don't
    # re-add it (double-count). Matches the chat endpoint + the batched path above.
    _record_metrics(prompt_toks, completion_toks)
    content = []
    visible_text = text
    # ALWAYS separate reasoning from the visible text. Native-thinking models
    # (Qwen3/DeepSeek-R1/GLM-Z1) emit CoT by DEFAULT, independent of the request's
    # thinking config — so gating extract_thinking on enable_thinking leaked the raw
    # <think>…</think> + the entire CoT into the visible text block whenever the client
    # didn't pass thinking={type:enabled}. chat.py/responses.py route reasoning
    # unconditionally; Anthropic was the lone outlier. extract_thinking is a no-op when
    # there are no think tags, so non-thinking models are unaffected.
    from ..streaming import extract_thinking

    thinking_text, visible_text = extract_thinking(text, req.model)
    if thinking_text:
        content.append({"type": "thinking", "thinking": thinking_text, "signature": ""})
    text_block: dict = {"type": "text", "text": visible_text}

    # Check for matched stop sequences — trim BEFORE tool call extraction.
    # Some engine paths trim the stop sequence themselves before returning
    # `finish_reason=="stop"`. In that case, recover by reporting the
    # caller's stop_sequences[0] (correct when only one stop was passed,
    # and "<stop_sequence>" as a placeholder otherwise) so `stop_reason`
    # is set truthfully per Anthropic's protocol.
    matched_stop = _detect_matched_stop(
        visible_text,
        stop,
        getattr(result, "finish_reason", None),
        stopped_by_stop_sequence=getattr(result, "stopped_by_stop_sequence", None),
    )
    if matched_stop and visible_text:
        idx = visible_text.find(matched_stop)
        if idx != -1:
            visible_text = visible_text[:idx]
            text_block["text"] = visible_text

    # Include logprobs in the text content block if requested
    if req.logprobs:
        _result_lp = _g("logprobs", None, None)
        if _result_lp:
            formatted_lp = _format_anthropic_logprobs(_result_lp)
            if formatted_lp:
                text_block["logprobs"] = formatted_lp

    content.append(text_block)

    # Extract tool calls from model output if tools were provided
    _suppress_tool_extraction = getattr(req, "_suppress_tools", False)
    has_tool_calls = False
    if _suppress_tool_extraction and req.tools:
        # tool_choice="none" suppresses tool_use emission, but the model may still
        # emit <tool_call> markup — strip it from the visible text so it doesn't leak (the
        # chat router cleans even under "none"; the old Anthropic path skipped cleanup).
        from ..streaming import clean_tool_call_markup

        text_block["text"] = clean_tool_call_markup(visible_text)
    if req.tools and not _suppress_tool_extraction:
        from ..streaming import clean_tool_call_markup, extract_tool_calls_model_aware

        tool_calls = extract_tool_calls_model_aware(visible_text, req.model)
        # enforce a forced/none-parallel tool_choice post-generation (parity with
        # chat's _enforce_tool_choice) — drop wrong-named / surplus calls.
        tool_calls = _enforce_anthropic_tool_choice(tool_calls, req.tool_choice)
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
                    inp = (
                        json.loads(tc["arguments"])
                        if isinstance(tc["arguments"], str)
                        else tc["arguments"]
                    )
                except (json.JSONDecodeError, TypeError):
                    inp = {}
                content.append(
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tc["name"],
                        "input": inp,
                    }
                )

    stop_reason = _map_stop_reason(
        finish_reason, matched_stop, has_tool_calls=has_tool_calls
    )

    # completion_toks already includes reasoning tokens (reasoning_tokens is the
    # detail subset, not an addend) — same fix as _non_stream_batched / chat.py.
    _reasoning_tok = _g("reasoning_tokens", None, 0) or 0
    _legacy_total_output = completion_toks
    _lg_cache_tok = getattr(engine, "_tokenizer", None) or getattr(
        engine, "tokenizer", None
    )
    _lg_input, _lg_creation, _lg_read = _anthropic_cache_usage(
        prompt_toks,
        cached_toks,
        _cacheable_prefix_token_count(
            getattr(req, "_anthropic_orig_system", req.system),
            kv_cache_breakpoints,
            _lg_cache_tok,
        ),
    )
    _legacy_usage: dict[str, Any] = {
        "input_tokens": _lg_input,
        "output_tokens": _legacy_total_output,
        "cache_creation_input_tokens": _lg_creation,
        "cache_read_input_tokens": _lg_read,
    }
    if _reasoning_tok > 0:
        _legacy_usage["output_tokens_details"] = {"reasoning_tokens": _reasoning_tok}

    return JSONResponse(
        {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": req.model,
            "stop_reason": stop_reason,
            "stop_sequence": _public_stop_sequence(matched_stop),
            "usage": _legacy_usage,
            **({"metadata": req.metadata} if req.metadata else {}),
        }
    )


async def _stream_anthropic(
    engine,
    messages,
    req,
    stop,
    request,
    is_batched=False,
    temp_files=None,
    lora_adapter=None,
    kv_cache_breakpoints=None,
) -> AsyncIterator[bytes]:
    """Anthropic SSE streaming with keepalive, disconnect detection, and tool-use deltas."""
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    enable_thinking = req.thinking and req.thinking.get("type") == "enabled"
    budget_tokens = req.thinking.get("budget_tokens") if req.thinking else None
    _logit_bias = _convert_logit_bias(req)
    # tool_choice="none" sets _suppress_tools (see above): the model is told not to
    # call tools and its prompt is emptied of them, so streaming must NOT extract
    # tool_use blocks either — otherwise stray tool-call markup becomes a tool_use
    # block + stop_reason="tool_use", which "none" forbids (non-streaming already
    # honors this via _suppress_tool_extraction).
    has_tools = (
        req.tools is not None
        and len(req.tools) > 0
        and not getattr(req, "_suppress_tools", False)
    )
    # thread the forced tool name + parallel cap so streaming enforces tool_choice
    # like the non-streaming path (parity with chat's ToolCallStreamer wiring).
    _tc = req.tool_choice if isinstance(req.tool_choice, dict) else None
    _forced_name = _tc.get("name") if (_tc and _tc.get("type") == "tool") else None
    _allow_parallel = not (_tc.get("disable_parallel_tool_use") if _tc else False)
    _tool_streamer = (
        ToolCallStreamer(
            model_name=req.model,
            forced_tool_name=_forced_name,
            allow_parallel=_allow_parallel,
        )
        if has_tools
        else None
    )
    block_index = 0
    thinking_block_started = False
    _thinking_block_idx = 0
    text_block_started = False
    _text_block_idx = 0
    tool_use_block_started = False
    _tool_block_idx = 0
    _tc_args_streamed = False  # did we already emit input_json_delta chunks for the
    # current tool block? (avoid re-emitting full args)
    _has_tool_calls = False  # persists across blocks (unlike tool_use_block_started)
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
        Uses the shared faithful accounting (input + creation + read == prompt).
        """
        # pass the REAL cacheable-prefix token count, not bool(breakpoints) (which
        # capped cache_creation_input_tokens at 1). The streaming scope has engine + req +
        # breakpoints, so this can be computed exactly like the non-stream path.
        _ms_tok = getattr(engine, "_tokenizer", None) or getattr(
            engine, "tokenizer", None
        )
        _ms_cacheable = _cacheable_prefix_token_count(
            getattr(req, "_anthropic_orig_system", req.system),
            kv_cache_breakpoints,
            _ms_tok,
        )
        _ms_input, cache_creation, cache_read = _anthropic_cache_usage(
            inp_tokens, cached_toks, _ms_cacheable
        )
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
                "usage": {
                    "input_tokens": _ms_input,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": cache_creation,
                    "cache_read_input_tokens": cache_read,
                },
            },
        }
        return f"event: message_start\ndata: {json.dumps(msg_start)}\n\n".encode()

    # content_block_start for thinking is NOT eagerly emitted — it opens
    # only when the first reasoning token arrives, avoiding empty thinking
    # blocks when the model decides not to think.

    async def _token_source():
        nonlocal input_tokens, output_tokens, block_index, cached_tokens
        nonlocal thinking_block_started, text_block_started, tool_use_block_started
        nonlocal \
            accumulated_text, \
            matched_stop, \
            _message_start_emitted, \
            _streaming_finish_reason, \
            reasoning_tok
        nonlocal _has_tool_calls
        # these were assigned inside _token_source WITHOUT a
        # nonlocal, so they shadowed the enclosing scope's copies (which stayed 0). The
        # *_started flags ARE nonlocal, so the error/MemoryError handlers in the outer
        # scope correctly saw a block was open but emitted content_block_stop with
        # index 0 — closing the wrong/already-closed block and orphaning the real one
        # (e.g. thinking at 0 + text at 1 → handler closed 0, left 1 open). Declaring
        # them nonlocal makes the handlers read the real open-block indices.
        nonlocal _thinking_block_idx, _text_block_idx, _tool_block_idx

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
                min_tokens=req.min_tokens,
                ignore_eos=req.ignore_eos,
                suppress_tokens=req.suppress_tokens,
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
                lora_adapter=lora_adapter,
                kv_cache_breakpoints=kv_cache_breakpoints,
            ):
                # Use engine's current_state (token-level tracking) for
                # thinking routing — more accurate than text-level ThinkingParser
                # which may miss model-specific tags like Qwen3.5's special tokens.
                _is_reasoning = getattr(output, "current_state", None) == "reasoning"
                _token_text = output.new_text

                if output.prompt_tokens and not input_tokens:
                    input_tokens = output.prompt_tokens
                if hasattr(output, "cached_tokens") and output.cached_tokens:
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

                # Thinking content. Route on _is_reasoning ALONE, not
                # `enable_thinking and _is_reasoning`. A native-thinking model emits CoT by
                # default; the old gate let reasoning tokens hit NEITHER branch when
                # the client didn't request thinking (falsy enable_thinking + _is_reasoning
                # true) → the entire CoT was silently DROPPED. Now reasoning always becomes
                # thinking deltas (mirrors the non-stream fix + chat.py).
                if _is_reasoning and _token_text:
                    if not thinking_block_started:
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'thinking', 'thinking': '', 'signature': ''}})}\n\n"
                        _thinking_block_idx = block_index
                        thinking_block_started = True
                    output_tokens += 1
                    reasoning_tok += 1
                    # Use engine's authoritative count when available
                    if hasattr(output, "reasoning_tokens") and output.reasoning_tokens:
                        reasoning_tok = max(reasoning_tok, output.reasoning_tokens)
                        output_tokens = max(
                            output_tokens,
                            reasoning_tok + (output_tokens - reasoning_tok),
                        )
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'thinking_delta', 'thinking': _token_text}})}\n\n"
                elif _token_text and not _is_reasoning:
                    # Visible text content (skip reasoning tokens when thinking is disabled)
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
                            logger.error(
                                "Anthropic streaming text exceeded 1MB — truncating"
                            )
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
                                _text_block_idx = block_index
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
                                    if (
                                        len(accumulated_text)
                                        > _MAX_STREAMING_TEXT_BUFFER
                                    ):
                                        logger.error(
                                            "Anthropic streaming text exceeded 1MB — truncating"
                                        )
                                        accumulated_text = accumulated_text[
                                            -_TRUNCATE_KEEP:
                                        ]
                                    _token_boundaries.append(len(accumulated_text))
                                    # Check for stop sequences after streamer text
                                    # is accumulated (the pre-streamer check at
                                    # line ~1139 operates on empty text when tools
                                    # are active, so we must check here instead).
                                    _stop_hit = False
                                    if stop:
                                        for seq in stop:
                                            if seq in accumulated_text:
                                                accumulated_text = accumulated_text[
                                                    : accumulated_text.find(seq)
                                                ]
                                                matched_stop = seq
                                                _stop_hit = True
                                                break
                                    if _stop_hit:
                                        # Use _token_boundaries to count how many
                                        # tokens are entirely within the trimmed
                                        # suffix — same logic as the non-tools path.
                                        _safe_end = len(accumulated_text)
                                        _tokens_to_trim = 0
                                        for _bi in range(
                                            len(_token_boundaries) - 1, -1, -1
                                        ):
                                            if _token_boundaries[_bi] > _safe_end:
                                                _tokens_to_trim += 1
                                            else:
                                                break
                                        _safe_len = len(accumulated_text) - _prev_len
                                        if _tokens_to_trim == 0 and _safe_len == 0:
                                            _tokens_to_trim = 1
                                        output_tokens = max(
                                            0, output_tokens - _tokens_to_trim
                                        )
                                        if _safe_len > 0:
                                            if not text_block_started:
                                                text_block_started = True
                                                _text_block_idx = block_index
                                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': _tc_out.text[:_safe_len]}})}\n\n"
                                    else:
                                        if not text_block_started:
                                            text_block_started = True
                                            _text_block_idx = block_index
                                            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': _tc_out.text}})}\n\n"
                                elif _tc_out.tool_call_start:
                                    # Incremental: streamer found tool name early.
                                    # Open content_block_start (tool_use) so the
                                    # client sees the tool name immediately and
                                    # can render a spinner before args arrive.
                                    if text_block_started:
                                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                                        block_index += 1
                                        text_block_started = False
                                    if not tool_use_block_started:
                                        tool_use_block_started = True
                                        _tc_args_streamed = False  # fresh tool block
                                        _tool_block_idx = block_index
                                        _has_tool_calls = True
                                        _tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'tool_use', 'id': _tool_id, 'name': _tc_out.tool_call_start.name, 'input': {}}})}\n\n"
                                elif _tc_out.tool_call_args_delta:
                                    # Incremental: streamer is emitting args
                                    # text chunks as the model generates them.
                                    _tc_args_streamed = True
                                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'input_json_delta', 'partial_json': _tc_out.tool_call_args_delta}})}\n\n"
                                elif _tc_out.tool_call:
                                    if not tool_use_block_started:
                                        if text_block_started:
                                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                                            block_index += 1
                                            text_block_started = False
                                        tool_use_block_started = True
                                        _tool_block_idx = block_index
                                        _has_tool_calls = True
                                        _tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'tool_use', 'id': _tool_id, 'name': _tc_out.tool_call.name, 'input': {}}})}\n\n"
                                    # Emit args via input_json_delta loop regardless of
                                    # whether content_block_start was emitted via
                                    # tool_call_start (early-name detection) or via
                                    # tool_call (full complete event). Forced-grammar
                                    # outputs come as one big token so tool_call fires
                                    # AFTER tool_call_start; without this loop the
                                    # streaming response would carry input:{} with zero
                                    # partial_json deltas.
                                    # Only re-emit the full args here when NO incremental
                                    # tool_call_args_delta chunks were already streamed for
                                    # this block (the forced-grammar single-token case);
                                    # otherwise the client would receive the partial_json
                                    # DUPLICATED (incremental deltas + full re-emit).
                                    _args_str = _tc_out.tool_call.arguments or "{}"
                                    if (
                                        _args_str
                                        and _args_str != "{}"
                                        and not _tc_args_streamed
                                    ):
                                        for _ci in range(0, len(_args_str), 8):
                                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'input_json_delta', 'partial_json': _args_str[_ci : _ci + 8]}})}\n\n"
                                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                                    block_index += 1
                                    tool_use_block_started = False
                                    _tc_args_streamed = False
                                    # do NOT break — a single token can carry
                                    # MORE than one complete tool call (forced-grammar /
                                    # one-shot output of <tool_call>…</tool_call><tool_call>…
                                    # </tool_call>). The streamer emits one tool_call output
                                    # per call; breaking after the first DROPPED every
                                    # subsequent call (and any trailing text) in this token.
                                    # Anthropic supports parallel tool_use blocks, and the
                                    # OpenAI chat path has no such break. Continue draining
                                    # this token's remaining outputs (each opens its own
                                    # tool_use block at the incremented block_index).
                        else:
                            # No tools — emit text directly
                            if not text_block_started:
                                text_block_started = True
                                _text_block_idx = block_index
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
                min_tokens=req.min_tokens,
                ignore_eos=req.ignore_eos,
                suppress_tokens=req.suppress_tokens,
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
                lora_adapter=lora_adapter,
                kv_cache_breakpoints=kv_cache_breakpoints,
            ):
                if (
                    hasattr(output, "prompt_tokens")
                    and output.prompt_tokens
                    and not input_tokens
                ):
                    input_tokens = output.prompt_tokens
                if hasattr(output, "cached_tokens") and output.cached_tokens:
                    cached_tokens = max(cached_tokens, output.cached_tokens)

                # Capture finish_reason from the last streaming output.
                # Use finish_reason whenever it's set (not just when finished=True)
                # because some engines set finish_reason without the finished flag.
                if (
                    hasattr(output, "finish_reason")
                    and output.finish_reason is not None
                ):
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
                _is_reasoning = getattr(output, "current_state", None) == "reasoning"
                _token_text = output.token_text

                if _is_reasoning and _token_text:  # route on _is_reasoning alone
                    # Thinking content via token-level state
                    if not thinking_block_started:
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'thinking', 'thinking': '', 'signature': ''}})}\n\n"
                        _thinking_block_idx = block_index
                        thinking_block_started = True
                    output_tokens += 1
                    reasoning_tok += 1
                    # Use engine's authoritative count when available
                    if hasattr(output, "reasoning_tokens") and output.reasoning_tokens:
                        reasoning_tok = max(reasoning_tok, output.reasoning_tokens)
                        output_tokens = max(
                            output_tokens,
                            reasoning_tok + (output_tokens - reasoning_tok),
                        )
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'thinking_delta', 'thinking': _token_text}})}\n\n"
                elif _token_text and not _is_reasoning:
                    # Visible text content (skip reasoning tokens when thinking is disabled)
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
                                    _text_block_idx = block_index
                                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                                _prev_len = len(accumulated_text)
                                accumulated_text += _tc_out.text
                                if len(accumulated_text) > _MAX_STREAMING_TEXT_BUFFER:
                                    logger.error(
                                        "Anthropic streaming text exceeded 1MB — truncating"
                                    )
                                    accumulated_text = accumulated_text[
                                        -_TRUNCATE_KEEP:
                                    ]
                                _token_boundaries.append(len(accumulated_text))
                                _stop_hit = False
                                if stop:
                                    for seq in stop:
                                        if seq in accumulated_text:
                                            accumulated_text = accumulated_text[
                                                : accumulated_text.find(seq)
                                            ]
                                            matched_stop = seq
                                            _stop_hit = True
                                            break
                                if _stop_hit:
                                    _safe_end = len(accumulated_text)
                                    _tokens_to_trim = 0
                                    for _bi in range(
                                        len(_token_boundaries) - 1, -1, -1
                                    ):
                                        if _token_boundaries[_bi] > _safe_end:
                                            _tokens_to_trim += 1
                                        else:
                                            break
                                    _safe_len = len(accumulated_text) - _prev_len
                                    if _tokens_to_trim == 0 and _safe_len == 0:
                                        _tokens_to_trim = 1
                                    output_tokens = max(
                                        0, output_tokens - _tokens_to_trim
                                    )
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
                                _tool_block_idx = block_index
                                _has_tool_calls = True
                                _tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'tool_use', 'id': _tool_id, 'name': _tc_out.tool_call.name, 'input': {}}})}\n\n"
                                _args_str = _tc_out.tool_call.arguments or "{}"
                                for _ci in range(0, len(_args_str), 8):
                                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'input_json_delta', 'partial_json': _args_str[_ci : _ci + 8]}})}\n\n"
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                                block_index += 1
                                tool_use_block_started = False
                                # do NOT break — the no-break fix on the batched path
                                # (above) was never propagated to this DEFAULT (legacy-Engine,
                                # is_batched=False) streaming path. A single token can carry
                                # multiple complete <tool_call>…</tool_call> blocks; breaking
                                # after the first DROPPED every subsequent parallel call (and
                                # any trailing text) in this token. Continue draining the rest.
                    else:
                        # No tool streamer — emit text directly
                        if not text_block_started:
                            text_block_started = True
                            _text_block_idx = block_index
                            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                        output_tokens += 1
                        _prev_len = len(accumulated_text)
                        accumulated_text += _token_text
                        if len(accumulated_text) > _MAX_STREAMING_TEXT_BUFFER:
                            logger.error(
                                "Anthropic streaming text exceeded 1MB — truncating"
                            )
                            accumulated_text = accumulated_text[-_TRUNCATE_KEEP:]
                        _token_boundaries.append(len(accumulated_text))

                        _stop_matched_this_token = False
                        if stop:
                            for seq in stop:
                                if seq in accumulated_text:
                                    accumulated_text = accumulated_text[
                                        : accumulated_text.find(seq)
                                    ]
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
        # This MUST run for BOTH the batched and non-batched branches.
        # In BUFFER_ALL mode (Mistral [TOOL_CALLS], Qwen, GLM block-form) and
        # for tool calls truncated at max_tokens, the streamer HOLDS the whole tool-call
        # buffer during process_token (yields nothing) and only parses + emits the calls
        # at flush(). It was nested inside the non-batched `else`, so the opt-in
        # engine-loop (is_batched=True) path NEVER flushed → tool calls silently dropped
        # and stop_reason fell back to end_turn (un-propagated to the batched
        # sibling). Dedented to run after the if/else — all referenced vars are shared.
        if has_tools and _tool_streamer:
            for _tc_out in _tool_streamer.flush():
                if _tc_out.text:
                    if not text_block_started:
                        text_block_started = True
                        _text_block_idx = block_index
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': _tc_out.text}})}\n\n"
                elif _tc_out.tool_call:
                    if text_block_started:
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                        block_index += 1
                        text_block_started = False
                    tool_use_block_started = True
                    _tool_block_idx = block_index
                    _has_tool_calls = True
                    _tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'tool_use', 'id': _tool_id, 'name': _tc_out.tool_call.name, 'input': {}}})}\n\n"
                    _args_str = _tc_out.tool_call.arguments or "{}"
                    for _ci in range(0, len(_args_str), 8):
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'input_json_delta', 'partial_json': _args_str[_ci : _ci + 8]}})}\n\n"
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                    block_index += 1
                    tool_use_block_started = False

        # If message_start was never emitted (engine produced zero outputs or
        # only outputs without prompt_tokens), emit it now with whatever we have.
        if not _message_start_emitted:
            _message_start_emitted = True
            yield _emit_message_start(input_tokens, cached_tokens)

        # If no content blocks were started at all (zero tokens), emit an empty text block
        # so that the response always has at least one content block (Anthropic protocol requirement).
        # Always use text block, NOT thinking — thinking blocks should only appear when
        # the model actually produces reasoning output.
        if (
            not text_block_started
            and not thinking_block_started
            and not tool_use_block_started
        ):
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            text_block_started = True
            _text_block_idx = block_index

        # Close last content block (only if one was actually started and not
        # already closed — tool_use blocks are closed inside the loop above).
        # also close tool_use block here if it was the LAST block
        # opened and is still pending — otherwise the safety-close at the bottom
        # emits content_block_stop AFTER message_delta, violating Anthropic spec
        # ordering (all content_block_stop must precede message_delta).
        nonlocal_idx = block_index
        if tool_use_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': _tool_block_idx})}\n\n"
            tool_use_block_started = False
        if text_block_started or thinking_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': nonlocal_idx})}\n\n"
            text_block_started = False
            thinking_block_started = False

        # message_delta (stop + usage)
        # Per Anthropic streaming spec, message_delta usage contains output_tokens
        # and optionally output_tokens_details (reasoning_tokens).
        # cache_creation_input_tokens / cache_read_input_tokens are in message_start
        # (emitted deferred above when the first engine output arrives).
        stop_reason = _map_stop_reason(
            _streaming_finish_reason, matched_stop, has_tool_calls=_has_tool_calls
        )
        _delta_usage: dict = {"output_tokens": output_tokens}
        if reasoning_tok > 0:
            _delta_usage["output_tokens_details"] = {"reasoning_tokens": reasoning_tok}
        delta_data = {
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": _public_stop_sequence(matched_stop),
            },
            "usage": _delta_usage,
        }
        yield f"event: message_delta\ndata: {json.dumps(delta_data)}\n\n"

    try:
        async for event in with_sse_keepalive(
            _token_source(),
            http_request=request,
            cancel_event=_anth_gen.cancel_event if _anth_gen else None,
        ):
            yield event.encode("utf-8") if isinstance(event, str) else event

        # Safety close: if content blocks are still open after the generator
        # completed (e.g. early exit due to max_tokens or generator GC),
        # close them before emitting message_stop. Under normal flow,
        # _token_source already closed all blocks — this is a defensive
        # duplicate that produces a no-op in that case because Anthropic
        # clients tolerate duplicate content_block_stop events gracefully.
        if tool_use_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': _tool_block_idx})}\n\n".encode()
        if text_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': _text_block_idx})}\n\n".encode()
        if thinking_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': _thinking_block_idx})}\n\n".encode()

        # Only emit message_stop on normal completion, NOT after errors
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n".encode()
    except MemoryError:
        # Emit message_start if it was never sent (error before first engine output)
        if not _message_start_emitted:
            _message_start_emitted = True
            yield _emit_message_start(input_tokens, cached_tokens)
        # If no content blocks were opened, emit an empty text block (Anthropic
        # spec requires at least one content block in every message).
        if (
            not text_block_started
            and not thinking_block_started
            and not tool_use_block_started
        ):
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n".encode()
            text_block_started = True
            _text_block_idx = block_index
        # Close ALL open blocks — the Anthropic spec requires every
        # content_block_start to have a matching content_block_stop.
        if tool_use_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': _tool_block_idx})}\n\n".encode()
        if text_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': _text_block_idx})}\n\n".encode()
        if thinking_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': _thinking_block_idx})}\n\n".encode()
        # Emit message_delta with stop_reason before message_stop (Anthropic protocol requirement)
        _error_stop_reason = _map_stop_reason(
            _streaming_finish_reason, None, has_tool_calls=_has_tool_calls
        )
        _error_delta_usage: dict = {"output_tokens": output_tokens}
        if reasoning_tok > 0:
            _error_delta_usage["output_tokens_details"] = {
                "reasoning_tokens": reasoning_tok
            }
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': _error_stop_reason, 'stop_sequence': None}, 'usage': _error_delta_usage})}\n\n".encode()
        error_event = {
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Out of GPU memory"},
        }
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode()
    except Exception:
        logger.error("Anthropic streaming error", exc_info=True)
        # Emit message_start if it was never sent (error before first engine output)
        if not _message_start_emitted:
            _message_start_emitted = True
            yield _emit_message_start(input_tokens, cached_tokens)
        # If no content blocks were opened, emit an empty text block (Anthropic
        # spec requires at least one content block in every message).
        if (
            not text_block_started
            and not thinking_block_started
            and not tool_use_block_started
        ):
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n".encode()
            text_block_started = True
            _text_block_idx = block_index
        # Close ALL open blocks — same logic as MemoryError handler.
        if tool_use_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': _tool_block_idx})}\n\n".encode()
        if text_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': _text_block_idx})}\n\n".encode()
        if thinking_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': _thinking_block_idx})}\n\n".encode()
        # Emit message_delta with stop_reason before message_stop (Anthropic protocol requirement)
        _exc_stop_reason = _map_stop_reason(
            _streaming_finish_reason, None, has_tool_calls=_has_tool_calls
        )
        _exc_delta_usage: dict = {"output_tokens": output_tokens}
        if reasoning_tok > 0:
            _exc_delta_usage["output_tokens_details"] = {
                "reasoning_tokens": reasoning_tok
            }
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': _exc_stop_reason, 'stop_sequence': None}, 'usage': _exc_delta_usage})}\n\n".encode()
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": "Internal server error"},
        }
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode()
    finally:
        _release_lora_adapter(engine, lora_adapter)
        if _anth_tracker is not None:
            try:
                _anth_tracker.unregister(message_id)
            except Exception:
                logger.debug("tracker unregister failed", exc_info=True)
        # Clean up temp files created for image blocks during streaming
        if temp_files:
            import os as _os

            for _tf_path in temp_files:
                with contextlib.suppress(OSError):
                    _os.unlink(_tf_path)
        # output_tokens already includes reasoning tokens (incremented once per
        # thinking token at line ~1163), so do NOT add reasoning_tok again.
        _record_metrics(input_tokens, output_tokens)


def _format_anthropic_logprobs(logprobs_list: list[dict] | None) -> list[dict] | None:
    """Format per-token logprobs from GenerationOutput into Anthropic Messages format.

    Anthropic returns logprobs as an array in each text content block:
    [{"token": str, "logprob": float, "top_logprobs": [{"token": str, "logprob": float}, ...]}]

    Returns None if no valid logprobs entries are found.
    """
    # Type/length guard (not truthiness): a raw mx.array would raise on `not x`.
    if not isinstance(logprobs_list, (list, tuple)) or len(logprobs_list) == 0:
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
        entries.append(
            {
                "token": lp_entry.get("token", ""),
                "logprob": lp_entry.get("logprob", 0.0),
                "top_logprobs": decoded_top,
            }
        )
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
            content={
                "type": "error",
                "error": {
                    "type": "not_found_error" if e.status_code == 404 else "api_error",
                    "message": e.detail,
                },
            },
        )

    tokenizer = getattr(engine, "_tokenizer", None)
    if tokenizer is None:
        return JSONResponse(
            status_code=503,
            content={
                "type": "error",
                "error": {
                    "type": "overloaded_error",
                    "message": "No tokenizer available",
                },
            },
        )

    # Apply chat template for accurate token counting (plain-text join undercounts
    # by missing role markers, special tokens, and generation prompt).
    messages = []
    # mirror create_message's system-lift. Generation
    # LIFTS any role="system" entries out of messages[] and MERGES them with the top-level
    # `system` into ONE canonical system block. count_tokens previously counted only the
    # top-level system here and left the lifted ones in the converted list below, rendering
    # a SECOND <|system|> wrapper per lifted message → an overcount vs the real prompt.
    # Collect them all now (top-level first, then in-message order) and drop them from the
    # converted messages below.
    _sys_parts: list[str] = []
    # include the cached_content prefix that generation prepends into the system
    # prompt (anthropic create_message), so input_tokens isn't undercounted by the entire
    # cached prefix. Same resolver as generation; mutate=False (estimation must not bump the
    # cache's usage). Cached text comes FIRST, matching generation's prepend order.
    if req.cached_content:
        try:
            _cc_text = _resolve_cached_content_text(
                req.cached_content, req.model, request, mutate=False
            )
            if _cc_text:
                _sys_parts.append(_cc_text)
        except HTTPException as _cc_e:
            return JSONResponse(
                status_code=_cc_e.status_code,
                content={
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": _cc_e.detail},
                },
            )
        except Exception:
            logger.debug("count_tokens cached_content resolve failed", exc_info=True)
    if req.system:
        _sys_parts.append(
            _extract_text_from_content(req.system)
            if isinstance(req.system, list)
            else req.system
        )
    for _m in req.messages:
        if getattr(_m, "role", None) == "system":
            _c = _m.content
            _sys_parts.append(
                _extract_text_from_content(_c) if isinstance(_c, list) else (_c or "")
            )
    _sys_parts = [p for p in _sys_parts if p]
    if _sys_parts:
        messages.append({"role": "system", "content": "\n\n".join(_sys_parts)})

    # Include tool definitions in the system prompt so that the token count
    # matches what the actual /messages endpoint would send. Mirrors the
    # tool-prompt injection in create_message (lines 532-552).
    if req.tools:
        tool_prompt = (
            "\n\nYou have access to the following tools. When you need to call a tool, "
        )
        tool_prompt += 'output a tool call in the following format:\n<tool_call\\>{"name": "...", "arguments": {...}}</tool_call\\>\n\n'
        tool_prompt += "Available tools:\n"
        for tool in req.tools:
            tool_prompt += f"- {tool.name}"
            if tool.description:
                tool_prompt += f": {tool.description}"
            if tool.input_schema:
                tool_prompt += f"\n  Parameters: {tool.input_schema}"
            tool_prompt += "\n"
        # mirror create_message's EXACT enforcement templates so
        # count_tokens doesn't under-report for tool_choice=any/tool (the old short text
        # was ~20-60 tokens shorter than what generation actually injects).
        if req.tool_choice:
            _tc_tool_names = ", ".join(
                n
                for n in (
                    (
                        getattr(t, "name", None)
                        or (t.get("name") if isinstance(t, dict) else None)
                    )
                    for t in (req.tools or [])
                )
                if n
            )
            if isinstance(req.tool_choice, dict):
                tc_type = req.tool_choice.get("type", "")
                if tc_type == "none":
                    tool_prompt = ""
                elif tc_type == "any":
                    tool_prompt += (
                        f"\nCRITICAL: The caller set tool_choice=any. You MUST invoke "
                        f"EXACTLY ONE of the following tools regardless of what the user said: {_tc_tool_names}. "
                        "Do NOT respond with text-only content. The FIRST tokens of your reply MUST be `<tool_call>`. "
                        "Format exactly (no prose before or after):\n"
                        '<tool_call>{"name": "<tool>", "arguments": {<args>}}</tool_call>\n'
                    )
                elif tc_type == "tool":
                    forced = req.tool_choice.get("name")
                    if forced:
                        tool_prompt += (
                            f"\nYou MUST call the tool '{forced}'. Emit ONLY a tool call, "
                            f"no prose. Format exactly:\n"
                            f'<tool_call>{{"name": "{forced}", "arguments": {{<args>}}}}</tool_call>\n'
                        )
            elif req.tool_choice == "any":
                tool_prompt += (
                    "\nYou MUST call at least one tool. Do NOT respond with only text. "
                    "Format exactly:\n"
                    '<tool_call>{"name": "<tool>", "arguments": {<args>}}</tool_call>\n'
                )
            elif req.tool_choice == "none":
                tool_prompt = ""
        if tool_prompt:
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] += tool_prompt
            else:
                messages.insert(0, {"role": "system", "content": tool_prompt.strip()})

    # Mirror create_message's thinking-instruction injection (lines 751-767) so
    # input_tokens reflects what /messages actually sends for thinking-enabled
    # requests — it was previously omitted, under-counting by the instruction block.
    # .
    if (
        req.thinking
        and isinstance(req.thinking, dict)
        and req.thinking.get("type") == "enabled"
    ):
        _budget = req.thinking.get("budget_tokens")
        _think_instruction = (
            "\n\nIMPORTANT FORMAT REQUIREMENT: You MUST begin every reply with "
            "an opening <think> tag, write your private step-by-step reasoning, "
            "then write a closing </think> tag, and only after that produce the "
            "final user-facing answer. Do not skip these tags. "
            "Example shape: <think>...reasoning...</think>final answer."
        )
        if isinstance(_budget, int) and _budget > 0:
            _think_instruction += f" Keep the reasoning inside <think>...</think> to roughly {_budget} tokens."
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = (
                messages[0].get("content") or ""
            ) + _think_instruction
        else:
            messages.insert(
                0, {"role": "system", "content": _think_instruction.strip()}
            )

    # Use the same conversion as /messages for accurate token counting.
    # _extract_text_from_content flattens tool_use/tool_result to text which
    # undercounts tokens compared to the actual OpenAI-format messages that
    # the /messages endpoint sends to the engine.
    has_images = any(_has_image_blocks(m.content) for m in req.messages)
    converted_msgs, _ct_temp_files = _convert_anthropic_messages(
        req.messages,
        has_images=has_images,
    )
    # the role="system" entries were lifted into the canonical system block
    # above (mirroring generation) — drop them here so the system wrapper isn't counted
    # twice.
    messages.extend(m for m in converted_msgs if m.get("role") != "system")
    try:
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback: if chat template fails (e.g. missing template), use plain join.
            # coerce non-str (image/multimodal LIST) content to text first —
            # str.join over a list-valued content raised an uncaught TypeError → 500 instead
            # of a count for any image request whose text template rejected list content.
            text_parts = [
                m["content"]
                if isinstance(m["content"], str)
                else _extract_text_from_content(m["content"])
                for m in messages
            ]
            prompt = "\n".join(text_parts)
        # avoid double-BOS (Gemma/Llama/Mistral) so count_tokens
        # matches the generation path's prompt_tokens, which now applies the same guard.
        _bos = getattr(tokenizer, "bos_token", None)
        _add = not (
            isinstance(_bos, str)
            and _bos
            and isinstance(prompt, str)
            and prompt.startswith(_bos)
        )
        try:
            tokens = tokenizer.encode(prompt, add_special_tokens=_add)
        except TypeError:
            tokens = tokenizer.encode(prompt)

        # ADD the per-image token estimate. The text tokenizer renders an
        # image_url block to ~0 tokens, but the real /messages path routes to the VLM engine
        # and bills _estimate_image_tokens() per image — so count_tokens grossly undercounted
        # (e.g. ~20 for a 4-image prompt the engine bills at ~2.3k), defeating client
        # budgeting. _ct_temp_files holds exactly one temp file per converted image. 576 ==
        # IMAGE_TOKEN_ESTIMATE (token_counter) == the VLM engine's conservative default.
        _img_tokens = len(_ct_temp_files) * 576
        # Anthropic's count_tokens response is just `{"input_tokens": N}` —
        # do not add a `type` discriminator (clients that strictly validate
        # against Anthropic's schema reject unknown keys).
        return {"input_tokens": len(tokens) + _img_tokens}
    finally:
        # _convert_anthropic_messages writes each base64 image to a NamedTemporaryFile
        # (delete=False); count_tokens never sends them to an engine, so unlink them
        # or every count_tokens-with-image call orphans /tmp files (the /messages path
        # cleans these up in its finally — this path previously discarded the list).
        if _ct_temp_files:
            import os as _ct_os

            for _ct_path in _ct_temp_files:
                with contextlib.suppress(OSError):
                    _ct_os.unlink(_ct_path)
