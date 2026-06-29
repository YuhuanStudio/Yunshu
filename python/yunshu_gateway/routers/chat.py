from __future__ import annotations

"""OpenAI Chat Completions compatible router.

Supports:
- Text chat completions (LLM mode)
- Vision chat completions with image input (VLM mode)
- Streaming and non-streaming
- enable_thinking parameter for reasoning models
- Tool calling with extraction from model output
- Context window validation
- SSE keepalive + disconnect guard
- Full OpenAI message format (text, image_url, content arrays)
"""

import contextlib
import copy
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from yunshu_engine.tool_call_streamer import ToolCallStreamer

from ..engine import get_engine, get_model_manager
from ..streaming import (
    clean_tool_call_markup,
    extract_thinking,
    extract_tool_calls_model_aware,
    format_openai_chunk,
    format_openai_done,
    format_openai_non_stream,
    format_openai_usage_chunk,
    run_with_disconnect_guard,
    validate_context_window,
    validate_prefill_memory,
    with_sse_keepalive,
)
from .models import _check_permission

logger = logging.getLogger(__name__)

_MAX_STREAMING_TEXT_BUFFER = 1 * 1024 * 1024  # 1MB safety limit
_TRUNCATE_KEEP = 512 * 1024  # Keep last 512KB for stop-sequence detection


def _validate_sampling_params(temperature: float, max_tokens: int, top_p: float) -> None:
    """Validate sampling parameters that the Pydantic Field constraints may not fully catch.

    Returns normally if valid, raises HTTPException(422) if invalid.

    Note: max_tokens=0 is allowed here (OpenAI API returns prompt_tokens only).
    The gateway handles this as a fast-path returning an empty completion.
    """
    if temperature < 0 or temperature > 2:
        raise HTTPException(
            status_code=422,
            detail=f"temperature must be in [0, 2], got {temperature}",
        )
    if max_tokens < 0 or max_tokens > 131072:
        raise HTTPException(
            status_code=422,
            detail=f"max_tokens must be in [0, 131072], got {max_tokens}",
        )
    # Pydantic Field declares `ge=0.0, le=1.0` (both inclusive)
    # — was inconsistent with this check which rejected 0.0. Accept the full
    # [0, 1] range to match Pydantic and OpenAI behavior. Filter NaN/Inf.
    import math
    if math.isnan(top_p) or math.isinf(top_p) or top_p < 0 or top_p > 1:
        raise HTTPException(
            status_code=422,
            detail=f"top_p must be in [0, 1], got {top_p}",
        )


def _generate_tool_call_id() -> str:
    """Generate a consistent tool call ID in OpenAI format.

    Uses format: call_{uuid_hex[:24]} — consistent across all code paths
    (LLM, VLM, multi-choice, streaming).
    """
    return f"call_{uuid.uuid4().hex[:24]}"


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
    results: list[dict] = []
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
    """Record token counts to metrics middleware and tracing counters.

    Does NOT call ServerMetrics.record_request_complete — the engine
    already records it (batched_engine.py fast / streaming / engine.py
    non-batched) with the FULL detail (cached_tokens, prefill/generation durations,
    model_id). The old duplicate call here (prompt/completion only, model_id="",
    durations=0) DOUBLED total_requests/prompt/completion in the billing-feeding totals and
    halved cache_efficiency_pct / throughput (doubled tokens, single-counted durations).
    """
    try:
        from ..middleware.metrics import get_metrics
        get_metrics().record_tokens(prompt_tokens, completion_tokens)
        get_metrics().record_inference()
    except Exception:
        logger.debug("metrics recording failed", exc_info=True)
    try:
        from yunshu_engine.tracing import get_metrics_v2
        get_metrics_v2().counter("yunshu_tokens_total", {"type": "prompt"}, prompt_tokens)
        get_metrics_v2().counter("yunshu_tokens_total", {"type": "completion"}, completion_tokens)
    except Exception:
        logger.debug("metrics recording failed", exc_info=True)
    # Attribute this request's tokens to the per-request box so the auth
    # middleware can enforce the tenant's tokens_per_minute quota at settle.
    try:
        from ..usage_context import record_billed_tokens
        record_billed_tokens((prompt_tokens or 0) + (completion_tokens or 0))
    except Exception:
        logger.debug("billed-token accounting failed", exc_info=True)


def _apply_lora_adapter(engine, adapter_id: str | None) -> str | None:
    """Apply a LoRA adapter to the engine for this request.

    Returns the adapter_id if loaded, None if not applicable.
    The caller must call _release_lora_adapter() after generation.

    Uses acquire_adapter/release_adapter (ref-counted) instead of
    load_adapter/unload_adapter to prevent concurrent-request eviction.
    """
    if not adapter_id:
        return None
    # A requested adapter that doesn't exist must be a hard error — NEVER silently
    # fall through to base-model output as a 200 OK (the caller asked for a fine-tuned
    # model and would have no way to know they got the wrong weights). Both paths below
    # raise HTTPException(404) when the adapter isn't registered/acquirable.
    lora_mgr = getattr(engine, 'get_lora_manager', lambda: None)()
    # LoRA concurrency keystone: for engines that self-manage LoRA (BatchedEngine),
    # do NOT acquire/apply here on the event loop — the engine acquires+applies inside its
    # executor closure, serialized with generation, so the model is never mutated off-thread
    # while another request generates. Just validate existence and pass the id through.
    # VLM/legacy engines (no flag) acquire here.
    if getattr(engine, '_self_manages_lora', False):
        # Validate the adapter is registered up-front so a typo/unknown id 404s instead of
        # silently generating on the base model inside the executor closure.
        if lora_mgr is not None and not lora_mgr.is_registered(adapter_id):
            raise HTTPException(
                status_code=404,
                detail=f"LoRA adapter '{adapter_id}' is not registered",
            )
        return adapter_id
    if lora_mgr is None:
        raise HTTPException(
            status_code=400,
            detail=f"LoRA adapter '{adapter_id}' requested but this engine has no LoRA manager",
        )
    if lora_mgr.acquire_adapter(adapter_id):
        return adapter_id
    raise HTTPException(
        status_code=404,
        detail=f"LoRA adapter '{adapter_id}' could not be acquired (not registered?)",
    )


def _release_lora_adapter(engine, adapter_id: str | None) -> None:
    """Release a LoRA adapter after generation completes.

    Uses release_adapter (decrements ref count) instead of unload_adapter.
    The adapter stays loaded for reuse until LRU eviction or explicit unload.
    """
    if not adapter_id:
        return
    # Self-managing engines release inside their executor closure (keystone).
    if getattr(engine, '_self_manages_lora', False):
        return
    lora_mgr = getattr(engine, 'get_lora_manager', lambda: None)()
    if lora_mgr is not None:
        lora_mgr.release_adapter(adapter_id)


# ── Request / Response schemas (OpenAI-compatible) ──


class TextContent(BaseModel):
    type: str = "text"
    text: str


class ImageURL(BaseModel):
    url: str


class ImageContent(BaseModel):
    type: str = "image_url"
    image_url: ImageURL


ContentPart = TextContent | ImageContent | dict


class ToolCallFunction(BaseModel):
    """Function call within a tool_call."""
    name: str
    arguments: str = ""


class ToolCall(BaseModel):
    """OpenAI tool_call structure in assistant messages."""
    id: str = ""
    type: str = "function"
    function: ToolCallFunction = ToolCallFunction(name="")


class ChatMessage(BaseModel):
    role: str
    content: str | list[ContentPart] | None = None
    # Tool call fields for multi-turn conversations (OpenAI spec)
    tool_calls: list[ToolCall] | None = None       # assistant messages with tool calls
    tool_call_id: str | None = None                 # tool role messages (result of a tool call)
    name: str | None = None                          # tool role messages (function name)

    @model_validator(mode="after")
    def validate_message(self):
        _VALID_ROLES = {"system", "user", "assistant", "tool", "function", "developer"}
        if self.role not in _VALID_ROLES:
            raise ValueError(f"messages: invalid role '{self.role}'. Must be one of: {', '.join(sorted(_VALID_ROLES))}")
        # Tool role messages must have tool_call_id
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("messages: tool role messages must have 'tool_call_id'")
        # For user/system/developer roles, reject explicitly-empty content
        # (empty string or empty list). None is permitted because downstream
        # extraction normalizes it to an empty string; only explicit empty
        # content is a malformed request.
        if self.role in ("user", "system", "developer"):
            if isinstance(self.content, str) and self.content != "" and not self.content.strip():
                raise ValueError(f"messages: {self.role} message content cannot be whitespace-only")
            if isinstance(self.content, list) and len(self.content) == 0:
                raise ValueError(f"messages: {self.role} message content list cannot be empty")
        return self


class ToolFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: dict | None = None


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
    logit_bias: dict[int, float] | None = None
    max_tokens: int = Field(default=512, ge=0, le=131072)
    max_completion_tokens: int | None = Field(default=None, ge=0, le=131072)
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: list[str] | None = None
    enable_thinking: bool | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: str | ToolChoiceFunction | None = None
    parallel_tool_calls: bool = True
    response_format: dict | None = None
    seed: int | None = None
    logprobs: bool = False
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    n: int = Field(default=1, ge=1, le=128)
    user: str | None = None
    # Advanced engine parameters
    spec_decode: bool = False
    thinking_budget: int | None = Field(default=None, ge=1, le=32768)
    reasoning_effort: str | None = None
    stop_token_ids: list[int] | None = None
    priority: int = Field(default=0, ge=0, le=100)
    xtc_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    xtc_threshold: float = Field(default=0.0, ge=0.0, le=0.5)  # engine requires [0,0.5]; le=1.0 made out-of-range 500 not 422
    # Serving parity:
    min_tokens: int = Field(default=0, ge=0)  # floor on generated tokens (mask EOS until reached)
    ignore_eos: bool = False  # keep generating past EOS to max_tokens
    suppress_tokens: list[int] | None = None  # hard-ban these token ids from output
    # per-prompt-token logprobs (eval/perplexity), parity with completions.
    prompt_logprobs: int | None = None
    grammar: dict | None = None  # {"type": "json", "schema": {...}} or {"type": "regex", "pattern": "..."}
    # vLLM/SGLang guided-decoding aliases. Clients/SDKs targeting those
    # engines send these named params; map them onto Yunshu's existing
    # grammar/json_schema plumbing (the constraint capability is already built).
    # SECURITY: cap length so a giant user regex/grammar can't drive
    # pathological NFA/parser construction (DoS) — the engine also bounds NFA size.
    guided_regex: str | None = Field(default=None, max_length=2048)
    guided_choice: list[str] | None = None
    guided_grammar: str | None = Field(default=None, max_length=32768)  # EBNF/Lark CFG
    guided_json: dict | None = None  # JSON schema
    lora_adapter: str | None = None  # LoRA adapter ID to apply for this request
    cached_content: str | None = None  # Gemini-style explicit context-cache handle to prepend (read)
    logits_processors: list | None = None  # User-provided custom logits processors
    timeout: float | None = Field(default=None, ge=1.0, le=600.0)  # Request timeout in seconds
    # vLLM/OpenAI-style chat-template overrides. Clients commonly send
    # {"enable_thinking": false} here (vLLM convention) — accept it and fold a
    # recognized key into the top-level field so it isn't silently ignored.
    chat_template_kwargs: dict | None = None

    @model_validator(mode="after")
    def validate_request(self):
        if not self.model or not self.model.strip():
            raise ValueError("model: field is required and cannot be empty")
        # Honor chat_template_kwargs.enable_thinking (vLLM convention)
        # when the top-level enable_thinking wasn't explicitly set, instead of
        # dropping it silently.
        if self.enable_thinking is None and isinstance(self.chat_template_kwargs, dict):
            _et = self.chat_template_kwargs.get("enable_thinking")
            if isinstance(_et, bool):
                self.enable_thinking = _et
        # Fold vLLM/SGLang guided_* aliases into the existing grammar /
        # response_format plumbing (only when the native field isn't already set).
        if self.grammar is None:
            if self.guided_regex is not None:
                self.grammar = {"type": "regex", "pattern": self.guided_regex}
            elif self.guided_choice:
                self.grammar = {"type": "choice", "choices": self.guided_choice}
            elif self.guided_grammar is not None:
                self.grammar = {"type": "cfg", "grammar": self.guided_grammar}
        if self.guided_json is not None and self.response_format is None:
            self.response_format = {"type": "json_schema",
                                    "json_schema": {"schema": self.guided_json}}
        if not self.messages:
            raise ValueError("messages: field is required and cannot be empty")
        # Reject system-only messages (no user message to respond to)
        _user_roles = {"user"}
        _user_msgs = [m for m in self.messages if m.role in _user_roles]
        if not _user_msgs:
            raise ValueError(
                "messages: must contain at least one message with role 'user'"
            )
        # Reject if EVERY user message has empty content (str that
        # is empty/whitespace). List content (multimodal w/ images) is allowed
        # even with empty text. Prevents the degenerate "nothing to respond to"
        # request that produces zero-token output.
        def _is_empty_content(c) -> bool:
            if c is None:
                return True
            if isinstance(c, str):
                return not c.strip()
            return False  # list content (multimodal) is non-empty by structure
        if all(_is_empty_content(m.content) for m in _user_msgs):
            raise ValueError(
                "messages: at least one user message must have non-empty content"
            )
        if self.stop and len(self.stop) > 16:
            raise ValueError("stop: maximum 16 stop sequences")
        if self.stop_token_ids and len(self.stop_token_ids) > 16:
            raise ValueError("stop_token_ids: maximum 16 stop token IDs")
        # Validate logit_bias values are within OpenAI's documented range
        if self.logit_bias:
            import math
            for k, v in self.logit_bias.items():
                # bool is subclass of int → `isinstance(True, (int,
                # float))` accepts. Explicitly exclude bool so {"50256": true}
                # is rejected (OpenAI spec requires numeric).
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError(f"logit_bias[{k}]: must be a finite number")
                if math.isnan(v) or math.isinf(v):
                    raise ValueError(f"logit_bias[{k}]: must be a finite number")
                if v < -100.0 or v > 100.0:
                    raise ValueError(
                        f"logit_bias[{k}]={v}: must be between -100 and 100"
                    )
        # Bound seed to 64-bit signed range (downstream samplers may overflow)
        if self.seed is not None and (self.seed < -(2**63) or self.seed >= 2**63):
            raise ValueError("seed: must fit within signed 64-bit integer range")
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
        # Validate logprobs/top_logprobs consistency
        if self.logprobs and self.top_logprobs is None:
            pass  # OK, top_logprobs defaults to None which is valid
        if not self.logprobs and self.top_logprobs is not None and self.top_logprobs > 0:
            raise ValueError("top_logprobs requires logprobs=true")
        # n > 1 with streaming is not supported (OpenAI returns error for this)
        if self.stream and self.n > 1:
            raise ValueError(
                "n > 1 is not supported when stream is True. "
                "Use non-streaming mode for multiple choices."
            )
        return self

    def effective_max_tokens(self) -> int:
        """Return max_completion_tokens if set, else max_tokens (OpenAI SDK compat)."""
        return self.max_completion_tokens if self.max_completion_tokens is not None else self.max_tokens


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
            # OpenAI strict mode: when strict=true, force additionalProperties=false
            # on the root schema so the constrained decoder rejects any keys not
            # declared in properties. We mutate a copy to avoid leaking changes
            # back to the request body.
            if js.get("strict") is True and isinstance(schema, dict):
                schema = copy.deepcopy(schema)
                if schema.get("type") == "object" or "properties" in schema:
                    schema["additionalProperties"] = False
            return schema
        return "json_object"

    return None


def _prepend_cached_content(messages: list[dict], cached_content: str | None,
                            req_model: str | None = None,
                            request: Request | None = None) -> list[dict]:
    """Gemini-style READ: prepend an explicit context-cache handle's stored
    messages so the automatic KVPrefixCache serves the warmed prefix. No-op if
    the handle is unset/expired (the request still runs, just without reuse)."""
    if not cached_content:
        return messages
    entry = None
    try:
        from ..explicit_cache import get_store
        entry = get_store().use(cached_content if cached_content.startswith("cachedContents/")
                                else f"cachedContents/{cached_content}")
    except Exception:
        logger.debug("cached_content lookup failed", exc_info=True)
        return messages
    if entry is None or not entry.messages:
        return messages
    # SECURITY (cross-tenant IDOR): the management routes check ownership but
    # this READ/consumption path didn't — so tenant B could prepend tenant A's private
    # cached context (a proprietary system prompt/document) into its own generation just by
    # guessing the handle, then exfiltrate it via the model output. Enforce ownership here
    # too: a handle owned by someone else is treated as not-found (ignored).
    if request is not None:
        _owner = getattr(entry, "owner", None)
        if _owner and _owner != "anonymous":
            try:
                from yunshu_control.audit_log import resolve_actor
                if resolve_actor(request) != _owner:
                    logger.warning("cached_content '%s' owned by another tenant — ignoring",
                                   cached_content)
                    return messages
            except Exception:
                logger.debug("cached_content ownership check failed — ignoring handle",
                             exc_info=True)
                return messages
    # A cached_content handle is model-specific — it was created against one
    # model's tokenizer and warmed into that model's KV prefix cache. Silently reusing it
    # with a DIFFERENT model gets zero KV reuse and injects cross-tokenizer text. Reject the
    # mismatch with a clear 400 instead of producing wrong/unwarmed output.
    if req_model and getattr(entry, 'model', None) and entry.model != req_model:
        raise HTTPException(
            status_code=400,
            detail=(f"cached_content '{cached_content}' was created for model "
                    f"'{entry.model}' and cannot be used with '{req_model}'"),
        )
    return list(entry.messages) + list(messages)


def _normalize_image_part(part: dict) -> dict:
    """Normalize OpenAI image content parts to the canonical ``image_url`` shape.

    The gateway accepts three image content types
    (image_url / image / image_data) and routes any of them to the VLM vision
    path, but the engine's _extract_images / _build_vlm_messages handle ONLY
    ``image_url``. So an ``image`` or ``image_data`` part passed _has_images()
    (→ routed to the VLM) but was then silently dropped → the model answered
    about an image it never saw (hallucination, the same failure class fixed for
    the file:// branch). Convert the other two types here so the engine sees a
    consistent format.
    """
    if not isinstance(part, dict):
        return part
    # The OpenAI BARE-STRING image_url variant
    # ({"type":"image_url","image_url":"https://…"}) is valid, but the engine's
    # _extract_images does part["image_url"].get("url") → AttributeError → 500 on a
    # str. An earlier fix coerced this only in _check_has_media (the SSRF pre-check), NOT here
    # on the path that feeds the engine, so the inference path still 500'd. Coerce the
    # bare string to the canonical {"url": …} object so the engine sees one shape.
    if part.get("type") == "image_url" and isinstance(part.get("image_url"), str):
        return {"type": "image_url", "image_url": {"url": part["image_url"]}}
    if part.get("type") not in ("image", "image_data"):
        return part
    url = part.get("url", "")
    iu = part.get("image_url")
    if not url and isinstance(iu, dict):
        url = iu.get("url", "")
    elif not url and isinstance(iu, str):
        url = iu
    if not url:
        url = part.get("data", "")
    if not url:
        # An image/image_data part with NO usable url/image_url/data was
        # still routed to the VLM (it has an image type, so _has_images → True), then
        # silently dropped by _extract_images (which only matches image_url) → the
        # model hallucinates / image order shifts. Fail loud instead (mirrors the
        # fail-loud invariant for unloadable media).
        raise ValueError(f"{part.get('type')} content part has no image url or data")
    # A bare base64 blob (no scheme) → wrap as a data URL the loader can decode.
    if not url.lower().startswith(("data:", "http://", "https://", "file://")):
        url = "data:image/png;base64," + url
    return {"type": "image_url", "image_url": {"url": url}}


def _extract_messages(msgs: list[ChatMessage]) -> list[dict]:
    """Convert ChatMessage objects to dicts, preserving multimodal content.

    Conversions:
    - String content -> {"role": ..., "content": "..."}
    - List content with image_url -> {"role": ..., "content": [{type: "text", ...}, {type: "image_url", ...}]}
    - Tool call fields (tool_calls, tool_call_id, name) preserved for multi-turn conversations
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
                    parts.append(_normalize_image_part(part))
                elif hasattr(part, "model_dump"):
                    parts.append(_normalize_image_part(part.model_dump()))
                elif isinstance(part, TextContent):
                    parts.append({"type": "text", "text": part.text})
                elif isinstance(part, ImageContent):
                    parts.append({"type": "image_url", "image_url": {"url": part.image_url.url}})
                else:
                    parts.append({"type": "text", "text": str(part)})
            # Collapse text-only content lists to a plain string so
            # downstream chat-template apply doesn't render `[{...}]` as
            # Python-repr text. Multi-modal (any image_url part) keeps the
            # list form so VLMEngine sees the structure intact.
            _has_non_text = any(
                isinstance(p, dict) and p.get("type") not in (None, "text")
                for p in parts
            )
            if parts and not _has_non_text:
                d["content"] = "\n".join(p.get("text", "") for p in parts if isinstance(p, dict))
            else:
                d["content"] = parts
        else:
            d["content"] = str(msg.content)
        # Preserve tool call fields for multi-turn conversations (OpenAI spec)
        if msg.tool_calls is not None:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        if msg.tool_call_id is not None:
            d["tool_call_id"] = msg.tool_call_id
        if msg.name is not None:
            d["name"] = msg.name
        result.append(d)
    return result


def _has_images(messages: list[dict]) -> bool:
    """Check if any message contains image content.

    Handles all OpenAI image content types:
    - image_url (with url field — http/https/data URLs)
    - image (with image_url or data field — some models)
    - image_data (base64 inline)
    """
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    ptype = part.get("type", "")
                    if ptype in ("image_url", "image", "image_data"):
                        # Validate URL scheme + host for image_url to prevent SSRF.
                        # Some SDKs send the bare-string form
                        # {"image_url": "https://…"} (not the {"url": …} object). The old
                        # `.get("image_url", {}).get("url")` then called .get on a str →
                        # AttributeError → opaque 500. Coerce the string form first.
                        _iu = part.get("image_url", {})
                        if isinstance(_iu, str):
                            _iu = {"url": _iu}
                        url = part.get("url", "") or (_iu.get("url", "") if isinstance(_iu, dict) else "")
                        if url and not _is_safe_image_url(url):
                            url_low = url.lower()
                            _safe = ("http://", "https://", "data:")
                            if not any(url_low.startswith(s) for s in _safe):
                                _msg = f"Image URL must use http, https, or data scheme, got '{url[:80]}'"
                            else:
                                _msg = f"Image URL blocked by SSRF protection (private/internal host): '{url[:80]}'"
                            raise HTTPException(status_code=400, detail=_msg)
                        return True
    return False


def _is_safe_image_url(url: str) -> bool:
    """Check if an image URL uses a safe scheme and does not target internal hosts.

    Blocks SSRF attacks by rejecting:
    - Non-http/https/data schemes
    - Private IPs (10.x, 172.16-31.x, 192.168.x)
    - Loopback (127.x, localhost)
    - Link-local (169.254.x, fe80::)
    - Cloud metadata (169.254.169.254)

    NB: uses urllib.parse instead of a custom regex so port-bearing
    URLs like http://127.0.0.1:8000/ are also classified correctly
    (the earlier regex greedily captured "127.0.0.1:8000" as the IP
    and then failed ipaddress.ip_address(), silently allowing the URL).
    """
    if not url:
        return True
    _SAFE_SCHEMES = ("http", "https", "data")
    url_stripped = url.strip()
    url_lower = url_stripped.lower()
    if url_lower.startswith("data:"):
        return True  # data: URLs are inline, no network fetch
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url_stripped)
    except Exception:
        return False
    if parsed.scheme.lower() not in _SAFE_SCHEMES:
        return False
    # urlparse handles IPv6 brackets, port stripping, userinfo, etc.
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return False
    # Block obvious internal hostnames
    if hostname in ("localhost", "localhost.localdomain", "ip6-localhost"):
        return False
    # IPv6 link-local hostnames carry a "%zone" suffix
    hostname_for_ip = hostname.split("%", 1)[0]
    import ipaddress
    try:
        ip = ipaddress.ip_address(hostname_for_ip)
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return False
    except ValueError:
        pass  # hostname, not IP — allow
    return True


def _extract_image_refs(messages: list[dict]) -> list[dict] | None:
    """Extract lightweight image references from messages.

    Returns raw image reference dicts (URLs or base64 data) without
    downloading, so the scheduler and dedup layers can hash them.
    """
    refs = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image_url", "image", "image_data"):
                    refs.append(part)
    return refs if refs else None


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
    tool_choice: str | ToolChoiceFunction | None = None,
    parallel_tool_calls: bool = True,
) -> list[dict]:
    """Inject tool definitions into the system prompt.

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
    # Note: emit explicit `<tool_call>{...}</tool_call>` tags + JSON shape so
    # the gateway parser (extract_tool_calls_v2 + Mistral/ChatML/Qwen
    # patterns) reliably picks them up. Without explicit format guidance,
    # models sometimes emit `{"tool_call":{"name":..,"arguments":..}}` or
    # raw `{"name":..}` JSON that older parser versions missed.
    _tool_format_example = (
        '\nFormat: <tool_call>{"name": "<tool_name>", "arguments": {<args_json>}}</tool_call>'
    )
    if isinstance(tool_choice, ToolChoiceFunction):
        forced_name = tool_choice.function.name
        tool_prompt += (
            f"\nYou MUST call the tool '{forced_name}'. "
            f"Do not respond with text — only output a tool call."
            f"{_tool_format_example}\n"
        )
    elif tool_choice == "auto" or tool_choice is None:
        tool_prompt += (
            "\nDecide whether to call a tool based on the user's request. "
            "If you can answer directly, do so. If you need a tool, use it."
            f"{_tool_format_example}\n"
        )
    elif tool_choice == "required":
        tool_prompt += (
            "\nYou MUST call at least one of the provided tools. "
            "Do NOT respond with only text — use a tool."
            f"{_tool_format_example}\n"
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


def _tool_choice_prefill(tool_choice: str | ToolChoiceFunction | None) -> str:
    """Return the assistant-turn PREFILL that structurally commits the model to a
    tool call for a forced tool_choice, or "" when no prefill applies.

    `_inject_tool_system_prompt` only ADVISES the model ("You MUST call the tool…"),
    which it can ignore by emitting plain text — and "required"/named-forced then can't
    be honored post-hoc (we can't fabricate tool arguments). Prefilling the opening
    tool-call marker onto the assistant turn (via continue_final_message in the engine's
    chat template) makes the model CONTINUE a tool call at generation time instead.
    Uses the documented `<tool_call>{"name": …, "arguments": …}</tool_call>` format that
    `_inject_tool_system_prompt` already instructs the model to emit.

    - "required"               → '<tool_call>\\n'  (model completes name + arguments)
    - {"name": "X"} (forced)   → '<tool_call>\\n{"name": "X", "arguments": {'  (args only)
    - "auto" / "none" / None    → ""  (no prefill — unchanged behavior)
    """
    if isinstance(tool_choice, ToolChoiceFunction):
        # json.dumps the name so an exotic tool name can't break the JSON shape.
        return '<tool_call>\n{"name": ' + json.dumps(tool_choice.function.name) + ', "arguments": {'
    if tool_choice == "required":
        return "<tool_call>\n"
    return ""


def _append_tool_prefill(messages: list[dict], prefill: str) -> list[dict]:
    """Append the tool-call PREFILL onto a trailing assistant turn so the engine's
    chat template keeps the assistant turn OPEN (continue_final_message) and the model
    generates a continuation of the tool-call markup. No-op when prefill is empty.

    If the last message is already an assistant string turn (e.g. a client-supplied
    prefill), the marker is appended to it; otherwise a fresh assistant turn is added.
    Returns a new list — the input is not mutated."""
    if not prefill:
        return messages
    messages = list(messages)
    last = messages[-1] if messages else None
    if (last is not None and last.get("role") == "assistant"
            and isinstance(last.get("content"), str)):
        last = dict(last)
        last["content"] = last["content"] + prefill
        messages[-1] = last
    else:
        messages.append({"role": "assistant", "content": prefill})
    return messages


def _enforce_tool_choice(tool_calls, tool_choice, parallel_tool_calls):
    """Post-generation enforcement of the OpenAI tool_choice / parallel_tool_calls
    contract. The system-prompt injection (`_inject_tool_system_prompt`) only *advises*
    the model — a model can ignore it. This applies the hard guarantees we can enforce
    deterministically on the extracted call list:

    - tool_choice == "none"  → never surface tool calls (suppress entirely).
    - named function choice   → drop any call whose name != the forced name.
    - parallel_tool_calls == False → keep at most the first call.

    Returns the (possibly empty / possibly None) adjusted list, preserving the input's
    None-vs-[] convention so callers can keep their existing `if tool_calls:` guards.
    The "required"/named-but-zero-calls case is NOT forced here (we cannot fabricate a
    call the model never produced); it is signalled separately by the caller."""
    if not tool_calls:
        return tool_calls
    if tool_choice == "none":
        return [] if isinstance(tool_calls, list) else None
    if isinstance(tool_choice, ToolChoiceFunction):
        forced = tool_choice.function.name
        tool_calls = [tc for tc in tool_calls if tc.get("name") == forced]
    if parallel_tool_calls is False and len(tool_calls) > 1:
        tool_calls = tool_calls[:1]
    return tool_calls


def _lp_bytes(entry: dict, decoded: str, tokenizer) -> list[int]:
    """The OpenAI logprobs `bytes` field must be the token's RAW UTF-8 bytes
    so clients can reassemble a multi-byte char (CJK/emoji) that byte-level BPE split
    across tokens. `decoded.encode("utf-8")` is WRONG for a split fragment (decode() of a
    lone fragment is U+FFFD → its bytes are the replacement char, losing the real bytes).
    Prefer the engine-provided raw bytes; else recover from the token id; else fall back
    to the decoded string (correct for whole, non-split tokens)."""
    b = entry.get("bytes")
    if isinstance(b, list):
        return b
    if tokenizer is not None and "token_id" in entry:
        try:
            from yunshu_engine.text_utils import token_id_to_bytes
            return token_id_to_bytes(tokenizer, entry["token_id"], decoded)
        except Exception:
            pass
    return list(decoded.encode("utf-8")) if decoded else []


def _format_logprobs(
    raw_logprobs: Any,
    tokenizer: Any,
    top_logprobs: int | None = None,
) -> dict | None:
    """Format logprobs from engine output into OpenAI Chat Completions format.

    OpenAI returns logprobs as:
    {"content": [{"token": "...", "logprob": -1.23, "top_logprobs": [{"token": "...", "logprob": -0.5}, ...]}]}
    """
    if raw_logprobs is None:
        return None
    # The engine-loop path can hand back a raw mx.array (not the per-token
    # dict list this formatter expects). `not <multi-element array>` raises
    # ValueError, so guard on type/length instead of truthiness, and skip
    # non-list formats safely (no logprobs rather than a 500).
    if not isinstance(raw_logprobs, (list, tuple)):
        return None
    if len(raw_logprobs) == 0:
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
                        token_str = str(lp["token_id"])
                # Decode top_logprobs with bytes field
                raw_top = lp.get("top_logprobs", [])
                if top_logprobs is not None and top_logprobs > 0:
                    raw_top = raw_top[:top_logprobs]
                decoded_top = []
                for tlp in raw_top:
                    if isinstance(tlp, dict):
                        tlp_token = tlp.get("token", "")
                        if not tlp_token and tokenizer and "token_id" in tlp:
                            with contextlib.suppress(Exception):
                                tlp_token = tokenizer.decode([tlp["token_id"]])
                        decoded_top.append({
                            "token": tlp_token,
                            "logprob": tlp.get("logprob", 0.0),
                            "bytes": _lp_bytes(tlp, tlp_token, tokenizer),
                        })
                    else:
                        decoded_top.append(tlp)
                entries.append({
                    "token": token_str,
                    "logprob": lp.get("logprob", 0.0),
                    "bytes": _lp_bytes(lp, token_str, tokenizer),
                    "top_logprobs": decoded_top,
                })
            elif isinstance(lp, (int, float)) and not isinstance(lp, bool):
                entries.append({
                    "token": "",
                    "logprob": float(lp),
                    "bytes": [],
                    "top_logprobs": [],
                })

    if not entries:
        return None

    return {"content": entries}


def _per_choice_seed(user_seed: int | None, idx: int) -> int:
    """Per-choice seed for n>1 sampling.

    When the caller provides a seed, derive a stable per-choice seed (`seed + idx`)
    so n>1 with the same seed is reproducible across runs. When the caller
    does NOT provide a seed, manufacture a fresh time-based seed for each
    choice — otherwise all n iterations would share `seed=None`, and the
    mlx-lm `categorical_sampling` `@mx.compile` cache traps the first call's
    PRNG state, producing identical token streams for choices 2..n. This is
    the same family of bug seen for VLM determinism.
    """
    if user_seed is not None:
        return (int(user_seed) + int(idx)) & ((1 << 63) - 1)
    import time as _t
    return (_t.time_ns() + int(idx) * 1_000_003) & ((1 << 63) - 1)


def _normalize_finish_reason(reason: str | None) -> str:
    """Normalize engine finish_reason to OpenAI-compatible values.

    The engine may produce internal finish reasons (abort, cancel, error,
    timeout, memory_limit) that are not valid OpenAI finish reasons.
    Map them to the closest OpenAI equivalent.
    """
    if not reason:
        return "stop"
    # Valid OpenAI finish reasons — pass through
    if reason in ("stop", "length", "tool_calls", "content_filter"):
        return reason
    # Internal reasons → OpenAI equivalents
    _INTERNAL_MAP = {
        "abort": "stop",
        "cancel": "stop",
        "error": "stop",
        # A timeout truncated the output mid-generation, so signal it as
        # "length" (a valid OpenAI value meaning "cut short") rather than "stop" — the
        # latter is indistinguishable from natural completion, so clients (and eval
        # harnesses) couldn't tell a timed-out partial response from a finished one.
        # Consistent with memory_limit/memory_exceeded, which already map to "length".
        "timeout": "length",
        "memory_limit": "length",
        "memory_exceeded": "length",
    }
    return _INTERNAL_MAP.get(reason, "stop")


async def _build_multi_choice(
    engine, req, messages, completion_id, is_batched, json_schema,
    cancel_event=None,
    lora_adapter=None,
    tool_prefill: str = "",
):
    """Build n > 1 completions by running parallel generation calls.

    ``tool_prefill`` is the prefill-forced tool_choice marker that was prepended to
    the prompt's assistant turn; each choice prepends it back onto its generated text
    before tool-call parsing (see the n=1 path)."""

    prompt_tok = 0
    completion_tok = 0
    reasoning_tok = 0
    cached_tok = 0
    choices = []

    async def _gen_one(idx: int):
        # Init before the branch — only the batched branch assigned it,
        # so n>1 on a non-batched (legacy) text engine hit UnboundLocalError at the
        # shared read below → every choice failed → 500. (Mirrors the n=1 path's
        # _n1_prompt_lp = None init.)
        _prompt_lp = None
        if is_batched:
            result = await engine.chat(
                messages=messages,
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
                seed=_per_choice_seed(req.seed, idx),
                enable_thinking=req.enable_thinking,
                json_schema=json_schema,
                spec_decode=req.spec_decode,
                thinking_budget=req.thinking_budget,
                stop_token_ids=req.stop_token_ids,
                reasoning_effort=req.reasoning_effort,
                xtc_probability=req.xtc_probability,
                xtc_threshold=req.xtc_threshold,
                priority=req.priority,
                logprobs=req.logprobs,
                top_logprobs=req.top_logprobs,
                logits_processors=req.logits_processors,
                lora_adapter=lora_adapter,
                cancel_event=cancel_event,
                timeout_seconds=req.timeout,
                min_tokens=req.min_tokens,
                ignore_eos=req.ignore_eos,
                suppress_tokens=req.suppress_tokens,
                prompt_logprobs=req.prompt_logprobs,
            )
            text = result.text
            pt = result.prompt_tokens
            ct = result.completion_tokens
            fr = _normalize_finish_reason(result.finish_reason)
            _prompt_lp = getattr(result, 'prompt_logprobs', None)
            lp = _format_logprobs(
                getattr(result, 'logprobs', None),
                getattr(engine, '_tokenizer', None),
                req.top_logprobs,
            )
        else:
            state = await engine.generate(
                prompt=messages,
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
                seed=_per_choice_seed(req.seed, idx),
                enable_thinking=req.enable_thinking,
                stop_token_ids=req.stop_token_ids,
                thinking_budget=req.thinking_budget,
                reasoning_effort=req.reasoning_effort,
                xtc_probability=req.xtc_probability,
                xtc_threshold=req.xtc_threshold,
                spec_decode=req.spec_decode,
                json_schema=json_schema,
                priority=req.priority,
                logprobs=req.logprobs,
                top_logprobs=req.top_logprobs,
                logits_processors=req.logits_processors,
                cancel_event=cancel_event,
                timeout_seconds=req.timeout,
                lora_adapter=lora_adapter,
            )
            text = state.generated_text
            pt = state.prompt_token_count
            ct = state.completion_token_count
            fr = _normalize_finish_reason(state.finish_reason)
            lp = _format_logprobs(
                getattr(state, 'logprobs', None),
                getattr(engine, '_tokenizer', None),
                req.top_logprobs,
            )
        # Stop-sequence overcount correction
        if req.stop and fr == "stop":
            for _seq in req.stop:
                if _seq and _seq in text:
                    _corrected = text[:text.find(_seq)]
                    _tok = getattr(engine, '_tokenizer', None)
                    if _tok:
                        try:
                            _cc = len(_tok.encode(_corrected))
                            if _cc < ct:
                                ct = _cc
                        except Exception:
                            pass
                    break

        # Prefill-forced tool_choice: prepend the prefilled marker back (see n=1 path)
        # so the parser sees complete `<tool_call>…` markup.
        if tool_prefill:
            text = tool_prefill + text

        thinking_content, regular_content = extract_thinking(text, req.model)
        cleaned = regular_content.strip()

        tool_calls = []
        if req.tools:
            _raw_calls = extract_tool_calls_model_aware(regular_content, req.model)
            tool_calls = _enforce_tool_choice(_raw_calls, req.tool_choice, req.parallel_tool_calls)
            # Clean markup whenever any was parsed (see n=1 path) — a suppressed
            # wrong-named tool's raw markup must not leak into content. Also clean when a
            # prefill was applied so the prefilled marker never leaks into content.
            if _raw_calls or tool_prefill:
                cleaned = clean_tool_call_markup(regular_content)
            if tool_calls:
                fr = "tool_calls"

        message = {"role": "assistant", "content": cleaned}
        if thinking_content:
            message["reasoning_content"] = thinking_content
        if tool_calls:
            from ..streaming import _sanitize_arguments
            message["tool_calls"] = [
                {"id": _generate_tool_call_id(), "type": "function", "function": {"name": tc["name"], "arguments": _sanitize_arguments(tc.get("arguments", {}))}}
                for i, tc in enumerate(tool_calls)
            ]

        _gen_result = result if is_batched else state
        # Both branches always execute exactly one, so result or state is always
        # set. However, the ternary reads from the *closure* scope which is
        # safe here — the if/else branches are guaranteed to execute.
        _rt = getattr(_gen_result, 'reasoning_tokens', 0) if _gen_result is not None else 0
        # Reconcile the reasoning_tokens detail with the extracted reasoning: when
        # the message carries reasoning_content but the engine's reasoning parser
        # didn't count it (model-specific format mismatch, e.g. a model that
        # doesn't natively emit <think>), derive the count from the thinking text
        # so usage matches the message. Capped at ct — reasoning is a subset of
        # the generated tokens, never an addend. Mirrors responses.py.
        if thinking_content and _rt == 0:
            _tok = getattr(engine, '_tokenizer', None)
            if _tok is not None:
                try:
                    _rt = min(len(_tok.encode(thinking_content)), ct)
                except Exception:
                    _rt = 0
        _ct_cached = getattr(_gen_result, 'cached_tokens', 0) if _gen_result is not None else 0
        choice = {"index": idx, "message": message, "finish_reason": fr}
        if lp:
            choice["logprobs"] = lp
        if _prompt_lp is not None:  # prompt_logprobs (eval/perplexity)
            choice["prompt_logprobs"] = _prompt_lp
        return idx, pt, ct, _rt, _ct_cached, choice

    # Run n>1 choices SEQUENTIALLY, not concurrently.
    # The MLX executor is single-threaded (max_workers=1), so concurrent
    # asyncio.gather just queues them anyway — but the interleaving corrupts
    # the model's KV cache state between choices, causing "Either input_embeddings
    # or prompt must be provided" errors and empty text on choice[1+].
    results = []
    for _i in range(req.n):
        try:
            results.append(await _gen_one(_i))
        except Exception as exc:
            # fix: was `except BaseException` which swallowed
            # CancelledError → client disconnect kept generating tokens.
            results.append(exc)

    errors = []
    for i, r in enumerate(results):
        if isinstance(r, BaseException):
            errors.append((i, r))
            logger.error(f"choice {i} failed: {r}", exc_info=r)
            continue
        idx, pt, ct, _rt, _ct_cached, choice = r
        choices.append(choice)
        prompt_tok = max(prompt_tok, pt)
        completion_tok += ct
        reasoning_tok += _rt
        cached_tok = max(cached_tok, _ct_cached)

    if not choices and errors:
        first_exc = errors[0][1]
        if isinstance(first_exc, MemoryError):
            return JSONResponse(
                status_code=507,
                content={"error": {"message": "Out of GPU memory", "type": "memory_error"}},
            )
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Internal server error", "type": "internal_error"}},
        )

    # Record metrics once for the entire n>1 request (not per-choice)
    if prompt_tok > 0 or completion_tok > 0 or reasoning_tok > 0:
        _record_metrics(prompt_tok, completion_tok)  # completion_tok already incl. reasoning

    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tok,
        # completion_tok (engine n_tok) ALREADY includes reasoning tokens;
        # reasoning_tok is the detail subset, not an addend (was double-counting).
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
    _check_permission(request, "can_infer")
    _validate_sampling_params(req.temperature, req.effective_max_tokens(), req.top_p)
    _rbac_key = getattr(request.state, "rbac_key", None)
    if _rbac_key is not None and not _rbac_key.can_access_model(req.model):
        raise HTTPException(status_code=403, detail=f"Model '{req.model}' not accessible with this API key")

    # Fast path: max_tokens=0 returns prompt_tokens only (OpenAI API behavior).
    # Estimate prompt tokens from tokenizer if available.
    _effective_mt = req.effective_max_tokens()
    if _effective_mt == 0:
        messages = _prepend_cached_content(_extract_messages(req.messages), req.cached_content, req.model, request)
        # Inject the tool system prompt BEFORE counting, so this prompt_tokens probe
        # matches what a real max_tokens>0 call reports (it injects at ~line 1241).
        # Without this, the documented "send max_tokens:0 to get prompt_tokens" probe
        # under-counts by the whole tool-prompt size when tools are present. .
        if req.tools:
            try:
                messages = _inject_tool_system_prompt(
                    messages, req.tools, req.tool_choice, req.parallel_tool_calls)
            except Exception:
                logger.debug("max_tokens=0 tool-prompt injection failed", exc_info=True)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        prompt_tok = 0
        # Resolve tokenizer in priority order: single-engine, then multi-model manager
        _tokenizer = None
        engine = get_engine()
        if engine and getattr(engine, 'is_loaded', False):
            _tokenizer = getattr(engine, '_tokenizer', None)
        if _tokenizer is None:
            try:
                manager = get_model_manager()
                if manager is not None:
                    entry = manager.get_entry(req.model)
                    if entry is not None:
                        if entry.is_loaded and entry.engine is not None:
                            _tokenizer = getattr(entry.engine, '_tokenizer', None)
                        # Fallback: load tokenizer-only via HF transformers
                        if _tokenizer is None:
                            try:
                                # SECURITY: gate trust_remote_code (RCE from a
                                # model dir's *.py) behind an explicit opt-in, default OFF.
                                _trc = os.environ.get("YUNSHU_TRUST_REMOTE_CODE", "").lower() in ("1", "true", "yes")
                                from transformers import AutoTokenizer
                                _tokenizer = AutoTokenizer.from_pretrained(
                                    entry.model_path, trust_remote_code=_trc
                                )
                            except Exception as _err:
                                logger.debug(f"max_tokens=0 fallback tokenizer load failed: {_err}")
            except Exception:
                pass
        if _tokenizer is not None:
            # Prefer chat template encoding (matches what generate() with
            # max_tokens>0 would report). Fall back to the heuristic
            # count_message_tokens if the tokenizer can't apply a template.
            try:
                templated = _tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                if isinstance(templated, str):
                    # avoid double-BOS so this probe matches the
                    # generation path's prompt_tokens (which now applies the same guard).
                    _bos = getattr(_tokenizer, "bos_token", None)
                    _add = not (isinstance(_bos, str) and _bos and templated.startswith(_bos))
                    try:
                        prompt_tok = len(_tokenizer.encode(templated, add_special_tokens=_add))
                    except TypeError:
                        prompt_tok = len(_tokenizer.encode(templated))
            except Exception:
                logger.debug("max_tokens=0 chat-template encode failed", exc_info=True)
            if prompt_tok == 0:
                try:
                    from yunshu_control.token_counter import count_message_tokens
                    prompt_tok = count_message_tokens(messages, _tokenizer)
                except Exception:
                    pass
        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "length",
            }],
            "usage": {
                "prompt_tokens": prompt_tok,
                "completion_tokens": 0,
                "total_tokens": prompt_tok,
            },
        })
    # Validate stop strings: reject empty strings (would match immediately)
    if req.stop:
        req.stop = [s for s in req.stop if s]
        if not req.stop:
            req.stop = None

    # Audit: log user field if provided (OpenAI spec: end-user tracking)
    if req.user:
        logger.info(f"[{getattr(request.state, 'request_id', '-')}] user={req.user}")

    # Structured tracing + logging
    from yunshu_engine.tracing import get_inference_tracer, get_structured_logger
    tracer = get_inference_tracer()
    slog = get_structured_logger()

    trace_id = f"chat-{uuid.uuid4().hex[:16]}"
    tracer.start_trace(trace_id, metadata={
        "model": req.model,
        "max_tokens": req.effective_max_tokens(),
        "temperature": req.temperature,
        "stream": req.stream,
        "endpoint": "/chat/completions",
    })
    tracer.span(trace_id, "prefill", {"model": req.model})
    slog.info("inference_request", model=req.model, trace_id=trace_id,
              max_tokens=req.effective_max_tokens(), stream=req.stream)

    messages = _prepend_cached_content(_extract_messages(req.messages), req.cached_content, req.model, request)
    has_images = _has_images(messages)
    has_audio = _has_audio(messages)
    has_video = _has_video(messages)

    # Route to VLM/Omni engine if images, audio, or video are present
    if has_images or has_audio or has_video:
        json_schema = _parse_response_format(req.response_format, req.grammar)
        return await _handle_vlm_chat(req, messages, request, json_schema=json_schema)

    # Check if the target model is a VLM/Omni (route through VLM handler)
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
        except (KeyError, Exception):
            raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found") from None

    # Inject tool definitions if provided
    if req.tools:
        messages = _inject_tool_system_prompt(messages, req.tools, req.tool_choice, req.parallel_tool_calls)

    # Parse response_format for structured output (JSON schema)
    json_schema = _parse_response_format(req.response_format, req.grammar)

    # Context window validation ()
    # Estimate prompt tokens for validation before generation
    try:
        tokenizer = getattr(engine, '_tokenizer', None)
        if tokenizer is not None:
            from yunshu_control.token_counter import (
                IMAGE_TOKEN_ESTIMATE,
                count_message_tokens,
            )

            from ..streaming import get_max_context_window, get_max_prefill_tokens

            # Cheap char-based UPPER bound first: a BPE token is always ≥1 char, so
            # token_count ≤ char_count. Encoding the full prompt with the real tokenizer
            # runs synchronously on the event loop and inflates TTFT (and scales with
            # prompt length), so only pay for the exact count when the cheap upper bound
            # is actually near a limit. The vast majority of prompts sit far below the
            # context window and are provably safe without any encode.
            char_upper = 0
            image_count = 0
            for m in messages:
                content = m.get("content", "")
                if isinstance(content, str):
                    char_upper += len(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                char_upper += len(block.get("text", ""))
                            elif block.get("type") in ("image_url", "image", "image_data"):
                                image_count += 1
            char_upper += image_count * IMAGE_TOKEN_ESTIMATE
            _max_ctx = get_max_context_window(req.model, engine) or 0
            _cap = get_max_prefill_tokens()
            # Provably safe if the upper bound clears both limits with 10% headroom
            # (covers chat-template framing tokens the char count doesn't see).
            _ctx_safe = (_max_ctx == 0) or (char_upper < _max_ctx * 0.9)
            _cap_safe = char_upper < _cap * 0.9
            if not (_ctx_safe and _cap_safe):
                # Near a limit — compute the precise count and enforce both guards.
                # count_message_tokens already adds IMAGE_TOKEN_ESTIMATE
                # per image block, so the old `+= image_count * IMAGE_TOKEN_ESTIMATE`
                # here double-counted images — a multi-image prompt just under the
                # window could be spuriously rejected. (The VLM sibling at ~1775
                # already does this correctly.)
                est_tokens = count_message_tokens(messages, tokenizer)
                validate_context_window(est_tokens, req.model, engine)
                # Reject huge under-window prompts that would OOM-crash the prefill
                # (uncatchable process death) before we attempt them.
                validate_prefill_memory(est_tokens)
    except HTTPException:
        raise
    except Exception:
        logger.debug("context window validation failed", exc_info=True)

    # Check if this is a BatchedEngine ()
    from yunshu_engine.batched_engine import BatchedEngine
    is_batched = isinstance(engine, BatchedEngine)

    # Prefill-forced tool_choice: for "required" / named-forced choices, prefill the
    # assistant turn with the opening tool-call marker so the model is STRUCTURALLY
    # committed to emitting a tool call (vs. the advisory-only system prompt, which a
    # model can ignore). Only the BatchedEngine chat template honors a trailing-assistant
    # prefill (continue_final_message); the legacy Engine always opens a fresh turn, so the
    # prefill wouldn't take — gate on is_batched and fall back to advise-only there. The
    # marker lives in the PROMPT, so the engine's generated text is only the continuation;
    # each path below prepends `_tool_prefill` back (or seeds the streamer with it) before
    # tool-call parsing so the parser sees complete `<tool_call>…` markup.
    _tool_prefill = ""
    if req.tools and is_batched:
        _tool_prefill = _tool_choice_prefill(req.tool_choice)
        if _tool_prefill:
            messages = _append_tool_prefill(messages, _tool_prefill)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if req.stream:
        if req.n > 1:
            return StreamingResponse(
                _stream_response_multi(
                    engine, messages, req, completion_id, request,
                    is_batched=is_batched, json_schema=json_schema,
                    tool_prefill=_tool_prefill,
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
                tool_prefill=_tool_prefill,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming with disconnect guard ()
    # Register with request tracker for cancellation support
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

    async def _build_response():
        loaded_adapter = _apply_lora_adapter(engine, req.lora_adapter)
        try:
            if req.n > 1:
                return await _build_multi_choice(
                    engine, req, messages, completion_id, is_batched, json_schema,
                    cancel_event=_ns_cancel_event,
                    lora_adapter=loaded_adapter,
                    tool_prefill=_tool_prefill,
                )

            try:
                _reasoning_tok = 0
                _cached_tok = 0
                _n1_prompt_lp = None  # prompt_logprobs for the n=1 path
                if is_batched:
                    result = await engine.chat(
                        messages=messages,
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
                        seed=req.seed,
                        enable_thinking=req.enable_thinking,
                        json_schema=json_schema,
                        logprobs=req.logprobs,
                        top_logprobs=req.top_logprobs,
                        spec_decode=req.spec_decode,
                        thinking_budget=req.thinking_budget,
                        stop_token_ids=req.stop_token_ids,
                        reasoning_effort=req.reasoning_effort,
                        xtc_probability=req.xtc_probability,
                        xtc_threshold=req.xtc_threshold,
                        priority=req.priority,
                        logits_processors=req.logits_processors,
                        cancel_event=_ns_cancel_event,
                        timeout_seconds=req.timeout,
                        lora_adapter=loaded_adapter,
                        min_tokens=req.min_tokens,
                        ignore_eos=req.ignore_eos,
                        suppress_tokens=req.suppress_tokens,
                        prompt_logprobs=req.prompt_logprobs,
                    )
                    raw_text = result.text
                    _n1_prompt_lp = getattr(result, 'prompt_logprobs', None)
                    prompt_tok = result.prompt_tokens
                    completion_tok = result.completion_tokens
                    finish = _normalize_finish_reason(result.finish_reason)
                    _reasoning_tok = getattr(result, 'reasoning_tokens', 0)
                    _cached_tok = getattr(result, 'cached_tokens', 0)
                    logprobs_data = _format_logprobs(
                        getattr(result, 'logprobs', None),
                        getattr(engine, '_tokenizer', None),
                        req.top_logprobs,
                    )
                else:
                    state = await engine.generate(
                        prompt=messages,
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
                        logits_processors=req.logits_processors,
                        cancel_event=_ns_cancel_event,
                        timeout_seconds=req.timeout,
                        lora_adapter=loaded_adapter,
                    )
                    raw_text = state.generated_text
                    prompt_tok = state.prompt_token_count
                    completion_tok = state.completion_token_count
                    finish = _normalize_finish_reason(state.finish_reason)
                    _reasoning_tok = getattr(state, 'reasoning_tokens', 0)
                    _cached_tok = getattr(state, 'cached_tokens', 0)
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
            except Exception:
                logger.error("engine inference failed", exc_info=True)
                return JSONResponse(
                    status_code=500,
                    content={"error": {"message": "Internal server error", "type": "internal_error"}},
                )

            # Stop-sequence overcount correction
            if req.stop and finish == "stop":
                for _seq in req.stop:
                    if _seq and _seq in raw_text:
                        _corrected = raw_text[:raw_text.find(_seq)]
                        _tok = getattr(engine, '_tokenizer', None)
                        if _tok:
                            try:
                                _cc = len(_tok.encode(_corrected))
                                if _cc < completion_tok:
                                    completion_tok = _cc
                            except Exception:
                                pass
                        break

            # Prefill-forced tool_choice: the opening tool-call marker was prefilled into
            # the PROMPT (continue_final_message), so raw_text is only the CONTINUATION.
            # Prepend the marker back so the parser sees complete `<tool_call>…` markup.
            if _tool_prefill:
                raw_text = _tool_prefill + raw_text

            # Extract thinking ()
            thinking_content, regular_content = extract_thinking(raw_text, req.model)

            # Reconcile reasoning_tokens with the extracted reasoning: when the
            # response carries reasoning_content but the engine's reasoning parser
            # didn't count it (model-specific format mismatch), derive the count
            # so the usage detail matches the message. Capped at completion_tok —
            # reasoning is a subset of the generated tokens, never an addend.
            if thinking_content and not _reasoning_tok:
                _rtok = getattr(engine, '_tokenizer', None)
                if _rtok is not None:
                    try:
                        _reasoning_tok = min(len(_rtok.encode(thinking_content)), completion_tok)
                    except Exception:
                        _reasoning_tok = 0

            # Extract tool calls using model-aware format detection (C15)
            tool_calls = []
            cleaned_content = regular_content
            if req.tools:
                _raw_calls = extract_tool_calls_model_aware(regular_content, req.model)
                tool_calls = _enforce_tool_choice(_raw_calls, req.tool_choice, req.parallel_tool_calls)
                # strip tool-call markup whenever ANY was parsed, not only when a
                # call SURVIVES enforcement. With a named/forced tool_choice the model may
                # emit markup for a DIFFERENT (suppressed) tool; gating cleanup on the
                # post-enforcement count left that raw <tool_call> markup in user-visible
                # content — a leak vs the streaming path, which drops suppressed calls.
                # Also clean when a prefill was applied (the prefilled <tool_call> marker
                # must not leak into content even if the model went off-script and no call
                # parsed).
                if _raw_calls or _tool_prefill:
                    cleaned_content = clean_tool_call_markup(regular_content)

            finish_reason = "tool_calls" if tool_calls else finish

            # Execute MCP tool calls if any (server-side tool execution)
            mcp_results = []
            if tool_calls:
                try:
                    mcp_results = await _try_execute_mcp_tools(tool_calls, request)
                except Exception:
                    logger.debug("MCP tool execution failed", exc_info=True)

            # completion_tok (engine count) already includes reasoning.
            _record_metrics(prompt_tok, completion_tok)

            # End tracing
            tracer.end_span(trace_id, "prefill")
            tracer.end_trace(trace_id, result={
                "prompt_tokens": prompt_tok,
                # completion_tok (engine n_tok) already includes reasoning tokens;
                # adding _reasoning_tok again double-counts (and diverged from the
                # user-facing usage below, which correctly uses completion_tok).
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
                # completion_tok already includes reasoning; reasoning attached
                # separately as completion_tokens_details below (was double-counted).
                completion_tokens=completion_tok,
                finish_reason=finish_reason,
                thinking_content=thinking_content if thinking_content else None,
                tool_calls=tool_calls if tool_calls else None,
                logprobs=logprobs_data,
            )

            # emit prompt_logprobs on the n=1 choice (the multi-choice
            # path already emits it; format_openai_non_stream doesn't know about it).
            if _n1_prompt_lp is not None:
                with contextlib.suppress(KeyError, IndexError, TypeError):
                    response_body["choices"][0]["prompt_logprobs"] = _n1_prompt_lp

            # Attach reasoning_tokens and cached_tokens to usage
            if _reasoning_tok:
                response_body.setdefault("usage", {})["completion_tokens_details"] = {
                    "reasoning_tokens": _reasoning_tok,
                }
            if _cached_tok:
                response_body.setdefault("usage", {})["prompt_tokens_details"] = {
                    "cached_tokens": _cached_tok,
                }

            # Attach MCP tool execution results (if any were executed)
            if mcp_results:
                response_body["mcp_tool_results"] = mcp_results

            return JSONResponse(response_body)
        finally:
            _release_lora_adapter(engine, loaded_adapter)

    try:
        return await run_with_disconnect_guard(
            request, _build_response(), cancel_event=_ns_cancel_event)
    finally:
        if _ns_tracker is not None:
            with contextlib.suppress(Exception):
                _ns_tracker.unregister(completion_id)


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
    load_error: str | None = None

    if manager is not None:
        from yunshu_engine.model_manager import ModelType

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
                        except Exception as e:
                            load_error = str(e)
                            logger.warning(f"VLM engine load failed for {entry.model_id}: {e}")
        else:
            # No model requested — accept any loaded VLM (legacy default behavior)
            for entry in manager.list_entries():
                if entry.is_loaded and isinstance(getattr(entry, 'engine', None), VLMEngine):
                    vlm_engine = entry.engine
                    break

    if vlm_engine is None:
        if load_error is not None:
            raise HTTPException(
                status_code=503,
                detail=f"Model '{req.model}' failed to load: {load_error}",
            )
        raise HTTPException(
            status_code=404,
            detail=f"VLM model '{req.model}' not registered or not loaded",
        )

    # VLM/multimodal chat previously BYPASSED the context + prefill
    # guards entirely (dispatched here before the text-path validation block),
    # yet multimodal prompts (long video/image token expansions) are the most
    # likely to be huge. Enforce the same guards; VLM context now resolves via
    # the engine _config dict path. Safe no-op if context can't be
    # resolved (won't false-reject) — only catches egregiously over-window prompts.
    try:
        from yunshu_control.token_counter import (
            IMAGE_TOKEN_ESTIMATE,
            count_message_tokens,
        )
        _vtok = getattr(vlm_engine, "_tokenizer", None)
        if _vtok is not None:
            _vlm_est = count_message_tokens(messages, _vtok)  # already counts image blocks
        else:
            _img = sum(
                1 for m in messages if isinstance(m.get("content"), list)
                for b in m["content"] if isinstance(b, dict) and b.get("type") == "image_url"
            )
            _vlm_est = sum(len(str(m.get("content", ""))) for m in messages) // 4
            _vlm_est += _img * IMAGE_TOKEN_ESTIMATE
        validate_context_window(_vlm_est, req.model, vlm_engine)
        validate_prefill_memory(_vlm_est)
    except HTTPException:
        raise
    except Exception:
        logger.debug("VLM context/prefill guard estimate failed", exc_info=True)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    # Inject tool definitions if provided
    if req.tools:
        messages = _inject_tool_system_prompt(messages, req.tools, req.tool_choice, req.parallel_tool_calls)

    if req.stream:
        if req.n > 1:
            raise HTTPException(
                status_code=400,
                detail="VLM streaming does not support n > 1. Use non-streaming mode for multiple choices.",
            )
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
        max_tokens=req.effective_max_tokens(),
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        min_p=req.min_p,
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
        logprobs=req.logprobs,
        top_logprobs=req.top_logprobs,
        spec_decode=req.spec_decode,
        priority=req.priority,
        logits_processors=req.logits_processors,
        timeout_seconds=req.timeout,
    )
    if json_schema:
        gen_kwargs["json_schema"] = json_schema

    tok = getattr(vlm_engine, '_tokenizer', None)
    # Pre-compute a text-only fallback prompt_tok for the case where the engine
    # returns 0 (e.g., timeout path). The authoritative count comes from the
    # engine result `r["prompt_tokens"]`, which already includes the per-image
    # vision-token estimate via vlm_engine._estimate_image_tokens().
    fallback_prompt_tok = 0
    if tok:
        try:
            prompt_text = vlm_engine._format_prompt(messages)
            fallback_prompt_tok = len(tok.encode(prompt_text))
        except Exception:
            logger.debug("prompt token count failed", exc_info=True)
    prompt_tok = 0

    async def _vlm_gen_one(idx: int):
        try:
            kwargs = {
                **gen_kwargs,
                "seed": _per_choice_seed(req.seed, idx),
            }
            r = await vlm_engine.generate(**kwargs)
        except MemoryError:
            return idx, None, "memory_error"
        except ValueError as e:
            # ValueError comes from image-download / input-validation; surface
            # the message so callers can see what URL/argument was rejected
            # instead of an opaque "inference_error" mask.
            logger.warning("VLM input validation failed: %s", e)
            return idx, None, f"input_error: {e}"
        except Exception as e:
            logger.error("VLM engine inference failed", exc_info=True)
            return idx, None, f"inference_error: {type(e).__name__}"

        content = r.get("text", "") or ""
        rt = r.get("reasoning_tokens", 0)
        ct = r.get("completion_tokens", 0) or (len(tok.encode(content)) if tok else max(1, len(content) // 4))
        # Use engine-reported prompt_tokens, which includes image-token estimate
        # from vlm_engine._estimate_image_tokens() .
        pt = r.get("prompt_tokens", 0) or fallback_prompt_tok
        finish_reason = _normalize_finish_reason(r.get("finish_reason"))

        # Extract thinking content. the LLM text path calls extract_thinking
        # UNCONDITIONALLY (enable_thinking defaults to None), so a reasoning-capable
        # VLM/Omni model that emits <think>…</think> by default had the raw think markup
        # left in user-visible content here while the text path stripped it (and VLM
        # streaming separates reasoning via current_state regardless). Strip unconditionally
        # to match — extract_thinking is a no-op when there is no think markup.
        thinking_content, content = extract_thinking(content, req.model)

        tool_calls = None
        if req.tools:
            _raw_calls = extract_tool_calls_model_aware(content, req.model)
            tool_calls = _enforce_tool_choice(_raw_calls, req.tool_choice, req.parallel_tool_calls)
            # clean markup whenever any was parsed (see LLM n=1 path) so a
            # suppressed wrong-named tool's markup doesn't leak into VLM content.
            if _raw_calls:
                content = clean_tool_call_markup(content)
            if tool_calls:
                finish_reason = "tool_calls"
        return idx, {
            "content": content.strip(),
            "reasoning_content": thinking_content,
            "reasoning_tokens": rt,
            "completion_tokens": ct,
            "prompt_tokens": pt,
            "finish_reason": finish_reason,
            "tool_calls": tool_calls,
        }, None

    n = max(req.n, 1)
    loaded_adapter = _apply_lora_adapter(vlm_engine, req.lora_adapter)
    gen_kwargs["lora_adapter"] = loaded_adapter

    # register for cancellation + client-disconnect stop. This non-streaming VLM
    # path previously had NO tracker entry, NO cancel_event, and NO disconnect guard, so
    # POST /v1/cancel 404'd for an in-flight image/video generation and a client disconnect
    # let the GPU run to max_tokens — the most expensive path to leave uncancellable. A
    # fresh sibling of the cancel keystone (the streaming + text non-stream paths
    # already do this). vlm_engine.generate reads cancel_event from kwargs and honors it.
    _vlm_tracker = None
    _vlm_cancel = None
    try:
        from yunshu_engine.request_tracker import get_request_tracker
        _vlm_tracker = get_request_tracker()
        _vlm_cancel = _vlm_tracker.register(completion_id, req.model).cancel_event
        gen_kwargs["cancel_event"] = _vlm_cancel
    except Exception:
        _vlm_tracker = None

    async def _run_all_choices():
        if n == 1:
            return [await _vlm_gen_one(0)]
        import asyncio
        _res = await asyncio.gather(*[_vlm_gen_one(i) for i in range(n)], return_exceptions=True)
        _valid = []
        for r in _res:
            if isinstance(r, BaseException):
                logger.error(f"VLM choice generation failed: {r}", exc_info=r)
            else:
                _valid.append(r)
        _valid.sort(key=lambda x: x[0])
        return _valid

    try:
        from ..streaming import run_with_disconnect_guard
        results = await run_with_disconnect_guard(
            request, _run_all_choices(), cancel_event=_vlm_cancel)
    finally:
        _release_lora_adapter(vlm_engine, loaded_adapter)
        if _vlm_tracker is not None:
            with contextlib.suppress(Exception):
                _vlm_tracker.unregister(completion_id)

    if not results:
        return JSONResponse(status_code=500, content={"error": {"message": "All choices failed", "type": "inference_error"}})

    # Check for errors
    for _idx, _data, err in results:
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
        # Use max across choices — engine prompt_tokens is per-call but should
        # be identical for the same input across n>1. Falls back to text-only
        # estimate if engine returned 0.
        prompt_tok = max(prompt_tok, data.get("prompt_tokens", 0))
        message = {"role": "assistant", "content": data["content"]}
        if data.get("reasoning_content"):
            message["reasoning_content"] = data["reasoning_content"]
        if data["tool_calls"]:
            from ..streaming import _sanitize_arguments
            message["tool_calls"] = [
                {"id": _generate_tool_call_id(), "type": "function", "function": {"name": tc["name"], "arguments": _sanitize_arguments(tc.get("arguments", {}))}}
                for i, tc in enumerate(data["tool_calls"])
            ]
        choices.append({
            "index": idx,
            "message": message,
            "finish_reason": data["finish_reason"],
        })

    # If every choice returned 0 (e.g., timeout), fall back to text-only count
    if prompt_tok == 0:
        prompt_tok = fallback_prompt_tok
    vlm_usage = {
        "prompt_tokens": prompt_tok,
        # VLM completion count is len(tokens) — all generated tokens, already
        # incl. reasoning; reasoning is the detail subset below (was double-added).
        "completion_tokens": total_completion_tok,
        "total_tokens": prompt_tok + total_completion_tok,
    }
    if total_reasoning_tok > 0:
        vlm_usage["completion_tokens_details"] = {"reasoning_tokens": total_reasoning_tok}

    # Record metrics for VLM non-streaming path
    if prompt_tok > 0 or total_completion_tok > 0 or total_reasoning_tok > 0:
        _record_metrics(prompt_tok, total_completion_tok)  # already incl. reasoning

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
    """SSE streaming for VLM engine ."""
    loaded_adapter = _apply_lora_adapter(vlm_engine, req.lora_adapter)

    # Register with request tracker for cancellation support (before _token_source
    # so cancel_event is available to pass into the engine)
    _vlm_tracker = None
    _vlm_gen = None
    try:
        from yunshu_engine.request_tracker import get_request_tracker
        _vlm_tracker = get_request_tracker()
        _vlm_gen = _vlm_tracker.register(completion_id, req.model)
    except Exception:
        _vlm_tracker = None
    _vlm_cancel_evt = _vlm_gen.cancel_event if _vlm_gen is not None else None
    vlm_prompt_tok = 0
    vlm_completion_tok = 0
    vlm_reasoning_tok = 0

    async def _token_source():
        nonlocal loaded_adapter, done_emitted, vlm_prompt_tok, vlm_completion_tok, vlm_reasoning_tok, metrics_recorded
        first_chunk = True
        vlm_cached_tok = 0
        vlm_last_finish_reason = None
        _vlm_streamed_text = ""  # track emitted text for stop-sequence correction
        include_usage = (
            req.stream_options is not None and req.stream_options.include_usage
        )
        stream_kwargs: dict[str, Any] = dict(
            messages=messages,
            max_tokens=req.effective_max_tokens(),
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            min_p=req.min_p,
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
            logprobs=req.logprobs,
            top_logprobs=req.top_logprobs,
            spec_decode=req.spec_decode,
            priority=req.priority,
            logits_processors=req.logits_processors,
            cancel_event=_vlm_cancel_evt,
            timeout_seconds=req.timeout,
            lora_adapter=loaded_adapter,
        )
        if json_schema:
            stream_kwargs["json_schema"] = json_schema
        async for output in vlm_engine.generate_stream(**stream_kwargs):
            if hasattr(output, 'completion_tokens') and output.completion_tokens is not None and output.completion_tokens > 0:
                vlm_completion_tok = output.completion_tokens
            elif hasattr(output, 'token_text') and output.token_text and getattr(output, 'current_state', None) != "reasoning":
                # Only count non-reasoning tokens toward completion_tok
                vlm_completion_tok += 1
            if hasattr(output, 'reasoning_tokens') and output.reasoning_tokens:
                vlm_reasoning_tok = output.reasoning_tokens
            if hasattr(output, 'cached_tokens') and output.cached_tokens:
                vlm_cached_tok = max(vlm_cached_tok, output.cached_tokens)
            if hasattr(output, 'prompt_tokens') and output.prompt_tokens:
                vlm_prompt_tok = output.prompt_tokens
            if output.finish_reason is not None:
                vlm_last_finish_reason = output.finish_reason
            # Track emitted text for stop-sequence correction
            _vlm_token_text = output.token_text or ""
            if _vlm_token_text and getattr(output, 'current_state', None) != "reasoning":
                _vlm_streamed_text += _vlm_token_text
                if len(_vlm_streamed_text) > _MAX_STREAMING_TEXT_BUFFER:
                    logger.error("VLM streaming text buffer exceeded 1MB — truncating")
                    _vlm_streamed_text = _vlm_streamed_text[-_TRUNCATE_KEEP:]
            # Detect stop-sequence overcount on final output
            if req.stop and vlm_last_finish_reason == "stop" and getattr(output, 'finished', False):
                for _seq in req.stop:
                    if _seq and _seq in _vlm_streamed_text:
                        _idx = _vlm_streamed_text.find(_seq)
                        _vlm_streamed_text = _vlm_streamed_text[:_idx]
                        _tok = getattr(vlm_engine, '_tokenizer', None)
                        if _tok:
                            try:
                                _correct_count = len(_tok.encode(_vlm_streamed_text))
                                if _correct_count < vlm_completion_tok:
                                    vlm_completion_tok = _correct_count
                            except Exception:
                                pass
                        break
            # Route thinking content based on engine's current_state
            _is_reasoning = getattr(output, 'current_state', None) == "reasoning"
            _vlm_token_text = output.token_text or ""
            _vlm_is_final = output.finish_reason is not None
            if _is_reasoning:
                if _vlm_token_text or not _vlm_is_final:
                    yield format_openai_chunk(
                        completion_id=completion_id,
                        model=req.model,
                        delta_content="",
                        thinking_content=_vlm_token_text,
                        finish_reason=None,
                        include_role=first_chunk,
                    )
                    first_chunk = False
            else:
                if _vlm_token_text or not _vlm_is_final:
                    yield format_openai_chunk(
                        completion_id=completion_id,
                        model=req.model,
                        delta_content=_vlm_token_text,
                        finish_reason=None,  # intermediate: always None
                        include_role=first_chunk,
                    )
                    first_chunk = False

        # Final chunk with finish_reason
        # If no tokens were emitted (first_chunk is still True), this is also
        # the first chunk and must include role=assistant per OpenAI spec.
        yield format_openai_chunk(
            completion_id=completion_id,
            model=req.model,
            delta_content="",
            finish_reason=_normalize_finish_reason(vlm_last_finish_reason),
            include_role=first_chunk,
        )

        if include_usage:
            tok = getattr(vlm_engine, '_tokenizer', None)
            if tok and not vlm_prompt_tok:
                try:
                    vlm_prompt_tok = len(tok.encode(vlm_engine._format_prompt(messages)))
                except Exception:
                    logger.debug("operation failed", exc_info=True)
            yield format_openai_usage_chunk(
                completion_id=completion_id,
                model=req.model,
                prompt_tokens=vlm_prompt_tok,
                completion_tokens=vlm_completion_tok,
                reasoning_tokens=vlm_reasoning_tok,
                cached_tokens=vlm_cached_tok,
            )

        # Record metrics for VLM streaming path
        if vlm_prompt_tok > 0 or vlm_completion_tok > 0 or vlm_reasoning_tok > 0:
            _record_metrics(vlm_prompt_tok, vlm_completion_tok)  # already incl. reasoning
        metrics_recorded = True

        done_emitted = True
        yield format_openai_done()
    done_emitted = False
    metrics_recorded = False
    try:
      async for event in with_sse_keepalive(
          _token_source(),
          http_request=request,
          cancel_event=_vlm_cancel_evt,
      ):
          yield event.encode("utf-8")
    except MemoryError:
        if _vlm_cancel_evt is not None:
            _vlm_cancel_evt.set()
        yield b'data: {"error": {"message": "Out of GPU memory", "type": "memory_error", "code": "oom"}}\n\n'
        if not done_emitted:
            yield b"data: [DONE]\n\n"
    except Exception:
        if _vlm_cancel_evt is not None:
            _vlm_cancel_evt.set()
        logger.error("VLM streaming error", exc_info=True)
        yield b'data: {"error": {"message": "Internal server error", "type": "internal_error"}}\n\n'
        if not done_emitted:
            yield b"data: [DONE]\n\n"
    finally:
      _release_lora_adapter(vlm_engine, loaded_adapter)
      if _vlm_tracker is not None:
          with contextlib.suppress(Exception):
              _vlm_tracker.unregister(completion_id)
      # Fallback metrics recording if generator raised before completing
      if not metrics_recorded and (vlm_prompt_tok > 0 or vlm_completion_tok > 0 or vlm_reasoning_tok > 0):
          with contextlib.suppress(Exception):
              _record_metrics(vlm_prompt_tok, vlm_completion_tok)  # already incl. reasoning


def _format_tool_call_chunk_multi(
    completion_id: str,
    model: str,
    choice_index: int,
    tc,
    tc_index: int,
    include_role: bool = False,
) -> str:
    """Format a tool_call as an OpenAI streaming chunk for a specific choice index."""
    delta: dict[str, Any] = {}
    if include_role:
        delta["role"] = "assistant"
    delta["content"] = None
    delta["tool_calls"] = [{
        "index": tc_index,
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.name,
            "arguments": tc.arguments,
        },
    }]
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": choice_index,
            "delta": delta,
            "finish_reason": None,
        }],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _format_tool_call_start_chunk(
    completion_id: str,
    model: str,
    tc_index: int,
    tc_id: str,
    tc_name: str,
    choice_index: int = 0,
    include_role: bool = False,
) -> str:
    """Format the first SSE chunk for a tool call (carries id + name, empty arguments).

    OpenAI streaming spec: the first tool_call delta includes id, type, and
    function.name.  function.arguments is an empty string.
    """
    delta: dict[str, Any] = {}
    if include_role:
        delta["role"] = "assistant"
    delta["content"] = None
    delta["tool_calls"] = [{
        "index": tc_index,
        "id": tc_id,
        "type": "function",
        "function": {
            "name": tc_name,
            "arguments": "",
        },
    }]
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": choice_index,
            "delta": delta,
            "finish_reason": None,
        }],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _format_tool_call_args_delta_chunk(
    completion_id: str,
    model: str,
    tc_index: int,
    args_delta: str,
    choice_index: int = 0,
) -> str:
    """Format an incremental tool call arguments SSE chunk.

    OpenAI streaming spec: subsequent deltas after the first carry only
    function.arguments with the new fragment (no id, no name, no type).
    """
    delta: dict[str, Any] = {}
    delta["content"] = None
    delta["tool_calls"] = [{
        "index": tc_index,
        "function": {
            "arguments": args_delta,
        },
    }]
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": choice_index,
            "delta": delta,
            "finish_reason": None,
        }],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


async def _stream_response_multi(
    engine,
    messages: list[dict],
    req: ChatCompletionRequest,
    completion_id: str,
    request: Request,
    is_batched: bool = False,
    json_schema: dict | str | None = None,
    tool_prefill: str = "",
) -> AsyncIterator[bytes]:
    """n>1 streaming: generate each choice sequentially, emit with correct index.

    On single-GPU systems parallel generation would serialize anyway, so we
    run choices one after another and interleave their SSE events.
    Each choice gets its own streaming loop with its `index` set correctly.

    When tools are provided, each choice gets its own ToolCallStreamer for
    independent per-choice tool call extraction and correct finish_reason.
    """
    from yunshu_engine.request_tracker import get_request_tracker
    tracker = None
    gen = None
    try:
        tracker = get_request_tracker()
        gen = tracker.register(completion_id, req.model)
    except Exception:
        tracker = None
    _multi_cancel_evt = gen.cancel_event if gen is not None else None
    include_usage = (
        req.stream_options is not None and req.stream_options.include_usage
    )
    use_tool_streamer = req.tools is not None and len(req.tools) > 0 and req.tool_choice != "none"
    total_prompt_tok = 0
    total_completion_tok = 0
    total_reasoning_tok = 0
    total_cached_tok = 0

    async def _token_source():
        nonlocal total_prompt_tok, total_completion_tok, total_reasoning_tok, total_cached_tok, done_emitted
        for choice_idx in range(req.n):
            if _multi_cancel_evt is not None and _multi_cancel_evt.is_set():
                yield _format_choice_chunk(
                    completion_id, req.model, choice_idx, "", "stop",
                )
                break
            first_chunk_for_choice = True
            choice_completion_tok = 0
            choice_finish_reason = None  # track actual finish_reason from engine
            choice_reasoning_tok = 0  # track per-choice reasoning tokens
            # Per-choice tool call streamer for independent tool call extraction.
            # thread tool_choice / parallel_tool_calls so streaming enforces
            # the same contract the non-streaming path does via _enforce_tool_choice.
            choice_tool_streamer = ToolCallStreamer(
                forced_tool_name=(req.tool_choice.function.name
                                  if isinstance(req.tool_choice, ToolChoiceFunction) else None),
                allow_parallel=req.parallel_tool_calls,
                model_name=req.model,  # hint for the BUFFER_ALL flush parser
            ) if use_tool_streamer else None
            choice_tool_call_index = 0
            choice_has_tool_call = False
            _choice_tc_args_streamed = False  # args delta emitted for current tool call?
            _choice_tc_start_emitted = False  # start chunk (id+name) emitted?
            _choice_streamed_text = ""  # track emitted text for stop-sequence correction
            # Prefill-forced tool_choice marker, re-seeded for each choice (every choice
            # generates fresh from the prefilled prompt). Fed onto the first streamer token.
            _choice_pending_prefill = tool_prefill

            if is_batched:
                stream = engine.stream_chat(
                    messages=messages,
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
                    json_schema=json_schema,
                    thinking_budget=req.thinking_budget,
                    reasoning_effort=req.reasoning_effort,
                    xtc_probability=req.xtc_probability,
                    xtc_threshold=req.xtc_threshold,
                    spec_decode=req.spec_decode,
                    priority=req.priority,
                    logprobs=req.logprobs,
                    top_logprobs=req.top_logprobs,
                    logits_processors=req.logits_processors,
                    cancel_event=_multi_cancel_evt,
                    timeout_seconds=req.timeout,
                    lora_adapter=loaded_adapter,
                    min_tokens=req.min_tokens,
                    ignore_eos=req.ignore_eos,
                    suppress_tokens=req.suppress_tokens,
                )
                async for output in stream:
                    if _multi_cancel_evt is not None and _multi_cancel_evt.is_set():
                        yield _format_choice_chunk(
                            completion_id, req.model, choice_idx, "", "stop",
                        )
                        done_emitted = True
                        yield format_openai_done()
                        return
                    if hasattr(output, 'prompt_tokens') and output.prompt_tokens:
                        total_prompt_tok = output.prompt_tokens
                    if hasattr(output, 'reasoning_tokens') and output.reasoning_tokens:
                        choice_reasoning_tok = output.reasoning_tokens
                    if hasattr(output, 'cached_tokens') and output.cached_tokens:
                        total_cached_tok = max(total_cached_tok, output.cached_tokens)
                    token_text = output.new_text
                    # emit prefill progress as SSE comment
                    _pf_prog = getattr(output, 'prefill_progress', None)
                    if _pf_prog is not None:
                        yield f": prefill-progress {_pf_prog[0]}/{_pf_prog[1]}\n\n"
                        continue
                    if hasattr(output, 'completion_tokens') and output.completion_tokens is not None and output.completion_tokens > 0:
                        choice_completion_tok = output.completion_tokens
                    elif token_text and getattr(output, 'current_state', None) != "reasoning":
                        # Only count non-reasoning tokens toward completion_tok
                        choice_completion_tok += 1
                    # Only set finish_reason on the final token from engine
                    fr = output.finish_reason
                    if fr is not None:
                        choice_finish_reason = fr
                    _chunk_lp = _format_chat_logprobs(output.logprobs, tokenizer=getattr(engine, "_tokenizer", None), top_logprobs=req.top_logprobs) if req.logprobs and hasattr(output, "logprobs") else None
                    # Track emitted text for stop-sequence correction
                    if token_text and getattr(output, 'current_state', None) != "reasoning":
                        _choice_streamed_text += token_text
                        if len(_choice_streamed_text) > _MAX_STREAMING_TEXT_BUFFER:
                            logger.error("Choice streaming text buffer exceeded 1MB — truncating")
                            _choice_streamed_text = _choice_streamed_text[-_TRUNCATE_KEEP:]
                    # Detect stop-sequence overcount on final output
                    if req.stop and choice_finish_reason == "stop" and getattr(output, "finished", False):
                        for _seq in req.stop:
                            if _seq and _seq in _choice_streamed_text:
                                _idx = _choice_streamed_text.find(_seq)
                                _choice_streamed_text = _choice_streamed_text[:_idx]
                                _tok = getattr(engine, '_tokenizer', None)
                                if _tok:
                                    try:
                                        _correct_count = len(_tok.encode(_choice_streamed_text))
                                        if _correct_count < choice_completion_tok:
                                            choice_completion_tok = _correct_count
                                    except Exception:
                                        pass
                                break
                    # Route thinking content based on SequenceStateMachine state
                    _is_reasoning = getattr(output, 'current_state', None) == "reasoning"
                    if _is_reasoning:
                        # Skip empty thinking chunk when engine already signaled finish.
                        if token_text or fr is None:
                            yield _format_choice_chunk(
                                completion_id, req.model, choice_idx,
                                "", None,
                                include_role=first_chunk_for_choice,
                                logprobs=_chunk_lp,
                                thinking_content=token_text,
                            )
                            first_chunk_for_choice = False
                    elif use_tool_streamer and choice_tool_streamer and token_text:
                        # Seed the prefill-forced tool-call marker onto the first token.
                        if _choice_pending_prefill:
                            token_text = _choice_pending_prefill + token_text
                            _choice_pending_prefill = ""
                        # Process through per-choice tool call streamer
                        for out in choice_tool_streamer.process_token(token_text):
                            if out.text:
                                yield _format_choice_chunk(
                                    completion_id, req.model, choice_idx,
                                    out.text, None,
                                    include_role=first_chunk_for_choice,
                                    logprobs=_chunk_lp,
                                )
                                first_chunk_for_choice = False
                            elif out.tool_call_start:
                                # First chunk for this tool call — carries id + name
                                yield _format_tool_call_start_chunk(
                                    completion_id, req.model,
                                    tc_index=choice_tool_call_index,
                                    tc_id=out.tool_call_start.id,
                                    tc_name=out.tool_call_start.name,
                                    choice_index=choice_idx,
                                    include_role=first_chunk_for_choice,
                                )
                                first_chunk_for_choice = False
                                _choice_tc_args_streamed = False
                                _choice_tc_start_emitted = True
                            elif out.tool_call_args_delta:
                                # Incremental arguments fragment
                                yield _format_tool_call_args_delta_chunk(
                                    completion_id, req.model,
                                    tc_index=choice_tool_call_index,
                                    args_delta=out.tool_call_args_delta,
                                    choice_index=choice_idx,
                                )
                                _choice_tc_args_streamed = True
                            elif out.tool_call:
                                # Complete tool call (closing tag found). (6th
                                # audit): emit full args if none streamed (single-chunk
                                # tool call) so arguments isn't empty client-side.
                                # a single-shot complete tool_call with NO preceding
                                # tool_call_start (GLM-4.x: the name is bare text so the streamer
                                # never emits a start — fixed the streamer half, this is the
                                # gateway half) must STILL send the id+name first, or an OpenAI
                                # streaming client can't key the call → drops it.
                                if not _choice_tc_start_emitted:
                                    yield _format_tool_call_start_chunk(
                                        completion_id, req.model,
                                        tc_index=choice_tool_call_index,
                                        tc_id=out.tool_call.id,
                                        tc_name=out.tool_call.name,
                                        choice_index=choice_idx,
                                        include_role=first_chunk_for_choice,
                                    )
                                if not _choice_tc_args_streamed:
                                    _full_args = (out.tool_call.arguments or "").strip()
                                    if _full_args and _full_args != "{}":
                                        yield _format_tool_call_args_delta_chunk(
                                            completion_id, req.model,
                                            tc_index=choice_tool_call_index,
                                            args_delta=_full_args,
                                            choice_index=choice_idx,
                                        )
                                choice_has_tool_call = True
                                choice_tool_call_index += 1
                                first_chunk_for_choice = False
                                _choice_tc_args_streamed = False
                                _choice_tc_start_emitted = False  # next index needs its own start
                    else:
                        # Skip empty intermediate chunk when engine already
                        # signaled finish (e.g. stop-on-first-token).
                        _is_final_from_engine = output.finish_reason is not None
                        if token_text or not _is_final_from_engine:
                            yield _format_choice_chunk(
                                completion_id, req.model, choice_idx,
                                token_text, None,  # intermediate: always None
                                include_role=first_chunk_for_choice,
                                logprobs=_chunk_lp,
                            )
                            first_chunk_for_choice = False
            else:
                stream = engine.generate_stream(
                    prompt=messages,
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
                    stop_token_ids=req.stop_token_ids,
                    thinking_budget=req.thinking_budget,
                    reasoning_effort=req.reasoning_effort,
                    xtc_probability=req.xtc_probability,
                    xtc_threshold=req.xtc_threshold,
                    spec_decode=req.spec_decode,
                    json_schema=json_schema,
                    priority=req.priority,
                    logprobs=req.logprobs,
                    top_logprobs=req.top_logprobs,
                    logits_processors=req.logits_processors,
                    cancel_event=_multi_cancel_evt,
                    timeout_seconds=req.timeout,
                    lora_adapter=loaded_adapter,
                    min_tokens=req.min_tokens,
                    ignore_eos=req.ignore_eos,
                    suppress_tokens=req.suppress_tokens,
                )
                async for output in stream:
                    if _multi_cancel_evt is not None and _multi_cancel_evt.is_set():
                        yield _format_choice_chunk(
                            completion_id, req.model, choice_idx, "", "stop",
                        )
                        done_emitted = True
                        yield format_openai_done()
                        return
                    if hasattr(output, 'prompt_tokens') and output.prompt_tokens:
                        total_prompt_tok = output.prompt_tokens
                    if hasattr(output, 'reasoning_tokens') and output.reasoning_tokens:
                        choice_reasoning_tok = output.reasoning_tokens
                    if hasattr(output, 'cached_tokens') and output.cached_tokens:
                        total_cached_tok = max(total_cached_tok, output.cached_tokens)
                    token_text = getattr(output, 'token_text', '')
                    if hasattr(output, 'completion_tokens') and output.completion_tokens is not None and output.completion_tokens > 0:
                        choice_completion_tok = output.completion_tokens
                    elif token_text and getattr(output, 'current_state', None) != "reasoning":
                        # Only count non-reasoning tokens toward completion_tok
                        choice_completion_tok += 1
                    # Only set finish_reason on the final token from engine
                    fr = getattr(output, 'finish_reason', None)
                    if fr is not None:
                        choice_finish_reason = fr
                    _chunk_lp = _format_chat_logprobs(output.logprobs, tokenizer=getattr(engine, "_tokenizer", None), top_logprobs=req.top_logprobs) if req.logprobs and hasattr(output, "logprobs") else None
                    # Track emitted text for stop-sequence correction
                    if token_text and getattr(output, 'current_state', None) != "reasoning":
                        _choice_streamed_text += token_text
                        if len(_choice_streamed_text) > _MAX_STREAMING_TEXT_BUFFER:
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
                                        if _correct_count < choice_completion_tok:
                                            choice_completion_tok = _correct_count
                                    except Exception:
                                        pass
                                break
                    # Route thinking content based on SequenceStateMachine state
                    _is_reasoning = getattr(output, 'current_state', None) == "reasoning"
                    if _is_reasoning:
                        # Skip empty thinking chunk when engine already signaled finish.
                        if token_text or fr is None:
                            yield _format_choice_chunk(
                                completion_id, req.model, choice_idx,
                                "", None,
                                include_role=first_chunk_for_choice,
                                logprobs=_chunk_lp,
                                thinking_content=token_text,
                            )
                            first_chunk_for_choice = False
                    elif use_tool_streamer and choice_tool_streamer and token_text:
                        # Seed the prefill-forced tool-call marker onto the first token.
                        if _choice_pending_prefill:
                            token_text = _choice_pending_prefill + token_text
                            _choice_pending_prefill = ""
                        # Process through per-choice tool call streamer
                        for out in choice_tool_streamer.process_token(token_text):
                            if out.text:
                                yield _format_choice_chunk(
                                    completion_id, req.model, choice_idx,
                                    out.text, None,
                                    include_role=first_chunk_for_choice,
                                    logprobs=_chunk_lp,
                                )
                                first_chunk_for_choice = False
                            elif out.tool_call_start:
                                # First chunk for this tool call — carries id + name
                                yield _format_tool_call_start_chunk(
                                    completion_id, req.model,
                                    tc_index=choice_tool_call_index,
                                    tc_id=out.tool_call_start.id,
                                    tc_name=out.tool_call_start.name,
                                    choice_index=choice_idx,
                                    include_role=first_chunk_for_choice,
                                )
                                first_chunk_for_choice = False
                                _choice_tc_args_streamed = False
                                _choice_tc_start_emitted = True
                            elif out.tool_call_args_delta:
                                # Incremental arguments fragment
                                yield _format_tool_call_args_delta_chunk(
                                    completion_id, req.model,
                                    tc_index=choice_tool_call_index,
                                    args_delta=out.tool_call_args_delta,
                                    choice_index=choice_idx,
                                )
                                _choice_tc_args_streamed = True
                            elif out.tool_call:
                                # Complete tool call (closing tag found). (6th
                                # audit): emit full args if none streamed (single-chunk
                                # tool call) so arguments isn't empty client-side.
                                # a single-shot complete tool_call with NO preceding
                                # tool_call_start (GLM-4.x: the name is bare text so the streamer
                                # never emits a start — fixed the streamer half, this is the
                                # gateway half) must STILL send the id+name first, or an OpenAI
                                # streaming client can't key the call → drops it.
                                if not _choice_tc_start_emitted:
                                    yield _format_tool_call_start_chunk(
                                        completion_id, req.model,
                                        tc_index=choice_tool_call_index,
                                        tc_id=out.tool_call.id,
                                        tc_name=out.tool_call.name,
                                        choice_index=choice_idx,
                                        include_role=first_chunk_for_choice,
                                    )
                                if not _choice_tc_args_streamed:
                                    _full_args = (out.tool_call.arguments or "").strip()
                                    if _full_args and _full_args != "{}":
                                        yield _format_tool_call_args_delta_chunk(
                                            completion_id, req.model,
                                            tc_index=choice_tool_call_index,
                                            args_delta=_full_args,
                                            choice_index=choice_idx,
                                        )
                                choice_has_tool_call = True
                                choice_tool_call_index += 1
                                first_chunk_for_choice = False
                                _choice_tc_args_streamed = False
                                _choice_tc_start_emitted = False  # next index needs its own start
                    else:
                        # Skip empty intermediate chunk when engine already
                        # signaled finish (e.g. stop-on-first-token).
                        _fr = getattr(output, 'finish_reason', None)
                        _is_final_from_engine = _fr is not None
                        if token_text or not _is_final_from_engine:
                            yield _format_choice_chunk(
                                completion_id, req.model, choice_idx,
                                token_text, None,  # intermediate: always None
                                include_role=first_chunk_for_choice,
                                logprobs=_chunk_lp,
                            )
                            first_chunk_for_choice = False

            # Flush any remaining content from per-choice tool streamer
            if use_tool_streamer and choice_tool_streamer:
                for out in choice_tool_streamer.flush():
                    if out.text:
                        yield _format_choice_chunk(
                            completion_id, req.model, choice_idx,
                            out.text, None,
                            include_role=first_chunk_for_choice,
                        )
                        first_chunk_for_choice = False
                    elif out.tool_call_start:
                        yield _format_tool_call_start_chunk(
                            completion_id, req.model,
                            tc_index=choice_tool_call_index,
                            tc_id=out.tool_call_start.id,
                            tc_name=out.tool_call_start.name,
                            choice_index=choice_idx,
                            include_role=first_chunk_for_choice,
                        )
                        first_chunk_for_choice = False
                    elif out.tool_call_args_delta:
                        yield _format_tool_call_args_delta_chunk(
                            completion_id, req.model,
                            tc_index=choice_tool_call_index,
                            args_delta=out.tool_call_args_delta,
                            choice_index=choice_idx,
                        )
                    elif out.tool_call:
                        choice_has_tool_call = True
                        choice_tool_call_index += 1
                        first_chunk_for_choice = False

            total_completion_tok += choice_completion_tok
            total_reasoning_tok += choice_reasoning_tok

            # Emit final chunk with actual finish_reason for this choice.
            # If tool calls were detected, override finish_reason to "tool_calls".
            if choice_has_tool_call:
                final_reason = "tool_calls"
            else:
                final_reason = _normalize_finish_reason(choice_finish_reason)
            # If no tokens were emitted for this choice, this is also the first
            # chunk for this choice and must include role=assistant per OpenAI spec.
            yield _format_choice_chunk(
                completion_id, req.model, choice_idx,
                "", final_reason,
                include_role=first_chunk_for_choice,
            )

        if include_usage:
            yield format_openai_usage_chunk(
                completion_id=completion_id,
                model=req.model,
                prompt_tokens=total_prompt_tok,
                completion_tokens=total_completion_tok,
                reasoning_tokens=total_reasoning_tok,
                cached_tokens=total_cached_tok,
            )
        done_emitted = True
        yield format_openai_done()
    done_emitted = False
    loaded_adapter = _apply_lora_adapter(engine, req.lora_adapter)
    try:
      async for event in with_sse_keepalive(
          _token_source(),
          http_request=request,
          cancel_event=_multi_cancel_evt,
      ):
          yield event.encode("utf-8")
    except MemoryError:
        if _multi_cancel_evt is not None:
            _multi_cancel_evt.set()
        yield b'data: {"error": {"message": "Out of GPU memory", "type": "memory_error", "code": "oom"}}\n\n'
        if not done_emitted:
            yield b"data: [DONE]\n\n"
    except Exception:
        if _multi_cancel_evt is not None:
            _multi_cancel_evt.set()
        logger.error("Chat multi-choice streaming error", exc_info=True)
        yield b'data: {"error": {"message": "Internal server error", "type": "internal_error"}}\n\n'
        if not done_emitted:
            yield b"data: [DONE]\n\n"
    finally:
      _release_lora_adapter(engine, loaded_adapter)
      if tracker is not None:
          with contextlib.suppress(Exception):
              tracker.unregister(completion_id)
      # Record middleware/tracing metrics on BOTH success and error. The old
      # `not done_emitted` guard skipped the SUCCESS path (done_emitted is set True
      # before [DONE]) → n>1 streaming silently under-counted observability tokens.
      # finally runs once; success and error are mutually exclusive, so no double-count.
      if total_prompt_tok > 0 or total_completion_tok > 0 or total_reasoning_tok > 0:
          with contextlib.suppress(Exception):
              _record_metrics(total_prompt_tok, total_completion_tok)  # already incl. reasoning


def _format_choice_chunk(
    completion_id: str,
    model: str,
    index: int,
    delta_content: str,
    finish_reason: str | None,
    include_role: bool = False,
    logprobs: dict | None = None,
    thinking_content: str | None = None,
) -> str:
    """Format an SSE chunk for a specific choice index."""
    delta: dict[str, Any] = {}
    if include_role:
        delta["role"] = "assistant"
    if include_role and not delta_content:
        delta["content"] = None
    else:
        delta["content"] = delta_content
    if thinking_content:
        delta["reasoning_content"] = thinking_content
    choice: dict[str, Any] = {
        "index": index,
        "delta": delta,
        "finish_reason": finish_reason,
    }
    if logprobs:
        choice["logprobs"] = logprobs
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [choice],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _format_chat_logprobs(
    logprobs_list: list[dict] | None,
    tokenizer=None,
    top_logprobs: int | None = None,
) -> dict | None:
    """Format per-token logprobs from GenerationOutput into OpenAI Chat format.

    Args:
        logprobs_list: Raw logprobs from engine output.
        tokenizer: Tokenizer for decoding token IDs.
        top_logprobs: Maximum number of top logprobs to return per token.
            When None, all available top logprobs are included (no truncation).
            When 0, top_logprobs are omitted entirely from the output.
    """
    # Type/length guard (not truthiness): a raw mx.array would raise on `not x`.
    if not isinstance(logprobs_list, (list, tuple)) or len(logprobs_list) == 0:
        return None
    content = []
    for lp_entry in logprobs_list:
        if not isinstance(lp_entry, dict):
            continue
        token_str = lp_entry.get("token", "")
        if not token_str and tokenizer and "token_id" in lp_entry:
            try:
                token_str = tokenizer.decode([lp_entry["token_id"]])
            except Exception:
                token_str = str(lp_entry["token_id"])
        raw_top = lp_entry.get("top_logprobs", [])
        # Apply truncation consistent with _format_logprobs (non-streaming path)
        if top_logprobs is not None and top_logprobs > 0:
            raw_top = raw_top[:top_logprobs]
        decoded_top = []
        for tlp in raw_top:
            tlp_token = tlp.get("token", "")
            if not tlp_token and tokenizer and "token_id" in tlp:
                try:
                    tlp_token = tokenizer.decode([tlp["token_id"]])
                except Exception:
                    tlp_token = str(tlp["token_id"])
            decoded_top.append({
                "token": tlp_token,
                "logprob": tlp.get("logprob", 0.0),
                "bytes": _lp_bytes(tlp, tlp_token, tokenizer),
            })
        content.append({
            "token": token_str,
            "logprob": lp_entry.get("logprob", 0.0),
            "bytes": _lp_bytes(lp_entry, token_str, tokenizer),
            "top_logprobs": decoded_top,
        })
    return {"content": content} if content else None


async def _stream_response(
    engine,
    messages: list[dict],
    req: ChatCompletionRequest,
    completion_id: str,
    request: Request,
    is_batched: bool = False,
    json_schema: dict | str | None = None,
    tool_prefill: str = "",
) -> AsyncIterator[bytes]:
    """SSE streaming response with keepalive and disconnect detection.

    for robust streaming.
    Supports both Engine (legacy) and BatchedEngine .
    Uses engine's current_state (from SequenceStateMachine) to route
    reasoning vs visible content, matching mlx-lm's server.py pattern.

    When tools are provided, uses ToolCallStreamer for incremental
    tool call detection — buffers tokens and emits tool_calls in
    OpenAI streaming delta format when detected.
    """
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
    _cancel_evt = _tracker_gen.cancel_event if _tracker_gen is not None else None
    use_tool_streamer = req.tools is not None and len(req.tools) > 0 and req.tool_choice != "none"
    # thread tool_choice / parallel_tool_calls so streaming enforces the same
    # contract the non-streaming path does via _enforce_tool_choice.
    tool_streamer = ToolCallStreamer(
        forced_tool_name=(req.tool_choice.function.name
                          if isinstance(req.tool_choice, ToolChoiceFunction) else None),
        allow_parallel=req.parallel_tool_calls,
        model_name=req.model,  # hint for the BUFFER_ALL flush parser
    ) if use_tool_streamer else None
    tool_call_index = 0  # Track index for streaming tool_calls delta
    has_emitted_tool_call = False
    _tc_args_streamed = False  # did we emit any args delta for the current tool call?
    include_usage = (
        req.stream_options is not None and req.stream_options.include_usage
    )
    prompt_tok = 0
    completion_tok = 0
    reasoning_tok = 0
    cached_tok = 0

    def _format_tool_call_chunk(tc, idx: int, include_role: bool = False) -> str:
        """Format a tool_call as an OpenAI streaming chunk with delta."""
        delta: dict[str, Any] = {}
        if include_role:
            delta["role"] = "assistant"
        delta["content"] = None
        delta["tool_calls"] = [{
            "index": idx,
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": tc.arguments,
            },
        }]
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
        # _tc_args_streamed (init'd in the outer scope) MUST be nonlocal: the batched
        # branch assigns it before the shared flush reads it, but the non-batched
        # branch does not — without nonlocal it becomes a local and the flush-path
        # read raises UnboundLocalError on a truncated tool-call (max_tokens cut the
        # call before its closing </tool_call> tag), crashing the stream.
        nonlocal tool_call_index, has_emitted_tool_call, prompt_tok, completion_tok, reasoning_tok, cached_tok, done_emitted, _tc_args_streamed
        first_chunk = True
        last_finish_reason = None  # track actual finish_reason from engine
        _streamed_text = ""  # track text emitted to client for stop-sequence correction
        # Prefill-forced tool_choice: the opening <tool_call> marker lives in the PROMPT
        # (continue_final_message), so the streamer must see it BEFORE the model's
        # continuation or it never enters tool-call state. Seed it onto the first token
        # fed to the streamer.
        _pending_tool_prefill = tool_prefill

        if is_batched:
            async for output in engine.stream_chat(
                messages=messages,
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
                seed=req.seed,
                enable_thinking=req.enable_thinking,
                json_schema=json_schema,
                thinking_budget=req.thinking_budget,
                reasoning_effort=req.reasoning_effort,
                xtc_probability=req.xtc_probability,
                xtc_threshold=req.xtc_threshold,
                priority=req.priority,
                logprobs=req.logprobs,
                top_logprobs=req.top_logprobs,
                spec_decode=req.spec_decode,
                logits_processors=req.logits_processors,
                cancel_event=_cancel_evt,
                timeout_seconds=req.timeout,
                lora_adapter=loaded_adapter,
                min_tokens=req.min_tokens,
                ignore_eos=req.ignore_eos,
                suppress_tokens=req.suppress_tokens,
            ):
                token_text = output.new_text
                # emit prefill progress as SSE comment for
                # client-side progress bars during long chunked prefills.
                _pf_prog = getattr(output, 'prefill_progress', None)
                if _pf_prog is not None:
                    yield f": prefill-progress {_pf_prog[0]}/{_pf_prog[1]}\n\n"
                    continue  # progress outputs carry no text
                if output.finish_reason is not None:
                    last_finish_reason = output.finish_reason
                if hasattr(output, 'prompt_tokens') and output.prompt_tokens:
                    prompt_tok = output.prompt_tokens
                if hasattr(output, 'reasoning_tokens') and output.reasoning_tokens:
                    reasoning_tok = output.reasoning_tokens
                if hasattr(output, 'cached_tokens') and output.cached_tokens:
                    cached_tok = max(cached_tok, output.cached_tokens)
                if hasattr(output, 'completion_tokens') and output.completion_tokens is not None and output.completion_tokens > 0:
                    completion_tok = output.completion_tokens
                elif token_text and getattr(output, 'current_state', None) != "reasoning":
                    # Only count non-reasoning tokens toward completion_tok
                    completion_tok += 1
                # Track streamed text for stop-sequence overcount correction
                if token_text and getattr(output, 'current_state', None) != "reasoning":
                    _streamed_text += token_text
                    if len(_streamed_text) > _MAX_STREAMING_TEXT_BUFFER:
                        logger.error("Streaming text buffer exceeded 1MB — truncating")
                        _streamed_text = _streamed_text[-_TRUNCATE_KEEP:]
                # Detect stop-sequence overcount: if stop sequences are provided
                # and the engine's finish_reason is "stop", the engine may have
                # overcounted completion_tok when a multi-token stop suffix was
                # matched (engine only decrements by 1 regardless of suffix length).
                if req.stop and last_finish_reason == "stop" and getattr(output, "finished", False):
                    for _seq in req.stop:
                        if _seq and _seq in _streamed_text:
                            _idx = _streamed_text.find(_seq)
                            _streamed_text = _streamed_text[:_idx]
                            # Use tokenizer to get accurate count of emitted tokens
                            _tok = getattr(engine, '_tokenizer', None)
                            if _tok:
                                try:
                                    _correct_count = len(_tok.encode(_streamed_text))
                                    if _correct_count < completion_tok:
                                        completion_tok = _correct_count
                                except Exception:
                                    pass
                            break

                _chunk_lp = _format_chat_logprobs(output.logprobs, tokenizer=getattr(engine, "_tokenizer", None), top_logprobs=req.top_logprobs) if req.logprobs and hasattr(output, "logprobs") else None

                # Route thinking content based on SequenceStateMachine state
                _is_reasoning = getattr(output, 'current_state', None) == "reasoning"
                _is_final_from_engine = output.finish_reason is not None
                if _is_reasoning:
                    # Skip empty thinking chunk when engine already signaled finish.
                    if token_text or not _is_final_from_engine:
                        yield format_openai_chunk(
                            completion_id=completion_id,
                            model=req.model,
                            delta_content="",
                            thinking_content=token_text,
                            finish_reason=None,
                            include_role=first_chunk,
                            logprobs=_chunk_lp,
                        )
                        first_chunk = False
                elif use_tool_streamer and tool_streamer and token_text:
                    # Seed the prefill-forced tool-call marker onto the first token.
                    if _pending_tool_prefill:
                        token_text = _pending_tool_prefill + token_text
                        _pending_tool_prefill = ""
                    # Run through tool call streamer
                    outputs = tool_streamer.process_token(token_text)
                    for out in outputs:
                        if out.text:
                            yield format_openai_chunk(
                                completion_id=completion_id,
                                model=req.model,
                                delta_content=out.text,
                                include_role=first_chunk,
                                logprobs=_chunk_lp,
                            )
                            first_chunk = False
                        elif out.tool_call_start:
                            # First chunk for this tool call — carries id + name
                            yield _format_tool_call_start_chunk(
                                completion_id, req.model,
                                tc_index=tool_call_index,
                                tc_id=out.tool_call_start.id,
                                tc_name=out.tool_call_start.name,
                                include_role=first_chunk,
                            )
                            first_chunk = False
                            _tc_args_streamed = False
                        elif out.tool_call_args_delta:
                            # Incremental arguments fragment
                            yield _format_tool_call_args_delta_chunk(
                                completion_id, req.model,
                                tc_index=tool_call_index,
                                args_delta=out.tool_call_args_delta,
                            )
                            _tc_args_streamed = True
                        elif out.tool_call:
                            # Complete tool call (closing tag found). (6th
                            # audit): when the body+closing tag arrive in one chunk
                            # (forced-grammar / single-token), NO tool_call_args_delta was
                            # ever streamed → the client got name with arguments="". Emit
                            # the full args once here if none streamed (mirrors the
                            # Anthropic path's fix). Guard avoids duplicating args.
                            if not _tc_args_streamed:
                                _full_args = (out.tool_call.arguments or "").strip()
                                if _full_args and _full_args != "{}":
                                    yield _format_tool_call_args_delta_chunk(
                                        completion_id, req.model,
                                        tc_index=tool_call_index,
                                        args_delta=_full_args,
                                    )
                            has_emitted_tool_call = True
                            tool_call_index += 1
                            first_chunk = False
                            _tc_args_streamed = False
                else:
                    # Skip empty intermediate chunk when engine already signaled
                    # finish (e.g. stop-on-first-token). The final chunk after
                    # this loop will carry finish_reason and include_role.
                    _is_final_from_engine = output.finish_reason is not None
                    if token_text or not _is_final_from_engine:
                        yield format_openai_chunk(
                            completion_id=completion_id,
                            model=req.model,
                            delta_content=token_text,
                            finish_reason=None,  # intermediate: always None
                            include_role=first_chunk,
                            logprobs=_chunk_lp,
                        )
                        first_chunk = False
        else:
            async for output in engine.generate_stream(
                prompt=messages,
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
                cancel_event=_cancel_evt,
                timeout_seconds=req.timeout,
                lora_adapter=loaded_adapter,
            ):
                # Track token counts for usage reporting
                if hasattr(output, 'prompt_tokens') and output.prompt_tokens:
                    prompt_tok = output.prompt_tokens
                if hasattr(output, 'completion_tokens') and output.completion_tokens is not None and output.completion_tokens > 0:
                    completion_tok = output.completion_tokens
                elif hasattr(output, 'token_text') and output.token_text and getattr(output, 'current_state', None) != "reasoning":
                    # Only count non-reasoning tokens toward completion_tok
                    completion_tok += 1
                if hasattr(output, 'reasoning_tokens') and output.reasoning_tokens:
                    reasoning_tok = output.reasoning_tokens
                if hasattr(output, 'cached_tokens') and output.cached_tokens:
                    cached_tok = max(cached_tok, output.cached_tokens)
                if output.finish_reason is not None:
                    last_finish_reason = output.finish_reason
                _chunk_lp = _format_chat_logprobs(output.logprobs, tokenizer=getattr(engine, "_tokenizer", None), top_logprobs=req.top_logprobs) if req.logprobs and hasattr(output, "logprobs") else None
                # Track streamed text for stop-sequence overcount correction
                _legacy_token_text = output.token_text or ""
                if _legacy_token_text and getattr(output, 'current_state', None) != "reasoning":
                    _streamed_text += _legacy_token_text
                    if len(_streamed_text) > _MAX_STREAMING_TEXT_BUFFER:
                        logger.error("Streaming text buffer exceeded 1MB — truncating")
                        _streamed_text = _streamed_text[-_TRUNCATE_KEEP:]
                # Detect stop-sequence overcount on final output
                if req.stop and last_finish_reason == "stop" and getattr(output, 'finished', False):
                    for _seq in req.stop:
                        if _seq and _seq in _streamed_text:
                            _idx = _streamed_text.find(_seq)
                            _streamed_text = _streamed_text[:_idx]
                            _tok = getattr(engine, '_tokenizer', None)
                            if _tok:
                                try:
                                    _correct_count = len(_tok.encode(_streamed_text))
                                    if _correct_count < completion_tok:
                                        completion_tok = _correct_count
                                except Exception:
                                    pass
                            break
                # Route based on SequenceStateMachine state (mlx-lm pattern)
                _is_final_from_engine = output.finish_reason is not None
                if getattr(output, 'current_state', None) == "reasoning":
                    _thinking_text = output.token_text
                    if _thinking_text or not _is_final_from_engine:
                        yield format_openai_chunk(
                            completion_id=completion_id,
                            model=req.model,
                            delta_content="",
                            thinking_content=_thinking_text,
                            finish_reason=None,  # intermediate: always None
                            include_role=first_chunk,
                            logprobs=_chunk_lp,
                        )
                        first_chunk = False
                else:
                    token_text = output.token_text
                    if use_tool_streamer and tool_streamer and token_text:
                        # Seed the prefill-forced tool-call marker onto the first token.
                        if _pending_tool_prefill:
                            token_text = _pending_tool_prefill + token_text
                            _pending_tool_prefill = ""
                        outputs = tool_streamer.process_token(token_text)
                        for out in outputs:
                            if out.text:
                                yield format_openai_chunk(
                                    completion_id=completion_id,
                                    model=req.model,
                                    delta_content=out.text,
                                    include_role=first_chunk,
                                    logprobs=_chunk_lp,
                                )
                                first_chunk = False
                            elif out.tool_call_start:
                                # First chunk for this tool call — carries id + name
                                yield _format_tool_call_start_chunk(
                                    completion_id, req.model,
                                    tc_index=tool_call_index,
                                    tc_id=out.tool_call_start.id,
                                    tc_name=out.tool_call_start.name,
                                    include_role=first_chunk,
                                )
                                first_chunk = False
                            elif out.tool_call_args_delta:
                                # Incremental arguments fragment
                                yield _format_tool_call_args_delta_chunk(
                                    completion_id, req.model,
                                    tc_index=tool_call_index,
                                    args_delta=out.tool_call_args_delta,
                                )
                            elif out.tool_call:
                                # Complete tool call (closing tag found)
                                has_emitted_tool_call = True
                                tool_call_index += 1
                                first_chunk = False
                    else:
                        # Skip empty intermediate chunk when engine already
                        # signaled finish (e.g. stop-on-first-token). The final
                        # chunk after this loop carries finish_reason + include_role.
                        _is_final_from_engine = output.finish_reason is not None
                        if token_text or not _is_final_from_engine:
                            yield format_openai_chunk(
                                completion_id=completion_id,
                                model=req.model,
                                delta_content=token_text,
                                finish_reason=None,  # intermediate: always None
                                include_role=first_chunk,
                                logprobs=_chunk_lp,
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
                        include_role=first_chunk,
                    )
                    first_chunk = False
                elif out.tool_call_start:
                    yield _format_tool_call_start_chunk(
                        completion_id, req.model,
                        tc_index=tool_call_index,
                        tc_id=out.tool_call_start.id,
                        tc_name=out.tool_call_start.name,
                        include_role=first_chunk,
                    )
                    first_chunk = False
                elif out.tool_call_args_delta:
                    yield _format_tool_call_args_delta_chunk(
                        completion_id, req.model,
                        tc_index=tool_call_index,
                        args_delta=out.tool_call_args_delta,
                    )
                    _tc_args_streamed = True
                elif out.tool_call:
                    # same empty-arguments fix on the flush path.
                    if not _tc_args_streamed:
                        _full_args = (out.tool_call.arguments or "").strip()
                        if _full_args and _full_args != "{}":
                            yield _format_tool_call_args_delta_chunk(
                                completion_id, req.model,
                                tc_index=tool_call_index,
                                args_delta=_full_args,
                            )
                    has_emitted_tool_call = True
                    tool_call_index += 1
                    first_chunk = False
                    _tc_args_streamed = False

        # Final chunk with finish_reason from engine
        # If no tokens were emitted (first_chunk is still True), this is also
        # the first chunk and must include role=assistant per OpenAI spec.
        if has_emitted_tool_call:
            final_reason = "tool_calls"
        else:
            final_reason = _normalize_finish_reason(last_finish_reason)
        yield format_openai_chunk(
            completion_id=completion_id,
            model=req.model,
            delta_content="",
            finish_reason=final_reason,
            include_role=first_chunk,
        )

        # Emit usage stats if stream_options.include_usage is true
        if include_usage:
            yield format_openai_usage_chunk(
                completion_id=completion_id,
                model=req.model,
                prompt_tokens=prompt_tok,
                completion_tokens=completion_tok,
                reasoning_tokens=reasoning_tok,
                cached_tokens=cached_tok,
            )

        done_emitted = True
        yield format_openai_done()
    done_emitted = False
    try:
      async for event in with_sse_keepalive(
          _token_source(),
          http_request=request,
          cancel_event=_cancel_evt,
      ):
          # the per-request StreamingResponseBuffer was written here but NEVER
          # read or flushed (the bytes actually sent are `encoded`), so once the 64KB ring
          # filled it logged a truncation WARNING on every subsequent chunk — pure dead
          # work + log spam on the hot streaming path. Removed; yield the encoded bytes.
          yield event.encode("utf-8")
    except MemoryError:
        if _cancel_evt is not None:
            _cancel_evt.set()
        yield b'data: {"error": {"message": "Out of GPU memory", "type": "memory_error", "code": "oom"}}\n\n'
        if not done_emitted:
            yield b"data: [DONE]\n\n"
    except Exception:
        if _cancel_evt is not None:
            _cancel_evt.set()
        logger.error("Chat streaming error", exc_info=True)
        err_payload = {"error": {"message": "Internal server error", "type": "internal_error"}}
        yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n".encode()
        if not done_emitted:
            yield b"data: [DONE]\n\n"
    finally:
      _release_lora_adapter(engine, loaded_adapter)
      if _tracker is not None:
          with contextlib.suppress(Exception):
              _tracker.unregister(completion_id)
      if prompt_tok > 0 or completion_tok > 0 or reasoning_tok > 0:
          with contextlib.suppress(Exception):
              _record_metrics(prompt_tok, completion_tok)  # completion_tok already incl. reasoning
