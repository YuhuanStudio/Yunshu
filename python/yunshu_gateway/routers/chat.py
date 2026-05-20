from __future__ import annotations
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
from .models import _check_permission
from yunshu_engine.tool_call_streamer import ToolCallStreamer
from yunshu_engine.gateway_optimizer import get_streaming_buffer, return_streaming_buffer

logger = logging.getLogger(__name__)

_MAX_STREAMING_TEXT_BUFFER = 1 * 1024 * 1024  # 1MB safety limit
_TRUNCATE_KEEP = 512 * 1024  # Keep last 512KB for stop-sequence detection


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


def _apply_lora_adapter(engine, adapter_id: str | None) -> str | None:
    """Apply a LoRA adapter to the engine for this request.

    Returns the adapter_id if loaded, None if not applicable.
    The caller must call _release_lora_adapter() after generation.

    Uses acquire_adapter/release_adapter (ref-counted) instead of
    load_adapter/unload_adapter to prevent concurrent-request eviction.
    """
    if not adapter_id:
        return None
    lora_mgr = getattr(engine, 'get_lora_manager', lambda: None)()
    if lora_mgr is None:
        logger.warning(f"LoRA adapter '{adapter_id}' requested but engine has no LoRA manager")
        return None
    if lora_mgr.acquire_adapter(adapter_id):
        return adapter_id
    logger.warning(f"Failed to acquire LoRA adapter '{adapter_id}'")
    return None


def _release_lora_adapter(engine, adapter_id: str | None) -> None:
    """Release a LoRA adapter after generation completes.

    Uses release_adapter (decrements ref count) instead of unload_adapter.
    The adapter stays loaded for reuse until LRU eviction or explicit unload.
    """
    if not adapter_id:
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


ContentPart = Union[TextContent, ImageContent, dict]


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
    content: Union[str, list[ContentPart], None] = None
    # Tool call fields for multi-turn conversations (OpenAI spec)
    tool_calls: Optional[list[ToolCall]] = None       # assistant messages with tool calls
    tool_call_id: Optional[str] = None                 # tool role messages (result of a tool call)
    name: Optional[str] = None                          # tool role messages (function name)


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
    max_completion_tokens: Optional[int] = Field(default=None, ge=1, le=131072)
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
    logits_processors: Optional[list] = None  # SAMP-2: User-provided custom logits processors
    timeout: Optional[float] = Field(default=None, ge=1.0, le=600.0)  # Request timeout in seconds

    @model_validator(mode="after")
    def validate_request(self):
        if not self.model or not self.model.strip():
            raise ValueError("model: field is required and cannot be empty")
        if not self.messages:
            raise ValueError("messages: field is required and cannot be empty")
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
            return schema
        return "json_object"

    return None


def _extract_messages(msgs: list[ChatMessage]) -> list[dict]:
    """Convert ChatMessage objects to dicts, preserving multimodal content.

    Follows oMLX's extract_multimodal_content pattern:
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
                        return True
    return False


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
                # Decode top_logprobs with bytes field
                raw_top = lp.get("top_logprobs", [])
                decoded_top = []
                for tlp in raw_top:
                    if isinstance(tlp, dict):
                        tlp_token = tlp.get("token", "")
                        if not tlp_token and tokenizer and "token_id" in tlp:
                            try:
                                tlp_token = tokenizer.decode([tlp["token_id"]])
                            except Exception:
                                pass
                        decoded_top.append({
                            "token": tlp_token,
                            "logprob": tlp.get("logprob", 0.0),
                            "bytes": list(tlp_token.encode("utf-8")) if tlp_token else [],
                        })
                    else:
                        decoded_top.append(tlp)
                entries.append({
                    "token": token_str,
                    "logprob": lp.get("logprob", 0.0),
                    "bytes": list(token_str.encode("utf-8")) if token_str else [],
                    "top_logprobs": decoded_top,
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
        "error": "length",
        "timeout": "length",
        "memory_limit": "length",
        "memory_exceeded": "length",
    }
    return _INTERNAL_MAP.get(reason, "stop")


async def _build_multi_choice(
    engine, req, messages, completion_id, is_batched, json_schema,
    cancel_event=None,
):
    """Build n > 1 completions by running parallel generation calls."""
    import asyncio

    prompt_tok = 0
    completion_tok = 0
    reasoning_tok = 0
    cached_tok = 0
    choices = []

    async def _gen_one(idx: int):
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
                seed=(req.seed + idx) if req.seed is not None else None,
                enable_thinking=req.enable_thinking,
                json_schema=json_schema,
                grammar=req.grammar,
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
                cancel_event=cancel_event,
                timeout_seconds=req.timeout,
            )
            text = result.text
            pt = result.prompt_tokens
            ct = result.completion_tokens
            fr = _normalize_finish_reason(result.finish_reason)
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
                seed=(req.seed + idx) if req.seed is not None else None,
                enable_thinking=req.enable_thinking,
                stop_token_ids=req.stop_token_ids,
                thinking_budget=req.thinking_budget,
                reasoning_effort=req.reasoning_effort,
                xtc_probability=req.xtc_probability,
                xtc_threshold=req.xtc_threshold,
                spec_decode=req.spec_decode,
                json_schema=json_schema,
                grammar=req.grammar,
                priority=req.priority,
                logprobs=req.logprobs,
                top_logprobs=req.top_logprobs,
                logits_processors=req.logits_processors,
                cancel_event=cancel_event,
                timeout_seconds=req.timeout,
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
            from ..streaming import _sanitize_arguments
            message["tool_calls"] = [
                {"id": _generate_tool_call_id(), "type": "function", "function": {"name": tc["name"], "arguments": _sanitize_arguments(tc.get("arguments", {}))}}
                for i, tc in enumerate(tool_calls)
            ]

        _gen_result = result if is_batched else state
        _rt = getattr(_gen_result, 'reasoning_tokens', 0)
        _ct_cached = getattr(_gen_result, 'cached_tokens', 0)
        choice = {"index": idx, "message": message, "finish_reason": fr}
        if lp:
            choice["logprobs"] = lp
        return idx, pt, ct, _rt, _ct_cached, choice

    results = await asyncio.gather(
        *[_gen_one(i) for i in range(req.n)], return_exceptions=True,
    )

    errors = []
    for i, r in enumerate(results):
        if isinstance(r, BaseException):
            errors.append((i, r))
            logger.error(f"choice {i} failed: {r}", exc_info=r)
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
            content={"error": {"message": "Internal server error", "type": "internal_error"}},
        )

    # Record metrics once for the entire n>1 request (not per-choice)
    if prompt_tok > 0 or completion_tok > 0:
        _record_metrics(prompt_tok, completion_tok)

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
    _check_permission(request, "can_infer")
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
    trace = tracer.start_trace(trace_id, metadata={
        "model": req.model,
        "max_tokens": req.effective_max_tokens(),
        "temperature": req.temperature,
        "stream": req.stream,
        "endpoint": "/chat/completions",
    })
    tracer.span(trace_id, "prefill", {"model": req.model})
    slog.info("inference_request", model=req.model, trace_id=trace_id,
              max_tokens=req.effective_max_tokens(), stream=req.stream)

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
                )

            try:
                _reasoning_tok = 0
                _cached_tok = 0
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
                        grammar=req.grammar,
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
                    )
                    raw_text = result.text
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
                        grammar=req.grammar,
                        logprobs=req.logprobs,
                        top_logprobs=req.top_logprobs,
                        priority=req.priority,
                        logits_processors=req.logits_processors,
                        cancel_event=_ns_cancel_event,
                        timeout_seconds=req.timeout,
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
            except Exception as e:
                logger.error("engine inference failed", exc_info=True)
                return JSONResponse(
                    status_code=500,
                    content={"error": {"message": "Internal server error", "type": "internal_error"}},
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
        return await run_with_disconnect_guard(request, _build_response())
    finally:
        if _ns_tracker is not None:
            try:
                _ns_tracker.unregister(completion_id)
            except Exception:
                pass


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
        grammar=req.grammar,
        timeout_seconds=req.timeout,
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
            kwargs = {
                **gen_kwargs,
                "seed": (req.seed + idx) if req.seed is not None else None,
            }
            r = await vlm_engine.generate(**kwargs)
        except MemoryError:
            return idx, None, "memory_error"
        except Exception as e:
            logger.error("VLM engine inference failed", exc_info=True)
            return idx, None, str(e)

        content = r.get("text", "") or ""
        rt = r.get("reasoning_tokens", 0)
        ct = r.get("completion_tokens", 0) or (len(tok.encode(content)) if tok else max(1, len(content) // 4))
        finish_reason = _normalize_finish_reason(r.get("finish_reason"))

        # Extract thinking content if enabled (consistent with LLM path)
        thinking_content = None
        if req.enable_thinking:
            thinking_content, content = extract_thinking(content, req.model)

        tool_calls = None
        if req.tools:
            tool_calls = extract_tool_calls_model_aware(content, req.model)
            if tool_calls:
                content = clean_tool_call_markup(content)
                finish_reason = "tool_calls"
        return idx, {
            "content": content.strip(),
            "reasoning_content": thinking_content,
            "reasoning_tokens": rt,
            "completion_tokens": ct,
            "finish_reason": finish_reason,
            "tool_calls": tool_calls,
        }, None

    n = max(req.n, 1)
    loaded_adapter = _apply_lora_adapter(vlm_engine, req.lora_adapter)
    try:
        if n == 1:
            results = [await _vlm_gen_one(0)]
        else:
            import asyncio
            results = await asyncio.gather(*[_vlm_gen_one(i) for i in range(n)], return_exceptions=True)
            # Filter out exceptions from gather
            valid_results = []
            for r in results:
                if isinstance(r, BaseException):
                    logger.error(f"VLM choice generation failed: {r}", exc_info=r)
                else:
                    valid_results.append(r)
            results = valid_results
            results.sort(key=lambda x: x[0])

    finally:
        _release_lora_adapter(vlm_engine, loaded_adapter)

    if not results:
        return JSONResponse(status_code=500, content={"error": {"message": "All choices failed", "type": "inference_error"}})

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

    vlm_usage = {
        "prompt_tokens": prompt_tok,
        "completion_tokens": total_completion_tok,
        "total_tokens": prompt_tok + total_completion_tok,
    }
    if total_reasoning_tok > 0:
        vlm_usage["completion_tokens_details"] = {"reasoning_tokens": total_reasoning_tok}

    # Record metrics for VLM non-streaming path
    if prompt_tok > 0 or total_completion_tok > 0:
        _record_metrics(prompt_tok, total_completion_tok)

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

    async def _token_source():
        nonlocal loaded_adapter, done_emitted, vlm_prompt_tok, vlm_completion_tok, metrics_recorded
        first_chunk = True
        vlm_reasoning_tok = 0
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
            grammar=req.grammar,
            cancel_event=_vlm_cancel_evt,
            timeout_seconds=req.timeout,
        )
        if json_schema:
            stream_kwargs["json_schema"] = json_schema
        async for output in vlm_engine.generate_stream(**stream_kwargs):
            if hasattr(output, 'completion_tokens') and output.completion_tokens is not None and output.completion_tokens > 0:
                vlm_completion_tok = output.completion_tokens
            elif hasattr(output, 'token_text') and output.token_text:
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
            if _vlm_token_text and not getattr(output, 'current_state', None) == "reasoning":
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
        if vlm_prompt_tok > 0 or vlm_completion_tok > 0:
            _record_metrics(vlm_prompt_tok, vlm_completion_tok)
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
    except Exception as e:
        if _vlm_cancel_evt is not None:
            _vlm_cancel_evt.set()
        logger.error("VLM streaming error", exc_info=True)
        yield b'data: {"error": {"message": "Internal server error", "type": "internal_error"}}\n\n'
        if not done_emitted:
            yield b"data: [DONE]\n\n"
    finally:
      _release_lora_adapter(vlm_engine, loaded_adapter)
      if _vlm_tracker is not None:
          try:
              _vlm_tracker.unregister(completion_id)
          except Exception:
              pass
      # Fallback metrics recording if generator raised before completing
      if not metrics_recorded and (vlm_prompt_tok > 0 or vlm_completion_tok > 0):
          try:
              _record_metrics(vlm_prompt_tok, vlm_completion_tok)
          except Exception:
              pass


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
    use_tool_streamer = req.tools is not None and len(req.tools) > 0
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
            # Per-choice tool call streamer for independent tool call extraction
            choice_tool_streamer = ToolCallStreamer() if use_tool_streamer else None
            choice_tool_call_index = 0
            choice_has_tool_call = False
            _choice_streamed_text = ""  # track emitted text for stop-sequence correction

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
                    grammar=req.grammar,
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
                    # vLLM pattern: emit prefill progress as SSE comment
                    _pf_prog = getattr(output, 'prefill_progress', None)
                    if _pf_prog is not None:
                        yield f": prefill-progress {_pf_prog[0]}/{_pf_prog[1]}\n\n".encode()
                        continue
                    if hasattr(output, 'completion_tokens') and output.completion_tokens is not None and output.completion_tokens > 0:
                        choice_completion_tok = output.completion_tokens
                    elif token_text:
                        choice_completion_tok += 1
                    # Only set finish_reason on the final token from engine
                    fr = output.finish_reason
                    if fr is not None:
                        choice_finish_reason = fr
                    _chunk_lp = _format_chat_logprobs(output.logprobs, tokenizer=getattr(engine, "_tokenizer", None)) if req.logprobs and hasattr(output, "logprobs") else None
                    # Track emitted text for stop-sequence correction
                    if token_text and not getattr(output, 'current_state', None) == "reasoning":
                        _choice_streamed_text += token_text
                        if len(_choice_streamed_text) > _MAX_STREAMING_TEXT_BUFFER:
                            logger.error("Choice streaming text buffer exceeded 1MB — truncating")
                            _choice_streamed_text = _choice_streamed_text[-_TRUNCATE_KEEP:]
                        if len(_choice_streamed_text) > _MAX_STREAMING_TEXT_BUFFER:
                            logger.error("Choice streaming text buffer exceeded 1MB — truncating")
                            _choice_streamed_text = _choice_streamed_text[-_TRUNCATE_KEEP:]
                    # Detect stop-sequence overcount on final output
                    if req.stop and choice_finish_reason == "stop" and output.finished:
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
                            elif out.tool_call:
                                yield _format_tool_call_chunk_multi(
                                    completion_id, req.model, choice_idx,
                                    out.tool_call, choice_tool_call_index,
                                    include_role=first_chunk_for_choice,
                                )
                                choice_tool_call_index += 1
                                choice_has_tool_call = True
                                first_chunk_for_choice = False
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
                    grammar=req.grammar,
                    priority=req.priority,
                    logprobs=req.logprobs,
                    top_logprobs=req.top_logprobs,
                    logits_processors=req.logits_processors,
                    cancel_event=_multi_cancel_evt,
                    timeout_seconds=req.timeout,
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
                    elif token_text:
                        choice_completion_tok += 1
                    # Only set finish_reason on the final token from engine
                    fr = getattr(output, 'finish_reason', None)
                    if fr is not None:
                        choice_finish_reason = fr
                    _chunk_lp = _format_chat_logprobs(output.logprobs, tokenizer=getattr(engine, "_tokenizer", None)) if req.logprobs and hasattr(output, "logprobs") else None
                    # Track emitted text for stop-sequence correction
                    if token_text and not getattr(output, 'current_state', None) == "reasoning":
                        _choice_streamed_text += token_text
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
                            elif out.tool_call:
                                yield _format_tool_call_chunk_multi(
                                    completion_id, req.model, choice_idx,
                                    out.tool_call, choice_tool_call_index,
                                    include_role=first_chunk_for_choice,
                                )
                                choice_tool_call_index += 1
                                choice_has_tool_call = True
                                first_chunk_for_choice = False
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
                    elif out.tool_call:
                        yield _format_tool_call_chunk_multi(
                            completion_id, req.model, choice_idx,
                            out.tool_call, choice_tool_call_index,
                            include_role=first_chunk_for_choice,
                        )
                        choice_tool_call_index += 1
                        choice_has_tool_call = True
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
    except Exception as e:
        if _multi_cancel_evt is not None:
            _multi_cancel_evt.set()
        logger.error("Chat multi-choice streaming error", exc_info=True)
        yield b'data: {"error": {"message": "Internal server error", "type": "internal_error"}}\n\n'
        if not done_emitted:
            yield b"data: [DONE]\n\n"
    finally:
      _release_lora_adapter(engine, loaded_adapter)
      if tracker is not None:
          try:
              tracker.unregister(completion_id)
          except Exception:
              pass
      if total_prompt_tok > 0 or total_completion_tok > 0:
          try:
              _record_metrics(total_prompt_tok, total_completion_tok)
          except Exception:
              pass


def _format_choice_chunk(
    completion_id: str,
    model: str,
    index: int,
    delta_content: str,
    finish_reason: Optional[str],
    include_role: bool = False,
    logprobs: Optional[dict] = None,
    thinking_content: Optional[str] = None,
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


def _format_chat_logprobs(logprobs_list: list[dict] | None, tokenizer=None) -> dict | None:
    """Format per-token logprobs from GenerationOutput into OpenAI Chat format."""
    if not logprobs_list:
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
                pass
        top_lps = lp_entry.get("top_logprobs", [])
        decoded_top = []
        for tlp in top_lps:
            tlp_token = tlp.get("token", "")
            if not tlp_token and tokenizer and "token_id" in tlp:
                try:
                    tlp_token = tokenizer.decode([tlp["token_id"]])
                except Exception:
                    pass
            decoded_top.append({
                "token": tlp_token,
                "logprob": tlp.get("logprob", 0.0),
                "bytes": list(tlp_token.encode("utf-8")) if tlp_token else [],
            })
        content.append({
            "token": token_str,
            "logprob": lp_entry.get("logprob", 0.0),
            "bytes": list(token_str.encode("utf-8")) if token_str else [],
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

    # Per-request streaming buffer for zero-alloc SSE ring buffering
    _stream_buf = None
    try:
        _stream_buf = get_streaming_buffer()
    except Exception:
        logger.debug("StreamingResponseBuffer creation failed", exc_info=True)

    async def _token_source():
        nonlocal tool_call_index, has_emitted_tool_call, prompt_tok, completion_tok, reasoning_tok, cached_tok, done_emitted
        first_chunk = True
        last_finish_reason = None  # track actual finish_reason from engine
        _streamed_text = ""  # track text emitted to client for stop-sequence correction
        _stop_overcount = 0  # tokens to subtract when stop sequence spans multiple tokens

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
                grammar=req.grammar,
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
            ):
                token_text = output.new_text
                # vLLM pattern: emit prefill progress as SSE comment for
                # client-side progress bars during long chunked prefills.
                _pf_prog = getattr(output, 'prefill_progress', None)
                if _pf_prog is not None:
                    yield f": prefill-progress {_pf_prog[0]}/{_pf_prog[1]}\n\n".encode()
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
                elif token_text:
                    completion_tok += 1
                # Track streamed text for stop-sequence overcount correction
                if token_text and not getattr(output, 'current_state', None) == "reasoning":
                    _streamed_text += token_text
                    if len(_streamed_text) > _MAX_STREAMING_TEXT_BUFFER:
                        logger.error("Streaming text buffer exceeded 1MB — truncating")
                        _streamed_text = _streamed_text[-_TRUNCATE_KEEP:]
                # Detect stop-sequence overcount: if stop sequences are provided
                # and the engine's finish_reason is "stop", the engine may have
                # overcounted completion_tok when a multi-token stop suffix was
                # matched (engine only decrements by 1 regardless of suffix length).
                if req.stop and last_finish_reason == "stop" and output.finished:
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
                                        _stop_overcount = completion_tok - _correct_count
                                        completion_tok = _correct_count
                                except Exception:
                                    pass
                            break

                _chunk_lp = _format_chat_logprobs(output.logprobs, tokenizer=getattr(engine, "_tokenizer", None)) if req.logprobs and hasattr(output, "logprobs") else None

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
                        elif out.tool_call:
                            yield _format_tool_call_chunk(out.tool_call, tool_call_index, include_role=first_chunk)
                            tool_call_index += 1
                            has_emitted_tool_call = True
                            first_chunk = False
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
                grammar=req.grammar,
                priority=req.priority,
                logprobs=req.logprobs,
                top_logprobs=req.top_logprobs,
                logits_processors=req.logits_processors,
                cancel_event=_cancel_evt,
                timeout_seconds=req.timeout,
            ):
                # Track token counts for usage reporting
                if hasattr(output, 'prompt_tokens') and output.prompt_tokens:
                    prompt_tok = output.prompt_tokens
                if hasattr(output, 'completion_tokens') and output.completion_tokens is not None and output.completion_tokens > 0:
                    completion_tok = output.completion_tokens
                elif hasattr(output, 'token_text') and output.token_text:
                    completion_tok += 1
                if hasattr(output, 'reasoning_tokens') and output.reasoning_tokens:
                    reasoning_tok = output.reasoning_tokens
                if hasattr(output, 'cached_tokens') and output.cached_tokens:
                    cached_tok = max(cached_tok, output.cached_tokens)
                if output.finish_reason is not None:
                    last_finish_reason = output.finish_reason
                _chunk_lp = _format_chat_logprobs(output.logprobs, tokenizer=getattr(engine, "_tokenizer", None)) if req.logprobs and hasattr(output, "logprobs") else None
                # Track streamed text for stop-sequence overcount correction
                _legacy_token_text = output.token_text or ""
                if _legacy_token_text and not getattr(output, 'current_state', None) == "reasoning":
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
                                        _stop_overcount = completion_tok - _correct_count
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
                            elif out.tool_call:
                                yield _format_tool_call_chunk(out.tool_call, tool_call_index, include_role=first_chunk)
                                tool_call_index += 1
                                has_emitted_tool_call = True
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
                elif out.tool_call:
                    yield _format_tool_call_chunk(out.tool_call, tool_call_index, include_role=first_chunk)
                    tool_call_index += 1
                    has_emitted_tool_call = True
                    first_chunk = False

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
          encoded = event.encode("utf-8")
          # Best-effort write to streaming buffer
          if _stream_buf is not None:
              try:
                  _stream_buf.write(encoded)
              except Exception:
                  logger.debug("StreamingResponseBuffer write failed", exc_info=True)
          yield encoded
    except MemoryError:
        if _cancel_evt is not None:
            _cancel_evt.set()
        yield b'data: {"error": {"message": "Out of GPU memory", "type": "memory_error", "code": "oom"}}\n\n'
        if not done_emitted:
            yield b"data: [DONE]\n\n"
    except Exception as e:
        if _cancel_evt is not None:
            _cancel_evt.set()
        logger.error("Chat streaming error", exc_info=True)
        err_payload = {"error": {"message": str(e)[:200], "type": "internal_error"}}
        yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n".encode("utf-8")
        if not done_emitted:
            yield b"data: [DONE]\n\n"
    finally:
      _release_lora_adapter(engine, loaded_adapter)
      if _tracker is not None:
          try:
              _tracker.unregister(completion_id)
          except Exception:
              pass
      if prompt_tok > 0 or completion_tok > 0:
          try:
              _record_metrics(prompt_tok, completion_tok)
          except Exception:
              pass
      # Log buffer stats and return to pool to prevent 64KB leak per request
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
          try:
              return_streaming_buffer(_stream_buf)
          except Exception:
              logger.debug("StreamingResponseBuffer return to pool failed", exc_info=True)
