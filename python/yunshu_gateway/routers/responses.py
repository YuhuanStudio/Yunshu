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
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from yunshu_engine import settings

from ..engine import get_engine, get_engine_for_model
from ..error_envelope import EngineStreamError, server_error_body
from ..usage_shapes import responses_usage

logger = logging.getLogger(__name__)

_MAX_STREAMING_TEXT_BUFFER = 1 * 1024 * 1024
_TRUNCATE_KEEP = 512 * 1024

# ──────────────────────────────────────────────────────────────────────
# Response store — keeps the most recent N responses for retrieval via
# GET /v1/responses/{id} when the client sent `store: true` on create.
# In-memory, bounded LRU; not persisted across process restarts.
# ──────────────────────────────────────────────────────────────────────
_RESPONSE_STORE_MAX = 1024
_response_store: OrderedDict[str, dict] = OrderedDict()
_response_store_lock = threading.Lock()


def _cache_kw(req, engine) -> dict:
    if not hasattr(engine, "_runner_input"):
        return {}
    return {
        "prompt_cache_key": req.prompt_cache_key,
        "prompt_cache_retention": req.prompt_cache_retention,
        "prompt_cache_options": req.prompt_cache_options,
    }


def _forced_by_grammar(req) -> bool:
    """A forced tool_choice (required / one function) that the tool-call grammar enforces."""
    forced = req.tool_choice == "required" or (
        isinstance(req.tool_choice, dict) and bool(req.tool_choice.get("name"))
    )
    return forced


def _chat_tool_choice(tool_choice: Any) -> Any:
    """Responses names a forced function flat (``{"type": "function", "name": ...}``); the chat
    tool prompt expects the chat shape, so a named choice becomes a ``ToolChoiceFunction``.
    An ``allowed_tools`` choice keeps only its ``mode``."""
    from .chat import ToolChoiceFunction, ToolFunction

    if not isinstance(tool_choice, dict):
        return tool_choice
    if tool_choice.get("name"):
        return ToolChoiceFunction(function=ToolFunction(name=tool_choice["name"]))
    mode = tool_choice.get("mode")
    return mode if isinstance(mode, str) else None


def _native_kw(req) -> dict:
    """``tools=`` for an engine whose chat template renders tool definitions itself; the
    forced tool_choice / parallel flag ride along so the tool-call grammar can enforce them."""
    tools = getattr(req, "_native_tools", None)
    if not tools:
        return {}
    kw = {"tools": tools}
    if _forced_by_grammar(req):
        kw["tool_choice"] = req.tool_choice
        kw["parallel_tool_calls"] = req.parallel_tool_calls
    return kw


def _seal_extra(req, text: str | None) -> dict:
    """``encrypted_content`` for a reasoning item, when the request asked for it (``include``)."""
    if text and "reasoning.encrypted_content" in (getattr(req, "include", None) or []):
        return {"encrypted_content": seal_reasoning(text)}
    return {}


def _store_response(response_id: str, payload: dict) -> None:
    """Insert/refresh a stored response (LRU evict to keep below the cap)."""
    with _response_store_lock:
        _response_store[response_id] = payload
        _response_store.move_to_end(response_id)
        while len(_response_store) > _RESPONSE_STORE_MAX:
            _response_store.popitem(last=False)


def _get_stored_response(response_id: str) -> dict | None:
    with _response_store_lock:
        payload = _response_store.get(response_id)
        if payload is not None:
            _response_store.move_to_end(response_id)
        return payload


def _delete_stored_response(response_id: str) -> bool:
    """Remove a stored response. Returns True if it existed."""
    with _response_store_lock:
        return _response_store.pop(response_id, None) is not None


def _resolve_owner(request) -> str:
    """Resolve the calling actor for response-store ownership.

    Mirrors the actor resolution every other per-handle router uses
    so stored responses can be ownership-gated on retrieval/cancel.
    """
    try:
        from yunshu_control.audit_log import resolve_actor

        return resolve_actor(request) or ""
    except Exception:
        return ""


def _is_admin(request) -> bool:
    """Admin/static-token/auth-disabled check (matches cancel.py:101-107)."""

    if settings.get_bool("YUNSHU_AUTH_DISABLED"):
        return True
    role_str = str(getattr(getattr(request, "state", None), "role", "") or "")
    return role_str.lower() in (
        "admin",
        "system",
        "owner",
    ) or role_str.upper().endswith("ADMIN")


def _owns_stored(request, payload: dict) -> bool:
    """True if the caller may read/cancel this stored response.

    Unowned (legacy) entries stay readable to avoid breaking older
    stores; admins bypass. Cross-tenant access to an owned entry is denied.
    """
    owner = payload.get("_owner")
    if not owner:
        return True
    if _is_admin(request):
        return True
    return owner == _resolve_owner(request)


def _config_echo(req: ResponsesRequest) -> dict:
    """The request's configuration as the Response object echoes it (OpenAI shape)."""
    reasoning = dict(req.reasoning) if isinstance(req.reasoning, dict) else {}
    if req.reasoning_effort and "effort" not in reasoning:
        reasoning["effort"] = req.reasoning_effort
    text = dict(req.text) if isinstance(req.text, dict) else {}
    text.setdefault("format", {"type": "text"})
    return {
        "instructions": req.instructions,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_output_tokens": req.max_output_tokens,
        "tools": [t.model_dump(exclude_none=True) for t in (req.tools or [])],
        "tool_choice": req.tool_choice or "auto",
        "parallel_tool_calls": req.parallel_tool_calls,
        "text": text,
        "reasoning": {"effort": reasoning.get("effort"), "summary": None}
        if reasoning
        else {"effort": None, "summary": None},
        "truncation": req.truncation or "disabled",
        "store": bool(req.store),
        "background": bool(req.background),
        "previous_response_id": req.previous_response_id,
        "service_tier": req.service_tier or "default",
        "user": req.user,
        "top_logprobs": req.top_logprobs or 0,
        "max_tool_calls": req.max_tool_calls,
        "prompt_cache_key": req.prompt_cache_key,
        "safety_identifier": req.safety_identifier,
        "metadata": req.metadata,
    }


def _auto_truncate(messages: list, tokens: int, limit: int, count) -> tuple[list, int]:
    """``truncation="auto"``: drop the oldest non-system messages (and the tool results
    that would be left without their call) until the prompt fits, keeping the last one."""
    msgs = list(messages)
    while tokens > limit:
        idx = next(
            (i for i, m in enumerate(msgs[:-1]) if m.get("role") != "system"), None
        )
        if idx is None:
            break
        del msgs[idx]
        while idx < len(msgs) - 1 and msgs[idx].get("role") == "tool":
            del msgs[idx]
        tokens = count(msgs)
    return msgs, tokens


def _public_stored(payload: dict) -> dict:
    """Strip internal plumbing keys (``_owner``, ``_input_messages``, …) before
    returning a stored response to the client."""
    return {k: v for k, v in payload.items() if not k.startswith("_")}


import contextlib

# A cut-short generation (token limit, the request's timeout, the memory guard) is
# "incomplete", never "completed".
_TRUNCATED_FINISH = ("length", "timeout", "memory_limit", "memory_exceeded")
_INCOMPLETE_FINISH = (
    *_TRUNCATED_FINISH,
    "cancel",
    "cancelled",
    "abort",
    "aborted",
    "error",
    "content_filter",
)


async def _vlm_to_responses(req, messages, request, logit_bias, own_input_messages):
    """Run a multimodal (image/audio) request through the chat VLM engine, then
    re-wrap its output in the OpenAI *Responses* shape — both the non-stream
    Response object and the response.* event stream — and honor ``store``.

    The VLM engine only speaks the Chat-Completions surface, so we always invoke it
    NON-streaming to get the full text, then emit a faithful (single-delta) Responses
    event sequence when the client asked for streaming. This trades incremental VLM
    token streaming for protocol correctness (the prior code emitted the wrong event
    family entirely). See."""
    import json as _json

    from .chat import (
        ChatCompletionRequest,
        ChatMessage,
        ToolDefinition,
        ToolFunction,
        _handle_vlm_chat,
    )

    chat_response_format = req.response_format or None
    chat_messages = [
        ChatMessage(role=m.get("role", "user"), content=m.get("content", ""))
        for m in messages
    ]
    # Force stream=False / n=1: we wrap a single completion into the Responses shape.
    chat_req = ChatCompletionRequest(
        model=req.model,
        messages=chat_messages,
        max_tokens=req.max_output_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        min_p=req.min_p,
        top_n_sigma=req.top_n_sigma,
        repetition_penalty=req.repetition_penalty,
        frequency_penalty=req.frequency_penalty,
        presence_penalty=req.presence_penalty,
        min_tokens=req.min_tokens,
        ignore_eos=req.ignore_eos,
        suppress_tokens=req.suppress_tokens,
        logit_bias=logit_bias,
        seed=req.seed,
        enable_thinking=req.enable_thinking,
        thinking_budget=req.thinking_budget,
        reasoning_effort=req.reasoning_effort,
        stop=req.stop,
        stop_token_ids=req.stop_token_ids,
        logprobs=req.logprobs,
        top_logprobs=req.top_logprobs,
        spec_decode=req.spec_decode,
        n=1,
        stream=False,
        stream_options=None,
        tools=[
            ToolDefinition(
                function=ToolFunction(
                    name=t.name, description=t.description, parameters=t.parameters
                )
            )
            for t in req.tools
        ]
        if req.tools
        else None,
        tool_choice=_chat_tool_choice(req.tool_choice),
        parallel_tool_calls=req.parallel_tool_calls,
        logits_processors=req.logits_processors,
        response_format=chat_response_format,
        xtc_probability=req.xtc_probability,
        xtc_threshold=req.xtc_threshold,
        lora_adapter=req.lora_adapter,
        priority=req.priority,
        user=req.user,
        timeout=req.timeout,
        grammar=req.grammar,
    )
    vlm_json_schema = _parse_response_format(chat_response_format, req.grammar)
    chat_resp = await _handle_vlm_chat(
        chat_req, messages, request, json_schema=vlm_json_schema
    )

    # _handle_vlm_chat returns a JSONResponse (non-stream forced above). Decode it.
    body = getattr(chat_resp, "body", None)
    chat_data = _json.loads(bytes(body)) if body is not None else (chat_resp or {})
    choice0 = (chat_data.get("choices") or [{}])[0]
    msg = choice0.get("message", {}) or {}
    text = msg.get("content") or ""
    reasoning = msg.get("reasoning_content")
    finish = choice0.get("finish_reason") or "stop"
    usage = chat_data.get("usage", {}) or {}
    pt = int(usage.get("prompt_tokens", 0) or 0)
    ct = int(usage.get("completion_tokens", 0) or 0)
    rt = int(
        (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
    )
    cached = int(
        (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    )
    status = "incomplete" if finish in _INCOMPLETE_FINISH else "completed"

    # Allocate the response id (consume the background forced id, like the text path).
    forced_id = getattr(request.state, "_forced_response_id", None)
    if forced_id:
        response_id = forced_id
        with contextlib.suppress(Exception):
            request.state._forced_response_id = None
    else:
        response_id = f"resp-{uuid.uuid4().hex[:24]}"

    text_s = text if getattr(req, "_custom_raw_input", False) else text.strip()
    msg_id = f"msg-{uuid.uuid4().hex[:24]}"
    content_parts = [{"type": "output_text", "text": text_s, "annotations": []}]
    # reasoning is a SEPARATE output item (rs_ id) preceding the message —
    # OpenAI Responses spec shape — not a content part of the message.
    rs_id = f"rs-{uuid.uuid4().hex[:24]}" if reasoning else None
    reasoning_item = (
        {
            "type": "reasoning",
            "id": rs_id,
            "summary": [{"type": "summary_text", "text": reasoning}],
            **_seal_extra(req, reasoning),
            "status": "completed",
        }
        if reasoning
        else None
    )
    message_item = {
        "type": "message",
        "id": msg_id,
        "role": "assistant",
        "content": content_parts,
        "status": "completed",
    }
    msg_idx = 1 if reasoning_item else 0
    final_output = ([reasoning_item] if reasoning_item else []) + [message_item]
    for call in msg.get("tool_calls") or []:
        fn = call.get("function") or {}
        final_output.append(
            {
                "type": "function_call",
                "id": f"fc-{uuid.uuid4().hex[:24]}",
                "call_id": call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                "name": fn.get("name", ""),
                "arguments": fn.get("arguments", "{}"),
                "status": "completed" if status == "completed" else "incomplete",
            }
        )
    payload = {
        "id": response_id,
        "object": "response",
        **_config_echo(req),
        "created_at": int(time.time()),
        "completed_at": int(time.time()),
        "model": req.model,
        "status": status,
        "output": final_output,
        "metadata": req.metadata,
        "usage": {
            "input_tokens": pt,
            "output_tokens": ct,
            "total_tokens": pt + ct,
            "output_tokens_details": {"reasoning_tokens": rt},
            "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": 0},
        },
    }
    if req.store:
        _persist = dict(payload)
        _persist["_input_messages"] = own_input_messages
        _persist["previous_response_id"] = req.previous_response_id
        _persist["_owner"] = _resolve_owner(request)
        _store_response(response_id, _persist)

    if not req.stream:
        return JSONResponse(payload)

    if msg.get("tool_calls"):
        from ..responses_client_tools import replay

        return StreamingResponse(replay(payload), media_type="text/event-stream")

    async def _replay():
        _seq = 0

        def _s():
            nonlocal _seq
            v = _seq
            _seq += 1
            return v

        yield format_responses_created(response_id, req.model, seq=_s())
        yield format_responses_in_progress(response_id, req.model, seq=_s())
        # emit the reasoning item (output_index 0) first when present, then the
        # message at output_index `msg_idx` (1 if reasoning, else 0).
        if reasoning_item is not None:
            yield format_responses_output_item_added(
                response_id,
                req.model,
                item_id=rs_id,
                output_index=0,
                seq=_s(),
                item_type="reasoning",
            )
            yield (
                "event: response.reasoning_summary_part.added\ndata: "
                + json.dumps(
                    {
                        "type": "response.reasoning_summary_part.added",
                        "item_id": rs_id,
                        "output_index": 0,
                        "summary_index": 0,
                        "part": {"type": "summary_text", "text": ""},
                        "sequence_number": _s(),
                    }
                )
                + "\n\n"
            )
            yield (
                "event: response.reasoning_summary_text.delta\ndata: "
                + json.dumps(
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": rs_id,
                        "output_index": 0,
                        "summary_index": 0,
                        "delta": reasoning,
                        "sequence_number": _s(),
                    }
                )
                + "\n\n"
            )
            yield (
                "event: response.reasoning_summary_text.done\ndata: "
                + json.dumps(
                    {
                        "type": "response.reasoning_summary_text.done",
                        "item_id": rs_id,
                        "output_index": 0,
                        "summary_index": 0,
                        "text": reasoning,
                        "sequence_number": _s(),
                    }
                )
                + "\n\n"
            )
            yield (
                "event: response.reasoning_summary_part.done\ndata: "
                + json.dumps(
                    {
                        "type": "response.reasoning_summary_part.done",
                        "item_id": rs_id,
                        "output_index": 0,
                        "summary_index": 0,
                        "part": {"type": "summary_text", "text": reasoning},
                        "sequence_number": _s(),
                    }
                )
                + "\n\n"
            )
            yield (
                "event: response.output_item.done\ndata: "
                + json.dumps(
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": reasoning_item,
                        "sequence_number": _s(),
                    }
                )
                + "\n\n"
            )
        yield format_responses_output_item_added(
            response_id, req.model, item_id=msg_id, output_index=msg_idx, seq=_s()
        )
        yield format_responses_content_part_added(
            item_id=msg_id, output_index=msg_idx, content_index=0, seq=_s()
        )
        if text_s:
            yield format_responses_text_delta(
                delta=text_s,
                item_id=msg_id,
                output_index=msg_idx,
                content_index=0,
                seq=_s(),
            )
        yield format_responses_text_done(
            text=text_s, item_id=msg_id, output_index=msg_idx, content_index=0, seq=_s()
        )
        yield format_responses_content_part_done(
            item_id=msg_id, text=text_s, output_index=msg_idx, content_index=0, seq=_s()
        )
        yield format_responses_output_item_done(
            item_id=msg_id, text=text_s, output_index=msg_idx, seq=_s()
        )
        if status == "incomplete":
            yield format_responses_incomplete(
                response_id,
                req.model,
                reason="max_output_tokens" if finish in _TRUNCATED_FINISH else finish,
                output=final_output,
                input_tokens=pt,
                output_tokens=ct,
                total_tokens=pt + ct,
                reasoning_tokens=rt,
                cached_tokens=cached,
                seq=_s(),
            )
        else:
            yield format_responses_completed(
                response_id,
                req.model,
                output=final_output,
                input_tokens=pt,
                output_tokens=ct,
                total_tokens=pt + ct,
                reasoning_tokens=rt,
                cached_tokens=cached,
                seq=_s(),
            )

    return StreamingResponse(
        _replay(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


from ..reasoning_token import reasoning_text_of, seal_reasoning
from ..server_tools.responses_loop import (
    create_with_server_tools_responses,
    function_tools,
    has_server_tools_responses,
    input_item_to_messages,
)
from ..streaming import (
    format_responses_completed,
    format_responses_content_part_added,
    format_responses_content_part_done,
    format_responses_created,
    format_responses_failed,
    format_responses_in_progress,
    format_responses_incomplete,
    format_responses_output_item_added,
    format_responses_output_item_done,
    format_responses_text_delta,
    format_responses_text_done,
    run_with_disconnect_guard,
)
from .chat import (
    _format_chat_logprobs,
    _normalize_finish_reason,
    _per_choice_seed,
    _record_metrics,
)
from .models import _check_permission

router = APIRouter(tags=["responses"])


class ResponseInputText(BaseModel):
    type: str = "message"
    role: str = "user"
    # Optional so tool-conversation items (function_call / function_call_output,
    # which carry no `content`) validate. A message-type item still requires
    # content — enforced by the validator below.
    content: str | list[dict] | None = None
    # Responses API tool-conversation item fields (type=function_call /
    # function_call_output) — let a client feed a prior tool call and its result
    # back in for multi-turn agent loops.
    call_id: str | None = None
    name: str | None = None
    arguments: Any = (
        None  # tool_search_call arguments are a JSON value, functions use strings
    )
    # function_call_output / custom_tool_call_output carry a string or a list of content items.
    output: Any = None

    # Every other Responses input item (reasoning, custom_tool_call, web_search_call, mcp_call,
    # mcp_list_tools, mcp_approval_request/response, compaction, item_reference, ...) keeps its
    # own fields; _convert_to_messages decides what the model sees.
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _require_content_for_messages(self):
        if self.type in ("message", "") and self.content is None:
            raise ValueError("content is required for message input items")
        if (
            self.type == "function_call"
            and self.arguments is not None
            and not isinstance(self.arguments, str)
        ):
            raise ValueError("function_call arguments must be a JSON string")
        if self.type == "custom_tool_call" and not isinstance(
            getattr(self, "input", None), str
        ):
            raise ValueError("custom_tool_call input must be a string")
        return self


class ResponseTool(BaseModel):
    """A declared tool. Besides ``function`` it can be ``web_search*``, ``mcp``, ``namespace``
    (Codex), ``custom`` ...; those keep their own fields (extra="allow") and have no ``name``."""

    type: str = "function"
    name: str = ""
    description: str | None = None
    parameters: dict | None = None

    model_config = ConfigDict(extra="allow")


class StreamOptions(BaseModel):
    """OpenAI stream_options parameter."""

    include_usage: bool = False


class ResponsesRequest(BaseModel):
    model: str
    input: str | list[ResponseInputText]
    instructions: str | None = None
    previous_response_id: str | None = None
    # Conversations API: a conversation id (or {id}); its items are prepended to `input` and this
    # turn is appended to it. Mutually exclusive with previous_response_id.
    conversation: str | dict | None = None
    # [{"type": "compaction", "compact_threshold": N}]: compact the input first when it exceeds N tokens.
    context_management: list[dict] | None = None
    max_output_tokens: int = Field(
        default_factory=lambda: int(settings.get("YUNSHU_DEFAULT_MAX_TOKENS")),
        ge=1,
        le=1048576,
    )
    # OpenAI Chat Completions legacy alias — accept silently and alias to
    # max_output_tokens so old client code doesn't run unbounded against
    # /v1/responses. Caught by validate_request hook below.
    max_completion_tokens: int | None = Field(default=None, ge=1, le=1048576)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    stream: bool = False
    n: int = Field(default=1, ge=1, le=128)
    tools: list[ResponseTool] | None = None
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool = True
    response_format: dict | None = None
    text: dict | None = None  # Responses API: {format: {type, name, schema, strict}}
    seed: int | None = None
    enable_thinking: bool | None = None
    thinking_budget: int | None = Field(default=None, ge=1, le=32768)
    reasoning_effort: str | None = None
    # OpenAI's canonical Responses API carries reasoning depth as a nested object
    # `reasoning: {effort: "high"}`. Accept it (the validator maps .effort →
    # reasoning_effort); without this field Pydantic silently dropped it and reasoning
    # models were served without thinking.
    reasoning: dict | None = None
    # WebSocket-mode prewarm (Codex): `generate: false` prefills the prompt prefix and
    # returns a response id without producing output.
    generate: bool | None = None
    # Fields real clients (Codex, the OpenAI SDK) send; ``include`` and ``max_tool_calls`` are
    # read by the server-tool loop, the rest are accepted for compatibility.
    include: list[str] | None = None
    max_tool_calls: int | None = Field(default=None, ge=1)
    prompt_cache_key: str | None = None
    client_metadata: dict | None = None
    service_tier: str | None = None
    repetition_penalty: float = Field(default=1.0, ge=0.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: dict[str, float] | None = None
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    top_n_sigma: float = Field(default=0.0, ge=0.0, le=10.0)
    stop: list[str] | None = None

    @field_validator("stop", mode="before")
    @classmethod
    def _stop_str_to_list(cls, v):
        return [v] if isinstance(v, str) else v

    stop_token_ids: list[int] | None = None
    logprobs: bool = False
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    spec_decode: bool = False
    xtc_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    xtc_threshold: float = Field(
        default=0.0, ge=0.0, le=0.5
    )  # engine requires [0,0.5]; le=1.0 made out-of-range 500 not 422
    # Extended sampling controls. These were honored on
    # /v1/chat/completions + /v1/completions but were never declared or plumbed here,
    # so a Responses request setting them got them SILENTLY ignored (suppress_tokens →
    # banned words appeared; min_tokens → empty/short output; ignore_eos → forced-length
    # broken). Same gap as chat.py:352-354.
    min_tokens: int = Field(default=0, ge=0)
    ignore_eos: bool = False
    suppress_tokens: list[int] | None = None
    grammar: dict | None = None
    lora_adapter: str | None = None
    stream_options: StreamOptions | None = None  # {"include_usage": true}
    # Whether to store the response for later retrieval via
    # GET /v1/responses/{id} or for previous_response_id chaining.
    # OpenAI's official Responses API defaults to True; match that so callers
    # who chain `previous_response_id` without explicitly setting `store:true`
    # don't silently lose conversation memory.
    store: bool | None = True
    # OpenAI background mode: when true, POST returns immediately with a `queued`
    # response and generation runs asynchronously; the client polls
    # GET /v1/responses/{id} (and may cancel via POST /v1/responses/{id}/cancel).
    # Requires storage (forced on internally). Not supported together with stream
    # (would need resumable SSE) — when both are set, falls through to streaming.
    background: bool = False
    user: str | None = None
    # OpenAI Responses echoes the client's `metadata` map back verbatim (up to
    # 16 string key/value pairs). It was previously declared nowhere and silently dropped
    # (Pydantic extra="ignore"), and every payload echoed {"user_id": user} instead — a
    # client using metadata for request correlation got nothing back. Now echoed.
    metadata: dict | None = None
    # OpenAI Responses fields that used to be dropped silently. ``truncation="auto"``
    # drops the oldest input items when the prompt would overflow the context window
    # (``"disabled"``, the default, answers 400); ``max_tool_calls`` caps the function
    # calls of one response; the rest are echoed on the Response object.
    truncation: Literal["auto", "disabled"] | None = None
    max_tool_calls: int | None = Field(default=None, ge=1)
    include: list[str] | None = None
    conversation: str | dict | None = None
    service_tier: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    prompt_cache_options: dict | None = None
    safety_identifier: str | None = None
    priority: int = Field(default=0, ge=0, le=100)
    logits_processors: list | None = None  # User-provided custom logits processors
    timeout: float | None = Field(
        default=None, ge=1.0, le=600.0
    )  # Request timeout in seconds

    @model_validator(mode="before")
    @classmethod
    def _null_means_unset(cls, data):
        from ..schemas.nulls import clean_request, fold_allowed_tools

        return fold_allowed_tools(clean_request(cls, data))

    @model_validator(mode="before")
    @classmethod
    def _text_format_to_response_format(cls, data):
        """Responses API structured output: ``text.format`` -> the chat-style field."""
        if isinstance(data, dict) and data.get("response_format") is None:
            fmt = (
                (data.get("text") or {}).get("format")
                if isinstance(data.get("text"), dict)
                else None
            )
            if isinstance(fmt, dict):
                if fmt.get("type") == "json_schema":
                    data = dict(data)
                    data["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": fmt.get("name", "response"),
                            "schema": fmt.get("schema", {}),
                            "strict": fmt.get("strict", True),
                        },
                    }
                elif fmt.get("type") == "json_object":
                    data = dict(data)
                    data["response_format"] = {"type": "json_object"}
        return data

    @model_validator(mode="after")
    def validate_request(self):
        if not self.model or not self.model.strip():
            raise ValueError("model: field is required and cannot be empty")
        # Alias legacy OpenAI Chat Completions max_completion_tokens →
        # Responses-native max_output_tokens. Caller may pass either.
        if self.max_completion_tokens is not None:
            object.__setattr__(
                self, "max_output_tokens", int(self.max_completion_tokens)
            )
        # Map the canonical nested reasoning:{effort} → flat reasoning_effort (the form
        # the engine reads) when the flat alias wasn't also sent.
        if self.reasoning_effort is None and isinstance(self.reasoning, dict):
            _eff = self.reasoning.get("effort")
            if isinstance(_eff, str) and _eff:
                object.__setattr__(self, "reasoning_effort", _eff)
        # Validate input: string must be non-empty, list must have elements
        if isinstance(self.input, str) and not self.input.strip():
            raise ValueError("input: cannot be empty or whitespace-only")
        # (a `generate: false` prewarm carries no input by design)
        if (
            isinstance(self.input, list)
            and not self.input
            and self.generate is not False
        ):
            raise ValueError("input: cannot be an empty list")
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
        # Validate grammar type if provided
        if self.grammar is not None:
            gtype = self.grammar.get("type") if isinstance(self.grammar, dict) else None
            if gtype not in ("json", "regex", "choice", "cfg", None):
                raise ValueError(
                    f"grammar.type: must be one of 'json', 'regex', 'choice', 'cfg', got '{gtype}'"
                )
            if gtype == "cfg" and "grammar" in self.grammar:
                from yunshu_engine.grammar_constraint import validate_constraint_spec

                validate_constraint_spec(self.grammar)
        if self.stop and len(self.stop) > 16:
            raise ValueError("stop: maximum 16 stop sequences")
        if self.stop and any(not s for s in self.stop):
            raise ValueError("stop: individual stop sequences must be non-empty")
        # bound metadata to OpenAI's limit (16 pairs) so an echoed map can't be
        # used to balloon stored-response memory.
        if self.metadata is not None and len(self.metadata) > 16:
            raise ValueError("metadata: maximum 16 key-value pairs")
        return self


def _input_file_text(block: dict) -> str:
    """Text of an ``input_file`` part: a text file's content inline, else a note the model can act on."""
    import base64 as _b64

    name = block.get("filename") or "file"
    data = block.get("file_data")
    if isinstance(data, str) and data:
        mime = "text/plain"
        payload = data
        if data.startswith("data:"):
            head, _, payload = data.partition(",")
            mime = head[5:].split(";")[0] or "text/plain"
        try:
            raw = _b64.b64decode(payload)
        except Exception:
            return ""
        if mime.startswith("text/") or mime in (
            "application/json",
            "application/xml",
            "application/x-yaml",
        ):
            return f"[file: {name}]\n{raw.decode('utf-8', errors='replace')}"
        return f"[file: {name} ({mime}) cannot be read as text by this model]"
    return ""


def _extract_input_text(content) -> str | list:
    """Flatten OpenAI Responses content blocks to text.

    Responses API content blocks look like:
      [{"type": "input_text", "text": "Hi"}, {"type": "input_image", "image_url":...}]
    The OpenAI chat protocol downstream expects either a string or a
    list of {type, text/image_url} blocks. Previously the handler kept
    the raw list-of-dicts and the model literally saw the stringified
    Python dict (BUG-1 from agent deep verify).
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    # build the multimodal list in DOCUMENT ORDER (append text AND media
    # as encountered). The old code appended media then insert(0, text) for every
    # text part — moving ALL text ahead of media AND reversing text order, so
    # "[text A, image, text B]" became "[B, A, image]". Position of text relative
    # to an image is semantically meaningful for VLMs (and chat/completions
    # preserves it), so the scrambled layout caused wrong grounding + cross-endpoint
    # divergence.
    text_parts: list[str] = []
    multimodal: list[dict] = []
    has_media = False
    for block in content:
        if not isinstance(block, dict):
            s = str(block)
            text_parts.append(s)
            multimodal.append({"type": "text", "text": s})
            continue
        btype = block.get("type", "")
        if btype in ("input_text", "text", "output_text"):
            t = block.get("text", "")
            if t:
                text_parts.append(t)
                part = {"type": "text", "text": t}
                if block.get("prompt_cache_breakpoint"):
                    part["prompt_cache_breakpoint"] = block["prompt_cache_breakpoint"]
                multimodal.append(part)
        elif btype == "input_image":
            has_media = True
            # Unwrap a nested {"url": ...} (clients reusing the chat shape) — a raw
            # dict would crash _extract_images' url.startswith or be dropped.
            iu = block.get("image_url")
            url = iu.get("url") if isinstance(iu, dict) else iu
            url = url or block.get("url")
            multimodal.append({"type": "image_url", "image_url": {"url": url}})
        elif btype == "input_file":
            txt = _input_file_text(block)
            if txt:
                text_parts.append(txt)
                multimodal.append({"type": "text", "text": txt})
        elif btype == "input_audio":
            has_media = True
            multimodal.append(
                {"type": "input_audio", "input_audio": block.get("input_audio", block)}
            )
    if has_media or any(p.get("prompt_cache_breakpoint") for p in multimodal):
        return multimodal
    return "\n".join(text_parts)


def _convert_to_messages(req: ResponsesRequest) -> list[dict]:
    """Convert Responses API input to OpenAI chat messages."""
    messages: list[dict[str, Any]] = []
    pending_reasoning: list[str] = []

    def _take_reasoning(msg: dict) -> dict:
        # Reasoning items the client sent back ride on the next assistant turn.
        if pending_reasoning and msg.get("role") == "assistant":
            msg["reasoning_content"] = "\n".join(pending_reasoning)
            pending_reasoning.clear()
        return msg

    if req.instructions:
        messages.append({"role": "system", "content": req.instructions})

    if isinstance(req.input, str):
        messages.append({"role": "user", "content": req.input})
    elif isinstance(req.input, list):
        for item in req.input:
            # Accept both dicts and pydantic ResponseInputText models.
            _get = (
                item.get
                if isinstance(item, dict)
                else (lambda k, d=None: getattr(item, k, d))
            )
            itype = _get("type", "message")
            if itype == "reasoning":
                txt = reasoning_text_of(
                    item.model_dump() if hasattr(item, "model_dump") else dict(item)
                )
                if txt:
                    pending_reasoning.append(txt)
                continue
            if itype not in ("message", "", "function_call", "function_call_output"):
                # web_search_call / mcp_call / custom tool calls become a tool-call pair; reasoning,
                # mcp_list_tools, approvals, compaction, item_reference ... are skipped.
                converted = input_item_to_messages(
                    item.model_dump() if hasattr(item, "model_dump") else dict(item)
                )
                messages.extend(_take_reasoning(m) for m in converted)
                continue
            # Tool-conversation items: feed a prior tool call + its result back so
            # multi-turn agent loops work. Previously these had no role/content and
            # became empty "user" messages (the call_id/output were dropped).
            if itype == "function_call_output":
                _out = _get("output", "")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": _get("call_id", "") or "",
                        "content": _out
                        if isinstance(_out, str)
                        else _extract_input_text(_out)
                        if isinstance(_out, list)
                        and any(
                            isinstance(p, dict)
                            and p.get("type")
                            in (
                                "input_text",
                                "input_image",
                                "input_file",
                                "input_audio",
                            )
                            for p in _out
                        )
                        else json.dumps(_out, ensure_ascii=False),
                    }
                )
                continue
            if itype == "function_call":
                messages.append(
                    _take_reasoning(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": _get("call_id", "") or _get("id", "") or "",
                                    "type": "function",
                                    "function": {
                                        "name": _get("name", "") or "",
                                        "arguments": _get("arguments", "") or "{}",
                                    },
                                }
                            ],
                        }
                    )
                )
                continue
            role = _get("role", "user") or "user"
            if role == "developer":
                role = "system"
            content = _get("content", "")
            messages.append(
                _take_reasoning({"role": role, "content": _extract_input_text(content)})
            )

    return messages


def _parse_response_format(rf: dict | None, grammar: dict | None = None) -> dict | None:
    spec = _parse_response_format_unchecked(rf, grammar)
    from fastapi import HTTPException

    from yunshu_engine.grammar_constraint import validate_constraint_spec

    try:
        validate_constraint_spec(spec)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return spec


def _parse_response_format_unchecked(
    rf: dict | None, grammar: dict | None = None
) -> dict | None:
    if grammar is not None:
        gtype = grammar.get("type")
        if gtype == "json":
            schema = grammar.get("schema")
            if schema:
                return schema
            return "json_object"
        return grammar
    if rf is None:
        return None
    rf_type = rf.get("type")
    if rf_type == "json_schema":
        js = rf.get("json_schema", {})
        return js.get("schema", js)
    elif rf_type == "json_object":
        return "json_object"
    return None


async def _start_background_response(req: ResponsesRequest, request: Request):
    """OpenAI background mode: store a `queued` response, kick off the full
    non-stream generation in a task under the same id, and return immediately.

    The runner re-enters ``create_response`` with background/stream off and a
    forced id, so generation reuses the entire tested path (sampling, tools,
    storage, request-tracker registration → cancel works). On failure the stored
    status is flipped to ``failed`` so pollers don't hang on ``queued`` forever.
    """
    import asyncio

    response_id = f"resp-{uuid.uuid4().hex[:24]}"
    owner = _resolve_owner(request)
    queued_payload = {
        "id": response_id,
        "object": "response",
        **_config_echo(req),
        "created_at": int(time.time()),
        "model": req.model,
        "status": "queued",
        "output": [],
        "metadata": req.metadata,
        "_owner": owner,
    }
    _store_response(response_id, queued_payload)

    def _mark_failed():
        cur = _get_stored_response(response_id)
        if cur is not None and cur.get("status") == "cancelled":
            return  # a cancel already ended it; the engine's cancel exception is not a failure
        _store_response(
            response_id,
            {
                **{k: v for k, v in queued_payload.items()},
                "status": "failed",
                "completed_at": int(time.time()),
                "error": {
                    "message": "background generation failed",
                    "type": "server_error",
                    "code": "internal_error",
                },
            },
        )

    async def _runner():
        # Flip queued → in_progress so a poll mid-generation reflects reality.
        cur = _get_stored_response(response_id)
        # honor a cancel that landed during the queued window (cancel_response
        # persisted status=cancelled because the tracker had no entry to signal yet) — bail
        # before burning any GPU work.
        if cur is not None and cur.get("status") == "cancelled":
            return
        if cur is not None and cur.get("status") == "queued":
            _in = dict(cur)
            _in["status"] = "in_progress"
            _store_response(response_id, _in)
        try:
            request.state._forced_response_id = response_id
            req2 = req.model_copy(
                update={"background": False, "store": True, "stream": False}
            )
            await create_response(req2, request)
            # create_response stores the terminal payload under response_id (store=True).
            # If it returned an error JSONResponse without storing, the entry is still
            # queued/in_progress → surface a failed status instead of hanging.
            final = _get_stored_response(response_id)
            if final is None or final.get("status") in ("queued", "in_progress"):
                _mark_failed()
        except Exception:
            logger.error("background response generation failed", exc_info=True)
            _mark_failed()
        finally:
            with contextlib.suppress(Exception):
                request.state._forced_response_id = None

    asyncio.create_task(_runner())
    return JSONResponse(_public_stored(queued_payload))


async def _prewarm_response(req: ResponsesRequest, request: Request):
    """``generate: false``: prefill the prompt prefix, return a response id, produce no output.

    Codex opens its Responses WebSocket with a prewarm frame (``generate: false``, empty ``input``,
    the full ``instructions`` and ``tools``), then sends the real request with ``previous_response_id``
    set to the prewarm's id. The point is the cache: the stable prefix (instructions + tool schemas,
    several thousand tokens) is prefilled before the user has typed anything. Here that is a real
    1-token generation through the normal path, whose prompt lands in the prefix cache; the tokens it
    produced are discarded. The stored response carries no input and no output, so chaining onto it
    replays nothing.
    """
    messages_present = bool(req.input) and (
        isinstance(req.input, str) or len(req.input) > 0
    )
    inner = req.model_copy(
        update={
            "generate": None,
            "stream": False,
            "background": False,
            "max_output_tokens": 1,
            "store": False,
            # Qwen-style templates need a user turn to render; the real request replaces it.
            "input": req.input
            if messages_present
            else [ResponseInputText(type="message", role="user", content="Hi")],
        }
    )
    resp = await create_response(inner, request)
    status_code = getattr(resp, "status_code", 200)
    if status_code >= 400:
        return resp
    try:
        prefilled = json.loads(bytes(resp.body))
    except Exception:
        prefilled = {}
    rid = f"resp-{uuid.uuid4().hex[:24]}"
    usage = prefilled.get("usage") or {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }
    usage = dict(usage)
    usage["output_tokens"] = 0
    usage["total_tokens"] = usage.get("input_tokens", 0)
    created = int(time.time())
    obj = {
        "id": rid,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "model": req.model,
        "output": [],
        "usage": usage,
        "previous_response_id": req.previous_response_id,
        "store": bool(req.store),
        "x_yunshu": {"prewarm": True},
    }
    _store_response(
        rid,
        {
            **obj,
            "_input_messages": [],
            "_owner": _resolve_owner(request),
        },
    )
    if not req.stream:
        return JSONResponse(obj)

    async def gen():
        started = {**obj, "status": "in_progress", "usage": None}
        events = [
            ("response.created", started),
            ("response.in_progress", started),
            ("response.completed", obj),
        ]
        for seq, (name, body) in enumerate(events):
            yield (
                f"event: {name}\ndata: "
                + json.dumps({"type": name, "response": body, "sequence_number": seq})
                + "\n\n"
            ).encode()

    return StreamingResponse(gen(), media_type="text/event-stream")


def _engine_templates_itself(engine) -> bool:
    """VLMEngine (every mlx-vlm model, the 27B included) applies its own chat template,
    family adapter and native tool rendering to the message list. Handing it a string that
    this router already templated wraps the whole ChatML transcript in a user message, and
    the model then reads a chat log inside a user turn."""
    from yunshu_engine.vlm_engine import VLMEngine

    return isinstance(engine, VLMEngine)


def _enforce_tool_choice(req: ResponsesRequest, calls: list[dict]) -> list[dict]:
    """The tool_choice / parallel_tool_calls / max_tool_calls contract on a finished call list
    (chat's ``_enforce_tool_choice``): a named function keeps only its own calls,
    ``parallel_tool_calls=false`` keeps the first, ``max_tool_calls`` caps the total."""
    named = req.tool_choice.get("name") if isinstance(req.tool_choice, dict) else None
    if named:
        calls = [c for c in calls if c.get("name") == named]
    if req.parallel_tool_calls is False:
        calls = calls[:1]
    if req.max_tool_calls:
        calls = calls[: req.max_tool_calls]
    return calls


def _apply_previous_response_chain(req, messages: list[dict], request) -> list[dict]:
    """Replay the stored ``previous_response_id`` chain (oldest first) into ``messages``: after a leading
    system message, before the new input. Shared by generation and ``/input_tokens``."""
    if not req.previous_response_id:
        return messages
    chain: list[dict] = []
    # Walk the chain back-to-front. Limit to 16 hops AND track visited ids to
    # prevent loops. : the 16-cap alone only bounded a cycle — a repeated
    # previous_response_id (e.g. a self-referential or A↔B pair) would still
    # prepend the same turn up to 16 times, duplicating context. The `_seen` set
    # stops the walk at the first repeat (completes the comment's stated intent).
    _prev_id = req.previous_response_id
    _seen: set[str] = set()
    for _ in range(16):
        if not _prev_id or _prev_id in _seen:
            break
        _seen.add(_prev_id)
        _prev = _get_stored_response(_prev_id)
        if not _prev:
            break
        # SECURITY: ownership-gate the chain parent. Without this a
        # tenant could chain off another tenant's stored response id to
        # replay (and then exfiltrate) its private input/output. Mirrors the
        # GET /responses/{id} ownership guard.
        if not _owns_stored(request, _prev):
            break
        # Build this hop's turn as a UNIT in natural
        # order (user input(s) then the assistant reply) and prepend the whole unit.
        # The old code prepended user input but APPENDED assistant output, so a
        # 2-hop chain produced [u1, u2, a2, a1] instead of [u1, a1, u2, a2] —
        # every prior assistant turn clustered at the tail in reverse. Single-hop
        # happened to be correct, which hid it. We walk newest→oldest, so prepending
        # each older turn-block ahead of newer ones yields chronological order.
        _turn: list[dict] = []
        _stored_input = _prev.get("_input_messages")
        if _stored_input:
            _turn.extend(_stored_input)
        _rs_pending: list[str] = []
        for out in _prev.get("output") or []:
            if out.get("type") == "reasoning":
                _rt = reasoning_text_of(out)
                if _rt:
                    _rs_pending.append(_rt)
                continue
            if out.get("role") == "assistant":
                for c in out.get("content") or []:
                    if c.get("type") == "output_text":
                        _amsg = {"role": "assistant", "content": c.get("text", "")}
                        if _rs_pending:
                            _amsg["reasoning_content"] = "\n".join(_rs_pending)
                            _rs_pending = []
                        _turn.append(_amsg)
            elif out.get("type") in (
                "web_search_call",
                "mcp_call",
                "custom_tool_call",
                "local_shell_call",
                "tool_search_call",
                "tool_search_output",
            ):
                _turn.extend(
                    input_item_to_messages(out, _prev.get("_server_tool_texts"))
                )
            elif out.get("type") == "function_call":
                # Replay the prior tool call (stored as a top-level item with no
                # "role") so a chained agent loop remembers it requested the
                # tool — previously dropped, breaking multi-turn tool use.
                _turn.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": out.get("call_id", "") or out.get("id", "") or "",
                                "type": "function",
                                "function": {
                                    "name": out.get("name", "") or "",
                                    "arguments": out.get("arguments", "") or "{}",
                                },
                            }
                        ],
                    }
                )
        chain[0:0] = _turn
        _prev_id = _prev.get("previous_response_id")
    if chain:
        # Inject chain after any system message but before the new user input
        if messages and messages[0].get("role") == "system":
            messages = [messages[0]] + chain + messages[1:]
        else:
            messages = chain + messages
    return messages


@router.post("/responses", response_model=None)
async def create_response(req: ResponsesRequest, request: Request):
    """OpenAI Responses API endpoint."""
    _check_permission(request, "can_infer")
    from ..capability_gate import enforce_responses

    enforce_responses(req)
    from ..responses_context import needs_state, run_stateful_response

    if needs_state(req):
        # conversation / compaction items / auto compaction: rewrite the request, re-enter here
        return await run_stateful_response(req, request, create_response)
    from ..streaming import RESPONSES_ECHO

    RESPONSES_ECHO.set(_config_echo(req))
    if isinstance(req.input, list):
        from ..files_store import FileRefError, has_file_refs, resolve_file_refs

        _raw_items = [
            i.model_dump() if hasattr(i, "model_dump") else i for i in req.input
        ]
        if has_file_refs(_raw_items):
            try:
                req = req.model_copy(
                    update={
                        "input": [
                            ResponseInputText(**i)
                            for i in resolve_file_refs(_raw_items)
                        ]
                    }
                )
            except FileRefError as exc:
                return JSONResponse(
                    status_code=exc.status,
                    content={
                        "error": {
                            "message": exc.message,
                            "type": "invalid_request_error",
                            "code": "file_not_found" if exc.status == 404 else None,
                        }
                    },
                )
    if req.generate is False:
        return await _prewarm_response(req, request)
    # OpenAI background mode: return a queued response immediately and run the
    # full (non-stream) generation asynchronously under the same id. Streaming +
    # background is not supported here (would require resumable SSE), so it only
    # triggers when stream is off.
    if has_server_tools_responses(req):
        return await create_with_server_tools_responses(req, request, create_response)
    if req.background and not req.stream:
        return await _start_background_response(req, request)
    from ..responses_client_tools import create_client_tools, declarations

    _client_defs = declarations(req.tools)
    if isinstance(req.input, list):
        for item in req.input:
            if item.type in ("tool_search_output", "additional_tools"):
                _client_defs.update(declarations(getattr(item, "tools", [])))
    _loaded_previous = False
    if req.previous_response_id:
        previous = _get_stored_response(req.previous_response_id)
        _loaded_previous = bool(
            previous
            and _owns_stored(request, previous)
            and previous.get("_loaded_tools")
        )
    if (
        any(
            d.get("type") in ("custom", "local_shell", "tool_search")
            for d in _client_defs.values()
        )
        or _loaded_previous
        or (
            isinstance(req.input, list)
            and any(
                i.type in ("tool_search_output", "additional_tools") for i in req.input
            )
            and bool(_client_defs)
        )
    ):
        return await create_client_tools(req, request, create_response)
    _fn_tools = function_tools(req.tools)
    if _fn_tools != req.tools:
        # namespaces flattened, non-function tools (custom, ...) dropped: the engine sees plain functions
        req = req.model_copy(update={"tools": _fn_tools})
    messages = _convert_to_messages(req)
    # Snapshot THIS hop's own input before previous_response_id chaining mutates
    # `messages` (MED). Storing the post-chain `messages` as
    # `_input_messages` made multi-hop chains replay older turns twice (the
    # walk-back both read this hop's stored input AND followed previous_response_id
    # to the same older response). We persist only this hop's new user input.
    # System/instructions are excluded so they don't reappear mid-conversation
    # when a later hop replays this turn.
    # The old filter ALSO stripped `assistant`, which silently
    # dropped a client-supplied assistant turn carrying `tool_calls` — leaving its
    # `tool` result ORPHANED on chain replay (a tool message with no preceding
    # assistant tool_calls), which corrupts the chat template (raise → plaintext
    # fallback / 500) or makes the model answer a call it never saw. It also lost
    # plain multi-turn assistant context. Keep this hop's own conversation turns
    # (user/assistant/tool); only system/instructions are excluded. This hop's own
    # input assistant turns are distinct from this response's GENERATED `output`
    # (replayed separately below), so keeping them cannot double-count.
    _own_input_messages = [m for m in messages if m.get("role") != "system"]

    # previous_response_id chaining: replay prior turns when store=true on
    # the predecessor. The Responses API contract says the chain provides
    # implicit conversation memory; without this the model sees only the
    # new turn and can't recall prior context.
    messages = _apply_previous_response_chain(req, messages, request)

    json_schema = _parse_response_format(req.response_format, req.grammar)

    # Convert logit_bias keys from str to int (API sends string keys,
    # engine expects int keys for tensor indexing).
    # SECURITY: validate magnitude + reject NaN/Inf (Pydantic field
    # has no bounds; matches chat.py:335-342 validator).
    _logit_bias = req.logit_bias
    if _logit_bias:
        import math

        for _bk, _bv in _logit_bias.items():
            if math.isnan(_bv) or math.isinf(_bv):
                raise HTTPException(
                    status_code=400,  # OpenAI uses 400 for invalid params
                    detail=f"logit_bias[{_bk}] must be finite, got {_bv}",
                )
            if _bv < -100 or _bv > 100:
                raise HTTPException(
                    status_code=400,  # OpenAI uses 400 for invalid params
                    detail=f"logit_bias[{_bk}] must be in [-100, 100], got {_bv}",
                )
        # Skip non-integer keys instead of crashing int(k) with 500 (a malformed
        # key is bad input, not a server error). Matches anthropic _convert_logit_bias.
        _converted = {}
        for k, v in _logit_bias.items():
            try:
                _converted[int(k)] = v
            except (ValueError, TypeError):
                logger.warning("Skipping non-integer logit_bias key: %r", k)
        _logit_bias = _converted or None

    # Structured tracing
    from yunshu_engine.tracing import get_inference_tracer, get_structured_logger

    tracer = get_inference_tracer()
    slog = get_structured_logger()
    trace_id = f"resp-{uuid.uuid4().hex[:16]}"
    tracer.start_trace(
        trace_id,
        metadata={
            "model": req.model,
            "stream": req.stream,
            "endpoint": "/responses",
        },
    )
    slog.info(
        "inference_request", model=req.model, trace_id=trace_id, stream=req.stream
    )

    engine = get_engine()
    # Single-model mode serves the loaded model under ANY requested name (placeholder),
    # exactly like /chat/completions — so accept the global engine when it's up instead
    # of 404-ing on a model-id mismatch (e.g. the default "local"). Only fall through to
    # the model manager (multi-model mode, where get_engine() is None).
    if engine is None or not engine.is_loaded:
        try:
            engine = await get_engine_for_model(req.model)
        except (KeyError, Exception) as e:
            # Don't leak the nested exception message — produces redundant
            # "Model 'X' not found: ... not found in model manager" output.
            logger.debug("responses model resolution failed for %s: %s", req.model, e)
            # When the client requested a stream, OpenAI clients expect SSE
            # framing even for terminal errors — returning a JSON 4xx body
            # breaks SSE parsers. Emit response.failed + [DONE].
            if req.stream:
                _err_response_id = f"resp-{uuid.uuid4().hex[:24]}"
                _err_model = req.model

                async def _model_not_found_stream():
                    yield format_responses_created(
                        _err_response_id, _err_model, seq=1
                    ).encode("utf-8")
                    yield format_responses_in_progress(
                        _err_response_id, _err_model, seq=2
                    ).encode("utf-8")
                    yield format_responses_failed(
                        _err_response_id,
                        _err_model,
                        error_code="model_not_found",
                        error_message=f"Model '{_err_model}' not found",
                        seq=3,
                    ).encode("utf-8")
                    yield b"data: [DONE]\n\n"

                return StreamingResponse(
                    _model_not_found_stream(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            raise HTTPException(
                status_code=404, detail=f"Model '{req.model}' not found"
            ) from None

    from ..model_guards import reject_embedding_only

    reject_embedding_only(engine, req.model)

    # Reject prompts over the context window (400) or too large to prefill (413),
    # before generation (see chat.py).
    try:
        from yunshu_control.token_counter import count_message_tokens

        from ..streaming import validate_context_window, validate_prefill_memory

        _tok = getattr(engine, "_tokenizer", None) or getattr(engine, "tokenizer", None)
        _est = count_message_tokens(messages, _tok)
        if req.truncation == "auto":
            from ..streaming import get_max_context_window

            _ctx = get_max_context_window(req.model, engine)
            if _ctx and _est > _ctx:
                _room = _ctx - min(req.max_output_tokens, _ctx // 4)
                _before, _est_before = messages, _est
                messages, _est = _auto_truncate(
                    messages,
                    _est,
                    _room,
                    lambda ms: count_message_tokens(ms, _tok),
                )
                if len(messages) < len(_before):
                    # never silent: what was dropped goes to x_yunshu.context_policy
                    from yunshu_engine.context_window import (
                        publish_report,
                        removal_report,
                    )

                    publish_report(
                        removal_report(
                            "truncation_auto",
                            _before,
                            messages,
                            _est_before,
                            _est,
                            _room,
                        )
                    )
        validate_context_window(_est, req.model, engine)
        validate_prefill_memory(_est)
        from ..token_budget import plan_for_engine

        _tb = plan_for_engine(
            _est, req.max_output_tokens, req.thinking_budget, req.model, engine
        )
        if _tb is not None and _tb.clamped:
            req.max_output_tokens = _tb.max_tokens_granted
            req.thinking_budget = _tb.thinking_budget_granted
    except HTTPException:
        raise
    except Exception:
        logger.debug("prompt-size validation skipped", exc_info=True)

    # Check for VLM/audio routing
    from .chat import _has_audio, _has_images

    has_media = _has_images(messages) or _has_audio(messages)
    if has_media:
        # previously this returned `_handle_vlm_chat(...)` verbatim — a
        # Chat-Completions object (object="chat.completion", choices[].message) for
        # non-stream and chat.completion.chunk SSE events for stream. That breaks
        # every OpenAI Responses SDK client (wrong object type, wrong output shape,
        # wrong event family) AND silently ignored `store` (so GET /v1/responses/{id}
        # and previous_response_id chaining 404'd). Funnel the VLM text through the
        # proper Responses object / event sequence and honor store.
        return await _vlm_to_responses(
            req,
            messages,
            request,
            _logit_bias,
            _own_input_messages,
        )

    # Inject tool definitions
    _resp_tool_prefill = ""  # prepended back onto the output before tool extraction
    if req.tools:
        from .chat import ToolDefinition, ToolFunction, _inject_tool_system_prompt

        tools = [
            ToolDefinition(
                function=ToolFunction(
                    name=t.name,
                    description=t.description,
                    parameters=t.parameters,
                )
            )
            for t in req.tools
        ]
        from .chat import _vlm_renders_tools

        if (
            req.tool_choice in (None, "auto") or _forced_by_grammar(req)
        ) and _vlm_renders_tools(engine):
            # The chat template renders the tools itself (Qwen3.x): the model's own tool-call
            # format, not an injected JSON prompt it may half-follow.
            req._native_tools = [t.model_dump() for t in tools]
        else:
            messages = _inject_tool_system_prompt(
                messages,
                tools,
                tool_choice=_chat_tool_choice(req.tool_choice),
                parallel_tool_calls=req.parallel_tool_calls,
                engine=engine,
            )
        # Structurally FORCE a required/named tool_choice via an assistant prefill — the
        # advisory injection alone lets the model emit plain text (so "required"/named
        # couldn't be honored). Non-stream only: the streaming path uses
        # ToolCallStreamer(forced_tool_name), and a shared prefill would leave its parser
        # without the opening marker. The prefill is prepended back before extraction.
        from .chat import _native_tools_active

        if (
            not req.stream
            and not getattr(req, "_native_tools", None)
            and not _native_tools_active()
        ):
            _tc = req.tool_choice
            if _tc == "required":
                _resp_tool_prefill = "<tool_call>\n"
            elif isinstance(_tc, dict) and _tc.get("name"):
                # Keep the JSON object complete for the model to generate, as
                # on Chat/Anthropic. A half-open arguments object encourages an
                # immediate empty closure; the injected prompt steers the name
                # and _enforce_tool_choice filters it after parsing.
                _resp_tool_prefill = "<tool_call>\n"
            if _resp_tool_prefill:
                from .chat import _append_tool_prefill

                messages = _append_tool_prefill(messages, _resp_tool_prefill, engine)

    # Background mode pre-allocates the id (so the queued response returned to the
    # client and the polled/cancellable generation share one id). Consume it once.
    _forced_id = getattr(request.state, "_forced_response_id", None)
    if _forced_id:
        response_id = _forced_id
        with contextlib.suppress(Exception):
            request.state._forced_response_id = None
    else:
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
            _stream_response(
                engine,
                req,
                messages,
                response_id,
                json_schema,
                loaded_adapter,
                request=request,
                own_input_messages=_own_input_messages,
            ),
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
        if not is_batched and not _engine_templates_itself(engine) and messages:
            _tokenizer = getattr(engine, "_tokenizer", None)
            if _tokenizer is not None and hasattr(_tokenizer, "apply_chat_template"):
                # bring this deprecated non-batched path closer to the engine's
                # hardened _apply_chat_template — remap OpenAI `developer`→`system` /
                # legacy `function`→`tool` and run _normalize_messages_for_chat_template
                # (dangling-<think> + tool-arg-JSON→dict) BEFORE the family adapter, so a
                # developer-role / tool-use Responses request on the legacy Engine isn't
                # passed raw to the template. (Assistant-prefill continue_final_message
                # remains a documented gap of this deprecated path; default = BatchedEngine.)
                _msgs = messages
                try:
                    from yunshu_engine.batched_engine import BatchedEngine as _BE

                    _msgs = [
                        (
                            {
                                **_m,
                                "role": (
                                    "system"
                                    if _m.get("role") == "developer"
                                    else "tool"
                                ),
                            }
                            if _m.get("role") in ("developer", "function")
                            else _m
                        )
                        for _m in messages
                    ]
                    _msgs = _BE._normalize_messages_for_chat_template(_msgs)
                except Exception:
                    _msgs = messages
                try:
                    from yunshu_engine.message_adapter import adapt_messages

                    _adapted = adapt_messages(
                        _msgs, getattr(engine, "model_name", "") or ""
                    )
                except Exception:
                    _adapted = _msgs
                try:
                    _tpl_kwargs: dict = {
                        "tokenize": False,
                        "add_generation_prompt": True,
                    }
                    if req.enable_thinking is not None:
                        _tpl_kwargs["enable_thinking"] = req.enable_thinking
                    _rendered = _tokenizer.apply_chat_template(_adapted, **_tpl_kwargs)
                    if _rendered:
                        _non_batched_prompt = _rendered
                except TypeError as _te:
                    if "enable_thinking" in str(_te):
                        _tpl_kwargs.pop("enable_thinking", None)
                        _rendered = _tokenizer.apply_chat_template(
                            _adapted, **_tpl_kwargs
                        )
                        if _rendered:
                            _non_batched_prompt = _rendered
                    else:
                        logger.debug(
                            "chat template failed for non-batched Responses path",
                            exc_info=True,
                        )
                except Exception:
                    logger.debug(
                        "chat template failed for non-batched Responses path",
                        exc_info=True,
                    )

        # ── n>1 support: generate n responses sequentially ──
        # Single GPU cannot parallelize multiple generations; they run
        # sequentially.  Each choice gets its own message output item.
        all_output_items: list[dict] = []
        total_pt = 0
        total_ct = 0
        total_reasoning_tokens = 0
        max_cached_tokens = 0
        last_finish_reason = "stop"
        # per-choice failure isolation. chat.py and completions.py both wrap each
        # choice so a transient mid-generation MemoryError/TimeoutError/RuntimeError on one
        # choice returns the OTHER (already-generated, billed) choices instead of 500/507-ing
        # the whole request; responses.py was the lone outlier (the deferred "reindent risk"
        # item). The engine call is the only statement that raises these, so it's wrapped;
        # only an all-choices-failed request re-raises (handled after the loop).
        _choice_errors: list[Exception] = []

        for choice_idx in range(req.n):
            result = None
            state = None

            if is_batched:
                # disconnect guard — set _ns_cancel_event on client disconnect so
                # the engine decode loop stops (chat.py had this; responses only registered
                # the event and never polled is_disconnected → ran to max_tokens/timeout).
                try:
                    result = await run_with_disconnect_guard(
                        request,
                        engine.chat(
                            messages=messages,
                            max_tokens=req.max_output_tokens,
                            temperature=req.temperature,
                            top_p=req.top_p,
                            top_k=req.top_k,
                            # per-choice seed for n>1 (was req.seed for EVERY choice →
                            # with explicit seed + temp>0 + n>1 all n choices got the same RNG
                            # base key → IDENTICAL token streams, still billed n×). chat/
                            # completions already offset by +idx; Responses was the outlier.
                            seed=_per_choice_seed(req.seed, choice_idx),
                            enable_thinking=req.enable_thinking,
                            thinking_budget=req.thinking_budget,
                            reasoning_effort=req.reasoning_effort,
                            repetition_penalty=req.repetition_penalty,
                            frequency_penalty=req.frequency_penalty,
                            presence_penalty=req.presence_penalty,
                            min_tokens=req.min_tokens,
                            ignore_eos=req.ignore_eos,
                            suppress_tokens=req.suppress_tokens,
                            logit_bias=_logit_bias,
                            min_p=req.min_p,
                            top_n_sigma=req.top_n_sigma,
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
                            lora_adapter=loaded_adapter,
                        ),
                        cancel_event=_ns_cancel_event,
                    )
                except HTTPException:
                    raise  # client-disconnect (499) etc. propagate, not a per-choice failure
                except Exception as _choice_exc:  # isolate this choice's failure
                    logger.warning(
                        "responses n>1: choice %d failed (%s); skipping",
                        choice_idx,
                        _choice_exc,
                    )
                    _choice_errors.append(_choice_exc)
                    continue
                if result is None:
                    raise HTTPException(status_code=499, detail="Client disconnected")
                text = result.text
                pt = result.prompt_tokens
                ct = result.completion_tokens
                finish_reason = _normalize_finish_reason(result.finish_reason)
                _reasoning_tokens = getattr(result, "reasoning_tokens", 0)
                _cached_tokens = getattr(result, "cached_tokens", 0)
            else:
                try:
                    state = await run_with_disconnect_guard(
                        request,
                        engine.generate(
                            **_native_kw(req),
                            **_cache_kw(req, engine),
                            prompt=_non_batched_prompt,
                            max_tokens=req.max_output_tokens,
                            temperature=req.temperature,
                            top_p=req.top_p,
                            top_k=req.top_k,
                            seed=_per_choice_seed(
                                req.seed, choice_idx
                            ),  # per-choice (see chat path)
                            enable_thinking=req.enable_thinking,
                            thinking_budget=req.thinking_budget,
                            reasoning_effort=req.reasoning_effort,
                            repetition_penalty=req.repetition_penalty,
                            frequency_penalty=req.frequency_penalty,
                            presence_penalty=req.presence_penalty,
                            min_tokens=req.min_tokens,
                            ignore_eos=req.ignore_eos,
                            suppress_tokens=req.suppress_tokens,
                            logit_bias=_logit_bias,
                            min_p=req.min_p,
                            top_n_sigma=req.top_n_sigma,
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
                            lora_adapter=loaded_adapter,
                        ),
                        cancel_event=_ns_cancel_event,
                    )
                except HTTPException:
                    raise  # client-disconnect (499) etc. propagate, not a per-choice failure
                except Exception as _choice_exc:  # isolate this choice's failure
                    logger.warning(
                        "responses n>1: choice %d failed (%s); skipping",
                        choice_idx,
                        _choice_exc,
                    )
                    _choice_errors.append(_choice_exc)
                    continue
                if state is None:
                    raise HTTPException(status_code=499, detail="Client disconnected")

                # VLMEngine returns a dict, BatchedEngine returns a state
                # object. Use a unified accessor that reads from either.
                def _get(name, fallback=None, default=None):
                    if isinstance(state, dict):
                        v = state.get(name)
                        if v is None and fallback:
                            v = state.get(fallback)
                        return default if v is None else v
                    v = getattr(state, name, None)
                    if v is None and fallback:
                        v = getattr(state, fallback, None)
                    return default if v is None else v

                text = _get("generated_text", "text", "")
                pt = _get("prompt_token_count", "prompt_tokens", 0)
                ct = _get("completion_token_count", "completion_tokens", 0)
                finish_reason = _normalize_finish_reason(
                    _get("finish_reason", None, None)
                )
                _reasoning_tokens = _get("reasoning_tokens", None, 0) or 0
                _cached_tokens = _get("cached_tokens", None, 0) or 0

            # Stop-sequence overcount correction
            if req.stop and finish_reason == "stop":
                for _seq in req.stop:
                    if _seq and _seq in text:
                        _corrected = text[: text.find(_seq)]
                        _tok = getattr(engine, "_tokenizer", None)
                        if _tok:
                            try:
                                _cc = len(_tok.encode(_corrected))
                                if _cc < ct:
                                    ct = _cc
                            except Exception:
                                pass
                        break

            # Extract thinking content for reasoning models
            from ..streaming import extract_thinking

            _thinking, text = (
                ("", text)
                if getattr(req, "_custom_raw_input", False)
                else extract_thinking(text, req.model)
            )
            if _thinking and _reasoning_tokens == 0:
                # cap at ct. output_tokens (= total_ct) already
                # INCLUDES reasoning (reasoning is reported as a subset detail), so
                # re-encoding the decoded thinking substring — which often yields MORE
                # tokens than the original slice — could make reasoning_tokens exceed
                # output_tokens, violating reasoning ≤ output. Matches chat.py's min(...,ct).
                _reasoning_tokens = (
                    min(len(engine._tokenizer.encode(_thinking)), ct)
                    if hasattr(engine, "_tokenizer") and engine._tokenizer
                    else 0
                )

            # Accumulate usage across all choices
            total_pt = pt  # prompt tokens are the same for every choice
            total_ct += ct
            total_reasoning_tokens += _reasoning_tokens
            max_cached_tokens = max(max_cached_tokens, _cached_tokens)
            last_finish_reason = finish_reason

            # Extract tool calls for this choice
            tool_calls = None
            if req.tools and req.tool_choice != "none":
                from yunshu_engine.tool_format import parse_tool_output, tool_formats

                # Prepend the forced-tool prefill (if any) so the markup is complete for
                # the parser — the model only generated the continuation after it.
                _parse_text = (
                    (_resp_tool_prefill + text) if _resp_tool_prefill else text
                )
                tool_calls, _clean = parse_tool_output(
                    _parse_text, tool_formats(engine), req.tools
                )
                if tool_calls:
                    text = _clean
                    tool_calls = _enforce_tool_choice(req, tool_calls)
                if tool_calls:
                    finish_reason = "tool_calls"

            # Build output item for this choice
            text_part = {
                "type": "output_text",
                "text": text
                if getattr(req, "_custom_raw_input", False)
                else text.strip(),
                "annotations": [],
            }
            # Include logprobs if requested
            # Responses API logprobs format: flat list of {"token", "logprob", "top_logprobs"}
            # NOT the Chat Completions {"content": [...]} wrapper.
            if req.logprobs:
                _result_lp = (
                    getattr(result, "logprobs", None)
                    if is_batched
                    else getattr(state, "logprobs", None)
                )
                if isinstance(_result_lp, (list, tuple)) and len(_result_lp) > 0:
                    _chat_lp = _format_chat_logprobs(
                        _result_lp, top_logprobs=req.top_logprobs
                    )
                    if _chat_lp:
                        # Unwrap from Chat Completions {"content": [...]} to flat list
                        text_part["logprobs"] = _chat_lp.get("content", [])
            content_parts = [text_part]
            # reasoning is a SEPARATE output item (OpenAI Responses spec:
            # type=reasoning, rs_ id) PRECEDING the message — matching the streaming path
            # — not nested as a message content part. A client iterating response.output
            # now finds a proper reasoning item (the official SDK shape).
            if _thinking:
                _reasoning_item = {
                    "type": "reasoning",
                    "id": f"rs-{uuid.uuid4().hex[:24]}",
                    "summary": [{"type": "summary_text", "text": _thinking}],
                    **_seal_extra(req, _thinking),
                    "status": "completed",
                }
                if req.n > 1:
                    _reasoning_item["index"] = choice_idx
                all_output_items.append(_reasoning_item)
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
                    all_output_items.append(
                        {
                            "type": "function_call",
                            "id": f"fc-{uuid.uuid4().hex[:24]}",
                            "call_id": f"call_{uuid.uuid4().hex[:8]}",
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                            # parity with the streaming path's function_call item
                            # (responses.py ~1465), which includes status — a non-stream
                            # client reconstructing output items shouldn't see a shape that
                            # differs from the streamed one by a missing field.
                            "status": "completed",
                        }
                    )

        # if EVERY choice failed (none produced an output item), re-raise the
        # last error so the outer 507 (MemoryError) / 500 handler responds — don't return a
        # 200 with empty output. A partial success (some choices made it) returns normally.
        if not all_output_items and _choice_errors:
            raise _choice_errors[-1]

        # total_ct (engine count) already includes reasoning — don't re-add
        # (non-streaming path; the streaming metrics below count non-reasoning
        # so they correctly add reasoning).
        _record_metrics(total_pt, total_ct)
        # End tracing
        tracer.end_trace(
            trace_id,
            result={
                "prompt_tokens": total_pt,
                "completion_tokens": total_ct,
                "choices": req.n,
            },
        )
        slog.info(
            "inference_complete",
            model=req.model,
            trace_id=trace_id,
            prompt_tokens=total_pt,
            completion_tokens=total_ct,
            choices=req.n,
        )

        # length OR an interrupted/cancelled/aborted/errored generation → "incomplete";
        # only a normal stop/eos/tool_calls/None is "completed" (mirrors streaming path).
        # a POST /v1/responses/{id}/cancel sets _ns_cancel_event, but the engine
        # reports finish_reason="stop" on cancel (not a distinct reason), so the map below
        # never matched → a CANCELLED background response was stored + polled as "completed".
        # The cancel_event is the authoritative signal; check it FIRST. This propagates the
        # streaming-path fix to the non-stream/background path (the cancel_response
        # docstring promises cancelled→incomplete; the non-stream path never delivered it).
        if _ns_cancel_event is not None and _ns_cancel_event.is_set():
            _response_status = (
                "cancelled"  # the Response status the spec names for a cancel
            )
        else:
            _response_status = (
                "incomplete"
                if last_finish_reason
                in (
                    *_TRUNCATED_FINISH,
                    "cancel",
                    "cancelled",
                    "abort",
                    "aborted",
                    "error",
                    "content_filter",
                )
                else "completed"
            )

        _response_payload = {
            "id": response_id,
            "object": "response",
            **_config_echo(req),
            "created_at": int(time.time()),
            "completed_at": int(time.time()),
            "model": req.model,
            "status": _response_status,
            "output": all_output_items,
            "metadata": req.metadata,
            # total_ct (engine count) already includes reasoning tokens.
            "usage": responses_usage(
                total_pt, total_ct, total_reasoning_tokens, max_cached_tokens
            ),
        }
        # Persist when the client requested storage so it can be retrieved
        # via GET /v1/responses/{id}. Stash the raw input messages and the
        # predecessor id so previous_response_id chaining can rebuild the
        # full conversation when the client follows up.
        if req.store:
            # a cancel can land while we were awaiting a model load — the
            # request is already in_progress in the store but not yet registered with
            # the request_tracker (registration happens AFTER get_engine_for_model's
            # await), so tracker.cancel found no entry and the cancel handler persisted
            # status="cancelled" directly (the branch). Our _ns_cancel_event was
            # never set (no tracker entry to signal), so the status map above resolved
            # "completed" — which would CLOBBER that cancel marker, and the client (who
            # already received a cancelled envelope) would then GET "completed" with full
            # output and the GPU work ran to completion. Honor a cancel that landed in
            # this pre-registration window: re-read and do NOT overwrite a terminal
            # "cancelled". No await between this read and the store → atomic, so the
            # marker can't slip in afterward. (The registered-path cancel goes through
            # _ns_cancel_event → "incomplete" above and never persists "cancelled", so
            # this is scoped strictly to the unregistered-window clobber.)
            _prior = _get_stored_response(response_id)
            if _prior is not None and _prior.get("status") == "cancelled":
                return JSONResponse(_public_stored(_prior))
            _persist_payload = dict(_response_payload)
            _persist_payload["_input_messages"] = _own_input_messages
            _persist_payload["previous_response_id"] = req.previous_response_id
            _persist_payload["_owner"] = _resolve_owner(request)
            _store_response(response_id, _persist_payload)
        return JSONResponse(_response_payload)
    except HTTPException:
        raise  # let the disconnect (499) propagate, not become a 500
    except MemoryError:
        return JSONResponse(
            status_code=507,
            content={
                "error": {
                    "message": "Insufficient GPU memory",
                    "type": "server_error",
                    "code": "insufficient_memory",
                }
            },
        )
    except Exception as e:
        logger.error(f"Responses API generation error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content=server_error_body())
    finally:
        _release_lora_adapter(engine, loaded_adapter)
        if _ns_tracker is not None:
            with contextlib.suppress(Exception):
                _ns_tracker.unregister(response_id)


async def _stream_response(
    engine,
    req,
    messages,
    response_id,
    json_schema,
    loaded_adapter=None,
    request=None,
    own_input_messages=None,
):
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
    from yunshu_engine.batched_engine import BatchedEngine

    from ..streaming import with_sse_keepalive
    from .chat import _release_lora_adapter

    is_batched = isinstance(engine, BatchedEngine)

    # Non-batched Engine path: apply chat template ourselves before passing
    # to engine.generate_stream().  The legacy Engine / EngineCore
    # _messages_to_text() does not call adapt_messages() and may strip
    # tool-related fields, producing garbage for tool-use conversations.
    _stream_prompt: str | list[dict] = messages
    if not is_batched and not _engine_templates_itself(engine) and messages:
        _tokenizer = getattr(engine, "_tokenizer", None)
        if _tokenizer is not None and hasattr(_tokenizer, "apply_chat_template"):
            # same role-remap + normalize as the non-stream sibling above.
            _msgs = messages
            try:
                from yunshu_engine.batched_engine import BatchedEngine as _BE

                _msgs = [
                    (
                        {
                            **_m,
                            "role": (
                                "system" if _m.get("role") == "developer" else "tool"
                            ),
                        }
                        if _m.get("role") in ("developer", "function")
                        else _m
                    )
                    for _m in messages
                ]
                _msgs = _BE._normalize_messages_for_chat_template(_msgs)
            except Exception:
                _msgs = messages
            try:
                from yunshu_engine.message_adapter import adapt_messages

                _adapted = adapt_messages(
                    _msgs, getattr(engine, "model_name", "") or ""
                )
            except Exception:
                _adapted = _msgs
            try:
                _tpl_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
                if req.enable_thinking is not None:
                    _tpl_kwargs["enable_thinking"] = req.enable_thinking
                _rendered = _tokenizer.apply_chat_template(_adapted, **_tpl_kwargs)
                if _rendered:
                    _stream_prompt = _rendered
            except TypeError as _te:
                if "enable_thinking" in str(_te):
                    _tpl_kwargs.pop("enable_thinking", None)
                    _rendered = _tokenizer.apply_chat_template(_adapted, **_tpl_kwargs)
                    if _rendered:
                        _stream_prompt = _rendered
                else:
                    logger.debug(
                        "chat template failed for non-batched streaming Responses path",
                        exc_info=True,
                    )
            except Exception:
                logger.debug(
                    "chat template failed for non-batched streaming Responses path",
                    exc_info=True,
                )

    _logit_bias = req.logit_bias
    if _logit_bias:
        # the STREAMING path skipped the finite + [-100,100] validation the
        # non-stream create_response does, but the engine's logit_bias processor does
        # NOT guard the bias VALUE — a streaming Responses request with
        # logit_bias={"50256": NaN}/1e9 made that logit NaN/Inf → softmax all-NaN →
        # garbage output instead of a clean 422. Same class fixed elsewhere.
        import math

        for _bk, _bv in _logit_bias.items():
            if math.isnan(_bv) or math.isinf(_bv):
                raise HTTPException(
                    status_code=400,  # OpenAI uses 400 for invalid params
                    detail=f"logit_bias[{_bk}] must be finite, got {_bv}",
                )
            if _bv < -100 or _bv > 100:
                raise HTTPException(
                    status_code=400,  # OpenAI uses 400 for invalid params
                    detail=f"logit_bias[{_bk}] must be in [-100, 100], got {_bv}",
                )
        _converted = {}
        for k, v in _logit_bias.items():
            try:
                _converted[int(k)] = v
            except (ValueError, TypeError):
                logger.warning("Skipping non-integer logit_bias key: %r", k)
        _logit_bias = _converted or None
    prompt_tok = 0
    completion_tok = 0
    reasoning_tok = 0
    cached_tok = 0

    # Register with request tracker for cancellation support.
    # Use the public response_id so POST /v1/responses/{id}/cancel can
    # signal this generation.
    _stream_id = response_id
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
    accumulated_thinking = ""

    # hold back tool-call markup from response.output_text.delta. Previously every
    # non-reasoning token was emitted verbatim as a delta — raw <tool_call>…</tool_call> /
    # ChatML markup leaked into the streamed visible text, while only the terminal .done
    # carried the cleaned text (markup-stripped). The chat router already routes streamed
    # tokens through a ToolCallStreamer; mirror that here so the delta stream stays clean.
    # accumulated_text still keeps the RAW text (used for end-of-stream tool extraction).
    _resp_tool_streamer = None
    if req.tools and req.tool_choice != "none":
        from yunshu_engine.tool_call_streamer import ToolCallStreamer

        _tc_forced = (
            req.tool_choice.get("name") if isinstance(req.tool_choice, dict) else None
        )
        from yunshu_engine.tool_format import tool_formats

        _resp_tool_streamer = ToolCallStreamer(
            tool_formats(engine),
            forced_tool_name=_tc_forced,
            allow_parallel=req.parallel_tool_calls,
            max_calls=req.max_tool_calls,
            tools=req.tools,
        )

    def _next_seq():
        nonlocal _seq
        _seq += 1
        return _seq

    _metrics_recorded = False
    _stream_last_finish_reason = None
    _content_part_added = False
    _output_item_added = False
    # reasoning is a SEPARATE output item (OpenAI Responses spec: type=reasoning,
    # rs_ id, output_index 0) PRECEDING the message (output_index 1 when reasoning present).
    # The message's output_item.added/content_part.added are therefore emitted LAZILY (once
    # we know whether reasoning came first) instead of unconditionally up front. These
    # flags live in the OUTER scope so the error/incomplete close paths can read them.
    _reasoning_item_added = False
    _reasoning_done_emitted = False
    _msg_idx = 0
    _rs_id = f"rs-{uuid.uuid4().hex[:24]}"
    try:

        async def _token_source():
            nonlocal \
                prompt_tok, \
                completion_tok, \
                reasoning_tok, \
                cached_tok, \
                accumulated_text, \
                accumulated_thinking, \
                _metrics_recorded, \
                _stream_last_finish_reason, \
                _content_part_added, \
                _output_item_added, \
                _reasoning_item_added, \
                _reasoning_done_emitted, \
                _msg_idx
            last_finish_reason = None

            # ── Lifecycle: response.created ──
            yield format_responses_created(response_id, req.model, seq=_next_seq())

            # ── Lifecycle: response.in_progress ──
            yield format_responses_in_progress(response_id, req.model, seq=_next_seq())

            # NOTE : no unconditional output_item.added here — the message item is
            # opened lazily by _ensure_msg_item() on the first non-reasoning token (or at
            # close), so a leading reasoning item can take output_index 0.

            def _ensure_reasoning_item():
                """Open the reasoning output item once (output_index 0)."""
                nonlocal _reasoning_item_added
                if _reasoning_item_added:
                    return
                _reasoning_item_added = True
                yield format_responses_output_item_added(
                    response_id,
                    req.model,
                    item_id=_rs_id,
                    output_index=0,
                    seq=_next_seq(),
                    item_type="reasoning",
                )
                yield (
                    "event: response.reasoning_summary_part.added\ndata: "
                    + json.dumps(
                        {
                            "type": "response.reasoning_summary_part.added",
                            "item_id": _rs_id,
                            "output_index": 0,
                            "summary_index": 0,
                            "part": {"type": "summary_text", "text": ""},
                            "sequence_number": _next_seq(),
                        }
                    )
                    + "\n\n"
                )

            def _close_reasoning_part():
                """Close the reasoning item once: summary text.done + part.done +
                output_item.done (reads accumulated_thinking at iteration time)."""
                nonlocal _reasoning_done_emitted
                if _reasoning_done_emitted or not _reasoning_item_added:
                    return
                _reasoning_done_emitted = True
                yield (
                    "event: response.reasoning_summary_text.done\ndata: "
                    + json.dumps(
                        {
                            "type": "response.reasoning_summary_text.done",
                            "item_id": _rs_id,
                            "output_index": 0,
                            "summary_index": 0,
                            "text": accumulated_thinking,
                            "sequence_number": _next_seq(),
                        }
                    )
                    + "\n\n"
                )
                yield (
                    "event: response.reasoning_summary_part.done\ndata: "
                    + json.dumps(
                        {
                            "type": "response.reasoning_summary_part.done",
                            "item_id": _rs_id,
                            "output_index": 0,
                            "summary_index": 0,
                            "part": {
                                "type": "summary_text",
                                "text": accumulated_thinking,
                            },
                            "sequence_number": _next_seq(),
                        }
                    )
                    + "\n\n"
                )
                yield (
                    "event: response.output_item.done\ndata: "
                    + json.dumps(
                        {
                            "type": "response.output_item.done",
                            "output_index": 0,
                            "item": {
                                "type": "reasoning",
                                "id": _rs_id,
                                "summary": [
                                    {
                                        "type": "summary_text",
                                        "text": accumulated_thinking,
                                    }
                                ],
                                **_seal_extra(req, accumulated_thinking),
                                "status": "completed",
                            },
                            "sequence_number": _next_seq(),
                        }
                    )
                    + "\n\n"
                )

            def _ensure_msg_item():
                """Open the message output item once, at _msg_idx (1 if a reasoning item
                precedes it, else 0)."""
                nonlocal _output_item_added, _content_part_added, _msg_idx
                if _output_item_added:
                    return
                _msg_idx = 1 if _reasoning_item_added else 0
                _output_item_added = True
                yield format_responses_output_item_added(
                    response_id,
                    req.model,
                    item_id=msg_id,
                    output_index=_msg_idx,
                    seq=_next_seq(),
                )
                yield format_responses_content_part_added(
                    item_id=msg_id,
                    output_index=_msg_idx,
                    content_index=0,
                    seq=_next_seq(),
                )
                _content_part_added = True

            def _emit_token(token_text, is_reasoning, lp=None):
                """Shared per-token emission (both the batched and non-batched loops): lazily
                opens the reasoning/message items and yields the correct delta at the right
                output_index. On the reasoning→text transition, closes the reasoning item."""
                if not token_text:
                    return
                if is_reasoning:
                    for _ev in _ensure_reasoning_item():
                        yield _ev
                    yield (
                        "event: response.reasoning_summary_text.delta\ndata: "
                        + json.dumps(
                            {
                                "type": "response.reasoning_summary_text.delta",
                                "item_id": _rs_id,
                                "output_index": 0,
                                "summary_index": 0,
                                "delta": token_text,
                                "sequence_number": _next_seq(),
                            }
                        )
                        + "\n\n"
                    )
                else:
                    for _ev in _close_reasoning_part():
                        yield _ev
                    for _ev in _ensure_msg_item():
                        yield _ev
                    if _resp_tool_streamer is not None:
                        # feed through the streamer; emit ONLY the clean-text chunks as
                        # deltas, holding back any tool-call markup (surfaced as function_call
                        # items at end-of-stream). logprobs are dropped on this path since the
                        # emitted text no longer aligns 1:1 with the source token.
                        for _so in _resp_tool_streamer.process_token(token_text):
                            if _so.text:
                                yield format_responses_text_delta(
                                    delta=_so.text,
                                    item_id=msg_id,
                                    output_index=_msg_idx,
                                    content_index=0,
                                    logprobs=None,
                                    seq=_next_seq(),
                                )
                    else:
                        yield format_responses_text_delta(
                            delta=token_text,
                            item_id=msg_id,
                            output_index=_msg_idx,
                            content_index=0,
                            logprobs=lp,
                            seq=_next_seq(),
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
                    min_tokens=req.min_tokens,
                    ignore_eos=req.ignore_eos,
                    suppress_tokens=req.suppress_tokens,
                    logit_bias=_logit_bias,
                    min_p=req.min_p,
                    top_n_sigma=req.top_n_sigma,
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
                    lora_adapter=loaded_adapter,
                ):
                    if getattr(output, "error", None):
                        raise EngineStreamError(str(output.error))
                    if hasattr(output, "prompt_tokens") and output.prompt_tokens:
                        prompt_tok = output.prompt_tokens
                    if hasattr(output, "reasoning_tokens") and output.reasoning_tokens:
                        reasoning_tok = output.reasoning_tokens
                    if hasattr(output, "cached_tokens") and output.cached_tokens:
                        cached_tok = max(cached_tok, output.cached_tokens)
                    if (
                        hasattr(output, "completion_tokens")
                        and output.completion_tokens
                    ):
                        completion_tok = max(completion_tok, output.completion_tokens)
                    elif (
                        output.new_text
                        and getattr(output, "current_state", None) != "reasoning"
                    ):
                        # Only count non-reasoning tokens toward completion_tok
                        completion_tok += 1
                    _is_reasoning = (
                        getattr(output, "current_state", None) == "reasoning"
                    )
                    if output.new_text:
                        if _is_reasoning:
                            accumulated_thinking += output.new_text
                        else:
                            accumulated_text += output.new_text
                    if len(accumulated_text) > _MAX_STREAMING_TEXT_BUFFER:
                        logger.error(
                            "Responses streaming text exceeded 1MB — truncating"
                        )
                        accumulated_text = accumulated_text[-_TRUNCATE_KEEP:]
                    if output.finish_reason is not None:
                        last_finish_reason = output.finish_reason

                    # ── Per-token emission via the shared lazy-item helper ──
                    if output.new_text:
                        _delta_lp = None
                        if req.logprobs and not _is_reasoning:
                            _flp = _format_chat_logprobs(
                                getattr(output, "logprobs", None),
                                tokenizer=getattr(engine, "_tokenizer", None),
                                top_logprobs=req.top_logprobs,
                            )
                            _delta_lp = _flp.get("content", []) if _flp else None
                        for _ev in _emit_token(
                            output.new_text, _is_reasoning, _delta_lp
                        ):
                            yield _ev
            else:
                async for output in engine.generate_stream(
                    **_native_kw(req),
                    **_cache_kw(req, engine),
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
                    min_tokens=req.min_tokens,
                    ignore_eos=req.ignore_eos,
                    suppress_tokens=req.suppress_tokens,
                    logit_bias=_logit_bias,
                    min_p=req.min_p,
                    top_n_sigma=req.top_n_sigma,
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
                    lora_adapter=loaded_adapter,
                ):
                    if getattr(output, "error", None):
                        raise EngineStreamError(str(output.error))
                    if hasattr(output, "prompt_tokens") and output.prompt_tokens:
                        prompt_tok = output.prompt_tokens
                    if hasattr(output, "reasoning_tokens") and output.reasoning_tokens:
                        reasoning_tok = output.reasoning_tokens
                    if hasattr(output, "cached_tokens") and output.cached_tokens:
                        cached_tok = max(cached_tok, output.cached_tokens)
                    token_text = getattr(output, "token_text", "")
                    _is_reasoning = (
                        getattr(output, "current_state", None) == "reasoning"
                    )
                    if (
                        hasattr(output, "completion_tokens")
                        and output.completion_tokens
                    ):
                        completion_tok = max(completion_tok, output.completion_tokens)
                    elif token_text and not _is_reasoning:
                        completion_tok += 1
                    if token_text:
                        if _is_reasoning:
                            accumulated_thinking += token_text
                        else:
                            accumulated_text += token_text
                    if len(accumulated_text) > _MAX_STREAMING_TEXT_BUFFER:
                        logger.error(
                            "Responses streaming text exceeded 1MB — truncating"
                        )
                        accumulated_text = accumulated_text[-_TRUNCATE_KEEP:]
                    if getattr(output, "finish_reason", None) is not None:
                        last_finish_reason = output.finish_reason

                    # ── Per-token emission via the shared lazy-item helper ──
                    if token_text:
                        _delta_lp2 = None
                        if req.logprobs and not _is_reasoning:
                            _flp2 = _format_chat_logprobs(
                                getattr(output, "logprobs", None),
                                tokenizer=getattr(engine, "_tokenizer", None),
                                top_logprobs=req.top_logprobs,
                            )
                            _delta_lp2 = _flp2.get("content", []) if _flp2 else None
                        for _ev in _emit_token(token_text, _is_reasoning, _delta_lp2):
                            yield _ev

            # flush any text the tool streamer was holding back (a partial markup
            # prefix that turned out to be plain text) before the message item is finalized.
            if _resp_tool_streamer is not None:
                for _so in _resp_tool_streamer.flush():
                    if _so.text:
                        for _ev in _close_reasoning_part():
                            yield _ev
                        for _ev in _ensure_msg_item():
                            yield _ev
                        yield format_responses_text_delta(
                            delta=_so.text,
                            item_id=msg_id,
                            output_index=_msg_idx,
                            content_index=0,
                            logprobs=None,
                            seq=_next_seq(),
                        )

            # Close the reasoning item if it never saw a following text token (all-reasoning
            # response, or the stream ended mid-reasoning) — otherwise it's unterminated. Then
            # ensure the message item exists (an all-reasoning response still emits an empty
            # message at output_index 1).
            for _ev in _close_reasoning_part():
                yield _ev
            for _ev in _ensure_msg_item():
                yield _ev

            # ── Stop-sequence overcount correction ──
            # The engine counts tokens up to and including the stop sequence, but
            # OpenAI API convention excludes stop tokens from completion_tok.
            if req.stop and last_finish_reason == "stop":
                for _stop_seq in req.stop:
                    if _stop_seq and _stop_seq in accumulated_text:
                        _idx = accumulated_text.find(_stop_seq)
                        accumulated_text = accumulated_text[:_idx]
                        _tok = getattr(engine, "_tokenizer", None)
                        if _tok:
                            try:
                                _correct_count = len(_tok.encode(accumulated_text))
                                if _correct_count < completion_tok:
                                    completion_tok = _correct_count
                            except Exception:
                                pass
                        break

            # ── Check for tool calls in the accumulated text (before closing lifecycles) ──
            tool_calls = None
            clean_text = accumulated_text
            if req.tools and req.tool_choice != "none":
                from yunshu_engine.tool_format import parse_tool_output, tool_formats

                tool_calls, _clean = parse_tool_output(
                    accumulated_text, tool_formats(engine), req.tools
                )
                if tool_calls:
                    clean_text = _clean
                    tool_calls = _enforce_tool_choice(req, tool_calls)

            # ── Lifecycle: response.output_text.done (use cleaned text). output_index is
            # _msg_idx (1 when a reasoning item precedes the message, else 0) —. ──
            yield format_responses_text_done(
                text=clean_text,
                item_id=msg_id,
                output_index=_msg_idx,
                content_index=0,
                seq=_next_seq(),
            )

            # ── Lifecycle: response.content_part.done (use cleaned text) ──
            yield format_responses_content_part_done(
                item_id=msg_id,
                text=clean_text,
                output_index=_msg_idx,
                content_index=0,
                seq=_next_seq(),
            )

            # ── Lifecycle: response.output_item.done (use cleaned text) ──
            yield format_responses_output_item_done(
                item_id=msg_id,
                text=clean_text,
                output_index=_msg_idx,
                seq=_next_seq(),
            )

            # ── Build final output for response.completed ── (: reasoning is a
            # SEPARATE output item preceding the message, matching the streamed events.)
            final_output = []
            if accumulated_thinking:
                final_output.append(
                    {
                        "type": "reasoning",
                        "id": _rs_id,
                        "summary": [
                            {"type": "summary_text", "text": accumulated_thinking}
                        ],
                        **_seal_extra(req, accumulated_thinking),
                        "status": "completed",
                    }
                )
            final_output.append(
                {
                    "type": "message",
                    "id": msg_id,
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": clean_text, "annotations": []}
                    ],
                    "status": "completed",
                }
            )

            # Append function_call items for detected tool calls
            if tool_calls:
                for _tc_idx, tc in enumerate(tool_calls):
                    fc_id = f"fc-{uuid.uuid4().hex[:24]}"
                    fc_call_id = f"call_{uuid.uuid4().hex[:8]}"
                    output_index = len(final_output)  # next output slot

                    # Lifecycle: response.output_item.added for the function_call. :
                    # carry call_id/name/arguments so a strict SDK client reading them off the
                    # `added` event (not just `.done`) sees the call.
                    yield format_responses_output_item_added(
                        response_id,
                        req.model,
                        item_id=fc_id,
                        output_index=output_index,
                        seq=_next_seq(),
                        item_type="function_call",
                        call_id=fc_call_id,
                        # tc is a dict ({"name","arguments"} from
                        # parse_tool_output), NOT an object — getattr() on a
                        # dict returns the default, so the fix silently emitted EMPTY
                        # name/arguments on the streaming output_item.added (re-breaking
                        # on the sibling path). Use subscript like the .done events
                        # below and the non-streaming path do.
                        name=tc["name"],
                        arguments=tc["arguments"],
                    )

                    # Lifecycle: response.function_call_arguments.done
                    _args_done_data = {
                        "type": "response.function_call_arguments.done",
                        "item_id": fc_id,
                        "output_index": output_index,
                        "call_id": fc_call_id,
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                        "sequence_number": _next_seq(),
                    }
                    yield (
                        "event: response.function_call_arguments.done\ndata: "
                        + json.dumps(_args_done_data, ensure_ascii=False)
                        + "\n\n"
                    )

                    # Lifecycle: response.output_item.done for the function_call
                    fc_item = {
                        "type": "function_call",
                        "id": fc_id,
                        "call_id": fc_call_id,
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                        "status": "completed",
                    }
                    _fc_done_data = {
                        "type": "response.output_item.done",
                        "output_index": output_index,
                        "item": fc_item,
                        "sequence_number": _next_seq(),
                    }
                    yield (
                        "event: response.output_item.done\ndata: "
                        + json.dumps(_fc_done_data, ensure_ascii=False)
                        + "\n\n"
                    )

                    final_output.append(fc_item)

            # ── Lifecycle: response.completed or response.incomplete ──
            # Per OpenAI Responses API spec, output_tokens includes ALL tokens
            # (visible + reasoning), with reasoning_tokens as a SUBSET detail.
            # completion_tok already holds the engine's total completion_tokens
            # (n_tok includes reasoning; only stop/suffix tokens are excluded — see
            # batched_engine.py:4368), so output_tokens == completion_tok. The
            # visible-only `elif completion_tok += 1` fallback never fires on the
            # real path (output.completion_tokens is always set). Previously this
            # ADDED reasoning_tok, double-counting it — the non-streaming path
            # (output_tokens=total_ct) does NOT add it, so streaming over-reported
            # usage by the reasoning-token count for every thinking-model request.
            _total_output_tok = completion_tok
            _stream_last_finish_reason = last_finish_reason
            # A non-normal termination must NOT report "completed": length → incomplete
            # (max_output_tokens), and an interrupted/cancelled/aborted/errored stream →
            # incomplete with its own reason. Only a genuine stop/eos/tool_calls/None
            # (normal end) is "completed". The old code mapped everything-but-length to
            # "completed", so a cancelled stream was stored + emitted as completed.
            _interrupt_reasons = {
                "cancel": "cancelled",
                "cancelled": "cancelled",
                "abort": "cancelled",
                "aborted": "cancelled",
                "error": "error",
                "content_filter": "content_filter",
            }
            if _cancel_evt is not None and _cancel_evt.is_set():
                # a POST /v1/responses/{id}/cancel sets this event, but every
                # engine streaming path emits finish_reason="stop" on cancel (not a
                # distinct reason), so the _interrupt_reasons map below never matched →
                # a CANCELLED stream was stored + emitted as "completed". The
                # cancel_event is the authoritative signal; check it first. (Restores
                # the cancelled→incomplete behavior on the streaming path,
                # which the cancel_response docstring promises but never delivered.)
                _terminal_status = "incomplete"
                yield format_responses_incomplete(
                    response_id=response_id,
                    model=req.model,
                    reason="cancelled",
                    output=final_output,
                    input_tokens=prompt_tok,
                    output_tokens=_total_output_tok,
                    total_tokens=prompt_tok + _total_output_tok,
                    reasoning_tokens=reasoning_tok,
                    cached_tokens=cached_tok,
                    seq=_next_seq(),
                )
            elif last_finish_reason in _TRUNCATED_FINISH:
                _terminal_status = "incomplete"
                yield format_responses_incomplete(
                    response_id=response_id,
                    model=req.model,
                    reason="max_output_tokens",
                    output=final_output,
                    input_tokens=prompt_tok,
                    output_tokens=_total_output_tok,
                    total_tokens=prompt_tok + _total_output_tok,
                    reasoning_tokens=reasoning_tok,
                    cached_tokens=cached_tok,
                    seq=_next_seq(),
                )
            elif last_finish_reason in _interrupt_reasons:
                _terminal_status = "incomplete"
                yield format_responses_incomplete(
                    response_id=response_id,
                    model=req.model,
                    reason=_interrupt_reasons[last_finish_reason],
                    output=final_output,
                    input_tokens=prompt_tok,
                    output_tokens=_total_output_tok,
                    total_tokens=prompt_tok + _total_output_tok,
                    reasoning_tokens=reasoning_tok,
                    cached_tokens=cached_tok,
                    seq=_next_seq(),
                )
            else:
                _terminal_status = "completed"
                yield format_responses_completed(
                    response_id=response_id,
                    model=req.model,
                    output=final_output,
                    input_tokens=prompt_tok,
                    output_tokens=_total_output_tok,
                    total_tokens=prompt_tok + _total_output_tok,
                    reasoning_tokens=reasoning_tok,
                    cached_tokens=cached_tok,
                    seq=_next_seq(),
                )

            # Persist final response when the client requested storage. Mirrors
            # the non-streaming path so GET /v1/responses/{id} succeeds.
            if getattr(req, "store", None):
                _usage_dict: dict = responses_usage(
                    prompt_tok, _total_output_tok, reasoning_tok, cached_tok
                )
                try:
                    _store_response(
                        response_id,
                        {
                            "id": response_id,
                            "object": "response",
                            **_config_echo(req),
                            "created_at": int(time.time()),
                            "completed_at": int(time.time()),
                            "model": req.model,
                            "status": _terminal_status,
                            "output": final_output,
                            "metadata": req.metadata,
                            "usage": _usage_dict,
                            # Stash the raw input + predecessor id so previous_response_id
                            # chaining can rebuild the full conversation on follow-up — the
                            # non-streaming path stores these (see ~line 764); without them a
                            # streamed-then-stored response breaks as a chaining parent.
                            "_input_messages": (
                                own_input_messages
                                if own_input_messages is not None
                                else [
                                    m
                                    for m in messages
                                    if m.get("role") not in ("assistant", "system")
                                ]
                            ),
                            "previous_response_id": req.previous_response_id,
                            "_owner": _resolve_owner(request)
                            if request is not None
                            else "",
                        },
                    )
                except Exception:
                    logger.debug("failed to store streaming response", exc_info=True)

            # completion_tok already includes reasoning tokens on the
            # batched path (reasoning_tok is a subset detail); adding it double-counted
            # the server-side completion metric (reasoning-double-count class,
            # un-fixed sibling). The non-streaming path + chat/anthropic already use
            # completion_tok alone.
            _record_metrics(prompt_tok, completion_tok)
            _metrics_recorded = True

            # [DONE] sentinel — required by SSE protocol to signal stream end
            yield "data: [DONE]\n\n"

        async for chunk in with_sse_keepalive(
            _token_source(), http_request=request, cancel_event=_cancel_evt
        ):
            yield chunk.encode("utf-8") if isinstance(chunk, str) else chunk
    except MemoryError:
        if _cancel_evt is not None:
            _cancel_evt.set()
        # Emit initial lifecycle events if they were never sent (error before
        # first engine output). Clients expect response.created before any
        # terminal event.
        if not _output_item_added and not _reasoning_item_added:
            yield format_responses_created(
                response_id, req.model, seq=_next_seq()
            ).encode("utf-8")
            yield format_responses_in_progress(
                response_id, req.model, seq=_next_seq()
            ).encode("utf-8")
        # close an open-but-unterminated reasoning item (output_index 0) before
        # reporting failure, so the SDK's reasoning item isn't left dangling.
        if _reasoning_item_added and not _reasoning_done_emitted:
            _reasoning_done_emitted = True
            yield (
                "event: response.reasoning_summary_text.done\ndata: "
                + json.dumps(
                    {
                        "type": "response.reasoning_summary_text.done",
                        "item_id": _rs_id,
                        "output_index": 0,
                        "summary_index": 0,
                        "text": accumulated_thinking,
                        "sequence_number": _next_seq(),
                    }
                )
                + "\n\n"
            ).encode("utf-8")
            yield (
                "event: response.reasoning_summary_part.done\ndata: "
                + json.dumps(
                    {
                        "type": "response.reasoning_summary_part.done",
                        "item_id": _rs_id,
                        "output_index": 0,
                        "summary_index": 0,
                        "part": {"type": "summary_text", "text": accumulated_thinking},
                        "sequence_number": _next_seq(),
                    }
                )
                + "\n\n"
            ).encode("utf-8")
            yield (
                "event: response.output_item.done\ndata: "
                + json.dumps(
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "type": "reasoning",
                            "id": _rs_id,
                            "summary": [
                                {"type": "summary_text", "text": accumulated_thinking}
                            ],
                            **_seal_extra(req, accumulated_thinking),
                            "status": "completed",
                        },
                        "sequence_number": _next_seq(),
                    }
                )
                + "\n\n"
            ).encode("utf-8")
        # Close open message lifecycle items (at _msg_idx) before reporting failure
        if _content_part_added:
            yield format_responses_content_part_done(
                msg_id,
                text=accumulated_text,
                output_index=_msg_idx,
                content_index=0,
                seq=_next_seq(),
            ).encode("utf-8")
        if _output_item_added:
            yield format_responses_output_item_done(
                msg_id,
                text=accumulated_text,
                output_index=_msg_idx,
                seq=_next_seq(),
            ).encode("utf-8")
        yield format_responses_failed(
            response_id,
            req.model,
            error_code="server_error",
            error_message="Insufficient GPU memory",
            input_tokens=prompt_tok,
            output_tokens=completion_tok,
            total_tokens=prompt_tok + completion_tok,
            reasoning_tokens=reasoning_tok,
            seq=_next_seq(),
        ).encode("utf-8")
        yield b"data: [DONE]\n\n"
        return
    except Exception as e:
        if _cancel_evt is not None:
            _cancel_evt.set()
        logger.error(f"Responses API streaming error: {e}", exc_info=True)
        # Emit initial lifecycle events if they were never sent (error before
        # first engine output). Clients expect response.created before any
        # terminal event.
        if not _output_item_added and not _reasoning_item_added:
            yield format_responses_created(
                response_id, req.model, seq=_next_seq()
            ).encode("utf-8")
            yield format_responses_in_progress(
                response_id, req.model, seq=_next_seq()
            ).encode("utf-8")
        # close an open-but-unterminated reasoning item (output_index 0) before
        # reporting failure, so the SDK's reasoning item isn't left dangling.
        if _reasoning_item_added and not _reasoning_done_emitted:
            _reasoning_done_emitted = True
            yield (
                "event: response.reasoning_summary_text.done\ndata: "
                + json.dumps(
                    {
                        "type": "response.reasoning_summary_text.done",
                        "item_id": _rs_id,
                        "output_index": 0,
                        "summary_index": 0,
                        "text": accumulated_thinking,
                        "sequence_number": _next_seq(),
                    }
                )
                + "\n\n"
            ).encode("utf-8")
            yield (
                "event: response.reasoning_summary_part.done\ndata: "
                + json.dumps(
                    {
                        "type": "response.reasoning_summary_part.done",
                        "item_id": _rs_id,
                        "output_index": 0,
                        "summary_index": 0,
                        "part": {"type": "summary_text", "text": accumulated_thinking},
                        "sequence_number": _next_seq(),
                    }
                )
                + "\n\n"
            ).encode("utf-8")
            yield (
                "event: response.output_item.done\ndata: "
                + json.dumps(
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "type": "reasoning",
                            "id": _rs_id,
                            "summary": [
                                {"type": "summary_text", "text": accumulated_thinking}
                            ],
                            **_seal_extra(req, accumulated_thinking),
                            "status": "completed",
                        },
                        "sequence_number": _next_seq(),
                    }
                )
                + "\n\n"
            ).encode("utf-8")
        # Close open message lifecycle items (at _msg_idx) before reporting failure
        if _content_part_added:
            yield format_responses_content_part_done(
                msg_id,
                text=accumulated_text,
                output_index=_msg_idx,
                content_index=0,
                seq=_next_seq(),
            ).encode("utf-8")
        if _output_item_added:
            yield format_responses_output_item_done(
                msg_id,
                text=accumulated_text,
                output_index=_msg_idx,
                seq=_next_seq(),
            ).encode("utf-8")
        yield format_responses_failed(
            response_id,
            req.model,
            error_code="server_error",
            error_message=str(e)
            if isinstance(e, EngineStreamError)
            else "Internal server error",
            input_tokens=prompt_tok,
            output_tokens=completion_tok,
            total_tokens=prompt_tok + completion_tok,
            reasoning_tokens=reasoning_tok,
            seq=_next_seq(),
        ).encode("utf-8")
        yield b"data: [DONE]\n\n"
        return
    finally:
        if _tracker is not None:
            with contextlib.suppress(Exception):
                _tracker.unregister(_stream_id)
        if loaded_adapter is not None:
            _release_lora_adapter(engine, loaded_adapter)
        if not _metrics_recorded and (prompt_tok > 0 or completion_tok > 0):
            with contextlib.suppress(Exception):
                # completion_tok already includes reasoning tokens (subset
                # detail) — adding reasoning_tok again double-counted the server
                # metric. See the in-loop record above.
                _record_metrics(prompt_tok, completion_tok)


# ──────────────────────────────────────────────────────────────────────
# Responses API — retrieval and cancellation
# ──────────────────────────────────────────────────────────────────────


@router.get("/responses/{response_id}", response_model=None)
async def get_response(response_id: str, request: Request):
    """Retrieve a stored response previously created with `store: true`.

    OpenAI-compatible: returns 404 if the response is unknown or was
    created without storage enabled.
    """
    _check_permission(request, "can_infer")
    payload = _get_stored_response(response_id)
    # SECURITY (cross-tenant IDOR): a stored response is owned by the
    # tenant that created it. Treat a cross-tenant access exactly like
    # not-found so the id space isn't enumerable. (Every other per-handle router
    # enforces this —; Responses was missed.)
    if payload is None or not _owns_stored(request, payload):
        # Return JSONResponse directly — using HTTPException with a dict
        # detail causes the global error middleware to stringify the dict
        # and re-wrap it (double-encoded error.message). See Bug-3 fix.
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"Response with id '{response_id}' not found",
                    "type": "invalid_request_error",
                    "code": "response_not_found",
                }
            },
        )
    # Strip internal plumbing (_owner, _input_messages) before returning.
    return JSONResponse(_public_stored(payload))


def _input_item_of(response_id: str, index: int, msg: dict) -> list[dict]:
    """The stored chat-format messages of a response as Responses input items (ids are stable:
    ``<response id>:<index>``). A tool-calling assistant turn becomes one ``function_call`` item per
    call, a ``tool`` message a ``function_call_output``; text becomes ``input_text`` parts."""
    role = msg.get("role", "user")
    if role == "tool":
        out = msg.get("content", "")
        return [
            {
                "id": f"fco_{response_id}_{index}",
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id", "") or "",
                "output": out if isinstance(out, str) else json.dumps(out),
                "status": "completed",
            }
        ]
    items: list[dict] = []
    content = msg.get("content")
    parts: list[dict[str, Any]] = []
    if isinstance(content, str) and content:
        parts = [{"type": "input_text", "text": content}]
    elif isinstance(content, list):
        for p in content:
            if isinstance(p, dict) and p.get("type") in ("text", "input_text"):
                parts.append({"type": "input_text", "text": p.get("text", "")})
            elif isinstance(p, dict) and p.get("type") in ("image_url", "input_image"):
                url = p.get("image_url")
                url = url.get("url") if isinstance(url, dict) else url
                parts.append(
                    {"type": "input_image", "image_url": url, "detail": "auto"}
                )
    if parts or not msg.get("tool_calls"):
        items.append(
            {
                "id": f"msg_{response_id}_{index}",
                "type": "message",
                "role": role,
                "content": parts,
                "status": "completed",
            }
        )
    for k, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function", {})
        items.append(
            {
                "id": f"fc_{response_id}_{index}_{k}",
                "type": "function_call",
                "call_id": tc.get("id", "") or "",
                "name": fn.get("name", ""),
                "arguments": fn.get("arguments", "") or "{}",
                "status": "completed",
            }
        )
    return items


@router.get("/responses/{response_id}/input_items", response_model=None)
async def list_response_input_items(
    response_id: str,
    request: Request,
    limit: int = 20,
    order: str = "desc",
    after: str | None = None,
    before: str | None = None,
):
    """The input items of a stored response (OpenAI ``GET /v1/responses/{id}/input_items``).

    This hop's own input only (``instructions`` and system / developer messages are not listed,
    like the hosted API); ``previous_response_id`` history belongs to the earlier responses.
    Cursor pagination with ``limit`` (1-100), ``order`` (``desc`` default) and ``after`` /
    ``before`` item ids."""
    _check_permission(request, "can_infer")
    payload = _get_stored_response(response_id)
    if payload is None or not _owns_stored(request, payload):
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"Response with id '{response_id}' not found",
                    "type": "invalid_request_error",
                    "code": "response_not_found",
                }
            },
        )
    if not 1 <= limit <= 100 or order not in ("asc", "desc"):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "limit must be 1-100 and order 'asc' or 'desc'",
                    "type": "invalid_request_error",
                    "param": "limit" if not 1 <= limit <= 100 else "order",
                    "code": "invalid_value",
                }
            },
        )
    items: list[dict] = list(payload.get("_client_input_items") or [])
    if "_client_input_items" not in payload:
        for i, m in enumerate(payload.get("_input_messages") or []):
            items.extend(_input_item_of(response_id, i, m))
    if order == "desc":
        items.reverse()
    ids = [it["id"] for it in items]
    for cursor, name in ((after, "after"), (before, "before")):
        if cursor is not None and cursor not in ids:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "message": f"Item '{cursor}' not found",
                        "type": "invalid_request_error",
                        "param": name,
                        "code": "item_not_found",
                    }
                },
            )
    lo = ids.index(after) + 1 if after is not None else 0
    hi = ids.index(before) if before is not None else len(items)
    items = items[lo:hi]
    page = items[:limit]
    return JSONResponse(
        {
            "object": "list",
            "data": page,
            "first_id": page[0]["id"] if page else None,
            "last_id": page[-1]["id"] if page else None,
            "has_more": len(items) > limit,
        }
    )


@router.delete("/responses/{response_id}", response_model=None)
async def delete_response(response_id: str, request: Request):
    """Delete a stored response (OpenAI-compatible DELETE /v1/responses/{id}).

    the endpoint was missing entirely — a stored response/conversation could
    never be explicitly deleted (only LRU-evicted after the store cap), so clients got 405
    and a stored conversation lingered until eviction (a minor privacy gap). Returns the
    OpenAI delete envelope. Ownership-gated exactly like GET (cross-tenant delete is denied as not-found so the id space stays non-enumerable, the IDOR class).
    """
    _check_permission(request, "can_infer")
    payload = _get_stored_response(response_id)
    if payload is None or not _owns_stored(request, payload):
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"Response with id '{response_id}' not found",
                    "type": "invalid_request_error",
                    "code": "response_not_found",
                }
            },
        )
    _delete_stored_response(response_id)
    return JSONResponse(
        {
            "id": response_id,
            "object": "response.deleted",
            "deleted": True,
        }
    )


@router.post("/responses/{response_id}/cancel", response_model=None)
async def cancel_response(response_id: str, request: Request):
    """Cancel an in-progress response generation.

    Signals the request tracker's cancel_event for the given response_id.
    The active generator polls the event between steps and exits with a
    cancelled finish_reason.  Idempotent: returns the stored payload (if
    any) on subsequent calls.
    """
    _check_permission(request, "can_infer")

    # SECURITY (cross-tenant cancel IDOR): mirror /v1/cancel's ownership
    # guard. Without it any tenant could cancel another tenant's in-flight
    # Responses generation by id (the SSE stream leaks the resp- id). Admins
    # bypass; an in-flight request with no recorded owner stays cancellable.
    cancelled = False
    try:
        from yunshu_engine.request_tracker import get_request_tracker

        tracker = get_request_tracker()
        if not _is_admin(request) and hasattr(tracker, "get_owner"):
            _owner = tracker.get_owner(response_id)
            if _owner and _owner != _resolve_owner(request):
                # Deny as not-found to avoid leaking the id's existence.
                return JSONResponse(
                    status_code=404,
                    content={
                        "error": {
                            "message": f"Response with id '{response_id}' not found or already finished",
                            "type": "invalid_request_error",
                            "code": "response_not_found",
                        }
                    },
                )
        cancelled = tracker.cancel(response_id)
    except Exception:
        logger.debug("cancel: request tracker unavailable", exc_info=True)

    stored = _get_stored_response(response_id)
    # A stored (already-finalized) response is also ownership-gated.
    if stored is not None and not _owns_stored(request, stored):
        stored = None
    if not cancelled and stored is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"Response with id '{response_id}' not found or already finished",
                    "type": "invalid_request_error",
                    "code": "response_not_found",
                }
            },
        )

    if stored is not None:
        # a cancel for a BACKGROUND response can arrive during the `queued` window
        # — before _runner registered with the request_tracker — so tracker.cancel found
        # nothing (cancelled=False) and previously the request ran to completion, losing the
        # cancel. Persist a cancel marker; _runner re-reads the stored status and bails
        # before generating. (An already-terminal stored status is surfaced unchanged.)
        # (also when the tracker was signalled: the reply is the cancelled response, as the
        # hosted API returns, not the in_progress snapshot; the generation finishing later sees
        # this marker and leaves it alone)
        if stored.get("status") in ("queued", "in_progress"):
            _cancel_payload = dict(stored)
            _cancel_payload["status"] = "cancelled"
            _cancel_payload["completed_at"] = int(time.time())
            _store_response(response_id, _cancel_payload)
            return JSONResponse(_public_stored(_cancel_payload))
        # If the response already finalized as completed/incomplete before
        # the cancel landed, surface that terminal state.
        return JSONResponse(_public_stored(stored))

    # In-flight: signalled but not yet finalized. Return a synthetic
    # cancelled envelope mirroring OpenAI's shape.
    return JSONResponse(
        {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "model": "",
            "status": "cancelled",
            "output": [],
        }
    )


def _count_prompt_tokens(tok, messages: list[dict], tools=None) -> int:
    """Tokens of the prompt the chat template renders for ``messages`` (+ native ``tools``), the way
    the engines render it: template, generation prompt, no double BOS; images at the engine's
    conservative per-image estimate."""
    from yunshu_control.token_counter import IMAGE_TOKEN_ESTIMATE

    kw = {"tools": tools} if tools else {}
    try:
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kw
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"chat template failed: {exc}"
        ) from None
    bos = getattr(tok, "bos_token", None)
    add = not (isinstance(bos, str) and bos and prompt.startswith(bos))
    try:
        n = len(tok.encode(prompt, add_special_tokens=add))
    except TypeError:
        n = len(tok.encode(prompt))
    images = sum(
        1
        for m in messages
        if isinstance(m.get("content"), list)
        for p in m["content"]
        if isinstance(p, dict)
        and p.get("type") in ("image_url", "input_image", "image")
    )
    return n + images * IMAGE_TOKEN_ESTIMATE


@router.post("/responses/input_tokens", response_model=None)
async def count_input_tokens(request: Request):
    """Count the input tokens a `/v1/responses` request would use (OpenAI input_tokens API).

    Builds the prompt the way ``create_response`` does (instructions, input items, conversation,
    compaction items, ``previous_response_id`` chain, function tools rendered natively when the
    template does it, injected otherwise) and counts the rendered template, so the number equals
    the ``usage.input_tokens`` of the same request (a real-server check compares them). Not
    counted: server-side tools (web_search / mcp definitions), the assistant prefill of a forced
    tool_choice on an engine without native tools, and the compaction a ``context_management``
    threshold would trigger."""
    from pydantic import ValidationError

    from ..conversations_store import ConversationError
    from ..responses_context import (
        ContextError,
        count_state_items,
        error_json,
        needs_state,
    )
    from .tokenize import _resolve_tokenizer

    _check_permission(request, "can_infer")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    body.setdefault("model", "")
    try:
        req = ResponsesRequest.model_validate(body)
    except ValidationError as exc:
        e = exc.errors()[0]
        loc = ".".join(str(x) for x in e.get("loc", ()))
        raise HTTPException(status_code=400, detail=f"{loc}: {e.get('msg')}") from None
    tok = _resolve_tokenizer(str(req.model or ""))
    if needs_state(req):
        try:
            items, prev_id = count_state_items(req, request)
        except (ContextError, ConversationError) as exc:
            return error_json(exc)
        req = req.model_copy(
            update={
                "input": [ResponseInputText(**i) for i in items],
                "conversation": None,
                "context_management": None,
                "previous_response_id": prev_id,
            }
        )
    fn_tools = function_tools(req.tools)
    if fn_tools != req.tools:
        req = req.model_copy(update={"tools": fn_tools})
    messages = _apply_previous_response_chain(req, _convert_to_messages(req), request)
    native = None
    if req.tools:
        from .chat import ToolDefinition, ToolFunction, _vlm_renders_tools

        defs = [
            ToolDefinition(
                function=ToolFunction(
                    name=t.name, description=t.description, parameters=t.parameters
                )
            )
            for t in req.tools
        ]
        engine = get_engine()
        if (
            req.tool_choice in (None, "auto") or _forced_by_grammar(req)
        ) and _vlm_renders_tools(engine):
            native = [t.model_dump() for t in defs]
        else:
            from .chat import _inject_tool_system_prompt

            messages = _inject_tool_system_prompt(
                messages,
                defs,
                tool_choice=_chat_tool_choice(req.tool_choice),
                parallel_tool_calls=req.parallel_tool_calls,
                engine=engine,
            )
            from yunshu_engine.batched_engine import _REQUEST_TOOLS

            native = _REQUEST_TOOLS.get()
    return {
        "object": "response.input_tokens",
        "input_tokens": _count_prompt_tokens(tok, messages, native),
    }
