"""Stateful context for ``POST /v1/responses``: Conversations, compaction items, auto compaction.

``create_response`` calls :func:`run_stateful_response` once, at its top, when a request carries a
``conversation``, a ``compaction`` input item or ``context_management``. The wrapper rewrites the request
into a plain one (conversation items prepended, compaction items expanded, optionally compacted first),
runs ``create_response`` again on it and then patches the outward response: ``conversation`` on the
object, the compaction item in ``output``, and the request / output items appended to the conversation
once. Because the wrapper sits outside everything else (server-tool loop, VLM path, stream and
non-stream), each path is covered by the single hook.

Compaction keeps every user / developer message verbatim and replaces the rest by one opaque
``compaction`` item that wraps a model-written summary (produced over loopback through this server's own
``/v1/chat/completions``).
"""

from __future__ import annotations

import contextlib
import json
import logging
import secrets
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from yunshu_engine import settings

from .conversations_store import ConversationError, get_store, normalize_item
from .reasoning_token import seal_compaction, unseal_compaction

logger = logging.getLogger(__name__)

SUMMARY_PREFIX = "Summary of the earlier conversation:\n"

COMPACT_PROMPT = (
    "You compress a conversation so that the work can continue from your summary alone. "
    "Summarize the conversation so far: the user's goals, decisions taken, files touched, "
    "tool calls and their key results, open tasks and next steps. Keep exact identifiers "
    "(names, paths, ids, numbers, commands, error messages) verbatim. Do not answer the "
    "conversation and do not add commentary; reply with the summary only."
)

_TOOL_RESULT_CHARS = 2000


class ContextError(Exception):
    def __init__(
        self,
        status: int,
        message: str,
        code: str | None = None,
        param: str | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code
        self.param = param


def error_json(exc: ContextError | ConversationError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status,
        content={
            "error": {
                "message": exc.message,
                "type": "invalid_request_error" if exc.status < 500 else "server_error",
                "param": exc.param,
                "code": exc.code,
            }
        },
    )


# ---------------------------------------------------------------------------
# items
# ---------------------------------------------------------------------------


def summary_message(text: str) -> dict:
    """A compaction summary as a Responses user message (never a second system message)."""
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": SUMMARY_PREFIX + text}],
    }


def make_compaction_item(summary: str) -> dict:
    return {
        "type": "compaction",
        "id": f"cmp_{secrets.token_hex(12)}",
        "encrypted_content": seal_compaction(summary),
    }


def expand_compaction_items(items: list[dict]) -> list[dict]:
    """``compaction`` items become summary messages; tokens that are not ours are dropped."""
    out: list[dict] = []
    for it in items:
        if isinstance(it, dict) and it.get("type") == "compaction":
            text = unseal_compaction(it.get("encrypted_content"))
            if text:
                out.append(summary_message(text))
            continue
        out.append(it)
    return out


def _item_dict(it: Any) -> dict:
    if hasattr(it, "model_dump"):
        return it.model_dump(exclude_none=True)
    return dict(it)


def request_items(req) -> list[dict]:
    """The request's own ``input`` as item dicts."""
    if isinstance(req.input, str):
        return [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": req.input}],
            }
        ]
    return [_item_dict(i) for i in req.input or []]


def _window(items: list[dict]) -> list[dict]:
    """Items from the last compaction item on (older ones are covered by its summary)."""
    for i in range(len(items) - 1, -1, -1):
        if items[i].get("type") == "compaction":
            return items[i:]
    return items


def _to_messages(items: list[dict], model: str = "local") -> list[dict]:
    from .routers.responses import ResponseInputText, ResponsesRequest

    req = ResponsesRequest.model_construct(
        model=model,
        input=[ResponseInputText(**i) for i in items],
        instructions=None,
    )
    from .routers.responses import _convert_to_messages

    return _convert_to_messages(req)


def _count_tokens(messages: list[dict]) -> int:
    from yunshu_control.token_counter import count_message_tokens

    from .engine import get_engine

    eng = get_engine()
    tok = getattr(eng, "_tokenizer", None) or getattr(eng, "tokenizer", None)
    return int(count_message_tokens(messages, tok))


# ---------------------------------------------------------------------------
# compaction
# ---------------------------------------------------------------------------


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for p in content or []:
        if isinstance(p, dict):
            if p.get("text"):
                parts.append(p["text"])
            elif p.get("type") in ("image_url", "input_image"):
                parts.append("[image]")
        else:
            parts.append(str(p))
    return "\n".join(parts)


def _transcript_line(msg: dict) -> str:
    role = msg.get("role", "user")
    if role == "tool":
        return f"tool result: {_text_of(msg.get('content'))[:_TOOL_RESULT_CHARS]}"
    lines = []
    text = _text_of(msg.get("content"))
    if text:
        lines.append(f"{role}: {text}")
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        lines.append(
            f"assistant called tool {fn.get('name')}({str(fn.get('arguments'))[:_TOOL_RESULT_CHARS]})"
        )
    return "\n".join(lines)


def _kept_item(item: dict) -> bool:
    return item.get("type", "message") in ("message", "") and item.get("role") in (
        "user",
        "developer",
        "system",
    )


def _msg_to_item(msg: dict) -> dict | None:
    """A stored chat message (previous_response_id chain) as a kept user message item."""
    if msg.get("role") not in ("user", "system"):
        return None
    content = msg.get("content")
    if isinstance(content, str):
        parts = [{"type": "input_text", "text": content}]
    else:
        parts = []
        for p in content or []:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "text":
                parts.append({"type": "input_text", "text": p.get("text", "")})
            elif p.get("type") == "image_url":
                url = (p.get("image_url") or {}).get("url")
                parts.append({"type": "input_image", "image_url": url})
    return {
        "type": "message",
        "role": "user" if msg["role"] == "user" else "developer",
        "content": parts,
    }


def stored_chain_messages(prev_id: str, request: Request) -> list[dict]:
    """Chat messages of a stored ``previous_response_id`` chain, oldest first."""
    from .routers.responses import _get_stored_response, _owns_stored

    hops: list[list[dict]] = []
    seen: set[str] = set()
    pid: str | None = prev_id
    for _ in range(16):
        if not pid or pid in seen:
            break
        seen.add(pid)
        p = _get_stored_response(pid)
        if not p or not _owns_stored(request, p):
            break
        turn = list(p.get("_input_messages") or [])
        for out in p.get("output") or []:
            if out.get("role") == "assistant":
                for c in out.get("content") or []:
                    if c.get("type") == "output_text":
                        turn.append({"role": "assistant", "content": c.get("text", "")})
            elif out.get("type") == "function_call":
                turn.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": out.get("call_id") or "",
                                "type": "function",
                                "function": {
                                    "name": out.get("name") or "",
                                    "arguments": out.get("arguments") or "{}",
                                },
                            }
                        ],
                    }
                )
        hops.append(turn)
        pid = p.get("previous_response_id")
    return [m for hop in reversed(hops) for m in hop]


def entries_from_items(items: list[dict]) -> list[tuple[dict | None, list[dict]]]:
    """(item to keep verbatim | None, chat messages for the transcript) per input item."""
    out: list[tuple[dict | None, list[dict]]] = []
    for it in items:
        if it.get("type") == "compaction":
            text = unseal_compaction(it.get("encrypted_content"))
            if text:
                out.append((None, [{"role": "user", "content": SUMMARY_PREFIX + text}]))
            continue
        if it.get("type") == "reasoning":
            continue
        msgs = _to_messages([it])
        out.append((it if _kept_item(it) else None, msgs))
    return out


def entries_from_messages(
    msgs: list[dict],
) -> list[tuple[dict | None, list[dict]]]:
    return [(_msg_to_item(m), [m]) for m in msgs]


SUMMARY_INSTRUCTION = (
    COMPACT_PROMPT
    + " The conversation so far is the messages above this one; do not continue it "
    "and do not call tools. If it starts with an earlier summary, fold it in and update "
    "what changed."
)

MAX_SUMMARY_PASSES = 8


def chat_tools(tools: Any) -> list[dict] | None:
    """Responses function tools as chat-completions tools, so the summary request renders the
    same template head (tool block) as the conversation and shares its prefix cache."""
    out = []
    for t in tools or []:
        if not isinstance(t, dict) or t.get("type") != "function" or not t.get("name"):
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description") or "",
                    "parameters": t.get("parameters")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return out or None


def _summary_request_messages(
    history: list[dict], instructions: str | None, guidance: str | None = None
) -> list[dict]:
    """The real messages first (the prefix the server already holds in its cache), the
    summarize instruction last."""
    ask = SUMMARY_INSTRUCTION
    if guidance:
        ask += "\n\nAdditional guidance:\n" + guidance
    head = [{"role": "system", "content": instructions}] if instructions else []
    return [*head, *history, {"role": "user", "content": ask}]


async def summarize(
    request: Request,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
) -> tuple[str, dict]:
    """Ask this server (loopback chat completions, same model) to summarize ``messages``
    (which already end with the summarize instruction)."""
    from .routers.ollama import _client

    body: dict = {
        "model": model,
        "stream": False,
        "temperature": 0.2,
        "max_tokens": int(settings.get("YUNSHU_COMPACT_MAX_TOKENS")),
        "enable_thinking": False,
        "messages": messages,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "none"
    async with _client(request) as c:
        r = await c.post("/v1/chat/completions", json=body)
    if r.status_code != 200:
        msg = "compaction summary request failed"
        with contextlib.suppress(Exception):
            msg = r.json()["error"]["message"]
        raise ContextError(r.status_code, msg, "compaction_failed")
    data = r.json()
    text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if not text.strip():
        raise ContextError(
            502, "compaction produced an empty summary", "compaction_failed"
        )
    u = data.get("usage") or {}
    pt = int(u.get("prompt_tokens", u.get("input_tokens", 0)) or 0)
    ct = int(u.get("completion_tokens", u.get("output_tokens", 0)) or 0)
    det = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
    return text.strip(), {
        "input_tokens": pt,
        "output_tokens": ct,
        "total_tokens": pt + ct,
        "input_tokens_details": {
            "cached_tokens": int(det.get("cached_tokens", 0) or 0),
            "cache_write_tokens": 0,
        },
        "output_tokens_details": {"reasoning_tokens": 0},
    }


def safe_boundaries(entries: list[tuple[dict | None, list[dict]]]) -> list[int]:
    """Entry indexes (> 0) where a user message starts and no tool call is still waiting for
    its result: the only places a history may be cut."""
    pending: set[str] = set()
    out = []
    for i, (_, msgs) in enumerate(entries):
        if i and not pending and msgs and msgs[0].get("role") == "user":
            out.append(i)
        for m in msgs:
            for tc in m.get("tool_calls") or []:
                pending.add(tc.get("id") or "")
            if m.get("role") == "tool":
                pending.discard(m.get("tool_call_id") or "")
    return out


def _context_budget(model: str) -> int | None:
    """Prompt tokens one summary call may hold: the context window minus the summary."""
    from .engine import get_engine
    from .streaming import get_max_context_window

    try:
        win = get_max_context_window(model, get_engine())
    except Exception:
        return None
    if not win:
        return None
    return max(256, int(win) - int(settings.get("YUNSHU_COMPACT_MAX_TOKENS")) - 256)


def _add_usage(total: dict, u: dict) -> None:
    for k in ("input_tokens", "output_tokens", "total_tokens"):
        total[k] = total.get(k, 0) + u.get(k, 0)
    for k in ("cached_tokens", "cache_write_tokens"):
        d = total.setdefault("input_tokens_details", {})
        d[k] = d.get(k, 0) + (u.get("input_tokens_details") or {}).get(k, 0)
    total.setdefault("output_tokens_details", {"reasoning_tokens": 0})


async def summarize_history(
    request: Request,
    model: str,
    entries: list[tuple[dict | None, list[dict]]],
    instructions: str | None = None,
    guidance: str | None = None,
    tools: list[dict] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[str, dict]:
    """Summary of ``entries``. One call when the whole history fits the model window (the
    messages are the prompt prefix, so the cache the conversation built is reused); otherwise
    up to ``MAX_SUMMARY_PASSES`` passes, each folding the previous summary plus the most whole
    exchanges that fit, cutting only at safe boundaries (a tool call never loses its result)."""
    budget = _context_budget(model)
    ctools = chat_tools(tools)
    pending = list(entries)
    summary_msg: list[dict] = []
    usage: dict = {}
    for n in range(1, MAX_SUMMARY_PASSES + 1):

        def build(k: int) -> list[dict]:
            hist = summary_msg + [m for _, ms in pending[:k] for m in ms]
            return _summary_request_messages(hist, instructions, guidance)

        cut = len(pending)
        if budget is not None and _count_tokens(build(cut)) > budget:
            bounds = [*safe_boundaries(pending), len(pending)]
            lo, hi = 0, len(bounds)
            while lo < hi:
                mid = (lo + hi) // 2
                if _count_tokens(build(bounds[mid])) <= budget:
                    lo = mid + 1
                else:
                    hi = mid
            if lo == 0:
                raise ContextError(
                    400,
                    "a single conversation exchange does not fit the model's context "
                    "window, so it cannot be summarized",
                    "context_length_exceeded",
                )
            cut = bounds[lo - 1]
        if progress:
            progress(n, MAX_SUMMARY_PASSES)
        text, u = await summarize(request, model, build(cut), ctools)
        _add_usage(usage, u)
        pending = pending[cut:]
        if not pending:
            return text, usage
        summary_msg = [{"role": "user", "content": SUMMARY_PREFIX + text}]
    raise ContextError(
        400,
        f"compaction needs more than {MAX_SUMMARY_PASSES} summary passes",
        "context_length_exceeded",
    )


async def compact_entries(
    request: Request,
    model: str,
    entries: list[tuple[dict | None, list[dict]]],
    instructions: str | None = None,
    *,
    guidance: str | None = None,
    tools: list[dict] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[dict], dict]:
    """Kept user/developer messages verbatim, then one ``compaction`` item; plus the usage."""
    kept = [it for it, _ in entries if it is not None]
    summary, usage = await summarize_history(
        request, model, entries, instructions, guidance, tools, progress
    )
    out = [normalize_item(it) for it in kept]
    out.append(make_compaction_item(summary))
    return out, usage


# ---------------------------------------------------------------------------
# the create_response wrapper
# ---------------------------------------------------------------------------


def needs_state(req) -> bool:
    if getattr(req, "conversation", None) or getattr(req, "context_management", None):
        return True
    return isinstance(req.input, list) and any(
        (i.get("type") if isinstance(i, dict) else getattr(i, "type", None))
        == "compaction"
        for i in req.input
    )


def _conversation_id(req) -> str | None:
    c = getattr(req, "conversation", None)
    if isinstance(c, dict):
        return c.get("id")
    return c or None


def _threshold(req) -> int | None:
    for cm in getattr(req, "context_management", None) or []:
        if isinstance(cm, dict) and cm.get("type") == "compaction":
            t = cm.get("compact_threshold")
            if isinstance(t, int) and t > 0:
                return t
    return None


def _terminal(name: str) -> bool:
    return name in ("response.completed", "response.incomplete")


def _finish(
    body: dict,
    conv_id: str | None,
    own_items: list[dict],
    compaction: dict | None,
) -> dict:
    """Patch a response object in place and append to the conversation."""
    if compaction is not None:
        body["output"] = [compaction, *(body.get("output") or [])]
    if conv_id:
        body["conversation"] = {"id": conv_id}
    return body


def _append(conv_id: str, own_items: list[dict], body: dict) -> None:
    items = [i for i in own_items if i.get("type") != "item_reference"]
    if body.get("status") in ("completed", "incomplete"):
        items += list(body.get("output") or [])
    try:
        get_store().add_items(conv_id, items)
    except ConversationError:
        logger.warning(
            "conversation %s: could not append items", conv_id, exc_info=True
        )


def _restore_store(body: dict, conv_id: str | None, compaction: dict | None) -> None:
    """The stored copy of the response carries the same additions."""
    from .routers.responses import _get_stored_response

    rid = body.get("id")
    stored = _get_stored_response(rid) if rid else None
    if stored is None:
        return
    if compaction is not None and not any(
        o.get("id") == compaction["id"] for o in stored.get("output") or []
    ):
        stored["output"] = [compaction, *(stored.get("output") or [])]
    if conv_id:
        stored["conversation"] = {"id": conv_id}


async def _stream(
    src: AsyncIterator[bytes | str],
    conv_id: str | None,
    own_items: list[dict],
    compaction: dict | None,
) -> AsyncIterator[bytes]:
    buf = b""
    async for chunk in src:
        buf += chunk.encode() if isinstance(chunk, str) else chunk
        while b"\n\n" in buf:
            block, buf = buf.split(b"\n\n", 1)
            yield _patch_event(block, conv_id, own_items, compaction) + b"\n\n"
    if buf:
        yield _patch_event(buf, conv_id, own_items, compaction)


def _patch_event(
    block: bytes, conv_id: str | None, own_items: list[dict], compaction: dict | None
) -> bytes:
    lines = block.decode("utf-8", errors="replace").split("\n")
    for i, ln in enumerate(lines):
        if not ln.startswith("data:"):
            continue
        try:
            data = json.loads(ln[5:].strip())
        except ValueError:
            continue
        resp = data.get("response") if isinstance(data, dict) else None
        if not isinstance(resp, dict) or resp.get("object") != "response":
            continue
        terminal = _terminal(str(data.get("type")))
        _finish(resp, conv_id, own_items, compaction if terminal else None)
        if terminal:
            if conv_id:
                _append(conv_id, own_items, resp)
            _restore_store(resp, conv_id, compaction)
        lines[i] = "data: " + json.dumps(data)
    return "\n".join(lines).encode("utf-8")


def count_state_items(req, request: Request) -> tuple[list[dict], str | None]:
    """The input items a stateful request would send (conversation items first, compaction items
    expanded) and its remaining ``previous_response_id``, without generating or writing anything:
    what ``POST /v1/responses/input_tokens`` counts. Raises ``ContextError`` / ``ConversationError``
    like generation."""
    conv_id = _conversation_id(req)
    if conv_id and req.previous_response_id:
        raise ContextError(
            400,
            "conversation cannot be combined with previous_response_id",
            "invalid_request",
            "conversation",
        )
    conv_items: list[dict] = []
    if conv_id:
        try:
            conv_items = _window(get_store().all_items(conv_id))
        except ConversationError as exc:
            exc.param = "conversation"
            raise
    items = expand_compaction_items(conv_items + request_items(req))
    return items, req.previous_response_id


async def run_stateful_response(req, request: Request, inner):
    """Wrapper called from ``create_response``; ``inner`` is ``create_response`` itself."""
    from .routers.models import _check_permission

    _check_permission(request, "can_infer")
    conv_id = _conversation_id(req)
    try:
        if conv_id and req.previous_response_id:
            raise ContextError(
                400,
                "conversation cannot be combined with previous_response_id",
                "invalid_request",
                "conversation",
            )
        conv_items: list[dict] = []
        if conv_id:
            try:
                conv_items = _window(get_store().all_items(conv_id))
            except ConversationError as exc:
                exc.param = "conversation"
                raise
        own_items = request_items(req)
        items = conv_items + own_items
        prev_id = req.previous_response_id
        compaction = None
        thr = _threshold(req)
        if thr is not None and req.generate is not False:
            model = req.model
            entries = entries_from_items(items)
            chain = stored_chain_messages(prev_id, request) if prev_id else []
            rendered = [m for m in chain] + [m for _, ms in entries for m in ms]
            if req.instructions:
                rendered = [{"role": "system", "content": req.instructions}, *rendered]
            if _count_tokens(rendered) > thr:
                all_entries = entries_from_messages(chain) + entries
                if req.stream:
                    return _compacting_stream(
                        req, request, inner, conv_id, own_items, items, all_entries
                    )
                out, _usage = await compact_entries(
                    request,
                    model,
                    all_entries,
                    req.instructions,
                    tools=_tool_dicts(req),
                )
                compaction = out[-1]
                items = out[:-1] + [
                    summary_message(
                        unseal_compaction(compaction["encrypted_content"]) or ""
                    )
                ]
                prev_id = None
        items = expand_compaction_items(items)
    except (ContextError, ConversationError) as exc:
        return error_json(exc)

    return await _respond(
        req, request, inner, items, prev_id, conv_id, own_items, compaction
    )


def _tool_dicts(req) -> list[dict]:
    return [t if isinstance(t, dict) else _item_dict(t) for t in (req.tools or [])]


def _sse_error(exc: ContextError | ConversationError) -> bytes:
    body = json.loads(error_json(exc).body)
    return (
        "event: error\ndata: " + json.dumps({"type": "error", **body}) + "\n\n"
    ).encode()


def _compacting_stream(
    req, request: Request, inner, conv_id, own_items, items, all_entries
) -> StreamingResponse:
    """Streaming request that needs compaction first: send the SSE headers at once and an SSE
    comment line per summary pass (and every few seconds while a pass runs) so the client's
    connection does not idle for the minutes a long summary takes, then relay the real response.
    Responses has no event for compaction progress (events need a response id that only exists
    once the real response starts), so progress is comments, which SDKs and proxies ignore."""
    import asyncio

    async def gen() -> AsyncIterator[bytes]:
        marks: list[tuple[int, int]] = []
        task = asyncio.ensure_future(
            compact_entries(
                request,
                req.model,
                all_entries,
                req.instructions,
                tools=_tool_dicts(req),
                progress=lambda n, total: marks.append((n, total)),
            )
        )
        yield b": compacting conversation\n\n"
        sent = 0
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=_PROGRESS_SECONDS)
                if len(marks) > sent:
                    sent = len(marks)
                    yield f": compacting pass {marks[-1][0]}\n\n".encode()
                elif not task.done():
                    yield b": compacting\n\n"
            out, _usage = task.result()
        except (ContextError, ConversationError) as exc:
            yield _sse_error(exc)
            return
        except BaseException:
            task.cancel()
            raise
        compaction = out[-1]
        new_items = out[:-1] + [
            summary_message(unseal_compaction(compaction["encrypted_content"]) or "")
        ]
        resp = await _respond(
            req,
            request,
            inner,
            expand_compaction_items(new_items),
            None,
            conv_id,
            own_items,
            compaction,
        )
        if isinstance(resp, StreamingResponse):
            async for chunk in resp.body_iterator:
                yield chunk.encode() if isinstance(chunk, str) else bytes(chunk)
        else:
            body = getattr(resp, "body", b"{}")
            yield b"event: error\ndata: " + (body or b"{}") + b"\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


_PROGRESS_SECONDS = 5.0


async def _respond(req, request, inner, items, prev_id, conv_id, own_items, compaction):
    from .routers.responses import ResponseInputText

    new_req = req.model_copy(
        update={
            "input": [ResponseInputText(**i) for i in items],
            "conversation": None,
            "context_management": None,
            "previous_response_id": prev_id,
        }
    )
    resp = await inner(new_req, request)

    if isinstance(resp, StreamingResponse):
        headers = {
            k: v for k, v in resp.headers.items() if k.lower() != "content-length"
        }
        return StreamingResponse(
            _stream(resp.body_iterator, conv_id, own_items, compaction),
            media_type=resp.media_type,
            headers=headers,
            status_code=resp.status_code,
        )
    if isinstance(resp, JSONResponse) and resp.status_code == 200:
        try:
            body = json.loads(resp.body)
        except ValueError:
            return resp
        if isinstance(body, dict) and body.get("object") == "response":
            _finish(body, conv_id, own_items, compaction)
            if conv_id:
                _append(conv_id, own_items, body)
            _restore_store(body, conv_id, compaction)
            return JSONResponse(body)
    return resp


# ---------------------------------------------------------------------------
# POST /v1/responses/compact
# ---------------------------------------------------------------------------


async def compact_endpoint(request: Request, body: dict) -> dict:
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ContextError(
            400, "model is required", "missing_required_parameter", "model"
        )
    instructions = body.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise ContextError(
            400, "instructions must be a string", "invalid_type", "instructions"
        )
    inp = body.get("input")
    prev = body.get("previous_response_id")
    if inp is None and not prev:
        raise ContextError(
            400,
            "input or previous_response_id is required",
            "missing_required_parameter",
            "input",
        )
    entries: list[tuple[dict | None, list[dict]]] = []
    if prev:
        from .routers.responses import _get_stored_response, _owns_stored

        p = _get_stored_response(prev)
        if p is None or not _owns_stored(request, p):
            raise ContextError(
                404,
                f"Response '{str(prev)[:80]}' not found",
                "response_not_found",
                "previous_response_id",
            )
        entries += entries_from_messages(stored_chain_messages(prev, request))
    if isinstance(inp, str):
        items = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": inp}],
            }
        ]
    elif isinstance(inp, list) and all(isinstance(i, dict) for i in inp):
        items = [dict(i) for i in inp]
    elif inp is None:
        items = []
    else:
        raise ContextError(
            400, "input must be a string or an array of items", "invalid_type", "input"
        )
    for it in items:
        it.setdefault("type", "message")
    try:
        entries += entries_from_items(items)
    except Exception as exc:
        raise ContextError(
            400, f"invalid input item: {exc}", "invalid_value", "input"
        ) from None
    if not entries:
        raise ContextError(400, "nothing to compact", "invalid_value", "input")
    tools = body.get("tools")
    if tools is not None and not isinstance(tools, list):
        raise ContextError(400, "tools must be an array", "invalid_type", "tools")
    # ``instructions`` is the conversation's own system prompt (as in POST /responses; Codex sends
    # its base instructions here), so it leads the summary request and shares the prefix cache.
    out, usage = await compact_entries(
        request, model, entries, instructions, tools=tools
    )
    return {
        "id": f"resp_{secrets.token_hex(12)}",
        "object": "response.compaction",
        "created_at": int(time.time()),
        "output": out,
        "usage": usage,
    }
