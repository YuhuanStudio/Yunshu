"""Anthropic Messages: server-side tools (``web_search_*``, ``web_fetch_*``) and the MCP connector
(``mcp_servers`` / ``mcp_toolset``), executed inside the generation loop.

The wrapper sits in front of the normal Messages handler (``inner``). For a request that declares
server tools it repeats: generate (the server tools are shown to the model as ordinary functions) ->
the model calls one -> run it here (async, off the MLX thread) -> append the result -> generate
again, until the model answers without a server tool call. The client sees one message whose
content interleaves the spec's blocks:

- ``server_tool_use`` + ``web_search_tool_result`` / ``web_fetch_tool_result``
- ``mcp_tool_use`` + ``mcp_tool_result``
- text blocks with ``citations`` (``web_search_result_location``) for the ``[n]`` markers the model
  wrote next to results it used.

Every round re-renders the same conversation plus the new turn, so the prefix cache serves the shared
prefix and a continuation only prefills the new tokens. Usage is summed over the rounds and carries
``server_tool_use.web_search_requests`` / ``web_fetch_requests``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from fastapi.responses import JSONResponse, StreamingResponse

from yunshu_engine import settings

from .mcp_connector import McpError
from .runtime import (
    WEB_FETCH_DESC,
    WEB_FETCH_SCHEMA,
    WEB_SEARCH_DESC,
    WEB_SEARCH_SCHEMA,
    ServerToolDef,
    ServerToolRuntime,
    ToolOutcome,
    decode_result,
    encode_result,
    format_search_text,
    mcp_tool_to_def,
    run_all,
    safe_fname,
)
from .search import SearchResult

logger = logging.getLogger(__name__)

Inner = Callable[[Any, Any], Awaitable[Any]]

_MARK = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")


def _is_web(t: dict, kind: str) -> bool:
    return str(t.get("type") or "").startswith(kind + "_")


def has_server_tools(req) -> bool:
    for t in req.tools or []:
        d = t.model_dump() if hasattr(t, "model_dump") else dict(t)
        if (
            _is_web(d, "web_search")
            or _is_web(d, "web_fetch")
            or d.get("type") == "mcp_toolset"
        ):
            return True
    return bool(getattr(req, "mcp_servers", None))


def _err(status: int, etype: str, msg: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"type": "error", "error": {"type": etype, "message": msg}},
    )


class _Setup:
    def __init__(self):
        self.defs: list[ServerToolDef] = []
        self.client_tools: list[dict] = []
        self.rt: ServerToolRuntime | None = None


async def _setup(req) -> _Setup | JSONResponse:
    """Split the declared tools into server tools (run here) and client tools (passed through)."""
    s = _Setup()
    tools = [
        t.model_dump(exclude_none=True) if hasattr(t, "model_dump") else dict(t)
        for t in (req.tools or [])
    ]
    toolsets: dict[str, dict] = {}
    for t in tools:
        ty = str(t.get("type") or "")
        if _is_web(t, "web_search"):
            s.defs.append(
                ServerToolDef(
                    "web_search",
                    "web_search",
                    WEB_SEARCH_DESC,
                    WEB_SEARCH_SCHEMA,
                    spec=t,
                )
            )
        elif _is_web(t, "web_fetch"):
            s.defs.append(
                ServerToolDef(
                    "web_fetch", "web_fetch", WEB_FETCH_DESC, WEB_FETCH_SCHEMA, spec=t
                )
            )
        elif ty == "mcp_toolset":
            toolsets[t.get("mcp_server_name", "")] = t
        else:
            s.client_tools.append(t)
    servers = list(getattr(req, "mcp_servers", None) or [])
    if servers and not settings.get("YUNSHU_MCP_CONNECTOR"):
        return _err(
            400,
            "invalid_request_error",
            "The MCP connector is disabled (YUNSHU_MCP_CONNECTOR=0).",
        )
    s.rt = ServerToolRuntime(s.defs)
    for srv in servers:
        name = srv.get("name") or ""
        url = srv.get("url") or ""
        if not name or not url:
            await s.rt.aclose()
            return _err(
                400,
                "invalid_request_error",
                "mcp_servers: each server needs a name and a url.",
            )
        try:
            conn = await s.rt.mcp_connect(
                name,
                url,
                authorization=srv.get("authorization_token"),
                headers=srv.get("headers"),
            )
            tools_ = await conn.list_tools()
        except McpError as e:
            await s.rt.aclose()
            return _err(
                400,
                "invalid_request_error",
                f"Unable to connect to MCP server '{name}': {e.message}",
            )
        # legacy tool_configuration on the server, or the newer mcp_toolset entry
        tc = srv.get("tool_configuration") or {}
        ts = toolsets.get(name) or {}
        enabled = tc.get("enabled", True)
        allowed = tc.get("allowed_tools")
        default_on = (ts.get("default_config") or {}).get("enabled", True)
        configs = ts.get("configs") or {}
        for t in tools_:
            on = enabled and default_on
            if allowed is not None:
                on = on and t.name in allowed
            if t.name in configs and "enabled" in configs[t.name]:
                on = bool(configs[t.name]["enabled"])
            if not on:
                continue
            fname = safe_fname(f"{name}__{t.name}")
            d = mcp_tool_to_def(name, t, fname)
            s.defs.append(d)
            s.rt.defs[fname] = d
    return s


# ── history: turn the spec's server-tool blocks back into plain tool_use / tool_result turns ──
def _result_text(block: dict) -> str:
    c = block.get("content")
    ty = block.get("type")
    if ty == "web_search_tool_result":
        if isinstance(c, dict):
            return f"Error: web search failed ({c.get('error_code', 'unavailable')})"
        res = []
        for x in c or []:
            d = decode_result(x.get("encrypted_content", "")) or {
                "title": x.get("title", ""),
                "url": x.get("url", ""),
                "snippet": "",
                "page_age": x.get("page_age"),
            }
            res.append(
                SearchResult(d["title"], d["url"], d["snippet"], d.get("page_age"))
            )
        return format_search_text("(earlier search)", res)
    if ty == "web_fetch_tool_result":
        if isinstance(c, dict) and c.get("type") == "web_fetch_tool_result_error":
            return f"Error: could not fetch ({c.get('error_code')})"
        doc = (c or {}).get("content") or {}
        src = doc.get("source") or {}
        return f"Fetched {(c or {}).get('url', '')}\n\n{src.get('data', '')}"
    parts = []
    for x in c or []:
        parts.append(x.get("text", "") if isinstance(x, dict) else str(x))
    return "\n".join(parts) if not isinstance(c, str) else c


def normalize_history(messages: list) -> list[dict]:
    """Assistant turns that carry server_tool_use / mcp_tool_use plus their results become the
    plain sequence the model saw: assistant(text, tool_use) -> user(tool_result) -> assistant(rest)."""
    out: list[dict] = []
    for m in messages:
        d = m.model_dump() if hasattr(m, "model_dump") else dict(m)
        c = d.get("content")
        if (
            d.get("role") != "assistant"
            or not isinstance(c, list)
            or not any(
                isinstance(b, dict)
                and b.get("type") in ("server_tool_use", "mcp_tool_use")
                for b in c
            )
        ):
            out.append(d)
            continue
        cur: list[dict] = []
        pending: list[dict] = []

        def flush():
            nonlocal cur, pending
            if cur:
                out.append({"role": "assistant", "content": cur})
            if pending:
                out.append({"role": "user", "content": pending})
            cur, pending = [], []

        for b in c:
            ty = b.get("type") if isinstance(b, dict) else None
            if ty in ("server_tool_use", "mcp_tool_use"):
                if pending:
                    flush()
                cur.append(
                    {
                        "type": "tool_use",
                        "id": b.get("id"),
                        "name": b.get("name")
                        if ty == "server_tool_use"
                        else safe_fname(f"{b.get('server_name')}__{b.get('name')}"),
                        "input": b.get("input", {}),
                    }
                )
            elif ty in (
                "web_search_tool_result",
                "web_fetch_tool_result",
                "mcp_tool_result",
            ):
                pending.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": b.get("tool_use_id"),
                        "content": _result_text(b),
                        "is_error": bool(b.get("is_error")),
                    }
                )
            else:
                if pending:
                    flush()
                cur.append(b)
        flush()
    return out


# ── SSE plumbing ──────────────────────────────────────────────────────────────
def _sse(name: str, data: dict) -> bytes:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


async def _parse_sse(body) -> AsyncIterator[tuple[str, Any]]:
    buf = ""
    async for chunk in body:
        buf += (
            chunk.decode("utf-8", "replace")
            if isinstance(chunk, bytes | bytearray)
            else chunk
        )
        buf = buf.replace("\r\n", "\n")
        while "\n\n" in buf:
            block, buf = buf.split("\n\n", 1)
            name, data = None, []
            for ln in block.split("\n"):
                if ln.startswith("event:"):
                    name = ln[6:].strip()
                elif ln.startswith("data:"):
                    data.append(ln[5:].strip())
            if not data:
                yield "__comment__", block
                continue
            try:
                obj = json.loads("\n".join(data))
            except ValueError:
                continue
            yield name or obj.get("type", ""), obj


def _new_id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex[:24]


class _Usage:
    def __init__(self):
        self.d = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

    def add(self, u: dict | None):
        for k in self.d:
            v = (u or {}).get(k)
            if isinstance(v, int):
                self.d[k] += v


def _sources_for(text: str, sources: list[SearchResult]) -> list[dict]:
    seen, cites = set(), []
    for m in _MARK.finditer(text):
        for n in re.split(r"\s*,\s*", m.group(1)):
            i = int(n)
            if i in seen or not (1 <= i <= len(sources)):
                continue
            seen.add(i)
            r = sources[i - 1]
            cites.append(
                {
                    "type": "web_search_result_location",
                    "url": r.url,
                    "title": r.title,
                    "encrypted_index": encode_result(r),
                    "cited_text": r.snippet[:150],
                }
            )
    return cites


def _render_result_block(d: ServerToolDef, tool_id: str, out: ToolOutcome) -> dict:
    if d.kind == "web_search":
        if out.is_error:
            return {
                "type": "web_search_tool_result",
                "tool_use_id": tool_id,
                "content": {
                    "type": "web_search_tool_result_error",
                    "error_code": out.error_code or "unavailable",
                },
            }
        return {
            "type": "web_search_tool_result",
            "tool_use_id": tool_id,
            "content": [
                {
                    "type": "web_search_result",
                    "url": r.url,
                    "title": r.title,
                    "encrypted_content": encode_result(r),
                    "page_age": r.page_age,
                }
                for r in out.results
            ],
        }
    if d.kind == "web_fetch":
        if out.is_error:
            return {
                "type": "web_fetch_tool_result",
                "tool_use_id": tool_id,
                "content": {
                    "type": "web_fetch_tool_result_error",
                    "error_code": out.error_code or "unavailable",
                },
            }
        f = out.fetched
        return {
            "type": "web_fetch_tool_result",
            "tool_use_id": tool_id,
            "content": {
                "type": "web_fetch_result",
                "url": f.url,
                "retrieved_at": f.retrieved_at,
                "content": {
                    "type": "document",
                    "title": f.title or f.url,
                    "citations": {"enabled": False},
                    "source": {
                        "type": "text",
                        "media_type": f.media_type,
                        "data": f.text,
                    },
                },
            },
        }
    content = [c for c in out.mcp_content if c.get("type") == "text"] or [
        {"type": "text", "text": out.text}
    ]
    return {
        "type": "mcp_tool_result",
        "tool_use_id": tool_id,
        "is_error": out.is_error,
        "content": content,
    }


async def run_stream(req, request, inner: Inner, setup: _Setup) -> AsyncIterator[bytes]:
    """The whole server-tool loop as one Anthropic SSE stream."""
    rt = setup.rt
    assert rt is not None
    max_iter = int(settings.get("YUNSHU_SERVER_TOOL_MAX_ITERATIONS"))
    fn_tools = [
        {"name": d.fname, "description": d.description, "input_schema": d.schema}
        for d in setup.defs
    ]
    history = normalize_history(req.messages)
    total = _Usage()
    out_index = 0
    started = False
    stop_reason = "end_turn"
    sources: list[SearchResult] = []
    stats = {"rounds": 0, "tools": []}
    msg_id = _new_id("msg_")
    try:
        for rnd in range(max_iter):
            stats["rounds"] += 1
            inner_req = _inner_request(req, history, setup.client_tools + fn_tools)
            resp = await inner(inner_req, request)
            if not isinstance(resp, StreamingResponse):
                body = resp.body.decode() if hasattr(resp, "body") else "{}"
                try:
                    err = json.loads(body).get("error", {})
                except ValueError:
                    err = {}
                if not started:
                    # nothing streamed yet: surface the engine's own error response
                    yield json.dumps(
                        {"__error__": resp.status_code, "body": body}
                    ).encode()
                    return
                yield _sse(
                    "error",
                    {
                        "type": "error",
                        "error": {
                            "type": err.get("type", "api_error"),
                            "message": err.get("message", "generation failed"),
                        },
                    },
                )
                return
            blocks_map: dict[int, int] = {}
            text_acc: dict[int, str] = {}
            text_pending: dict[int, str] = {}
            tool_blocks: dict[int, dict] = {}
            model_turn: list[
                dict
            ] = []  # blocks as the model produced them (for the next round)
            turn_stop = "end_turn"
            round_usage: dict = {}
            async for name, ev in _parse_sse(resp.body_iterator):
                if name == "__comment__":
                    yield (ev + "\n\n").encode()
                    continue
                t = ev.get("type")
                if t == "message_start":
                    m = ev.get("message", {})
                    round_usage.update(m.get("usage") or {})
                    if not started:
                        started = True
                        msg_id = m.get("id", msg_id)
                        yield _sse(
                            "message_start",
                            {
                                "type": "message_start",
                                "message": {
                                    **m,
                                    "content": [],
                                    "stop_reason": None,
                                    "stop_sequence": None,
                                    "usage": {**m.get("usage", {}), "output_tokens": 1},
                                },
                            },
                        )
                elif t == "content_block_start":
                    cb = ev["content_block"]
                    i = ev["index"]
                    if cb.get("type") == "text":
                        # A text block is opened lazily, on its first non-blank text: the chat
                        # template's "\n\n" after </think> must not become its own (or a leading)
                        # part of the reply, the API never emits blank text blocks.
                        text_acc[i] = ""
                        text_pending[i] = ""
                        continue
                    blocks_map[i] = out_index
                    if cb.get("type") == "tool_use" and cb.get("name") in rt.defs:
                        d = rt.defs[cb["name"]]
                        sid = _new_id("mcptoolu_" if d.kind == "mcp" else "srvtoolu_")
                        if d.kind == "mcp":
                            outb = {
                                "type": "mcp_tool_use",
                                "id": sid,
                                "name": d.tool_name,
                                "server_name": d.server_name,
                                "input": {},
                            }
                        else:
                            outb = {
                                "type": "server_tool_use",
                                "id": sid,
                                "name": d.kind,
                                "input": {},
                            }
                        tool_blocks[i] = {
                            "def": d,
                            "id": sid,
                            "orig_id": cb.get("id"),
                            "json": "",
                            "server": True,
                        }
                        yield _sse(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": out_index,
                                "content_block": outb,
                            },
                        )
                    else:
                        if cb.get("type") == "tool_use":
                            tool_blocks[i] = {"server": False, "json": "", "block": cb}
                        yield _sse(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": out_index,
                                "content_block": cb,
                            },
                        )
                    out_index += 1
                elif t == "content_block_delta":
                    i = ev["index"]
                    dl = ev["delta"]
                    if dl.get("type") == "text_delta" and i in text_acc:
                        text_acc[i] += dl.get("text", "")
                        if i not in blocks_map:
                            pend = text_pending.get(i, "") + dl.get("text", "")
                            if not pend.strip():
                                text_pending[i] = pend
                                continue
                            text_pending.pop(i, None)
                            blocks_map[i] = out_index
                            out_index += 1
                            yield _sse(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": blocks_map[i],
                                    "content_block": {"type": "text", "text": ""},
                                },
                            )
                            dl = {"type": "text_delta", "text": pend.lstrip()}
                    if dl.get("type") == "input_json_delta" and i in tool_blocks:
                        tool_blocks[i]["json"] += dl.get("partial_json", "")
                    if i not in blocks_map:
                        continue
                    yield _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": blocks_map[i],
                            "delta": dl,
                        },
                    )
                elif t == "content_block_stop":
                    i = ev["index"]
                    if i in text_acc and text_acc[i].strip():
                        model_turn.append({"type": "text", "text": text_acc[i].strip()})
                    if i not in blocks_map:
                        continue  # a blank text block that was never opened
                    if i in text_acc and text_acc[i] and sources:
                        for c in _sources_for(text_acc[i], sources):
                            yield _sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": blocks_map[i],
                                    "delta": {"type": "citations_delta", "citation": c},
                                },
                            )
                    yield _sse(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": blocks_map[i]},
                    )
                elif t == "message_delta":
                    turn_stop = ev.get("delta", {}).get("stop_reason") or turn_stop
                    round_usage.update(ev.get("usage") or {})
                elif t == "error":
                    yield _sse("error", ev)
                    return
            total.add(round_usage)
            # blocks of this model turn -> assistant message for history; run server calls
            calls = []
            for tb in tool_blocks.values():
                try:
                    args = json.loads(tb["json"]) if tb["json"] else {}
                except ValueError:
                    args = {}
                tb["args"] = args
                if tb["server"]:
                    calls.append(tb)
            client_calls = [tb for tb in tool_blocks.values() if not tb["server"]]
            if not calls:
                stop_reason = turn_stop
                break
            outcomes = await run_all(
                rt, [(tb["def"].fname, tb["args"]) for tb in calls]
            )
            assistant_blocks = [b for b in model_turn]
            result_msgs = []
            for tb, oc in zip(calls, outcomes, strict=True):
                d = tb["def"]
                if d.kind == "web_search" and not oc.is_error:
                    off = len(sources)
                    sources.extend(oc.results)
                    oc.text = format_search_text(
                        oc.query or "", oc.results, start=off + 1
                    )
                stats["tools"].append(
                    {
                        "tool": d.fname,
                        "kind": d.kind,
                        "ok": not oc.is_error,
                        "error_code": oc.error_code,
                        "ms": round(oc.elapsed * 1000, 1),
                        "query": oc.query,
                        "provider": oc.provider,
                        **({"hint": oc.hint} if oc.hint else {}),
                    }
                )
                block = _render_result_block(d, tb["id"], oc)
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": out_index,
                        "content_block": block,
                    },
                )
                yield _sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": out_index},
                )
                out_index += 1
                assistant_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tb["id"],
                        "name": d.fname,
                        "input": tb["args"],
                    }
                )
                result_msgs.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tb["id"],
                        "content": oc.text,
                        "is_error": oc.is_error,
                    }
                )
            for tb in client_calls:
                assistant_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tb["block"].get("id"),
                        "name": tb["block"].get("name"),
                        "input": tb["args"],
                    }
                )
            history = history + [{"role": "assistant", "content": assistant_blocks}]
            if client_calls:
                stop_reason = "tool_use"
                break
            history.append({"role": "user", "content": result_msgs})
            if rnd == max_iter - 1:
                stop_reason = "pause_turn"
        usage = dict(total.d)
        st = {}
        if rt.counts["web_search"]:
            st["web_search_requests"] = rt.counts["web_search"]
        if rt.counts["web_fetch"]:
            st["web_fetch_requests"] = rt.counts["web_fetch"]
        if st:
            usage["server_tool_use"] = st
        yield _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": usage,
                "x_yunshu": {"server_tools": stats},
            },
        )
        yield _sse("message_stop", {"type": "message_stop"})
    finally:
        with contextlib.suppress(Exception):
            await rt.aclose()


def _inner_request(req, history: list[dict], tools: list[dict]):
    """A fresh request for one generation round: the running history, function-style tools."""
    from ..routers.anthropic import AnthropicMessage, AnthropicTool

    return req.model_copy(
        update={
            "messages": [
                AnthropicMessage(role=m["role"], content=m["content"]) for m in history
            ],
            "tools": [AnthropicTool(**t) for t in tools] or None,
            "stream": True,
            "mcp_servers": None,
        }
    )


async def assemble_message(events: AsyncIterator[bytes]) -> dict:
    """Fold the SSE stream into a non-streaming Message (used when the client did not stream)."""
    msg: dict = {}
    blocks: dict[int, dict] = {}
    jbuf: dict[int, str] = {}
    usage: dict = {}
    extra: dict = {}
    buf = b""
    async for chunk in events:
        buf += chunk
    async for name, ev in _parse_sse(_aiter([buf])):
        if name == "__comment__":
            continue
        t = ev.get("type")
        if t == "message_start":
            msg = dict(ev["message"])
        elif t == "content_block_start":
            blocks[ev["index"]] = dict(ev["content_block"])
            jbuf[ev["index"]] = ""
        elif t == "content_block_delta":
            i, dl = ev["index"], ev["delta"]
            b = blocks[i]
            dt = dl.get("type")
            if dt == "text_delta":
                b["text"] = b.get("text", "") + dl["text"]
            elif dt == "thinking_delta":
                b["thinking"] = b.get("thinking", "") + dl["thinking"]
            elif dt == "signature_delta":
                b["signature"] = dl["signature"]
            elif dt == "input_json_delta":
                jbuf[i] += dl["partial_json"]
            elif dt == "citations_delta":
                b.setdefault("citations", []).append(dl["citation"])
        elif t == "content_block_stop":
            i = ev["index"]
            if jbuf.get(i):
                with contextlib.suppress(ValueError):
                    blocks[i]["input"] = json.loads(jbuf[i])
        elif t == "message_delta":
            msg["stop_reason"] = ev["delta"].get("stop_reason")
            msg["stop_sequence"] = ev["delta"].get("stop_sequence")
            usage = ev.get("usage", {})
            extra = ev.get("x_yunshu", {})
    msg["content"] = [blocks[i] for i in sorted(blocks)]
    msg["usage"] = {**msg.get("usage", {}), **usage}
    if extra:
        msg["x_yunshu"] = extra
    return msg


async def _aiter(items):
    for x in items:
        yield x


async def create_with_server_tools(req, request, inner: Inner):
    """Entry point from the Messages handler; returns a Response."""
    setup = await _setup(req)
    if isinstance(setup, JSONResponse):
        return setup
    gen = run_stream(req, request, inner, setup)
    if req.stream:
        return StreamingResponse(
            gen, media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )
    first = await gen.__anext__()
    if first.startswith(b'{"__error__"'):
        info = json.loads(first)
        return JSONResponse(
            status_code=info["__error__"], content=json.loads(info["body"] or "{}")
        )

    async def rest():
        yield first
        async for c in gen:
            yield c

    msg = await assemble_message(rest())
    return JSONResponse(content=msg)


__all__ = [
    "create_with_server_tools",
    "has_server_tools",
    "normalize_history",
    "assemble_message",
]
