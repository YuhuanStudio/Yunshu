"""OpenAI Responses: server-side tools (``web_search*``) and the remote MCP tool (``{"type":"mcp"}``),
executed inside the generation loop.

Same design as :mod:`anthropic_loop`: the wrapper sits in front of the normal ``create_response``
handler (``inner``). For a request that declares server tools it repeats: generate (server tools
are shown to the model as ordinary function tools, inner call is always ``stream=True``) -> parse
the inner Responses SSE -> when the model calls a server tool, run it here and emit the spec's
items -> append call + result to the history -> generate again. The client gets ONE response
(stream or JSON) whose ``output`` interleaves:

- ``web_search_call`` (``action.query``, optional ``sources``) and a ``message`` whose text
  carries ``url_citation`` annotations for the ``[n]`` markers the model wrote,
- ``mcp_list_tools``, ``mcp_call`` and ``mcp_approval_request`` items,
- everything else the model produced (reasoning, messages, the client's own function calls).

Inner rounds are called with ``store=False``; the merged response is stored here (under the
outward id) so ``previous_response_id`` chains see it.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from fastapi.responses import JSONResponse, StreamingResponse

from yunshu_engine import settings

from .mcp_connector import McpError
from .runtime import (
    WEB_SEARCH_DESC,
    WEB_SEARCH_SCHEMA,
    ServerToolDef,
    ServerToolRuntime,
    format_search_text,
    mcp_tool_to_def,
    parse_tool_args,
    run_all,
    safe_fname,
)
from .search import SearchResult

logger = logging.getLogger(__name__)

Inner = Callable[[Any, Any], Awaitable[Any]]

_MARK = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")
_SECRET_KEYS = ("authorization", "headers")


def _new_id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex[:24]


def _dump(t: Any) -> dict:
    if hasattr(t, "model_dump"):
        return t.model_dump(exclude_none=True)
    return dict(t)


def _is_web(t: dict) -> bool:
    return str(t.get("type") or "").startswith("web_search")


def has_server_tools_responses(req) -> bool:
    for t in req.tools or []:
        d = _dump(t)
        if _is_web(d) or d.get("type") == "mcp":
            return True
    return False


def function_tools(tools):
    """The tools the engine's chat template can use: plain named functions. Codex ``namespace``
    tools are flattened to their children; freeform custom tools use an internal input function.
    Returns the input object itself when nothing needed changing."""
    if not tools:
        return tools
    from ..routers.responses import ResponseTool

    out: list = []
    changed = False
    for t in tools:
        d = _dump(t)
        ty = d.get("type") or "function"
        if ty == "function" and d.get("name"):
            out.append(t)
        elif ty == "custom":
            from ..custom_tools import custom_function

            changed = True
            out.append(ResponseTool(**custom_function(d)))
        elif ty == "namespace":
            changed = True
            children = [
                ResponseTool(**c) for c in d.get("tools") or [] if isinstance(c, dict)
            ]
            out.extend(function_tools(children) or [])
        else:
            changed = True
    if not changed:
        return tools
    return out or None


# ── input items -> chat messages ──────────────────────────────────────────────
def _json_text(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def input_item_to_messages(item: dict, texts: dict | None = None) -> list[dict]:
    """Chat messages for a Responses item that is not a plain message / function_call(_output):
    a ``web_search_call`` / ``mcp_call`` / custom tool call becomes an assistant tool call plus its
    tool result (when the item carries enough), everything else contributes nothing."""
    ty = item.get("type")
    iid = item.get("id") or item.get("call_id") or ""
    texts = texts or {}

    def pair(name: str, args: str, result: str) -> list[dict]:
        return [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": iid,
                        "type": "function",
                        "function": {"name": name, "arguments": args or "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": iid, "content": result},
        ]

    if ty == "web_search_call":
        action = item.get("action") or {}
        q = action.get("query")
        if not q:
            return []
        res = texts.get(iid)
        if res is None:
            res = f'Web search for "{q}" ({item.get("status") or "completed"}).'
            urls = [
                s.get("url")
                for s in action.get("sources") or []
                if isinstance(s, dict) and s.get("url")
            ]
            if urls:
                res += "\nSources:\n" + "\n".join(urls)
        return pair("web_search", json.dumps({"query": q}, ensure_ascii=False), res)
    if ty == "mcp_call":
        name = item.get("name")
        if not name:
            return []
        fname = safe_fname(f"{item.get('server_label') or ''}__{name}")
        if item.get("error"):
            res = f"Error: {_json_text(item['error'])}"
        else:
            res = _json_text(
                item.get("output") if item.get("output") is not None else ""
            )
        return pair(fname, _json_text(item.get("arguments") or "{}"), res)
    if ty == "custom_tool_call_output":
        return [
            {
                "role": "tool",
                "tool_call_id": item.get("call_id") or "",
                "content": _json_text(item.get("output") or ""),
            }
        ]
    if ty == "custom_tool_call":
        return [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": item.get("call_id") or iid,
                        "type": "function",
                        "function": {
                            "name": item.get("name") or "",
                            "arguments": json.dumps(
                                {"input": item.get("input") or ""}, ensure_ascii=False
                            ),
                        },
                    }
                ],
            }
        ]
    return []


# ── errors ────────────────────────────────────────────────────────────────────
def _err(status: int, msg: str, code: str | None = None, param: str | None = None):
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": msg,
                "type": "invalid_request_error",
                "param": param,
                "code": code,
            }
        },
    )


# ── setup ─────────────────────────────────────────────────────────────────────
def _tool_names(v: Any) -> list[str] | None:
    if v is None:
        return None
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, dict):
        tn = v.get("tool_names")
        return [str(x) for x in tn] if isinstance(tn, list) else None
    return None


def needs_approval(spec: dict, tool: str) -> bool:
    ra = spec.get("require_approval", "always")
    if ra == "never":
        return False
    if isinstance(ra, dict):
        return tool not in (_tool_names(ra.get("never")) or [])
    return True


class _Setup:
    def __init__(self):
        self.defs: list[ServerToolDef] = []
        self.client_tools: list[dict] = []
        self.rt: ServerToolRuntime | None = None
        self.mcp_specs: dict[str, dict] = {}
        self.mcp_lists: dict[str, list[dict]] = {}


def _input_items(req) -> list[dict]:
    if not isinstance(req.input, list):
        return []
    return [_dump(i) for i in req.input]


def _stored_chain(req, request) -> list[dict]:
    """Stored payloads along previous_response_id (newest first, ownership-gated)."""
    from ..routers.responses import _get_stored_response, _owns_stored

    out, seen, pid = [], set(), req.previous_response_id
    for _ in range(16):
        if not pid or pid in seen:
            break
        seen.add(pid)
        p = _get_stored_response(pid)
        if not p or not _owns_stored(request, p):
            break
        out.append(p)
        pid = p.get("previous_response_id")
    return out


async def _setup(req) -> _Setup | JSONResponse:
    s = _Setup()
    for t in (_dump(x) for x in (req.tools or [])):
        ty = t.get("type") or "function"
        if _is_web(t):
            filt = t.get("filters") or {}
            spec: dict = {}
            if filt.get("allowed_domains"):
                spec["allowed_domains"] = list(filt["allowed_domains"])
            if t.get("user_location"):
                spec["user_location"] = t["user_location"]
            if t.get("search_context_size"):
                spec["search_context_size"] = t["search_context_size"]
            s.defs.append(
                ServerToolDef(
                    "web_search",
                    "web_search",
                    WEB_SEARCH_DESC,
                    WEB_SEARCH_SCHEMA,
                    spec=spec,
                )
            )
        elif ty == "mcp":
            if t.get("connector_id") and not t.get("server_url"):
                return _err(
                    400,
                    "OpenAI-hosted connectors (connector_id) are not supported; "
                    "pass the MCP server's URL as server_url.",
                    "unsupported_connector",
                    "tools",
                )
            label = t.get("server_label") or ""
            if not label or not t.get("server_url"):
                return _err(
                    400,
                    "mcp tool: server_label and server_url are required.",
                    "missing_required_parameter",
                    "tools",
                )
            s.mcp_specs[label] = t
        else:
            s.client_tools.append(t)
    if s.mcp_specs and not settings.get("YUNSHU_MCP_CONNECTOR"):
        return _err(400, "The MCP tool is disabled (YUNSHU_MCP_CONNECTOR=0).")
    s.rt = ServerToolRuntime(s.defs)
    for label, t in s.mcp_specs.items():
        try:
            conn = await s.rt.mcp_connect(
                label,
                t["server_url"],
                authorization=t.get("authorization"),
                headers=t.get("headers"),
            )
            tools_ = await conn.list_tools()
        except McpError as e:
            await s.rt.aclose()
            return JSONResponse(
                status_code=424,
                content={
                    "error": {
                        "message": f"Error retrieving tool list from MCP server: '{label}'. "
                        f"{e.message}",
                        "type": "invalid_request_error",
                        "param": "tools",
                        "code": "external_connector_error",
                    }
                },
            )
        allowed = _tool_names(t.get("allowed_tools"))
        listed = []
        for mt in tools_:
            if allowed is not None and mt.name not in allowed:
                continue
            fname = safe_fname(f"{label}__{mt.name}")
            d = mcp_tool_to_def(label, mt, fname)
            d.spec = t
            s.defs.append(d)
            s.rt.defs[fname] = d
            listed.append(
                {
                    "name": mt.name,
                    "description": mt.description or None,
                    "input_schema": mt.input_schema,
                    "annotations": mt.annotations,
                }
            )
        s.mcp_lists[label] = listed
    return s


# ── SSE plumbing ──────────────────────────────────────────────────────────────
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
            if isinstance(obj, dict):
                yield name or obj.get("type", ""), obj


def annotations_for(text: str, sources: list[SearchResult]) -> list[dict]:
    """``url_citation`` annotations for each ``[n]`` marker that names a known source."""
    out = []
    for m in _MARK.finditer(text or ""):
        for n in re.split(r"\s*,\s*", m.group(1)):
            i = int(n)
            if 1 <= i <= len(sources):
                r = sources[i - 1]
                out.append(
                    {
                        "type": "url_citation",
                        "start_index": m.start(),
                        "end_index": m.end(),
                        "url": r.url,
                        "title": r.title,
                    }
                )
    return out


class _Run:
    """Mutable state of one outward response."""

    def __init__(self, req, setup: _Setup):
        self.req = req
        self.setup = setup
        self.id = f"resp-{uuid.uuid4().hex[:24]}"
        self.created_at = int(time.time())
        self.seq = -1
        self.output: list[dict] = []
        self.usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }
        self.sources: list[SearchResult] = []
        self.tool_texts: dict[str, str] = {}
        self.stats: dict = {"rounds": 0, "tools": [], "round_usage": []}
        self.final: dict | None = None
        self.error: tuple[int, dict] | None = None
        self.status = "completed"
        self.incomplete_reason: str | None = None
        self.total_calls = 0

    def ev(self, _t: str, **fields) -> bytes:
        self.seq += 1
        name = _t
        data = {"type": name, **fields, "sequence_number": self.seq}
        return (
            f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()
        )

    def add_usage(self, u: dict | None):
        u = u or {}
        # per round: a continuation should read the shared prefix from the cache
        self.stats["round_usage"].append(
            {
                "input_tokens": u.get("input_tokens"),
                "cached_tokens": (u.get("input_tokens_details") or {}).get(
                    "cached_tokens"
                ),
                "output_tokens": u.get("output_tokens"),
            }
        )
        for k in ("input_tokens", "output_tokens", "total_tokens"):
            if isinstance(u.get(k), int):
                self.usage[k] += u[k]
        for grp, key in (
            ("input_tokens_details", "cached_tokens"),
            ("output_tokens_details", "reasoning_tokens"),
        ):
            v = (u.get(grp) or {}).get(key)
            if isinstance(v, int):
                self.usage[grp][key] += v

    def echo_tools(self) -> list[dict]:
        out = []
        for t in self.req.tools or []:
            d = (
                t.model_dump(exclude_unset=True)
                if hasattr(t, "model_dump")
                else dict(t)
            )
            out.append({k: v for k, v in d.items() if k not in _SECRET_KEYS})
        return out

    def envelope(self, status: str, output: list, terminal: bool) -> dict:
        req = self.req
        r: dict = {
            "id": self.id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": req.model,
            "output": output,
            "instructions": req.instructions,
            "metadata": req.metadata or {},
            "tools": self.echo_tools(),
            "tool_choice": req.tool_choice or "auto",
            "parallel_tool_calls": req.parallel_tool_calls,
            "previous_response_id": req.previous_response_id,
            "store": bool(req.store),
            "error": None,
            "incomplete_details": None,
        }
        if terminal:
            r["usage"] = self.usage
            r["x_yunshu"] = {"server_tools": self.stats}
            if status == "completed":
                r["completed_at"] = int(time.time())
            if status == "incomplete":
                r["incomplete_details"] = {"reason": self.incomplete_reason}
        return r

    def add_item(self, item: dict) -> int:
        self.output.append(item)
        return len(self.output) - 1


def _args_str(v: Any) -> str:
    if isinstance(v, str):
        return v
    return json.dumps(v or {}, ensure_ascii=False)


def _parse_args(v: Any) -> dict:
    return parse_tool_args(v)


def _approval_items(req, request) -> tuple[list[dict], set[str], list[dict]]:
    """(approval decisions, labels whose tool list is already known, executed approval ids).

    A decision is ``{"request": <mcp_approval_request item>, "approve": bool, "reason": str}``."""
    items = _input_items(req)
    chain = _stored_chain(req, request)
    requests: dict[str, dict] = {}
    known: set[str] = set()
    done: set[str] = set()
    for it in items:
        if it.get("type") == "mcp_approval_request" and it.get("id"):
            requests[it["id"]] = it
        elif it.get("type") == "mcp_list_tools" and it.get("server_label"):
            known.add(it["server_label"])
        elif it.get("type") == "mcp_call" and it.get("approval_request_id"):
            done.add(it["approval_request_id"])
    for p in chain:
        for it in p.get("output") or []:
            if it.get("type") == "mcp_approval_request" and it.get("id"):
                requests.setdefault(it["id"], it)
            elif it.get("type") == "mcp_list_tools" and it.get("server_label"):
                known.add(it["server_label"])
            elif it.get("type") == "mcp_call" and it.get("approval_request_id"):
                done.add(it["approval_request_id"])
    decisions = []
    for it in items:
        if it.get("type") != "mcp_approval_response":
            continue
        rid = it.get("approval_request_id")
        if not rid or rid in done or rid not in requests:
            continue
        done.add(rid)
        decisions.append(
            {
                "request": requests[rid],
                "approve": bool(it.get("approve")),
                "reason": it.get("reason"),
            }
        )
    return decisions, known, []


def _inner_request(req, items: list[dict], tools: list[dict], first: bool, defs=()):
    from ..routers.responses import ResponseInputText, ResponseTool

    base = list(req.input) if isinstance(req.input, list) else None
    if base is None:
        base = [ResponseInputText(type="message", role="user", content=req.input)]
    tc = req.tool_choice
    if first and tc in (None, "auto"):
        # an explicit "search the web ..." / "fetch <url>" in the last user turn steers round one
        from .runtime import explicit_tool_request

        last = ""
        for it in reversed(base):
            role = getattr(it, "role", None)
            if role == "user":
                c = getattr(it, "content", None) or ""
                last = (
                    c
                    if isinstance(c, str)
                    else " ".join(p.get("text", "") for p in c if isinstance(p, dict))
                )
                break
        want = explicit_tool_request(last, defs)
        if want:
            tc = {"type": "function", "name": want}
    if not first:
        tc = "auto"
    elif isinstance(tc, dict) and tc.get("type") in (
        "web_search",
        "web_search_preview",
    ):
        tc = {"type": "function", "name": "web_search"}
    elif isinstance(tc, dict) and tc.get("type") == "mcp":
        tc = "auto"
    return req.model_copy(
        update={
            "input": base + [ResponseInputText(**i) for i in items],
            "tools": [ResponseTool(**t) for t in tools] or None,
            "tool_choice": tc if tools else None,
            "stream": True,
            "store": False,
            "background": False,
        }
    )


async def run_stream(req, request, inner: Inner, setup: _Setup, run: _Run):
    rt = setup.rt
    assert rt is not None
    max_iter = int(settings.get("YUNSHU_SERVER_TOOL_MAX_ITERATIONS"))
    cap = req.max_tool_calls
    include_sources = "web_search_call.action.sources" in (req.include or [])
    hist: list[dict] = []
    fn_tools = [
        {"name": d.fname, "description": d.description, "parameters": d.schema}
        for d in setup.defs
    ]
    try:
        yield run.ev(
            "response.created", response=run.envelope("in_progress", [], False)
        )
        yield run.ev(
            "response.in_progress", response=run.envelope("in_progress", [], False)
        )

        # mcp_list_tools for servers whose list the client has not already seen
        decisions, known, _ = _approval_items(req, request)
        for label, listed in setup.mcp_lists.items():
            if label in known:
                continue
            item = {
                "id": _new_id("mcpl_"),
                "type": "mcp_list_tools",
                "server_label": label,
                "tools": [],
            }
            idx = run.add_item(item)
            yield run.ev("response.output_item.added", output_index=idx, item=item)
            yield run.ev(
                "response.mcp_list_tools.in_progress",
                item_id=item["id"],
                output_index=idx,
            )
            item = {**item, "tools": listed}
            run.output[idx] = item
            yield run.ev(
                "response.mcp_list_tools.completed",
                item_id=item["id"],
                output_index=idx,
            )
            yield run.ev("response.output_item.done", output_index=idx, item=item)

        # approvals the client answered: execute (or deny) the calls we paused on
        for dec in decisions:
            r = dec["request"]
            label, tname = r.get("server_label"), r.get("name")
            fname = safe_fname(f"{label}__{tname}")
            call = {
                "id": _new_id("mcp_"),
                "type": "mcp_call",
                "server_label": label,
                "name": tname,
                "arguments": _args_str(r.get("arguments")),
                "output": None,
                "error": None,
                "approval_request_id": r.get("id"),
                "status": "in_progress",
            }
            idx = run.add_item(call)
            yield run.ev("response.output_item.added", output_index=idx, item=call)
            yield run.ev(
                "response.mcp_call.in_progress", item_id=call["id"], output_index=idx
            )
            if dec["approve"] and fname in rt.defs:
                async for b in _finish_mcp(run, rt, idx, call, fname, cap):
                    yield b
            else:
                why = (
                    "The tool call was denied."
                    if not dec["approve"]
                    else f"MCP tool '{tname}' is not available."
                )
                if dec["approve"] is False and dec.get("reason"):
                    why = f"{why} Reason: {dec['reason']}"
                async for b in _fail_mcp(run, idx, call, why):
                    yield b
            hist += _pair(
                call["id"], fname, call["arguments"], run.tool_texts[call["id"]]
            )

        for rnd in range(max_iter):
            run.stats["rounds"] += 1
            capped = cap is not None and run.total_calls >= cap
            tools = setup.client_tools + ([] if capped else fn_tools)
            inner_req = _inner_request(
                req, hist, tools, first=(rnd == 0 and not hist), defs=setup.defs
            )
            with contextlib.suppress(Exception):
                request.state._forced_response_id = run.id
            resp = await inner(inner_req, request)
            if not isinstance(resp, StreamingResponse):
                body = resp.body.decode() if hasattr(resp, "body") else "{}"
                try:
                    err = json.loads(body)
                except ValueError:
                    err = {}
                run.error = (getattr(resp, "status_code", 500), err)
                e = (err or {}).get("error") or {}
                run.status = "failed"
                r = run.envelope("failed", run.output, True)
                r["error"] = {
                    "code": e.get("code") or "server_error",
                    "message": e.get("message") or "generation failed",
                }
                run.final = r
                yield run.ev("response.failed", response=r)
                yield b"data: [DONE]\n\n"
                return

            omap: dict[int, int] = {}
            swallowed: dict[int, dict] = {}
            ann_stash: dict[tuple[int, int], list[dict]] = {}
            round_text = ""
            client_call = False
            round_status = "completed"
            failed_ev = None
            body_it = resp.body_iterator
            try:
                async for name, ev in _parse_sse(body_it):
                    if name == "__comment__":
                        yield (ev + "\n\n").encode()
                        continue
                    t = ev.get("type") or name
                    if t in ("response.created", "response.in_progress"):
                        continue
                    if t in (
                        "response.completed",
                        "response.incomplete",
                        "response.failed",
                    ):
                        r = ev.get("response") or {}
                        run.add_usage(r.get("usage"))
                        if t == "response.incomplete":
                            round_status = "incomplete"
                            run.incomplete_reason = (
                                r.get("incomplete_details") or {}
                            ).get("reason") or "max_output_tokens"
                        elif t == "response.failed":
                            round_status = "failed"
                            failed_ev = r.get("error") or {}
                        continue
                    if t == "error":
                        round_status = "failed"
                        failed_ev = ev.get("error") or {"message": ev.get("message")}
                        continue
                    i = ev.get("output_index")
                    if t == "response.output_item.added":
                        it = ev.get("item") or {}
                        if (
                            it.get("type") == "function_call"
                            and it.get("name") in rt.defs
                        ):
                            swallowed[i] = {
                                "def": rt.defs[it["name"]],
                                "args": it.get("arguments") or "",
                            }
                            continue
                        omap[i] = run.add_item(it)
                        if it.get("type") == "function_call":
                            client_call = True
                    if i is not None and i in swallowed:
                        sw = swallowed[i]
                        if t == "response.function_call_arguments.delta":
                            sw["args"] += ev.get("delta", "")
                        elif t == "response.function_call_arguments.done":
                            sw["args"] = ev.get("arguments", sw["args"])
                        elif t == "response.output_item.done":
                            sw["args"] = (ev.get("item") or {}).get("arguments") or sw[
                                "args"
                            ]
                        continue
                    if i is not None:
                        if i not in omap:
                            omap[i] = len(run.output)
                            run.output.append({})
                        ev = {**ev, "output_index": omap[i]}
                    oi = ev.get("output_index")
                    if t == "response.output_text.done" and run.sources:
                        anns = annotations_for(ev.get("text", ""), run.sources)
                        ci = ev.get("content_index", 0)
                        ann_stash[(oi, ci)] = anns
                        for ai, a in enumerate(anns):
                            yield run.ev(
                                "response.output_text.annotation.added",
                                item_id=ev.get("item_id"),
                                output_index=oi,
                                content_index=ci,
                                annotation_index=ai,
                                annotation=a,
                            )
                    elif t == "response.content_part.done" and run.sources:
                        part = dict(ev.get("part") or {})
                        anns = ann_stash.get(
                            (oi, ev.get("content_index", 0))
                        ) or annotations_for(part.get("text", ""), run.sources)
                        part["annotations"] = anns
                        ev = {**ev, "part": part}
                    elif t == "response.output_item.done":
                        it = dict(ev.get("item") or {})
                        if it.get("type") == "message":
                            parts = []
                            for ci, p in enumerate(it.get("content") or []):
                                if p.get("type") == "output_text":
                                    round_text += p.get("text", "")
                                    if run.sources:
                                        p = {
                                            **p,
                                            "annotations": ann_stash.get((oi, ci))
                                            or annotations_for(
                                                p.get("text", ""), run.sources
                                            ),
                                        }
                                parts.append(p)
                            it["content"] = parts
                        ev = {**ev, "item": it}
                        run.output[oi] = it
                    ev = {k: v for k, v in ev.items() if k != "sequence_number"}
                    ev.pop("type", None)
                    yield run.ev(t, **ev)
            finally:
                with contextlib.suppress(Exception):
                    aclose = getattr(body_it, "aclose", None)
                    if aclose:
                        await aclose()

            if round_status == "failed":
                run.status = "failed"
                r = run.envelope("failed", run.output, True)
                r["error"] = {
                    "code": (failed_ev or {}).get("code") or "server_error",
                    "message": (failed_ev or {}).get("message") or "generation failed",
                }
                run.final = r
                yield run.ev("response.failed", response=r)
                yield b"data: [DONE]\n\n"
                return
            if round_status == "incomplete":
                run.status = "incomplete"
                break
            calls = [swallowed[k] for k in sorted(swallowed)]
            has_client_call = client_call
            if not calls:
                break

            # ── announce, execute, finish this round's server calls ──────────────────────
            if round_text.strip():
                hist.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": round_text}],
                    }
                )
            paused = False
            running: list[dict] = []
            for c in calls:
                d: ServerToolDef = c["def"]
                args = _parse_args(c["args"])
                c["args_obj"] = args
                if d.kind == "web_search":
                    item = {
                        "id": _new_id("ws_"),
                        "type": "web_search_call",
                        "status": "in_progress",
                        "action": {
                            "type": "search",
                            "query": str(args.get("query", "")),
                        },
                    }
                    idx = run.add_item(item)
                    c.update(item=item, idx=idx)
                    yield run.ev(
                        "response.output_item.added", output_index=idx, item=item
                    )
                    yield run.ev(
                        "response.web_search_call.in_progress",
                        item_id=item["id"],
                        output_index=idx,
                    )
                    yield run.ev(
                        "response.web_search_call.searching",
                        item_id=item["id"],
                        output_index=idx,
                    )
                    running.append(c)
                elif needs_approval(d.spec, d.tool_name or ""):
                    item = {
                        "id": _new_id("mcpr_"),
                        "type": "mcp_approval_request",
                        "server_label": d.server_name,
                        "name": d.tool_name,
                        "arguments": _args_str(args),
                    }
                    idx = run.add_item(item)
                    yield run.ev(
                        "response.output_item.added", output_index=idx, item=item
                    )
                    yield run.ev(
                        "response.output_item.done", output_index=idx, item=item
                    )
                    paused = True
                else:
                    item = {
                        "id": _new_id("mcp_"),
                        "type": "mcp_call",
                        "server_label": d.server_name,
                        "name": d.tool_name,
                        "arguments": _args_str(args),
                        "output": None,
                        "error": None,
                        "approval_request_id": None,
                        "status": "in_progress",
                    }
                    idx = run.add_item(item)
                    c.update(item=item, idx=idx)
                    yield run.ev(
                        "response.output_item.added", output_index=idx, item=item
                    )
                    yield run.ev(
                        "response.mcp_call.in_progress",
                        item_id=item["id"],
                        output_index=idx,
                    )
                    yield run.ev(
                        "response.mcp_call_arguments.delta",
                        item_id=item["id"],
                        output_index=idx,
                        delta=item["arguments"],
                    )
                    yield run.ev(
                        "response.mcp_call_arguments.done",
                        item_id=item["id"],
                        output_index=idx,
                        arguments=item["arguments"],
                    )
                    running.append(c)
            # calls past the per-request cap are answered without running
            room = max(cap - run.total_calls, 0) if cap is not None else len(running)
            to_run = running[:room]
            skipped = running[room:]
            outcomes = await run_all(
                rt, [(c["def"].fname, c["args_obj"]) for c in to_run]
            )
            run.total_calls += len(to_run)
            for c, oc in zip(to_run, outcomes, strict=True):
                d = c["def"]
                item, idx = c["item"], c["idx"]
                if d.kind == "web_search":
                    if not oc.is_error:
                        off = len(run.sources)
                        run.sources.extend(oc.results)
                        oc.text = format_search_text(
                            oc.query or "", oc.results, start=off + 1
                        )
                    item = {**item, "status": "failed" if oc.is_error else "completed"}
                    if include_sources and not oc.is_error:
                        item["action"] = {
                            **item["action"],
                            "sources": [
                                {"type": "url", "url": r.url} for r in oc.results
                            ],
                        }
                    run.output[idx] = item
                    run.tool_texts[item["id"]] = oc.text
                    if not oc.is_error:
                        yield run.ev(
                            "response.web_search_call.completed",
                            item_id=item["id"],
                            output_index=idx,
                        )
                    yield run.ev(
                        "response.output_item.done", output_index=idx, item=item
                    )
                else:
                    item = {
                        **item,
                        "status": "failed" if oc.is_error else "completed",
                        "output": None if oc.is_error else oc.text,
                        "error": (oc.error_message or oc.text) if oc.is_error else None,
                    }
                    run.output[idx] = item
                    run.tool_texts[item["id"]] = oc.text
                    yield run.ev(
                        "response.mcp_call.failed"
                        if oc.is_error
                        else "response.mcp_call.completed",
                        item_id=item["id"],
                        output_index=idx,
                    )
                    yield run.ev(
                        "response.output_item.done", output_index=idx, item=item
                    )
                run.stats["tools"].append(
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
                hist += _pair(
                    item["id"],
                    d.fname,
                    _args_str(c["args_obj"]),
                    oc.text,
                )
            for c in skipped:
                item, idx = c["item"], c["idx"]
                msg = "Error: the request's max_tool_calls was reached."
                run.tool_texts[item["id"]] = msg
                if c["def"].kind == "web_search":
                    item = {**item, "status": "failed"}
                else:
                    item = {**item, "status": "failed", "error": msg}
                run.output[idx] = item
                yield run.ev("response.output_item.done", output_index=idx, item=item)
                hist += _pair(item["id"], c["def"].fname, "{}", msg)

            if paused or has_client_call:
                break
            if rnd == max_iter - 1:
                run.status = "incomplete"
                run.incomplete_reason = "max_tool_calls"

        terminal = (
            "response.completed" if run.status == "completed" else "response.incomplete"
        )
        run.final = run.envelope(run.status, run.output, True)
        _persist(req, request, run)
        yield run.ev(terminal, response=run.final)
        yield b"data: [DONE]\n\n"
    finally:
        with contextlib.suppress(Exception):
            request.state._forced_response_id = None
        with contextlib.suppress(Exception):
            await rt.aclose()


def _pair(call_id: str, fname: str, args: str, result: str) -> list[dict]:
    return [
        {"type": "function_call", "call_id": call_id, "name": fname, "arguments": args},
        {"type": "function_call_output", "call_id": call_id, "output": result},
    ]


async def _finish_mcp(run: _Run, rt, idx: int, call: dict, fname: str, cap):
    [oc] = await run_all(rt, [(fname, _parse_args(call["arguments"]))])
    run.total_calls += 1
    call = {
        **call,
        "status": "failed" if oc.is_error else "completed",
        "output": None if oc.is_error else oc.text,
        "error": (oc.error_message or oc.text) if oc.is_error else None,
    }
    run.output[idx] = call
    run.tool_texts[call["id"]] = oc.text
    yield run.ev(
        "response.mcp_call.failed" if oc.is_error else "response.mcp_call.completed",
        item_id=call["id"],
        output_index=idx,
    )
    yield run.ev("response.output_item.done", output_index=idx, item=call)


async def _fail_mcp(run: _Run, idx: int, call: dict, why: str):
    call = {**call, "status": "failed", "error": why}
    run.output[idx] = call
    run.tool_texts[call["id"]] = f"Error: {why}"
    yield run.ev("response.mcp_call.failed", item_id=call["id"], output_index=idx)
    yield run.ev("response.output_item.done", output_index=idx, item=call)


def _persist(req, request, run: _Run):
    if not req.store or run.final is None:
        return
    from ..routers.responses import (
        _convert_to_messages,
        _resolve_owner,
        _store_response,
    )

    payload = dict(run.final)
    payload["_input_messages"] = [
        m for m in _convert_to_messages(req) if m.get("role") != "system"
    ]
    payload["_owner"] = _resolve_owner(request)
    payload["_server_tool_texts"] = dict(run.tool_texts)
    _store_response(run.id, payload)


async def create_with_server_tools_responses(req, request, inner: Inner):
    """Entry point from the Responses handler; returns a Response."""
    if req.background:
        return _err(
            400,
            "background mode is not supported together with server-side tools "
            "(web_search / mcp); send the request without background.",
            "unsupported_parameter",
            "background",
        )
    setup = await _setup(req)
    if isinstance(setup, JSONResponse):
        return setup
    run = _Run(req, setup)
    gen = run_stream(req, request, inner, setup, run)
    if req.stream:
        return StreamingResponse(
            gen, media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )
    async for _ in gen:
        pass
    if run.error is not None:
        return JSONResponse(status_code=run.error[0], content=run.error[1])
    return JSONResponse(content=run.final or {})


__all__ = [
    "create_with_server_tools_responses",
    "has_server_tools_responses",
    "function_tools",
    "input_item_to_messages",
    "annotations_for",
]
