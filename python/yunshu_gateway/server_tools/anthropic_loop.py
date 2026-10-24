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
from datetime import UTC
from typing import Any

from fastapi.responses import JSONResponse, StreamingResponse

from yunshu_engine import settings

from ..routers.responses import _is_admin
from .mcp_connector import McpError
from .runtime import (
    CODE_BASH_DESC,
    CODE_BASH_SCHEMA,
    CODE_EDITOR_DESC,
    CODE_EDITOR_SCHEMA,
    CODE_PY_SCHEMA,
    WEB_FETCH_DESC,
    WEB_FETCH_SCHEMA,
    WEB_SEARCH_DESC,
    WEB_SEARCH_SCHEMA,
    ServerToolDef,
    ServerToolRuntime,
    ToolOutcome,
    decode_result,
    encode_result,
    explicit_tool_request,
    format_search_text,
    mcp_tool_to_def,
    parse_tool_args,
    python_tool_description,
    run_all,
    safe_fname,
)
from .search import Passage, SearchResult

logger = logging.getLogger(__name__)

Inner = Callable[[Any, Any], Awaitable[Any]]

_MARK = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")


def _is_web(t: dict, kind: str) -> bool:
    return str(t.get("type") or "").startswith(kind + "_")


_CODE_TYPE = re.compile(r"^code_execution_\d{8}$")
CODE_RESULT_TYPES = (
    "bash_code_execution_tool_result",
    "text_editor_code_execution_tool_result",
    "code_execution_tool_result",
)
CODE_TOOL_NAMES = (
    "bash_code_execution",
    "text_editor_code_execution",
    "code_execution",
)


def _is_code(t: dict) -> bool:
    return bool(_CODE_TYPE.match(str(t.get("type") or "")))


def _callers(t: dict) -> list[str]:
    ac = t.get("allowed_callers")
    return [str(x) for x in ac] if isinstance(ac, list) and ac else ["direct"]


def _is_ptc(t: dict) -> bool:
    """A client tool the code execution tool may call from code (programmatic tool calling)."""
    return not _is_code(t) and any(c.startswith("code_execution_") for c in _callers(t))


def _code_subtools(version: str, ptc: bool) -> tuple[bool, bool, bool]:
    """(bash+editor, python, python state persists) per tool version.

    * ``20250522``: Python only, a fresh interpreter per call;
    * ``20250825``: bash + text editor, plus Python (fresh per call) when the request lets the code
      call client tools;
    * ``20260120`` and later: bash + text editor + Python whose process (REPL state) persists
      between calls in a container.
    """
    if version == "20250522":
        return (False, True, False)
    if version == "20250825":
        return (True, ptc, False)
    return (True, True, True)


def has_server_tools(req) -> bool:
    for t in req.tools or []:
        d = t.model_dump() if hasattr(t, "model_dump") else dict(t)
        if (
            _is_web(d, "web_search")
            or _is_code(d)
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
        self.messages: list | None = (
            None  # request messages with container_upload blocks replaced
        )
        # set when this request answers a code cell that is waiting for client tool results
        self.resume: dict | None = None
        self.admin = False


async def _setup(req, request=None) -> _Setup | JSONResponse:
    """Split the declared tools into server tools (run here) and client tools (passed through)."""
    s = _Setup()
    tools = [
        t.model_dump(exclude_none=True) if hasattr(t, "model_dump") else dict(t)
        for t in (req.tools or [])
    ]
    toolsets: dict[str, dict] = {}
    code_tool: dict | None = None
    ptc_tools: list[dict] = []
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
        elif _is_code(t):
            code_tool = t
        elif ty == "mcp_toolset":
            toolsets[t.get("mcp_server_name", "")] = t
        else:
            if _is_ptc(t):
                ptc_tools.append(t)
                if "direct" not in _callers(t):
                    continue  # callable from code only: the model is not offered it directly
            s.client_tools.append(t)
    if ptc_tools and code_tool is None:
        return _err(
            400,
            "invalid_request_error",
            "A tool lists a code_execution_* caller in allowed_callers, but the request does not "
            "declare the code execution tool.",
        )
    if code_tool is not None:
        version = str(code_tool.get("type"))[len("code_execution_") :]
        bash_ed, py, persistent = _code_subtools(version, bool(ptc_tools))
        if bash_ed:
            s.defs.append(
                ServerToolDef(
                    "bash_code_execution",
                    "code_execution",
                    CODE_BASH_DESC,
                    CODE_BASH_SCHEMA,
                    spec=code_tool,
                )
            )
            s.defs.append(
                ServerToolDef(
                    "text_editor_code_execution",
                    "code_execution",
                    CODE_EDITOR_DESC,
                    CODE_EDITOR_SCHEMA,
                    spec=code_tool,
                )
            )
        if py:
            s.defs.append(
                ServerToolDef(
                    "code_execution",
                    "code_execution",
                    python_tool_description(persistent, ptc_tools),
                    CODE_PY_SCHEMA,
                    spec=code_tool,
                )
            )
    servers = list(getattr(req, "mcp_servers", None) or [])
    if servers and not settings.get("YUNSHU_MCP_CONNECTOR"):
        return _err(
            400,
            "invalid_request_error",
            "The MCP connector is disabled (YUNSHU_MCP_CONNECTOR=0).",
        )
    s.rt = ServerToolRuntime(s.defs)
    s.rt.ptc_tools = ptc_tools
    if request is not None:
        from ..routers.responses import _resolve_owner

        s.rt.owner = _resolve_owner(request)
        s.admin = _is_admin(request)
    if code_tool is not None:
        bad = _setup_code(req, s) or _setup_ptc(req, s, code_tool)
        if bad is not None:
            await s.rt.aclose()
            return bad
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


def _setup_code(req, s: _Setup) -> JSONResponse | None:
    """Gate the code execution tool and resolve its container; an error response, or None."""
    from ..sandbox import ContainerError, SandboxUnavailableError, get_executor
    from ..sandbox.containers import owner_ok

    if not settings.get("YUNSHU_CODE_EXECUTION"):
        return _err(
            400,
            "invalid_request_error",
            "The code execution tool is disabled on this server. Set YUNSHU_CODE_EXECUTION=1 "
            "(it runs commands inside the sandbox-runtime, srt).",
        )
    rt = s.rt
    assert rt is not None
    ex = get_executor()
    try:
        ex.srt()
    except SandboxUnavailableError as e:
        return _err(503, "api_error", str(e))
    cont = req.container
    cid = cont.get("id") if isinstance(cont, dict) else cont
    if cid:
        try:
            box = ex.manager.get(str(cid))
        except ContainerError as e:
            return _err(400, "invalid_request_error", e.message)
        if not owner_ok(box, rt.owner, s.admin):
            return _err(
                400,
                "invalid_request_error",
                f"Container '{cid}' was not found.",
            )
        rt.container = box
    # container_upload blocks: stage the files into the container, tell the model their names
    uploads: list[str] = []
    msgs: list[dict] = []
    for m in req.messages:
        d = m.model_dump() if hasattr(m, "model_dump") else dict(m)
        c = d.get("content")
        if isinstance(c, list) and any(
            isinstance(b, dict) and b.get("type") == "container_upload" for b in c
        ):
            new = []
            for b in c:
                if isinstance(b, dict) and b.get("type") == "container_upload":
                    uploads.append(str(b.get("file_id") or ""))
                    new.append({"type": "text", "text": "[file uploaded: {}]"})
                else:
                    new.append(b)
            d["content"] = new
        msgs.append(d)
    if uploads:
        from ..files_store import FileStoreError

        try:
            if rt.container is None:
                rt.container = ex.manager.create(owner=rt.owner)
            names = [ex.stage_file(rt.container, f) for f in uploads]
        except FileStoreError as e:
            return _err(
                404 if e.status == 404 else 400,
                "not_found_error" if e.status == 404 else "invalid_request_error",
                str(e),
            )
        it = iter(names)
        for d in msgs:
            if isinstance(d.get("content"), list):
                for b in d["content"]:
                    if (
                        b.get("type") == "text"
                        and b.get("text") == "[file uploaded: {}]"
                    ):
                        b["text"] = (
                            f"[file uploaded to the working directory: {next(it)}]"
                        )
        s.messages = msgs
    return None


def _tool_result_text(b: dict) -> str:
    c = b.get("content")
    if isinstance(c, str):
        return c
    return "\n".join(
        x.get("text", "")
        for x in c or []
        if isinstance(x, dict) and x.get("type") == "text"
    )


def _is_ptc_caller(b: dict) -> bool:
    c = b.get("caller")
    return isinstance(c, dict) and str(c.get("type") or "").startswith(
        "code_execution_"
    )


def caller_type(code_tool: dict) -> str:
    """The ``caller.type`` for tool_use blocks the code issued. The SDK models the 20250825 and
    20260120 callers; later versions report 20260120."""
    ty = str(code_tool.get("type"))
    return (
        ty
        if ty in ("code_execution_20250825", "code_execution_20260120")
        else (
            "code_execution_20260120"
            if ty > "code_execution_20260120"
            else "code_execution_20250825"
        )
    )


def _setup_ptc(req, s: _Setup, code_tool: dict) -> JSONResponse | None:
    """Is this request the client's answer to tool calls the code made (programmatic tool calling)?"""
    from ..sandbox import ContainerError, get_manager, get_pool
    from ..sandbox.repl import PAUSE_TTL

    rt = s.rt
    assert rt is not None
    msgs = [
        m.model_dump() if hasattr(m, "model_dump") else dict(m) for m in req.messages
    ]
    ptc_ids: set[str] = set()
    for m in msgs:
        if m.get("role") == "assistant" and isinstance(m.get("content"), list):
            ptc_ids |= {
                b["id"]
                for b in m["content"]
                if isinstance(b, dict)
                and b.get("type") == "tool_use"
                and _is_ptc_caller(b)
                and b.get("id")
            }
    if not ptc_ids or not msgs or msgs[-1].get("role") != "user":
        return None
    last = msgs[-1].get("content")
    answers = {
        b.get("tool_use_id"): b
        for b in (last if isinstance(last, list) else [])
        if isinstance(b, dict) and b.get("type") == "tool_result"
    }
    want = ptc_ids & set(answers)
    if not want:
        return None
    pool = get_pool()
    sess = pool.peek(rt.container.id) if rt.container is not None else None
    if sess is None or not (sess.paused and sess.meta.get("ptc")):
        sess = pool.find_pending(want)
    gone = _err(
        400,
        "invalid_request_error",
        "The code execution that made these tool calls is no longer running (it waits "
        f"{int(PAUSE_TTL)} s "
        "for the results, or the container expired). Send the request again from the user turn.",
    )
    if sess is None or not sess.paused:
        return gone
    ptc = sess.meta.get("ptc") or {}
    calls: dict[str, str] = ptc.get("calls") or {}
    if not set(calls) <= want:
        missing = sorted(set(calls) - want)
        return _err(
            400,
            "invalid_request_error",
            "Missing tool_result for tool_use id(s) " + ", ".join(missing) + ".",
        )
    try:
        rt.container = get_manager().get(sess.c.id)
    except ContainerError:
        return gone
    results = {}
    for tid, cid in calls.items():
        b = answers[tid]
        results[cid] = (_tool_result_text(b), bool(b.get("is_error")))
    s.resume = {
        "srv_id": ptc.get("srv_id"),
        "results": results,
        "ids": set(calls),
        "caller": caller_type(code_tool),
    }
    return None


# ── history: turn the spec's server-tool blocks back into plain tool_use / tool_result turns ──
def replay_search_results(block: dict) -> list[SearchResult]:
    content = block.get("content")
    if not isinstance(content, list):
        return []
    results = []
    for item in content:
        d = decode_result(item.get("encrypted_content", "")) or {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": "",
            "page_age": item.get("page_age"),
        }
        results.append(
            SearchResult(
                d["title"],
                d["url"],
                d["snippet"],
                d.get("page_age"),
                passages=[Passage(**p) for p in d.get("passages", [])],
                content_hash=d.get("content_hash", ""),
                fetched=d.get("fetched", False),
            )
        )
    return results


def _result_text(block: dict, start: int = 1) -> str:
    c = block.get("content")
    ty = block.get("type")
    if ty == "web_search_tool_result":
        if isinstance(c, dict):
            return f"Error: web search failed ({c.get('error_code', 'unavailable')})"
        return format_search_text(
            "(earlier search)", replay_search_results(block), start=start
        )
    if ty == "web_fetch_tool_result":
        if isinstance(c, dict) and c.get("type") == "web_fetch_tool_result_error":
            return f"Error: could not fetch ({c.get('error_code')})"
        doc = (c or {}).get("content") or {}
        src = doc.get("source") or {}
        return f"Fetched {(c or {}).get('url', '')}\n\n{src.get('data', '')}"
    if ty in CODE_RESULT_TYPES:
        if isinstance(c, dict):
            ct = str(c.get("type") or "")
            if ct.endswith("_error"):
                return f"Error: {c.get('error_code', 'unavailable')}"
            if ct == "bash_code_execution_result" or ct == "code_execution_result":
                out = [f"exit code: {c.get('return_code', 0)}"]
                if c.get("stdout"):
                    out.append("stdout:\n" + c["stdout"])
                if c.get("stderr"):
                    out.append("stderr:\n" + c["stderr"])
                names = [
                    x.get("file_id", "")
                    for x in c.get("content") or []
                    if isinstance(x, dict)
                ]
                if names:
                    out.append("files saved: " + ", ".join(names))
                return "\n".join(out)
            if ct.startswith("text_editor_code_execution_"):
                return str(c.get("content") or json.dumps(c))
        return str(c)
    parts = []
    for x in c or []:
        parts.append(x.get("text", "") if isinstance(x, dict) else str(x))
    return "\n".join(parts) if not isinstance(c, str) else c


def drop_programmatic_calls(messages: list) -> list:
    """Tool calls the code made (``caller`` is a code_execution tool) and their results are not part
    of what the model saw: it saw one code execution call and its final result. Remove them, drop
    the messages that became empty and join the assistant turns they separated."""
    dumped = [m.model_dump() if hasattr(m, "model_dump") else dict(m) for m in messages]
    ids = {
        b["id"]
        for m in dumped
        if m.get("role") == "assistant" and isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "tool_use" and _is_ptc_caller(b)
    }
    if not ids:
        return messages
    out: list[dict] = []
    for m in dumped:
        c = m.get("content")
        if isinstance(c, list):
            keep = [
                b
                for b in c
                if not (
                    isinstance(b, dict)
                    and (
                        (b.get("type") == "tool_use" and b.get("id") in ids)
                        or (
                            b.get("type") == "tool_result"
                            and b.get("tool_use_id") in ids
                        )
                    )
                )
            ]
            if len(keep) != len(c):
                if not keep:
                    continue
                m = {**m, "content": keep}
        if (
            out
            and out[-1]["role"] == m["role"] == "assistant"
            and isinstance(out[-1].get("content"), list)
            and isinstance(m.get("content"), list)
        ):
            out[-1] = {**out[-1], "content": out[-1]["content"] + m["content"]}
        else:
            out.append(m)
    return out


def normalize_history(
    messages: list, sources: list[SearchResult] | None = None
) -> list[dict]:
    """Assistant turns that carry server_tool_use / mcp_tool_use plus their results become the
    plain sequence the model saw: assistant(text, tool_use) -> user(tool_result) -> assistant(rest)."""
    out: list[dict] = []
    if sources is None:
        sources = []
    messages = drop_programmatic_calls(messages)
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
                *CODE_RESULT_TYPES,
            ):
                start = len(sources) + 1
                if ty == "web_search_tool_result":
                    sources.extend(replay_search_results(b))
                pending.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": b.get("tool_use_id"),
                        "content": _result_text(b, start=start),
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
                    "cited_text": (r.passages[0].text if r.passages else r.snippet)[
                        :150
                    ],
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
    if d.kind == "code_execution":
        r = out.code
        editor = d.fname == "text_editor_code_execution"
        py = d.fname == "code_execution"
        rt = (
            "text_editor_code_execution_tool_result"
            if editor
            else "code_execution_tool_result"
            if py
            else "bash_code_execution_tool_result"
        )
        if out.is_error or r is None:
            code = out.error_code or "unavailable"
            if py and code not in (
                "invalid_tool_input",
                "unavailable",
                "too_many_requests",
                "execution_time_exceeded",
            ):
                code = "unavailable"  # the Python tool's error vocabulary has no file-size code
            return {
                "type": rt,
                "tool_use_id": tool_id,
                "content": {"type": rt + "_error", "error_code": code},
            }
        if py:
            return {
                "type": rt,
                "tool_use_id": tool_id,
                "content": {
                    "type": "code_execution_result",
                    "stdout": r.stdout,
                    "stderr": r.stderr,
                    "return_code": r.return_code,
                    "content": [
                        {"type": "code_execution_output", "file_id": f["file_id"]}
                        for f in r.files
                    ],
                },
            }
        if editor:
            return {"type": rt, "tool_use_id": tool_id, "content": r.data}
        return {
            "type": rt,
            "tool_use_id": tool_id,
            "content": {
                "type": "bash_code_execution_result",
                "stdout": r.stdout,
                "stderr": r.stderr,
                "return_code": r.return_code,
                "content": [
                    {"type": "bash_code_execution_output", "file_id": f["file_id"]}
                    for f in r.files
                ],
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


def _ptc_events(
    srv_id: str, calls: list[dict], caller: str, out_index: int, mapping: dict
):
    """tool_use blocks for the calls the code is waiting on; fills ``mapping`` (tool_use id -> call).
    Returns (events, next out_index)."""
    ev: list[bytes] = []
    for c in calls:
        tid = _new_id("toolu_")
        mapping[tid] = c["call_id"]
        ev.append(
            _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": out_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tid,
                        "name": c["name"],
                        "input": {},
                        "caller": {"type": caller, "tool_id": srv_id},
                    },
                },
            )
        )
        ev.append(
            _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": out_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(c["input"], ensure_ascii=False),
                    },
                },
            )
        )
        ev.append(
            _sse(
                "content_block_stop", {"type": "content_block_stop", "index": out_index}
            )
        )
        out_index += 1
    return ev, out_index


def _remember_pause(rt: ServerToolRuntime, srv_id: str, mapping: dict) -> None:
    from ..sandbox import get_pool

    if rt.container is not None:
        sess = get_pool().peek(rt.container.id)
        if sess is not None:
            sess.meta["ptc"] = {"srv_id": srv_id, "calls": dict(mapping)}


async def run_stream(req, request, inner: Inner, setup: _Setup) -> AsyncIterator[bytes]:
    """The whole server-tool loop as one Anthropic SSE stream."""
    rt = setup.rt
    assert rt is not None
    max_iter = int(settings.get("YUNSHU_SERVER_TOOL_MAX_ITERATIONS"))
    fn_tools = [
        {"name": d.fname, "description": d.description, "input_schema": d.schema}
        for d in setup.defs
    ]
    sources: list[SearchResult] = []
    history = normalize_history(setup.messages or req.messages, sources=sources)
    total = _Usage()
    out_index = 0
    started = False
    stop_reason = "end_turn"
    stats = {"rounds": 0, "tools": [], "round_usage": []}
    msg_id = _new_id("msg_")
    paused = False
    try:
        rs = setup.resume
        if rs is not None:
            # the client answered tool calls the code made: continue that cell first
            started = True
            yield _sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "model": req.model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 1},
                    },
                },
            )
            oc = await rt.python_resume(rs["results"])
            rt.counts["code_execution"] += 1
            d = rt.defs["code_execution"]
            if oc.code is not None and oc.code.calls:
                mapping: dict[str, str] = {}
                evs, out_index = _ptc_events(
                    rs["srv_id"], oc.code.calls, rs["caller"], out_index, mapping
                )
                for e in evs:
                    yield e
                _remember_pause(rt, rs["srv_id"], mapping)
                stop_reason, paused = "tool_use", True
            else:
                block = _render_result_block(d, rs["srv_id"], oc)
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
                tr = {
                    "type": "tool_result",
                    "tool_use_id": rs["srv_id"],
                    "content": oc.text,
                    "is_error": oc.is_error,
                }
                if history and history[-1]["role"] == "user":
                    c0 = history[-1]["content"]
                    history[-1] = {
                        "role": "user",
                        "content": [
                            *(
                                [{"type": "text", "text": c0}]
                                if isinstance(c0, str)
                                else c0
                            ),
                            tr,
                        ],
                    }
                else:
                    history.append({"role": "user", "content": [tr]})
        for rnd in range(0 if paused else max_iter):
            used = int(total.d.get("output_tokens") or 0)
            if used >= req.max_tokens:
                stop_reason = "max_tokens"
                break
            stats["rounds"] += 1
            inner_req = _inner_request(
                req,
                history,
                setup.client_tools + fn_tools,
                rnd == 0,
                setup.defs,
                used_output=used,
            )
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
                                "name": d.fname
                                if d.kind == "code_execution"
                                else d.kind,
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
            # Per-round prompt accounting: a continuation should read the shared prefix from the
            # cache (cache_read_input_tokens) and prefill only the new turn (input_tokens).
            stats["round_usage"].append(
                {
                    k: round_usage.get(k)
                    for k in (
                        "input_tokens",
                        "cache_read_input_tokens",
                        "output_tokens",
                    )
                }
            )
            # blocks of this model turn -> assistant message for history; run server calls
            calls = []
            for tb in tool_blocks.values():
                try:
                    args = json.loads(tb["json"]) if tb["json"] else {}
                except ValueError:
                    args = {}
                tb["args"] = args
                # what the tool runs on: tolerant of truncated / bare-string arguments
                tb["exec_args"] = (
                    args
                    if isinstance(args, dict) and args
                    else parse_tool_args(tb["json"])
                )
                if tb["server"]:
                    calls.append(tb)
            client_calls = [tb for tb in tool_blocks.values() if not tb["server"]]
            if not calls:
                stop_reason = turn_stop
                break
            outcomes = await run_all(
                rt, [(tb["def"].fname, tb["exec_args"]) for tb in calls]
            )
            assistant_blocks = [b for b in model_turn]
            result_msgs = []
            waiting = None
            for tb, oc in zip(calls, outcomes, strict=True):
                d = tb["def"]
                if d.kind == "code_execution" and oc.code is not None and oc.code.calls:
                    waiting = (tb, oc)  # the code awaits client tools: no result yet
                    continue
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
            if waiting is not None:
                tb, oc = waiting
                mapping = {}
                evs, out_index = _ptc_events(
                    tb["id"],
                    oc.code.calls,
                    caller_type(tb["def"].spec),
                    out_index,
                    mapping,
                )
                for e in evs:
                    yield e
                _remember_pause(rt, tb["id"], mapping)
                stop_reason = "tool_use"
                break
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
        delta: dict[str, Any] = {"stop_reason": stop_reason, "stop_sequence": None}
        if rt.container is not None:
            from datetime import datetime

            box = rt.container
            with contextlib.suppress(Exception):
                box = _container_now(rt.container)
            delta["container"] = {
                "id": box.id,
                "expires_at": datetime.fromtimestamp(box.expires_at, UTC)
                .isoformat()
                .replace("+00:00", "Z"),
            }
        yield _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": delta,
                "usage": usage,
                "x_yunshu": {"server_tools": stats},
            },
        )
        yield _sse("message_stop", {"type": "message_stop"})
    finally:
        with contextlib.suppress(Exception):
            await rt.aclose()


def _container_now(c):
    """The container with its idle timer refreshed by this request."""
    from ..sandbox import get_manager

    return get_manager().get(c.id)


def _last_user_text(history: list[dict]) -> str:
    """Text of the last user turn, or "" when it only carries tool results."""
    if not history or history[-1].get("role") != "user":
        return ""
    c = history[-1].get("content")
    if isinstance(c, str):
        return c
    parts = []
    for b in c or []:
        if isinstance(b, dict) and b.get("type") == "tool_result":
            return ""
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
    return "\n".join(parts)


def _inner_request(
    req,
    history: list[dict],
    tools: list[dict],
    first: bool,
    defs=(),
    used_output: int = 0,
):
    """A fresh request for one generation round: the running history, function-style tools.

    ``tool_choice`` only shapes the first round (a forced tool would otherwise loop): later rounds are
    ``auto``, and an explicit "search the web ..." in the last user turn steers round one to the tool.
    """
    from ..routers.anthropic import AnthropicMessage, AnthropicTool

    tc = req.tool_choice
    if (
        isinstance(tc, dict)
        and tc.get("type") == "tool"
        and tc.get("name") == "code_execution"
    ):
        # the declared tool stands for its sub-tools
        names = {d.fname for d in defs}
        tc = {
            **tc,
            "name": "bash_code_execution"
            if "bash_code_execution" in names
            else "code_execution",
        }
    if not first:
        tc = {"type": "auto"} if tools else None
    elif tc in (None, "auto", {"type": "auto"}):
        want = explicit_tool_request(_last_user_text(history), defs)
        if want:
            tc = {"type": "tool", "name": want}
    return req.model_copy(
        update={
            # the caller's cap covers the whole outward response, not each round
            "max_tokens": max(1, req.max_tokens - used_output),
            "messages": [
                AnthropicMessage(role=m["role"], content=m["content"]) for m in history
            ],
            "tools": [AnthropicTool(**t) for t in tools] or None,
            "tool_choice": tc if tools else None,
            "stream": True,
            "mcp_servers": None,
            "container": None,
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
            if ev["delta"].get("container"):
                msg["container"] = ev["delta"]["container"]
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
    setup = await _setup(req, request)
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
