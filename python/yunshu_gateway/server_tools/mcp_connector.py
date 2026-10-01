"""Server-side MCP client for the connector features (Anthropic ``mcp_servers``, OpenAI Responses
``{"type": "mcp"}`` tools).

One :class:`McpConnection` is opened per request and closed when the request ends: it speaks MCP
over streamable HTTP (a POST per JSON-RPC message; the reply is JSON or an SSE stream, the
``Mcp-Session-Id`` is echoed) and falls back to the older HTTP+SSE transport (GET the stream, read
the ``endpoint`` event, POST there, replies arrive on the stream). Every call has a timeout.

This is separate from ``yunshu_engine.mcp_client`` (a startup-configured pool of stdio/HTTP servers
for the engine's own tool use): connector servers are named by the API request itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field

import httpx

from yunshu_engine import settings
from yunshu_engine.netguard import (
    Target,
    UrlNotAllowedError,
    join_url,
    origin_of,
    pin_request,
    resolve_target,
    same_origin,
)

PROTOCOL_VERSION = "2025-06-18"


class McpError(Exception):
    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass
class McpTool:
    name: str
    description: str = ""
    input_schema: dict = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    annotations: dict | None = None


@dataclass
class McpCallResult:
    text: str
    content: list[dict]
    is_error: bool
    raw: dict


def _sse_messages(text: str):
    for block in text.replace("\r\n", "\n").split("\n\n"):
        data = [ln[5:].strip() for ln in block.splitlines() if ln.startswith("data:")]
        if data:
            try:
                yield json.loads("\n".join(data))
            except ValueError:
                continue


def _reply_for(msg, rid: int) -> dict | None:
    """Classify one JSON-RPC message against the request id ``rid``.

    ``None``: not the reply (a server notification, a server-to-client request, another id).
    Raises :class:`McpError` for a message that claims to be the reply but is malformed.
    """
    if not isinstance(msg, dict):
        raise McpError("MCP server sent a JSON-RPC message that is not an object")
    if "method" in msg or "id" not in msg:
        return None
    if type(msg["id"]) is not int or msg["id"] != rid:
        return None
    if msg.get("jsonrpc") != "2.0":
        raise McpError("MCP reply is not JSON-RPC 2.0")
    has_result, has_error = "result" in msg, "error" in msg
    if has_result == has_error:
        raise McpError("MCP reply must carry exactly one of result and error")
    if has_error:
        if not isinstance(msg["error"], dict):
            raise McpError("MCP reply error is not an object")
    elif not isinstance(msg["result"], dict):
        raise McpError("MCP reply result is not an object")
    return msg


def _unwrap(msg: dict):
    if "error" in msg:
        e = msg["error"]
        code = e.get("code")
        raise McpError(
            str(e.get("message", "MCP error")), code if isinstance(code, int) else None
        )
    return msg["result"]


class McpConnection:
    def __init__(
        self,
        url: str,
        *,
        headers: dict | None = None,
        authorization: str | None = None,
        timeout: float | None = None,
        allow_private: bool | None = None,
        max_bytes: int | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        self.url = url
        self.headers = {
            "Accept": "application/json, text/event-stream",
            **(headers or {}),
        }
        if authorization:
            self.headers["Authorization"] = (
                authorization if " " in authorization else f"Bearer {authorization}"
            )
        self.timeout = timeout or float(settings.get("YUNSHU_MCP_CONNECTOR_TIMEOUT"))
        self.max_bytes = max_bytes or int(
            settings.get("YUNSHU_MCP_CONNECTOR_MAX_BYTES")
        )
        self.allow_private = (
            bool(settings.get("YUNSHU_MCP_CONNECTOR_ALLOW_PRIVATE"))
            if allow_private is None
            else allow_private
        )
        self._client = client
        self._own = client is None
        self._sid: str | None = None
        self._id = 0
        self._legacy_post: str | None = None
        self._legacy_task: asyncio.Task | None = None
        self._legacy_waiters: dict[int, asyncio.Future] = {}
        self._targets: dict[tuple, Target] = {}
        self.server_info: dict = {}

    # ── lifecycle ────────────────────────────────────────────────────────────
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *exc):
        await self.close()

    async def connect(self):
        """Open the connection. On any failure (or cancellation) everything this connection owns
        is closed before the exception propagates."""
        try:
            await self._connect()
        except BaseException:
            await self.close()
            raise

    async def _connect(self):
        try:
            async with asyncio.timeout(self.timeout):
                await self._target(self.url)
        except TimeoutError as e:
            raise McpError(f"MCP connect timed out after {self.timeout:g}s") from e
        if self._client is None:
            # trust_env off: a proxy from the environment would bypass the pinned address
            self._client = httpx.AsyncClient(
                timeout=self.timeout, follow_redirects=False, trust_env=False
            )
        init = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "yunshu", "version": "1"},
        }
        try:
            res = await self._request("initialize", init)
        except _NeedsLegacyError:
            await self._open_legacy()
            res = await self._request("initialize", init)
        self.server_info = (res or {}).get("serverInfo", {}) or {}
        await self._notify("notifications/initialized", {})

    async def close(self):
        if self._legacy_task:
            self._legacy_task.cancel()
            with contextlib.suppress(BaseException):
                await self._legacy_task
            self._legacy_task = None
        for fut in self._legacy_waiters.values():
            if not fut.done():
                fut.cancel()
        self._legacy_waiters.clear()
        if self._sid and self._client and not self._legacy_post:
            with contextlib.suppress(Exception):
                await self._call(
                    "DELETE",
                    self.url,
                    headers={**self.headers, "Mcp-Session-Id": self._sid},
                    timeout=3,
                )
        if self._own and self._client:
            await self._client.aclose()
        self._client = None

    # ── public calls ─────────────────────────────────────────────────────────
    async def list_tools(self) -> list[McpTool]:
        tools: list[McpTool] = []
        cursor = None
        for _ in range(20):
            res = await self._request(
                "tools/list", {"cursor": cursor} if cursor else {}
            )
            for t in (res or {}).get("tools", []):
                tools.append(
                    McpTool(
                        t.get("name", ""),
                        t.get("description", "") or "",
                        t.get("inputSchema") or {"type": "object", "properties": {}},
                        t.get("annotations"),
                    )
                )
            cursor = (res or {}).get("nextCursor")
            if not cursor:
                break
        return tools

    async def call_tool(self, name: str, arguments: dict) -> McpCallResult:
        res = (
            await self._request(
                "tools/call", {"name": name, "arguments": arguments or {}}
            )
            or {}
        )
        content = res.get("content") or []
        parts = []
        for c in content:
            if c.get("type") == "text":
                parts.append(c.get("text", ""))
            elif c.get("type") in ("image", "audio"):
                parts.append(f"[{c['type']}: {c.get('mimeType', '')}]")
            elif c.get("type") == "resource":
                r = c.get("resource") or {}
                parts.append(r.get("text") or f"[resource {r.get('uri', '')}]")
            else:
                parts.append(json.dumps(c))
        if not parts and res.get("structuredContent") is not None:
            parts.append(json.dumps(res["structuredContent"]))
        return McpCallResult("\n".join(parts), content, bool(res.get("isError")), res)

    # ── transport ────────────────────────────────────────────────────────────
    def _hdr(self) -> dict:
        h = dict(self.headers)
        if self._sid:
            h["Mcp-Session-Id"] = self._sid
            h["MCP-Protocol-Version"] = PROTOCOL_VERSION
        return h

    async def _target(self, url: str) -> Target:
        """Resolve a URL once per origin, check it, and keep the checked address: every request
        to that origin connects to it (a second DNS answer is never consulted)."""
        try:
            key = origin_of(url)
            tgt = self._targets.get(key)
            if tgt is None:
                tgt = await resolve_target(url, allow_private=self.allow_private)
                self._targets[key] = tgt
            return tgt
        except UrlNotAllowedError as e:
            raise McpError(f"MCP server url not allowed: {e}") from e

    async def _pinned(self, url: str, headers: dict) -> tuple[str, dict, dict]:
        pinned, extra, ext = pin_request(await self._target(url), url)
        return pinned, {**headers, **extra}, ext

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            raise McpError("MCP connection is closed")
        return self._client

    async def _call(self, method: str, url: str, *, headers: dict, **kw):
        pinned, hdrs, ext = await self._pinned(url, headers)
        return await self._http().request(
            method, pinned, headers=hdrs, extensions=ext, **kw
        )

    async def _open(self, method: str, url: str, *, headers: dict, **kw):
        pinned, hdrs, ext = await self._pinned(url, headers)
        return self._http().stream(method, pinned, headers=hdrs, extensions=ext, **kw)

    async def _notify(self, method: str, params: dict):
        body = {"jsonrpc": "2.0", "method": method, "params": params}
        with contextlib.suppress(httpx.HTTPError):  # notifications are best-effort
            await self._call(
                "POST",
                self._legacy_post or self.url,
                json=body,
                headers=self._hdr(),
                timeout=self.timeout,
            )

    async def _request(self, method: str, params: dict):
        self._id += 1
        rid = self._id
        body = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
        try:
            async with asyncio.timeout(self.timeout):
                if self._legacy_post:
                    return await self._legacy_request(rid, body)
                return await self._post_request(rid, body)
        except TimeoutError as e:
            raise McpError(f"MCP {method} timed out after {self.timeout:g}s") from e
        except httpx.HTTPError as e:
            raise McpError(f"MCP {method} failed: {type(e).__name__}: {e}") from e

    def _too_large(self) -> McpError:
        return McpError(f"MCP reply too large (over {self.max_bytes} bytes)")

    async def _post_request(self, rid: int, body: dict):
        ctx = await self._open("POST", self.url, json=body, headers=self._hdr())
        async with ctx as r:
            if body["method"] == "initialize" and r.status_code in (404, 405):
                raise _NeedsLegacyError
            if r.status_code == 401 or r.status_code == 403:
                raise McpError(
                    f"MCP server refused authorization (HTTP {r.status_code})",
                    r.status_code,
                )
            if r.status_code >= 400:
                raise McpError(
                    f"MCP server returned HTTP {r.status_code}", r.status_code
                )
            if 300 <= r.status_code < 400:
                raise McpError(
                    f"MCP server redirected (HTTP {r.status_code}); not followed"
                )
            if r.headers.get("mcp-session-id"):
                self._sid = r.headers["mcp-session-id"]
            ctype = r.headers.get("content-type", "").lower()
            if "text/event-stream" in ctype:
                buf = ""
                async for chunk in r.aiter_text():
                    buf = (buf + chunk).replace("\r\n", "\n")
                    if len(buf) > self.max_bytes:
                        raise self._too_large()
                    while "\n\n" in buf:
                        block, buf = buf.split("\n\n", 1)
                        for msg in _sse_messages(block + "\n\n"):
                            hit = _reply_for(msg, rid)
                            if hit is not None:
                                return _unwrap(hit)
                for msg in _sse_messages(buf):
                    hit = _reply_for(msg, rid)
                    if hit is not None:
                        return _unwrap(hit)
                raise McpError("MCP server closed the stream without a response")
            raw = bytearray()
            async for (
                chunk
            ) in r.aiter_bytes():  # decoded bytes: a gzip bomb counts expanded
                raw += chunk
                if len(raw) > self.max_bytes:
                    raise self._too_large()
            try:
                msg = json.loads(raw)
            except ValueError as e:
                raise McpError("MCP server returned a non-JSON reply") from e
            for m in msg if isinstance(msg, list) else [msg]:
                hit = _reply_for(m, rid)
                if hit is not None:
                    return _unwrap(hit)
            raise McpError("MCP reply does not answer the request id")

    # ── legacy HTTP+SSE ──────────────────────────────────────────────────────
    async def _open_legacy(self):
        ready: asyncio.Future = asyncio.get_running_loop().create_future()

        def handle(block: str):
            ev, data = None, []
            for line in block.split("\n"):
                if line.startswith("event:"):
                    ev = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].strip())
            payload = "\n".join(data)
            if ev == "endpoint" and not ready.done():
                ready.set_result(payload)
            elif payload:
                with contextlib.suppress(ValueError):
                    m = json.loads(payload)
                    rid = m.get("id") if isinstance(m, dict) else None
                    fut = (
                        self._legacy_waiters.pop(rid, None)
                        if type(rid) is int
                        else None
                    )
                    if fut and not fut.done():
                        fut.set_result(m)

        async def reader():
            err: McpError | None = None
            try:
                ctx = await self._open(
                    "GET",
                    self.url,
                    headers={**self.headers, "Accept": "text/event-stream"},
                    timeout=None,
                )
                async with ctx as r:
                    if r.status_code >= 400:
                        raise McpError(
                            f"MCP SSE endpoint returned HTTP {r.status_code}"
                        )
                    buf = ""
                    async for chunk in r.aiter_text():
                        buf = (buf + chunk).replace("\r\n", "\n")
                        if len(buf) > self.max_bytes:
                            raise self._too_large()
                        while "\n\n" in buf:
                            block, buf = buf.split("\n\n", 1)
                            handle(block)
            except McpError as e:
                err = e
            except Exception as e:  # noqa: BLE001
                err = McpError(f"MCP SSE stream failed: {e}")
            # reached on normal EOF too (a cancelled reader skips this): nobody may wait forever
            err = err or McpError("MCP SSE stream closed")
            if not ready.done():
                ready.set_exception(err)
            for fut in list(self._legacy_waiters.values()):
                if not fut.done():
                    fut.set_exception(err)

        self._legacy_task = asyncio.create_task(reader())
        try:
            payload = await asyncio.wait_for(ready, self.timeout)
        except TimeoutError as e:
            raise McpError("MCP SSE server sent no endpoint event") from e
        try:
            endpoint = join_url(self.url, payload)
        except UrlNotAllowedError as e:
            raise McpError(f"MCP SSE endpoint not allowed: {e}") from e
        # the legacy transport POSTs where the server says; credentials are bound to the origin
        # that was configured, so another origin (or port, or scheme) is refused outright
        if not same_origin(self.url, endpoint):
            raise McpError(
                "MCP SSE endpoint is on a different origin than the server url"
            )
        await self._target(endpoint)
        self._legacy_post = endpoint

    async def _legacy_request(self, rid: int, body: dict):
        fut = asyncio.get_running_loop().create_future()
        self._legacy_waiters[rid] = fut
        try:
            if self._legacy_task is None or self._legacy_task.done():
                raise McpError("MCP SSE stream closed")
            if self._legacy_post is None:
                raise McpError("MCP SSE endpoint not announced")
            r = await self._call(
                "POST",
                self._legacy_post,
                json=body,
                headers=self._hdr(),
                timeout=self.timeout,
            )
            if r.status_code >= 400:
                raise McpError(
                    f"MCP server returned HTTP {r.status_code}", r.status_code
                )
            msg = await fut  # bounded by the caller's deadline; EOF/errors resolve it
        finally:
            self._legacy_waiters.pop(rid, None)
        hit = _reply_for(msg, rid)
        if hit is None:
            raise McpError("MCP reply does not answer the request id")
        return _unwrap(hit)


class _NeedsLegacyError(Exception):
    pass
