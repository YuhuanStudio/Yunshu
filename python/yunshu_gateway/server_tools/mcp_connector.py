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
import time
from dataclasses import dataclass, field
from urllib.parse import urljoin

import httpx

from yunshu_engine import settings

from .netguard import UrlNotAllowedError, resolve_target

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


class McpConnection:
    def __init__(
        self,
        url: str,
        *,
        headers: dict | None = None,
        authorization: str | None = None,
        timeout: float | None = None,
        allow_private: bool | None = None,
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
        self.server_info: dict = {}

    # ── lifecycle ────────────────────────────────────────────────────────────
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *exc):
        await self.close()

    async def connect(self):
        try:
            await resolve_target(self.url, allow_private=self.allow_private)
        except UrlNotAllowedError as e:
            raise McpError(f"MCP server url not allowed: {e}") from e
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout, follow_redirects=False
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
        if self._sid and self._client and not self._legacy_post:
            with contextlib.suppress(Exception):
                await self._client.delete(
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

    async def _notify(self, method: str, params: dict):
        body = {"jsonrpc": "2.0", "method": method, "params": params}
        with contextlib.suppress(httpx.HTTPError):  # notifications are best-effort
            await self._client.post(
                self._legacy_post or self.url,
                json=body,
                headers=self._hdr(),
                timeout=self.timeout,
            )

    async def _request(self, method: str, params: dict):
        self._id += 1
        rid = self._id
        body = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
        deadline = time.monotonic() + self.timeout
        try:
            if self._legacy_post:
                return await self._legacy_request(rid, body, deadline)
            async with asyncio.timeout(self.timeout):
                return await self._post_request(rid, body)
        except TimeoutError as e:
            raise McpError(f"MCP {method} timed out after {self.timeout:g}s") from e
        except httpx.HTTPError as e:
            raise McpError(f"MCP {method} failed: {type(e).__name__}: {e}") from e

    async def _post_request(self, rid: int, body: dict):
        async with self._client.stream(
            "POST", self.url, json=body, headers=self._hdr()
        ) as r:
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
            if r.headers.get("mcp-session-id"):
                self._sid = r.headers["mcp-session-id"]
            ctype = r.headers.get("content-type", "").lower()
            if "text/event-stream" in ctype:
                buf = ""
                async for chunk in r.aiter_text():
                    buf += chunk
                    while "\n\n" in buf.replace("\r\n", "\n"):
                        buf = buf.replace("\r\n", "\n")
                        block, buf = buf.split("\n\n", 1)
                        for msg in _sse_messages(block + "\n\n"):
                            if isinstance(msg, dict) and msg.get("id") == rid:
                                return self._unwrap(msg)
                for msg in _sse_messages(buf):
                    if isinstance(msg, dict) and msg.get("id") == rid:
                        return self._unwrap(msg)
                raise McpError("MCP server closed the stream without a response")
            raw = await r.aread()
            try:
                msg = json.loads(raw)
            except ValueError as e:
                raise McpError("MCP server returned a non-JSON reply") from e
            if isinstance(msg, list):
                msg = next((m for m in msg if m.get("id") == rid), {})
            return self._unwrap(msg)

    @staticmethod
    def _unwrap(msg: dict):
        if "error" in msg:
            e = msg["error"] or {}
            raise McpError(str(e.get("message", "MCP error")), e.get("code"))
        return msg.get("result")

    # ── legacy HTTP+SSE ──────────────────────────────────────────────────────
    async def _open_legacy(self):
        ready: asyncio.Future = asyncio.get_running_loop().create_future()

        async def reader():
            try:
                async with self._client.stream(
                    "GET",
                    self.url,
                    headers={**self.headers, "Accept": "text/event-stream"},
                    timeout=None,
                ) as r:
                    if r.status_code >= 400:
                        if not ready.done():
                            ready.set_exception(
                                McpError(
                                    f"MCP SSE endpoint returned HTTP {r.status_code}"
                                )
                            )
                        return
                    ev, data = None, []
                    async for line in r.aiter_lines():
                        if line.startswith("event:"):
                            ev = line[6:].strip()
                        elif line.startswith("data:"):
                            data.append(line[5:].strip())
                        elif line == "":
                            payload = "\n".join(data)
                            if ev == "endpoint" and not ready.done():
                                ready.set_result(urljoin(self.url, payload))
                            elif payload:
                                with contextlib.suppress(ValueError):
                                    m = json.loads(payload)
                                    fut = (
                                        self._legacy_waiters.pop(m.get("id"), None)
                                        if isinstance(m, dict)
                                        else None
                                    )
                                    if fut and not fut.done():
                                        fut.set_result(m)
                            ev, data = None, []
            except Exception as e:  # noqa: BLE001
                if not ready.done():
                    ready.set_exception(McpError(f"MCP SSE stream failed: {e}"))
                for fut in self._legacy_waiters.values():
                    if not fut.done():
                        fut.set_exception(McpError("MCP SSE stream closed"))

        self._legacy_task = asyncio.create_task(reader())
        try:
            self._legacy_post = await asyncio.wait_for(ready, self.timeout)
        except TimeoutError as e:
            raise McpError("MCP SSE server sent no endpoint event") from e
        await resolve_target(self._legacy_post, allow_private=self.allow_private)

    async def _legacy_request(self, rid: int, body: dict, deadline: float):
        fut = asyncio.get_running_loop().create_future()
        self._legacy_waiters[rid] = fut
        r = await self._client.post(
            self._legacy_post, json=body, headers=self._hdr(), timeout=self.timeout
        )
        if r.status_code >= 400:
            self._legacy_waiters.pop(rid, None)
            raise McpError(f"MCP server returned HTTP {r.status_code}", r.status_code)
        try:
            msg = await asyncio.wait_for(fut, max(0.1, deadline - time.monotonic()))
        except TimeoutError as e:
            self._legacy_waiters.pop(rid, None)
            raise McpError(
                f"MCP {body['method']} timed out after {self.timeout:g}s"
            ) from e
        return self._unwrap(msg)


class _NeedsLegacyError(Exception):
    pass
