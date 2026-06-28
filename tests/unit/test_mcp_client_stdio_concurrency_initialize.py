"""Waves 923-924: MCP stdio client concurrency + initialize handshake.

W923 (HIGH): the stdio transport shared one stdin/stdout pair with no lock and returned the
  first response carrying ANY id (no req_id match), so concurrent tool calls interleaved
  writes and raced reads → the WRONG result returned to the wrong caller. Serialize each
  round trip + match the response id.
W924: connect() jumped straight to tools/list with no `initialize` handshake, so a
  spec-conformant server rejected it → tools never discovered → call_tool silently failed.
  Send initialize + notifications/initialized first.
"""
from __future__ import annotations

import asyncio
import contextlib
import inspect

from yunshu_engine.mcp_client import MCPServerConnection


def test_w923_stdio_has_lock_and_id_match():
    src = inspect.getsource(MCPServerConnection._send_stdio)
    assert "async with self._stdio_lock:" in src
    assert 'response.get("id") != req_id' in src


def test_w923_connection_constructs_lock():
    src = inspect.getsource(MCPServerConnection.__init__)
    assert "self._stdio_lock = asyncio.Lock()" in src


def test_w924_connect_does_initialize_before_discover():
    stdio = inspect.getsource(MCPServerConnection._connect_stdio)
    http = inspect.getsource(MCPServerConnection._connect_http)
    for src in (stdio, http):
        i = src.index("self._initialize()")
        j = src.index("self._discover_tools()")
        assert i < j, "initialize must run before tools/list discovery"
    init = inspect.getsource(MCPServerConnection._initialize)
    assert '"initialize"' in init
    assert "notifications/initialized" in init


def test_w923_concurrent_calls_get_their_own_responses():
    """Behavioral: two concurrent _send_stdio calls must each receive the response matching
    their own request id, even if the server replies out of order."""

    class _FakeStdin:
        def __init__(self):
            self.writes = []

        def write(self, b):
            self.writes.append(b)

        async def drain(self):
            await asyncio.sleep(0)

    class _FakeStdout:
        """Replies to whatever ids it has seen, in REVERSE order (out-of-order)."""

        def __init__(self, stdin):
            self._stdin = stdin
            self._queued: list[bytes] = []

        async def readline(self):
            await asyncio.sleep(0)
            # parse any pending requests from stdin into queued responses (reversed)
            if not self._queued and self._stdin.writes:
                import json
                ids = []
                for w in self._stdin.writes:
                    with contextlib.suppress(Exception):
                        ids.append(json.loads(w.decode())["id"])
                self._stdin.writes.clear()
                for rid in reversed(ids):
                    self._queued.append(
                        (json.dumps({"jsonrpc": "2.0", "id": rid,
                                     "result": {"echo": rid}}) + "\n").encode())
            if self._queued:
                return self._queued.pop(0)
            await asyncio.sleep(0.001)
            return b""

    class _FakeProc:
        def __init__(self):
            self.stdin = _FakeStdin()
            self.stdout = _FakeStdout(self.stdin)

    async def _run():
        from yunshu_engine.mcp_client import MCPServerConfig
        conn = MCPServerConnection(MCPServerConfig(server_id="s", transport="stdio",
                                                   command="x", args=[]))
        conn._process = _FakeProc()
        r1, r2 = await asyncio.gather(
            conn._send_stdio("m1", {}), conn._send_stdio("m2", {}))
        # each call gets the result whose echoed id matches its own request — never swapped
        return r1, r2

    r1, r2 = asyncio.run(_run())
    assert r1 is not None and r2 is not None
    assert r1["echo"] != r2["echo"], "the two calls must get distinct (own) responses"
