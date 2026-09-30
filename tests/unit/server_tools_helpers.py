"""Shared fakes for the server-side tool tests: a scripted generation handler, a fake search
provider, and a fake MCP server (streamable HTTP and legacy SSE) running on a loopback port."""

from __future__ import annotations

import importlib.util
import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from fastapi.responses import StreamingResponse

from yunshu_gateway.server_tools import search as search_mod

_TINY = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "research"
    / "agent_compat"
    / "tiny_mcp.py"
)
_spec = importlib.util.spec_from_file_location("tiny_mcp", _TINY)
tiny_mcp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tiny_mcp)


def sse_bytes(name: str, data: dict) -> bytes:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()


def message_events(
    blocks: list[dict], stop: str = "end_turn", usage=None
) -> list[bytes]:
    """SSE events shaped like the engine's own Messages stream."""
    u = usage or {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    ev = [
        sse_bytes(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_inner",
                    "type": "message",
                    "role": "assistant",
                    "model": "m",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {**u, "output_tokens": 1},
                },
            },
        )
    ]
    for i, b in enumerate(blocks):
        if b["type"] == "text":
            ev.append(
                sse_bytes(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": i,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            )
            ev.append(
                sse_bytes(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {"type": "text_delta", "text": b["text"]},
                    },
                )
            )
        elif b["type"] == "thinking":
            ev.append(
                sse_bytes(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": i,
                        "content_block": {"type": "thinking", "thinking": ""},
                    },
                )
            )
            ev.append(
                sse_bytes(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {"type": "thinking_delta", "thinking": b["thinking"]},
                    },
                )
            )
        else:
            ev.append(
                sse_bytes(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": i,
                        "content_block": {
                            "type": "tool_use",
                            "id": b["id"],
                            "name": b["name"],
                            "input": {},
                        },
                    },
                )
            )
            ev.append(
                sse_bytes(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(b["input"]),
                        },
                    },
                )
            )
        ev.append(
            sse_bytes("content_block_stop", {"type": "content_block_stop", "index": i})
        )
    ev.append(
        sse_bytes(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop, "stop_sequence": None},
                "usage": {"output_tokens": u["output_tokens"]},
            },
        )
    )
    ev.append(sse_bytes("message_stop", {"type": "message_stop"}))
    return ev


class ScriptedInner:
    """Stands in for the Messages generation handler: answers round N with rounds[N]."""

    def __init__(self, rounds: list[tuple[list[dict], str]]):
        self.rounds = list(rounds)
        self.requests: list = []

    async def __call__(self, req, request):
        self.requests.append(req)
        blocks, stop = (
            self.rounds.pop(0)
            if self.rounds
            else ([{"type": "text", "text": "done"}], "end_turn")
        )

        async def gen():
            for e in message_events(blocks, stop):
                yield e

        return StreamingResponse(gen(), media_type="text/event-stream")


class FakeSearch(search_mod.SearchProvider):
    name = "fake"

    def __init__(self, results=None, error: search_mod.SearchError | None = None):
        self.results = (
            results
            if results is not None
            else [
                search_mod.SearchResult(
                    "Yunshu",
                    "https://example.com/yunshu",
                    "Yunshu is a local inference engine.",
                    "1 day ago",
                ),
                search_mod.SearchResult(
                    "MLX",
                    "https://ml-explore.github.io/mlx",
                    "MLX is an array framework.",
                ),
            ]
        )
        self.error = error
        self.calls: list[dict] = []

    async def search(
        self,
        query,
        *,
        limit,
        allowed_domains=None,
        blocked_domains=None,
        user_location=None,
        client,
    ):
        self.calls.append(
            {
                "query": query,
                "allowed": allowed_domains,
                "blocked": blocked_domains,
                "loc": user_location,
            }
        )
        if self.error:
            raise self.error
        return self.results


class FakeMcpHttp:
    """Loopback MCP server. mode='streamable' (JSON or SSE replies) or 'sse' (legacy endpoint transport)."""

    def __init__(self, mode="streamable", token=None, sse_reply=False):
        self.mode, self.token, self.sse_reply = mode, token, sse_reply
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        self._streams: list[queue.Queue] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _empty(self, code):
                self.send_response(code)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def _auth_ok(self):
                return (
                    not outer.token
                    or self.headers.get("Authorization") == f"Bearer {outer.token}"
                )

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                msg = json.loads(self.rfile.read(n) or b"{}")
                outer.requests.append(msg)
                outer.headers.append(dict(self.headers))
                if not self._auth_ok():
                    return self._empty(401)
                if outer.mode == "sse":
                    if not self.path.startswith("/messages"):
                        return self._empty(404)
                    r = tiny_mcp.handle(msg)
                    if r is not None:
                        for q in outer._streams:
                            q.put(r)
                    return self._empty(202)
                r = tiny_mcp.handle(msg)
                if r is None:
                    return self._empty(202)
                if outer.sse_reply:
                    b = f"event: message\ndata: {json.dumps(r)}\n\n".encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                else:
                    b = json.dumps(r).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                self.send_header("Mcp-Session-Id", "s1")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if outer.mode != "sse":
                    return self._empty(405)
                if not self._auth_ok():
                    return self._empty(401)
                q: queue.Queue = queue.Queue()
                outer._streams.append(q)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(b"event: endpoint\ndata: /messages?sid=1\n\n")
                self.wfile.flush()
                try:
                    while True:
                        try:
                            r = q.get(timeout=0.2)
                        except queue.Empty:
                            if getattr(outer, "stopped", False):
                                return
                            continue
                        self.wfile.write(
                            f"event: message\ndata: {json.dumps(r)}\n\n".encode()
                        )
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return

            def do_DELETE(self):
                self._empty(200)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}" + (
            "/sse" if mode == "sse" else "/mcp"
        )
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.stopped = True
        self.httpd.shutdown()
        self.httpd.server_close()
