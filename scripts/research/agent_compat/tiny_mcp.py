"""Tiny MCP test server: stdio (default) or streamable HTTP (--http PORT). Tools: echo, add.

Used as the fake MCP server for the agent census and the connector tests. Pure stdlib.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOOLS = [
    {
        "name": "echo",
        "description": "Echo the text back.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "add",
        "description": "Add two integers.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
]


def handle(msg: dict):
    """Return a JSON-RPC response dict, or None for notifications."""
    method, mid, p = msg.get("method"), msg.get("id"), msg.get("params") or {}
    if mid is None:
        return None
    if method == "initialize":
        res = {
            "protocolVersion": p.get("protocolVersion", "2025-06-18"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "tiny-mcp", "version": "1.0"},
        }
    elif method == "tools/list":
        res = {"tools": TOOLS}
    elif method == "tools/call":
        a = p.get("arguments") or {}
        if p.get("name") == "echo":
            text = str(a.get("text", ""))
        elif p.get("name") == "add":
            text = str(int(a.get("a", 0)) + int(a.get("b", 0)))
        else:
            return {
                "jsonrpc": "2.0",
                "id": mid,
                "error": {"code": -32602, "message": "unknown tool"},
            }
        res = {"content": [{"type": "text", "text": text}], "isError": False}
    elif method == "ping":
        res = {}
    else:
        return {
            "jsonrpc": "2.0",
            "id": mid,
            "error": {"code": -32601, "message": "method not found"},
        }
    return {"jsonrpc": "2.0", "id": mid, "result": res}


def serve_stdio():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        r = handle(json.loads(line))
        if r is not None:
            sys.stdout.write(json.dumps(r) + "\n")
            sys.stdout.flush()


def serve_http(port: int, token: str | None = None):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _empty(self, code):
            self.send_response(code)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            msg = json.loads(self.rfile.read(n) or b"{}")
            if token and self.headers.get("Authorization") != f"Bearer {token}":
                return self._empty(401)
            r = handle(msg)
            if r is None:
                return self._empty(202)
            b = json.dumps(r).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Mcp-Session-Id", "tiny-session")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            self._empty(405)

        def do_DELETE(self):
            self._empty(200)

    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", type=int)
    ap.add_argument("--token")
    a = ap.parse_args()
    if a.http:
        serve_http(a.http, a.token)
    else:
        serve_stdio()
