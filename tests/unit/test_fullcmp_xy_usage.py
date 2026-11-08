"""tfbench send() reads x_yunshu from the final usage object when the top level has none (the gateway
puts it there on the include_usage chunk; wire-1106 yunshu conc requests showed 'unavailable')."""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import tfbench  # noqa: E402

SPEC = {"accepted": 5, "drafted": 8, "rounds": 3}


def _serve(final_chunk):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = {
                "id": "x",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": "a"}}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _final(usage):
    return {
        "id": "x",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
        "usage": usage,
    }


def _body():
    return {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "stream": True,
    }


def test_send_reads_x_yunshu_from_the_final_usage_object():
    usage = {
        "prompt_tokens": 1024,
        "completion_tokens": 4,
        "total_tokens": 1028,
        "x_yunshu": {"speculative": SPEC},
    }
    srv = _serve(_final(usage))
    try:
        r = tfbench.send(f"http://127.0.0.1:{srv.server_port}", _body())
    finally:
        srv.shutdown()
    assert r["_xy_speculative"] == SPEC


def test_send_top_level_x_yunshu_still_wins_and_absent_stays_none():
    top = {
        "id": "x",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "x_yunshu": {"speculative": SPEC},
    }
    srv = _serve(top)
    try:
        r = tfbench.send(f"http://127.0.0.1:{srv.server_port}", _body())
    finally:
        srv.shutdown()
    assert r["_xy_speculative"] == SPEC
    srv = _serve(
        _final({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
    )
    try:
        r2 = tfbench.send(f"http://127.0.0.1:{srv.server_port}", _body())
    finally:
        srv.shutdown()
    assert r2["_xy_speculative"] is None
