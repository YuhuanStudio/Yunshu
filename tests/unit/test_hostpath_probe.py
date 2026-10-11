import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
import hostpath_probe as hp  # noqa: E402


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for _ in range(3):
            c = {"choices": [{"delta": {"content": "x"}}]}
            self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
        u = {"choices": [], "usage": {"completion_tokens": 3, "prompt_tokens": 5}}
        self.wfile.write(b"data: " + json.dumps(u).encode() + b"\n\n")
        if not self.server.truncate:
            self.wfile.write(b"data: [DONE]\n\n")


def serve(truncate):
    s = HTTPServer(("127.0.0.1", 0), H)
    s.truncate = truncate
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


def test_launcher_compiles():
    compile(hp.LAUNCHER_PATCH, "launcher", "exec")


def test_stream_measures_and_fails_closed():
    s = serve(False)
    r = hp.stream(s.server_port, hp.body_for("m", "hi", 3))
    assert r["ct"] == 3 and r["chunks"] == 3 and r["ttft_ms"] >= r["head_ms"]
    s.shutdown()
    s = serve(True)
    try:
        hp.stream(s.server_port, hp.body_for("m", "hi", 3))
    except RuntimeError:
        pass
    else:
        raise AssertionError("truncated stream must fail")
    s.shutdown()
