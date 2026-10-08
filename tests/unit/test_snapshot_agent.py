"""CPU-only checks for snapshot agent sampling and terminal evidence."""

import json
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from snapshot_agent import GreedyProxy, greedy_body, validate  # noqa: E402


def test_greedy_rewrite_preserves_tools_and_prompt():
    original = {
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"function": {"name": "read"}}],
        "temperature": 0.7,
    }
    body = json.loads(greedy_body(json.dumps(original).encode()))
    assert body["messages"] == original["messages"]
    assert body["tools"] == original["tools"]
    assert body["temperature"] == 0 and body["top_p"] == 1


def test_greedy_proxy_transmits_correct_content_length():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen.append(json.loads(body))
            response = b'{"choices":[],"usage":{}}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    proxy = GreedyProxy(f"http://127.0.0.1:{server.server_port}").start()
    try:
        request = urllib.request.Request(
            proxy.url + "/v1/chat/completions",
            data=b'{"model":"fixture","temperature":0.7}',
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 200
        assert seen[0]["temperature"] == 0 and seen[0]["top_p"] == 1
    finally:
        proxy.stop()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_missing_or_non_greedy_agent_evidence_fails_closed():
    row = {
        "type": "run",
        "requests": 1,
        "task": "fixture",
        "request_log": [{"sampling": {"temperature": 0, "top_p": 1}}],
    }
    assert validate([row, {"complete": True}], "fixture") == (True, "")
    assert not validate([row], "fixture")[0]
    assert not validate([row, {"complete": True}], "different")[0]
    row["request_log"][0]["sampling"]["temperature"] = 0.7
    assert not validate([row, {"complete": True}], "fixture")[0]
