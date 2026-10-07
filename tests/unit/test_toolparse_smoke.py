"""Probe CPU preflight: real HTTP transport with a fake server and fail-closed judge."""

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "toolparse_smoke", Path(__file__).parents[2] / "scripts/research/toolparse_smoke.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("stream"):
            reply = (
                "data: "
                + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_0",
                                            "function": {
                                                "name": "weather",
                                                "arguments": '{"city":"Taipei"}',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                )
                + "\n\ndata: [DONE]\n"
            )
        else:
            message = (
                {
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "function": {
                                "name": "weather",
                                "arguments": '{"city":"Taipei"}',
                            },
                        }
                    ]
                }
                if body.get("tools")
                else {
                    "content": 'free <result>{"city":"Taipei"}</result>',
                    "reasoning_content": "自由推理",
                }
            )
            reply = json.dumps({"choices": [{"message": message}]})
        self.send_response(200)
        self.end_headers()
        self.wfile.write(reply.encode())

    def log_message(self, *args):
        pass


def test_probe_with_fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        rows = probe.run(f"http://127.0.0.1:{server.server_port}", "fake")
        assert len(rows) == 3 and all(r["ok"] for r in rows)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_probe_fails_without_trigger_or_stream_arguments():
    with pytest.raises(ValueError):
        probe.judge("structural", {"choices": [{"message": {"content": "prose only"}}]})
    with pytest.raises(ValueError):
        probe.judge("stream", "data: [DONE]\n")
    args = probe.parser().parse_args(["--model", "fake", "--out", "out.json"])
    assert args.port == 18996
