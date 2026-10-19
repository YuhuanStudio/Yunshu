"""Competitor probe schema and HTTP preflight without MLX or model weights."""

import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from scripts.research.tf110_probe import post, validate


@pytest.fixture
def reply():
    return {
        "choices": [{"finish_reason": "length", "message": {"content": "fixture"}}],
        "usage": {"prompt_tokens": 1024, "completion_tokens": 128},
        "tensorfold": {"sampling": "greedy", "drafts": True, "token_sha256": "a" * 64},
        "speculative": {"rounds": 16, "accepted": 112, "drafted": 240},
    }


def test_validate_native_schema(reply):
    assert validate(reply, 128, True)["speculative"]["rounds"] == 16


@pytest.mark.parametrize(
    "field,value",
    [("choices", []), ("usage", {}), ("tensorfold", {}), ("speculative", {})],
)
def test_missing_evidence_fails(reply, field, value):
    broken = copy.deepcopy(reply)
    broken[field] = value
    with pytest.raises((ValueError, IndexError)):
        validate(broken, 128, True)


def test_plain_must_be_engaged(reply):
    with pytest.raises(ValueError):
        validate(reply, 128, False)
    reply["tensorfold"]["drafts"] = False
    assert validate(reply, 128, False)["runtime"]["drafts"] is False


def test_http_cpu_preflight(reply):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["temperature"] == 0
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(reply).encode())

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert (
            validate(post(server.server_port, {"temperature": 0}), 128, True)[
                "token_sha256"
            ]
            == "a" * 64
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
