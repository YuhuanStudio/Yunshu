"""The observe real-check script judges a fake engine correctly (CPU only)."""

from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "observe_check", ROOT / "scripts/research/observe_check.py"
)
oc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oc)


def _server(observe=True, missing_route=None, telemetry="ok"):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            path = self.path.split("?")[0]
            code, body = 200, {}
            if path.startswith("/debug/"):
                if path.rsplit("/", 1)[1] == missing_route:
                    code = 404
            elif path == "/v1/yunshu/status":
                body = {"observe": observe}
            elif path == "/v1/yunshu/config":
                body = {
                    "settings": [
                        {"name": n, "source": "observe" if observe else "default"}
                        for n in oc.OBSERVED
                    ]
                }
            elif path == "/v1/yunshu/host":
                body = {"telemetry": {"state": telemetry}}
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _home(tmp_path):
    logs = tmp_path / ".yunshu" / "logs"
    logs.mkdir(parents=True)
    (logs / "serve.jsonl").write_text('{"a":1}\n')
    (logs / "audit.log").write_text("")
    (logs / "stream_capture.jsonl").write_text("{}\n")
    return tmp_path


LP = {"choices": [{"logprobs": {"content": [{"token_id": 5}]}}]}


def test_all_live_passes(tmp_path):
    srv = _server()
    base = f"http://127.0.0.1:{srv.server_port}"
    assert oc.verify(base, _home(tmp_path), [[{"request_id": "r"}]], LP) == []


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        ({"missing_route": "kv-cache"}, "/debug/kv-cache -> 404"),
        ({"observe": False}, "observe=False"),
        ({"telemetry": "unknown"}, "telemetry state=unknown"),
    ],
)
def test_each_missing_feature_is_reported(tmp_path, kwargs, needle):
    srv = _server(**kwargs)
    base = f"http://127.0.0.1:{srv.server_port}"
    bad = oc.verify(base, _home(tmp_path), [[{}]], LP)
    assert any(needle in b for b in bad), bad


def test_missing_files_ids_and_in_flight(tmp_path):
    srv = _server()
    base = f"http://127.0.0.1:{srv.server_port}"
    bad = oc.verify(base, tmp_path, [], {"choices": [{"logprobs": {"content": [{}]}}]})
    text = " | ".join(bad)
    for needle in ("serve log", "audit.log", "stream_capture", "token_id", "in-flight"):
        assert needle in text, text
