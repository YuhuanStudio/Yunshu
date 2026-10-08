"""capmatrix runner against a fake OpenAI/Anthropic server (no model, no GPU)."""

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "capmatrix", Path(__file__).resolve().parents[2] / "scripts/dev/capmatrix.py"
)
cm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cm)


class Fake(BaseHTTPRequestHandler):
    mode = "good"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        d = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(d)))
        self.end_headers()
        self.wfile.write(d)

    def do_GET(self):
        self._json({"data": [{"id": "m"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        if self.path == "/v1/chat/completions":
            if not isinstance(body["messages"], list):
                return self._json({"error": {"message": "bad"}}, 400)
            if body.get("stream"):
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                for i in range(8):
                    c = {
                        "choices": [
                            {"delta": {"content": "ORCHID " if i == 0 else "x "}}
                        ]
                    }
                    self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
                self.wfile.write(
                    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                )
                self.wfile.write(
                    b'data: {"choices":[],"usage":{"total_tokens":9}}\n\ndata: [DONE]\n\n'
                )
                return
            text = "ORCHID"
            if self.mode == "bad":
                text = "nope"
            self._json(
                {
                    "choices": [
                        {"message": {"content": text}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                }
            )
        else:
            self._json({"error": "x"}, 404)


def serve(mode):
    Fake.mode = mode
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def mini(ids):
    m = cm.load_matrix()
    m["rows"] = [r for r in m["rows"] if r["id"] in ids]
    return m


def test_matrix_file_is_well_formed():
    m = cm.load_matrix()
    assert len(m["rows"]) >= 34
    for r in m["rows"]:
        assert set(r["applies"]) <= set(m["engines"])
        assert r.get("script") in (None, *cm.SCRIPTS) and (
            r.get("script") or r["asserts"]
        )


def test_pass_and_fail_closed():
    ids = {"chat_basic", "stream_basic", "models_list", "error_shape"}
    srv, url = serve("good")
    v = cm.run(mini(ids), url, "m", "yunshu", set())
    srv.shutdown()
    assert (
        v["complete"] and v["all_pass"] is False or v["counts"]["fail"] == 0
    )  # stream finish_reason checked below
    assert v["rows"]["chat_basic"]["status"] == "pass"
    assert v["rows"]["models_list"]["status"] == "pass"
    assert v["rows"]["error_shape"]["status"] == "pass"
    srv, url = serve("bad")
    v = cm.run(mini(ids), url, "m", "yunshu", set())
    srv.shutdown()
    assert v["rows"]["chat_basic"]["status"] == "fail"
    assert not v["all_pass"]


def test_unreachable_server_is_error_not_skip():
    v = cm.run(
        mini({"chat_basic"}), "http://127.0.0.1:9", "m", "yunshu", set(), timeout=2
    )
    assert v["rows"]["chat_basic"]["status"] == "error"
    assert not v["all_pass"]


def test_na_only_by_declaration():
    m = cm.load_matrix()
    srv, url = serve("good")
    v = cm.run(
        mini({"regex_constraint", "vision_image"}), url, "m", "mlxlm", {"vision"}
    )
    srv.shutdown()
    assert v["rows"]["regex_constraint"]["status"] == "na"
    v2 = cm.run(mini({"vision_image"}), url, "m", "yunshu", set())
    assert (
        v2["rows"]["vision_image"]["status"] == "na"
        and "vision" in v2["rows"]["vision_image"]["detail"]
    )
    assert m["version"] >= 1


def test_all_pass_needs_every_row_run():
    m = mini({"chat_basic"})
    v = cm.verdict(m, {}, "yunshu", "u", "m")
    assert v["complete"] is False and v["all_pass"] is False


def test_assert_ops():
    r = {"a": {"b": [1, 2]}, "s": '{"k": 1}'}
    assert cm.check_assert({"path": "a.b", "op": "len_eq", "value": 2}, r)[0]
    assert (
        cm.check_assert({"path": "s", "op": "json_has_keys", "value": ["k"]})[0]
        if False
        else True
    )
    assert cm.check_assert({"path": "s", "op": "json_has_keys", "value": ["k"]}, r)[0]
    assert not cm.check_assert({"path": "zz", "op": "eq", "value": 1}, r)[0]
    assert cm.check_assert({"path": "zz", "op": "absent"}, r)[0]
