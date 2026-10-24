"""CPU check of scripts/research/apigaps_real.py helpers (no server, no model)."""

import importlib.util
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

p = pathlib.Path(__file__).parents[2] / "scripts/research/apigaps_real.py"
spec = importlib.util.spec_from_file_location("apigaps_real", p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_expect_raises_fail():
    mod.expect(True, "x")
    with pytest.raises(mod.Fail):
        mod.expect(False, "boom")


def test_wait_ready_fails_fast_when_server_exited():
    class P:
        returncode = 3

        def poll(self):
            return 3

    with pytest.raises(mod.Fail):
        mod.wait_ready("http://127.0.0.1:1", P(), secs=5)


def _stub(need_token):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            ok = (not need_token) or self.headers.get("Authorization") == (
                f"Bearer {mod.TOKEN}"
            )
            self.send_response(200 if ok else 401)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.mark.parametrize("token", [None, mod.TOKEN])
def test_check_host_matches_the_auth_mode(token):
    srv = _stub(need_token=token is not None)
    try:
        res = {}
        mod.check_host(f"http://127.0.0.1:{srv.server_port}", res, token)
        assert res["host_without_token"] == (401 if token else 200)
    finally:
        srv.shutdown()


def test_check_host_fails_when_a_tokenless_server_refuses():
    srv = _stub(need_token=True)
    try:
        with pytest.raises(mod.Fail):
            mod.check_host(f"http://127.0.0.1:{srv.server_port}", {}, None)
    finally:
        srv.shutdown()
