"""An upstream the server calls on the model's behalf (an MCP server, a page for web_fetch, a
search provider) fails mid-request: the answer is a tool error the model can read and the stream
still ends with exactly one terminal, never a 500, a hang or an empty body."""

from __future__ import annotations

import http.server
import json
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_gateway.routers import anthropic
from yunshu_gateway.server_tools import search

from ..server_tools_helpers import FakeMcpHttp, FakeSearch, ScriptedInner, tiny_mcp
from .harness import assert_one_terminal

TU = "toolu_model1"


def call(name, inp, id_=TU):
    return {"type": "tool_use", "id": id_, "name": name, "input": inp}


def text(t):
    return {"type": "text", "text": t}


@pytest.fixture
def make(monkeypatch):
    search.set_provider_for_tests(None)

    def _make(rounds):
        inner = ScriptedInner(rounds)
        monkeypatch.setattr(anthropic, "create_message", inner)
        app = FastAPI()
        app.include_router(anthropic.router, prefix="/v1")
        return TestClient(app), inner

    yield _make
    search.set_provider_for_tests(None)


def _mcp_body(url, stream):
    return {
        "model": "m",
        "max_tokens": 64,
        "stream": stream,
        "messages": [{"role": "user", "content": "add"}],
        "mcp_servers": [{"type": "url", "url": url, "name": "tiny"}],
    }


@pytest.fixture
def flaky_mcp(monkeypatch):
    """A fake MCP server whose ``tools/call`` fails the way a real upstream does."""
    srv = FakeMcpHttp()
    real = tiny_mcp.handle
    mode = {"on": "ok"}

    def handle(msg):
        if msg.get("method") == "tools/call":
            if mode["on"] == "drop":
                raise ConnectionError(
                    "upstream died"
                )  # the handler thread ends: no reply
            if mode["on"] == "slow":
                time.sleep(3.0)
        return real(msg)

    monkeypatch.setattr(tiny_mcp, "handle", handle)
    monkeypatch.setenv("YUNSHU_MCP_CONNECTOR_TIMEOUT", "1")
    yield srv, mode
    srv.stop()


@pytest.mark.parametrize("fault", ["drop", "slow"])
@pytest.mark.parametrize("stream", [True, False])
def test_mcp_tool_call_failure_is_a_tool_error_not_a_hang(
    make, flaky_mcp, fault, stream
):
    srv, mode = flaky_mcp
    mode["on"] = fault
    c, _ = make(
        [
            ([call("tiny__add", {"a": 2, "b": 40})], "tool_use"),
            ([text("The tool failed.")], "end_turn"),
        ]
    )
    t0 = time.monotonic()
    r = c.post("/v1/messages", json=_mcp_body(srv.url, stream))
    assert (
        time.monotonic() - t0 < 6.0
    )  # the connector timeout, not the upstream's 3 s sleep
    assert r.status_code == 200
    if stream:
        assert assert_one_terminal("anthropic", r.text) == "message_stop"
        assert '"is_error": true' in r.text
        assert "The tool failed." in r.text
    else:
        content = r.json()["content"]
        res = next(b for b in content if b["type"] == "mcp_tool_result")
        assert res["is_error"] is True
        assert content[-1]["type"] == "text"


@pytest.fixture
def page(monkeypatch):
    """A loopback page server: ``status`` and ``delay`` pick the fault."""
    cfg = {"status": 200, "delay": 0.0}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            time.sleep(cfg["delay"])
            b = b"<html><title>Doc</title><body>hello</body></html>"
            self.send_response(cfg["status"])
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("YUNSHU_WEB_FETCH_ALLOW_PRIVATE", "1")
    monkeypatch.setenv("YUNSHU_WEB_FETCH_TIMEOUT", "1")
    yield f"http://127.0.0.1:{srv.server_address[1]}/d", cfg
    srv.shutdown()
    srv.server_close()


FETCH = {"type": "web_fetch_20250910", "name": "web_fetch"}


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("fault", ["refused", "http500", "slow"])
def test_web_fetch_failure_is_a_tool_error_result(make, page, fault, stream):
    url, cfg = page
    if fault == "refused":
        url = "http://127.0.0.1:1/never"
    elif fault == "http500":
        cfg["status"] = 500
    else:
        cfg["delay"] = 3.0
    c, _ = make(
        [
            ([call("web_fetch", {"url": url})], "tool_use"),
            ([text("could not read it")], "end_turn"),
        ]
    )
    t0 = time.monotonic()
    r = c.post(
        "/v1/messages",
        json={
            "model": "m",
            "max_tokens": 64,
            "stream": stream,
            "messages": [{"role": "user", "content": "read it"}],
            "tools": [FETCH],
        },
    )
    assert time.monotonic() - t0 < 6.0
    assert r.status_code == 200
    if stream:
        assert assert_one_terminal("anthropic", r.text) == "message_stop"
        events = [
            json.loads(d)
            for ln in r.text.split("\n")
            if ln.startswith("data: ")
            for d in [ln[6:]]
        ]
        errs = [
            e["content_block"]["content"]
            for e in events
            if e.get("type") == "content_block_start"
            and e["content_block"].get("type") == "web_fetch_tool_result"
        ]
        assert errs and errs[0]["type"] == "web_fetch_tool_result_error"
    else:
        content = r.json()["content"]
        res = next(b for b in content if b["type"] == "web_fetch_tool_result")
        assert res["content"]["type"] == "web_fetch_tool_result_error"
        assert res["content"]["error_code"]
        assert content[-1]["type"] == "text"


@pytest.mark.parametrize("stream", [True, False])
def test_search_provider_failure_is_a_tool_error_result(make, stream):
    search.set_provider_for_tests(
        FakeSearch(error=search.SearchError("unavailable", "provider is down"))
    )
    c, _ = make(
        [
            ([call("web_search", {"query": "x"})], "tool_use"),
            ([text("search failed")], "end_turn"),
        ]
    )
    r = c.post(
        "/v1/messages",
        json={
            "model": "m",
            "max_tokens": 64,
            "stream": stream,
            "messages": [{"role": "user", "content": "find"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        },
    )
    assert r.status_code == 200
    if stream:
        assert assert_one_terminal("anthropic", r.text) == "message_stop"
        assert "web_search_tool_result_error" in r.text
    else:
        content = r.json()["content"]
        res = next(b for b in content if b["type"] == "web_search_tool_result")
        assert res["content"]["type"] == "web_search_tool_result_error"
