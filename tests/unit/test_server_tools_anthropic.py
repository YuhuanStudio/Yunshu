"""Anthropic Messages server tools: web_search, web_fetch and the MCP connector, run inside the
generation loop (scripted generation handler, fake provider, fake MCP server)."""

from __future__ import annotations

import json
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_gateway.routers import anthropic
from yunshu_gateway.server_tools import search
from yunshu_gateway.server_tools.anthropic_loop import normalize_history

from .server_tools_helpers import FakeMcpHttp, FakeSearch, ScriptedInner

TU = "toolu_model1"


def call(name, inp, id_=TU):
    return {"type": "tool_use", "id": id_, "name": name, "input": inp}


def text(t):
    return {"type": "text", "text": t}


@pytest.fixture
def make(monkeypatch):
    """client(rounds) -> (TestClient, ScriptedInner)"""
    search.set_provider_for_tests(None)

    def _make(rounds):
        inner = ScriptedInner(rounds)
        monkeypatch.setattr(anthropic, "create_message", inner)
        app = FastAPI()
        app.include_router(anthropic.router, prefix="/v1")
        return TestClient(app), inner

    yield _make
    search.set_provider_for_tests(None)


def body(tools, **kw):
    return {
        "model": "m",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "find yunshu"}],
        "tools": tools,
        **kw,
    }


WS = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}


def test_web_search_non_stream(make):
    search.set_provider_for_tests(fake := FakeSearch())
    c, inner = make(
        [
            ([call("web_search", {"query": "yunshu"})], "tool_use"),
            ([text("Yunshu is a local engine [1].")], "end_turn"),
        ]
    )
    r = c.post("/v1/messages", json=body([WS]))
    assert r.status_code == 200, r.text
    m = r.json()
    kinds = [b["type"] for b in m["content"]]
    assert kinds == ["server_tool_use", "web_search_tool_result", "text"]
    stu, res, txt = m["content"]
    assert (
        stu["name"] == "web_search"
        and stu["input"] == {"query": "yunshu"}
        and stu["id"].startswith("srvtoolu_")
    )
    assert res["tool_use_id"] == stu["id"]
    assert [x["url"] for x in res["content"]] == [
        "https://example.com/yunshu",
        "https://ml-explore.github.io/mlx",
    ]
    assert all(
        x["type"] == "web_search_result" and x["encrypted_content"]
        for x in res["content"]
    )
    cit = txt["citations"]
    assert len(cit) == 1 and cit[0]["type"] == "web_search_result_location"
    assert cit[0]["url"] == "https://example.com/yunshu" and cit[0][
        "cited_text"
    ].startswith("Yunshu is")
    assert m["stop_reason"] == "end_turn"
    assert m["usage"]["server_tool_use"] == {"web_search_requests": 1}
    assert (
        m["usage"]["input_tokens"] == 20 and m["usage"]["output_tokens"] == 10
    )  # summed over rounds
    assert m["x_yunshu"]["server_tools"]["rounds"] == 2
    assert len(m["x_yunshu"]["server_tools"]["round_usage"]) == 2
    assert fake.calls[0]["query"] == "yunshu"
    # round 2 saw the tool call and its result, and no server-tool declaration leaked as a type
    second = inner.requests[1]
    roles = [x.role for x in second.messages]
    assert roles == ["user", "assistant", "user"]
    assert second.messages[1].content[0]["type"] == "tool_use"
    tr = second.messages[2].content[0]
    assert (
        tr["type"] == "tool_result"
        and "[1] Yunshu" in tr["content"]
        and "[2] MLX" in tr["content"]
    )
    first_tools = inner.requests[0].tools
    assert [t.name for t in first_tools] == ["web_search"] and first_tools[
        0
    ].input_schema["required"] == ["query"]
    assert inner.requests[0].stream is True and inner.requests[0].mcp_servers is None


def test_web_search_stream_events(make):
    search.set_provider_for_tests(FakeSearch())
    c, _ = make(
        [
            ([text("Let me look."), call("web_search", {"query": "q"})], "tool_use"),
            ([text("Result [2].")], "end_turn"),
        ]
    )
    with c.stream("POST", "/v1/messages", json=body([WS], stream=True)) as r:
        raw = r.read().decode()
    events = [
        (blk.split("\n")[0][7:], json.loads(blk.split("\n")[1][6:]))
        for blk in raw.strip().split("\n\n")
    ]
    names = [n for n, _ in events]
    assert names[0] == "message_start" and names[-2:] == [
        "message_delta",
        "message_stop",
    ]
    starts = [
        (e["index"], e["content_block"]["type"])
        for n, e in events
        if n == "content_block_start"
    ]
    assert starts == [
        (0, "text"),
        (1, "server_tool_use"),
        (2, "web_search_tool_result"),
        (3, "text"),
    ]
    # indices are dense and every block is closed in order
    assert [e["index"] for n, e in events if n == "content_block_stop"] == [0, 1, 2, 3]
    deltas = [e for n, e in events if n == "content_block_delta" and e["index"] == 1]
    assert json.loads("".join(d["delta"]["partial_json"] for d in deltas)) == {
        "query": "q"
    }
    cit = [
        e["delta"]
        for n, e in events
        if n == "content_block_delta" and e["delta"]["type"] == "citations_delta"
    ]
    assert (
        len(cit) == 1
        and cit[0]["citation"]["url"] == "https://ml-explore.github.io/mlx"
    )
    last = events[-2][1]
    assert last["usage"]["server_tool_use"]["web_search_requests"] == 1
    assert last["delta"]["stop_reason"] == "end_turn"


def test_no_provider_is_unavailable_with_hint(make, monkeypatch):
    for k in (
        "YUNSHU_SEARXNG_URL",
        "YUNSHU_BRAVE_API_KEY",
        "YUNSHU_TAVILY_API_KEY",
        "YUNSHU_EXA_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)
    c, inner = make(
        [
            ([call("web_search", {"query": "q"})], "tool_use"),
            ([text("I cannot search.")], "end_turn"),
        ]
    )
    m = c.post("/v1/messages", json=body([WS])).json()
    res = m["content"][1]
    assert res["content"] == {
        "type": "web_search_tool_result_error",
        "error_code": "unavailable",
    }
    tool = m["x_yunshu"]["server_tools"]["tools"][0]
    assert "YUNSHU_SEARXNG_URL" in tool["hint"] and tool["ok"] is False
    # the model is told why, so it can explain the setup to the user
    assert "YUNSHU_SEARXNG_URL" in inner.requests[1].messages[2].content[0]["content"]
    assert (
        "server_tool_use" not in m["usage"]
        or m["usage"]["server_tool_use"].get("web_search_requests") == 1
    )


def test_max_uses_exceeded(make):
    search.set_provider_for_tests(FakeSearch())
    tool = {**WS, "max_uses": 1}
    c, _ = make(
        [
            ([call("web_search", {"query": "a"}, "t1")], "tool_use"),
            ([call("web_search", {"query": "b"}, "t2")], "tool_use"),
            ([text("ok")], "end_turn"),
        ]
    )
    m = c.post("/v1/messages", json=body([tool])).json()
    kinds = [b["type"] for b in m["content"]]
    assert kinds == [
        "server_tool_use",
        "web_search_tool_result",
        "server_tool_use",
        "web_search_tool_result",
        "text",
    ]
    assert isinstance(m["content"][1]["content"], list)
    assert m["content"][3]["content"] == {
        "type": "web_search_tool_result_error",
        "error_code": "max_uses_exceeded",
    }
    assert m["usage"]["server_tool_use"] == {"web_search_requests": 1}


def test_domain_filters_and_user_location_reach_provider(make):
    fake = FakeSearch()
    search.set_provider_for_tests(fake)
    tool = {
        **WS,
        "allowed_domains": ["example.com"],
        "user_location": {"type": "approximate", "country": "TW"},
    }
    c, _ = make(
        [
            ([call("web_search", {"query": "q"})], "tool_use"),
            ([text("done")], "end_turn"),
        ]
    )
    m = c.post("/v1/messages", json=body([tool])).json()
    assert [x["url"] for x in m["content"][1]["content"]] == [
        "https://example.com/yunshu"
    ]
    assert (
        fake.calls[0]["allowed"] == ["example.com"]
        and fake.calls[0]["loc"]["country"] == "TW"
    )


def test_client_tool_call_ends_turn_with_tool_use(make):
    search.set_provider_for_tests(FakeSearch())
    client_tool = {
        "name": "get_time",
        "description": "time",
        "input_schema": {"type": "object", "properties": {}},
    }
    c, inner = make(
        [
            (
                [call("web_search", {"query": "q"}, "t1"), call("get_time", {}, "t2")],
                "tool_use",
            )
        ]
    )
    m = c.post("/v1/messages", json=body([WS, client_tool])).json()
    kinds = [b["type"] for b in m["content"]]
    assert (
        kinds == ["server_tool_use", "tool_use", "web_search_tool_result"]
        or kinds[-1] == "tool_use"
        or "tool_use" in kinds
    )
    assert m["stop_reason"] == "tool_use"
    assert {t.name for t in inner.requests[0].tools} == {"web_search", "get_time"}
    assert len(inner.requests) == 1


def test_plain_requests_bypass_the_loop():
    from yunshu_gateway.server_tools.anthropic_loop import has_server_tools

    base = {
        "model": "m",
        "max_tokens": 5,
        "messages": [{"role": "user", "content": "x"}],
    }
    plain = anthropic.AnthropicMessagesRequest(
        **base, tools=[{"name": "f", "input_schema": {"type": "object"}}]
    )
    assert not has_server_tools(plain)
    assert not has_server_tools(anthropic.AnthropicMessagesRequest(**base))
    assert has_server_tools(
        anthropic.AnthropicMessagesRequest(
            **base, tools=[{"type": "web_fetch_20250910", "name": "web_fetch"}]
        )
    )
    assert has_server_tools(
        anthropic.AnthropicMessagesRequest(
            **base, mcp_servers=[{"name": "a", "url": "http://x"}]
        )
    )


def test_pause_turn_after_iteration_limit(make, monkeypatch):
    monkeypatch.setenv("YUNSHU_SERVER_TOOL_MAX_ITERATIONS", "2")
    search.set_provider_for_tests(FakeSearch())
    c, inner = make(
        [
            ([call("web_search", {"query": "a"}, "t1")], "tool_use"),
            ([call("web_search", {"query": "b"}, "t2")], "tool_use"),
            ([text("never")], "end_turn"),
        ]
    )
    m = c.post("/v1/messages", json=body([WS])).json()
    assert m["stop_reason"] == "pause_turn" and len(inner.requests) == 2


def test_history_round_trip_of_server_blocks(make):
    """A later request carries the earlier server_tool_use / result blocks; the model sees plain turns."""
    search.set_provider_for_tests(FakeSearch())
    c, inner = make(
        [
            ([call("web_search", {"query": "q"})], "tool_use"),
            ([text("A [1].")], "end_turn"),
        ]
    )
    first = c.post("/v1/messages", json=body([WS])).json()
    c2, inner2 = make([([text("second answer")], "end_turn")])
    msgs = [
        {"role": "user", "content": "find yunshu"},
        {"role": "assistant", "content": first["content"]},
        {"role": "user", "content": "and more?"},
    ]
    r = c2.post("/v1/messages", json={**body([WS]), "messages": msgs})
    assert r.status_code == 200
    seen = inner2.requests[0].messages
    assert [m.role for m in seen] == ["user", "assistant", "user", "assistant", "user"]
    assert (
        seen[1].content[0]["type"] == "tool_use"
        and seen[2].content[0]["type"] == "tool_result"
    )
    assert (
        "Yunshu is a local inference engine" in seen[2].content[0]["content"]
    )  # rebuilt from encrypted_content
    assert seen[3].content[0]["type"] == "text"


def test_normalize_history_handles_error_results():
    msgs = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "s1",
                    "name": "web_search",
                    "input": {"query": "q"},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "s1",
                    "content": {
                        "type": "web_search_tool_result_error",
                        "error_code": "unavailable",
                    },
                },
                {"type": "text", "text": "sorry"},
            ],
        }
    ]
    out = normalize_history(msgs)
    assert [m["role"] for m in out] == ["assistant", "user", "assistant"]
    assert "unavailable" in out[1]["content"][0]["content"]


# ── web_fetch ────────────────────────────────────────────────────────────────


def test_web_fetch_blocked_then_allowed(make, monkeypatch):
    import http.server

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            b = b"<html><title>Doc</title><body>hello from the page</body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/d"
    tool = {"type": "web_fetch_20250910", "name": "web_fetch"}
    try:
        c, inner = make(
            [
                ([call("web_fetch", {"url": url})], "tool_use"),
                ([text("sorry")], "end_turn"),
            ]
        )
        m = c.post("/v1/messages", json=body([tool])).json()
        assert (
            m["content"][0]["type"] == "server_tool_use"
            and m["content"][0]["name"] == "web_fetch"
        )
        assert m["content"][1]["content"] == {
            "type": "web_fetch_tool_result_error",
            "error_code": "url_not_allowed",
        }
        monkeypatch.setenv("YUNSHU_WEB_FETCH_ALLOW_PRIVATE", "1")
        c, inner = make(
            [
                ([call("web_fetch", {"url": url})], "tool_use"),
                ([text("It says hello.")], "end_turn"),
            ]
        )
        m = c.post("/v1/messages", json=body([tool])).json()
        res = m["content"][1]
        assert (
            res["type"] == "web_fetch_tool_result"
            and res["content"]["type"] == "web_fetch_result"
        )
        assert res["content"]["url"] == url and res["content"]["retrieved_at"]
        doc = res["content"]["content"]
        assert doc["type"] == "document" and doc["title"] == "Doc"
        assert doc["source"] == {
            "type": "text",
            "media_type": "text/plain",
            "data": "hello from the page",
        }
        assert m["usage"]["server_tool_use"] == {"web_fetch_requests": 1}
        assert (
            "hello from the page" in inner.requests[1].messages[2].content[0]["content"]
        )
    finally:
        srv.shutdown()
        srv.server_close()


# ── MCP connector ────────────────────────────────────────────────────────────


def test_mcp_servers_connector(make):
    srv = FakeMcpHttp(token="tok")
    try:
        c, inner = make(
            [
                ([call("tiny__add", {"a": 2, "b": 40})], "tool_use"),
                ([text("It is 42.")], "end_turn"),
            ]
        )
        req = body(
            [],
            mcp_servers=[
                {
                    "type": "url",
                    "url": srv.url,
                    "name": "tiny",
                    "authorization_token": "tok",
                }
            ],
        )
        req.pop("tools")
        m = c.post("/v1/messages", json=req).json()
        kinds = [b["type"] for b in m["content"]]
        assert kinds == ["mcp_tool_use", "mcp_tool_result", "text"]
        use, res = m["content"][:2]
        assert (
            use["name"] == "add"
            and use["server_name"] == "tiny"
            and use["input"] == {"a": 2, "b": 40}
        )
        assert use["id"].startswith("mcptoolu_")
        assert res == {
            "type": "mcp_tool_result",
            "tool_use_id": use["id"],
            "is_error": False,
            "content": [{"type": "text", "text": "42"}],
        }
        assert {t.name for t in inner.requests[0].tools} == {"tiny__echo", "tiny__add"}
        assert inner.requests[1].messages[2].content[0]["content"] == "42"
        assert any(h.get("Authorization") == "Bearer tok" for h in srv.headers)
    finally:
        srv.stop()


def test_mcp_tool_filtering_and_toolset(make):
    srv = FakeMcpHttp()
    try:
        # legacy tool_configuration.allowed_tools
        c, inner = make([([text("x")], "end_turn")])
        req = {
            "model": "m",
            "max_tokens": 9,
            "messages": [{"role": "user", "content": "hi"}],
            "mcp_servers": [
                {
                    "type": "url",
                    "url": srv.url,
                    "name": "tiny",
                    "tool_configuration": {"enabled": True, "allowed_tools": ["echo"]},
                }
            ],
        }
        assert c.post("/v1/messages", json=req).status_code == 200
        assert [t.name for t in inner.requests[0].tools] == ["tiny__echo"]
        # newer mcp_toolset with per-tool disable
        c, inner = make([([text("x")], "end_turn")])
        req = {
            "model": "m",
            "max_tokens": 9,
            "messages": [{"role": "user", "content": "hi"}],
            "mcp_servers": [{"type": "url", "url": srv.url, "name": "tiny"}],
            "tools": [
                {
                    "type": "mcp_toolset",
                    "mcp_server_name": "tiny",
                    "configs": {"echo": {"enabled": False}},
                }
            ],
        }
        assert c.post("/v1/messages", json=req).status_code == 200
        assert [t.name for t in inner.requests[0].tools] == ["tiny__add"]
    finally:
        srv.stop()


def test_mcp_connection_failure_is_a_400(make):
    c, _ = make([])
    req = {
        "model": "m",
        "max_tokens": 9,
        "messages": [{"role": "user", "content": "hi"}],
        "mcp_servers": [
            {"type": "url", "url": "http://127.0.0.1:1/mcp", "name": "dead"}
        ],
    }
    r = c.post("/v1/messages", json=req)
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
    assert "dead" in r.json()["error"]["message"]


def test_mcp_connector_can_be_disabled(make, monkeypatch):
    monkeypatch.setenv("YUNSHU_MCP_CONNECTOR", "0")
    c, _ = make([])
    req = {
        "model": "m",
        "max_tokens": 9,
        "messages": [{"role": "user", "content": "hi"}],
        "mcp_servers": [{"type": "url", "url": "http://127.0.0.1:1/mcp", "name": "x"}],
    }
    assert c.post("/v1/messages", json=req).status_code == 400


def test_mcp_tool_error_and_history(make):
    srv = FakeMcpHttp()
    try:
        c, _ = make(
            [([call("tiny__nope", {})], "tool_use"), ([text("failed")], "end_turn")]
        )
        req = {
            "model": "m",
            "max_tokens": 9,
            "messages": [{"role": "user", "content": "hi"}],
            "mcp_servers": [{"type": "url", "url": srv.url, "name": "tiny"}],
        }
        m = c.post("/v1/messages", json=req).json()
        # the model called a name that is not an MCP tool: the client sees it as a plain tool_use
        assert m["content"][0]["type"] == "tool_use" and m["stop_reason"] == "tool_use"
    finally:
        srv.stop()
    out = normalize_history(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "mcp_tool_use",
                        "id": "m1",
                        "name": "add",
                        "server_name": "tiny",
                        "input": {"a": 1},
                    },
                    {
                        "type": "mcp_tool_result",
                        "tool_use_id": "m1",
                        "is_error": False,
                        "content": [{"type": "text", "text": "1"}],
                    },
                ],
            }
        ]
    )
    assert (
        out[0]["content"][0]["name"] == "tiny__add"
        and out[1]["content"][0]["content"] == "1"
    )


def test_thinking_adaptive_and_extra_fields_validate():
    """Claude Code sends thinking {type: adaptive}, output_config, context_management."""
    r = anthropic.AnthropicMessagesRequest(
        model="m",
        max_tokens=100,
        messages=[{"role": "user", "content": "x"}],
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        context_management={"edits": [{"type": "clear_thinking_20251015"}]},
        tools=[
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5,
                "user_location": {"type": "approximate", "country": "US"},
            },
            {"type": "mcp_toolset", "mcp_server_name": "s"},
        ],
    )
    assert r.tools[0].model_dump()["max_uses"] == 5 and r.tools[1].name == ""


def test_blank_text_blocks_are_not_emitted(make):
    """The template's newlines after </think> must not become text blocks or leading whitespace."""
    search.set_provider_for_tests(FakeSearch())
    c, _ = make(
        [
            (
                [
                    {"type": "thinking", "thinking": "hmm"},
                    text("\n\n"),
                    call("web_search", {"query": "q"}),
                ],
                "tool_use",
            ),
            ([text("\n\nAnswer [1].")], "end_turn"),
        ]
    )
    m = c.post("/v1/messages", json=body([WS])).json()
    assert [b["type"] for b in m["content"]] == [
        "thinking",
        "server_tool_use",
        "web_search_tool_result",
        "text",
    ]
    assert m["content"][-1]["text"] == "Answer [1]."


def test_mid_conversation_system_message_with_image_request_does_not_crash(monkeypatch):
    """Claude Code sends role=system reminders mid-conversation; with an image in the request every
    message's content became a parts list and joining the lifted system text raised a 500."""
    from fastapi import HTTPException

    async def no_engine(model_id):
        raise HTTPException(status_code=404, detail="no engine in this test")

    monkeypatch.setattr(anthropic, "_resolve_engine", no_engine)
    app = FastAPI()
    app.include_router(anthropic.router, prefix="/v1")
    png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    r = TestClient(app, raise_server_exceptions=False).post(
        "/v1/messages",
        json={
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": png,
                            },
                        },
                        {"type": "text", "text": "look"},
                    ],
                },
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "<reminder>7 left</reminder>"}
                    ],
                },
            ],
        },
    )
    assert r.status_code == 404, (
        r.text
    )  # reached engine resolution: the system lift did not crash


def test_explicit_search_request_steers_only_the_first_round(make):
    """A local model sometimes answers from memory when told to search: an explicit ask forces round one."""
    search.set_provider_for_tests(FakeSearch())
    c, inner = make(
        [
            ([call("web_search", {"query": "q"})], "tool_use"),
            ([text("done")], "end_turn"),
        ]
    )
    r = c.post(
        "/v1/messages",
        json={
            "model": "m",
            "max_tokens": 50,
            "tools": [WS],
            "messages": [
                {
                    "role": "user",
                    "content": "Perform a web search for the query: yunshu",
                }
            ],
        },
    )
    assert r.status_code == 200
    assert inner.requests[0].tool_choice == {"type": "tool", "name": "web_search"}
    assert inner.requests[1].tool_choice == {"type": "auto"}
    # no explicit ask: the model decides
    c, inner = make([([text("hi")], "end_turn")])
    c.post(
        "/v1/messages",
        json={
            "model": "m",
            "max_tokens": 50,
            "tools": [WS],
            "messages": [{"role": "user", "content": "say hello"}],
        },
    )
    assert inner.requests[0].tool_choice is None


def test_explicit_tool_request_patterns():
    from yunshu_gateway.server_tools.runtime import ServerToolDef, explicit_tool_request

    defs = [
        ServerToolDef("web_search", "web_search", "", {}),
        ServerToolDef("web_fetch", "web_fetch", "", {}),
    ]
    assert (
        explicit_tool_request("Please search the web for mlx news", defs)
        == "web_search"
    )
    assert (
        explicit_tool_request("Perform a web search for the query: x", defs)
        == "web_search"
    )
    assert (
        explicit_tool_request("fetch https://example.com/a and summarize", defs)
        == "web_fetch"
    )
    assert explicit_tool_request("what is a web server?", defs) is None
    assert explicit_tool_request("tell me about https://example.com", defs) is None
    assert (
        explicit_tool_request("search the web", defs[1:]) is None
    )  # tool not declared
