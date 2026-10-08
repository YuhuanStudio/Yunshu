"""Responses API server tools: web_search and the remote MCP tool, run inside the generation loop
(scripted generation handler, fake search provider, fake MCP server)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_gateway.routers import responses
from yunshu_gateway.routers.responses import ResponsesRequest, _convert_to_messages
from yunshu_gateway.server_tools import search
from yunshu_gateway.server_tools.responses_loop import (
    annotations_for,
    function_tools,
)

from .server_tools_helpers import FakeMcpHttp, FakeSearch
from .server_tools_responses_helpers import (
    ScriptedResponses,
    call,
    parse_sse,
    reasoning,
    text,
)

CODEX_BODIES = (
    Path("/Users/yuhuan/Documents/YuhuanStudio/Yunshu")
    / "docs"
    / "research"
    / "runs"
    / "2026-09-30-agent-census"
    / "cx_shell"
    / "requests.jsonl"
)


@pytest.fixture
def make(monkeypatch):
    """make(rounds) -> (TestClient, ScriptedResponses)"""
    search.set_provider_for_tests(None)

    def _make(rounds):
        inner = ScriptedResponses(rounds)
        monkeypatch.setattr(responses, "create_response", inner)
        app = FastAPI()
        app.include_router(responses.router, prefix="/v1")
        return TestClient(app), inner

    yield _make
    search.set_provider_for_tests(None)


def body(tools, **kw):
    return {"model": "m", "input": "find yunshu", "tools": tools, "store": False, **kw}


WS = {"type": "web_search"}


def types(resp):
    return [i["type"] for i in resp["output"]]


# ── web_search ────────────────────────────────────────────────────────────────
def test_web_search_non_stream(make):
    search.set_provider_for_tests(fake := FakeSearch())
    c, inner = make(
        [
            [call("web_search", {"query": "yunshu"})],
            [text("Yunshu is a local engine [1]. MLX too [2].")],
        ]
    )
    r = c.post(
        "/v1/responses",
        json=body(
            [WS],
            include=["web_search_call.action.sources"],
        ),
    )
    assert r.status_code == 200, r.text
    m = r.json()
    assert types(m) == ["web_search_call", "message"]
    ws, msg = m["output"]
    assert ws["status"] == "completed" and ws["id"].startswith("ws_")
    assert ws["action"]["type"] == "search" and ws["action"]["query"] == "yunshu"
    assert ws["action"]["sources"] == [
        {"type": "url", "url": "https://example.com/yunshu"},
        {"type": "url", "url": "https://ml-explore.github.io/mlx"},
    ]
    t = msg["content"][0]["text"]
    anns = msg["content"][0]["annotations"]
    assert [a["url"] for a in anns] == [
        "https://example.com/yunshu",
        "https://ml-explore.github.io/mlx",
    ]
    for a in anns:
        assert a["type"] == "url_citation" and t[a["start_index"] : a["end_index"]] in (
            "[1]",
            "[2]",
        )
        assert a["title"]
    assert m["status"] == "completed" and m["object"] == "response"
    assert m["usage"]["input_tokens"] == 20 and m["usage"]["output_tokens"] == 10
    assert m["usage"]["total_tokens"] == 30
    assert m["usage"]["input_tokens_details"]["cached_tokens"] == 4
    assert m["usage"]["output_tokens_details"]["reasoning_tokens"] == 2
    assert m["x_yunshu"]["server_tools"]["rounds"] == 2
    assert m["tools"] == [WS]
    assert fake.calls[0]["query"] == "yunshu" and fake.calls[0]["allowed"] is None
    # round 1 saw web_search as a plain function tool and no web_search-typed tool
    first = inner.requests[0]
    assert [t.name for t in first.tools] == ["web_search"]
    assert first.stream is True and first.store is False
    assert inner.forced_ids[0] == m["id"] == inner.forced_ids[1]
    # round 2 got the call + result appended to the input
    items = inner.requests[1].input
    assert [i.type for i in items[-2:]] == ["function_call", "function_call_output"]
    assert "[1] Yunshu" in items[-1].output and items[-2].name == "web_search"


def test_web_search_allowed_domains_reach_provider(make):
    search.set_provider_for_tests(fake := FakeSearch())
    c, _ = make([[call("web_search", {"query": "q"})], [text("ok")]])
    tool = {**WS, "filters": {"allowed_domains": ["example.com"]}}
    m = c.post("/v1/responses", json=body([tool])).json()
    assert fake.calls[0]["allowed"] == ["example.com"]
    assert m["output"][0]["status"] == "completed"


def test_web_search_without_sources_by_default(make):
    search.set_provider_for_tests(FakeSearch())
    c, _ = make([[call("web_search", {"query": "q"})], [text("ok [1]")]])
    m = c.post("/v1/responses", json=body([WS])).json()
    assert "sources" not in m["output"][0]["action"]


def test_web_search_stream_events(make):
    search.set_provider_for_tests(FakeSearch())
    c, _ = make(
        [
            [
                reasoning("hmm"),
                text("Let me look."),
                call("web_search", {"query": "q"}),
            ],
            [text("Yunshu is local [1].")],
        ]
    )
    r = c.post("/v1/responses", json=body([WS], stream=True))
    assert r.headers["content-type"].startswith("text/event-stream")
    evs = parse_sse(r.text)
    names = [n for n, _ in evs]
    assert names[:2] == ["response.created", "response.in_progress"]
    assert names[-1] == "response.completed"
    assert r.text.rstrip().endswith("data: [DONE]")
    # dense sequence numbers, dense output indexes
    assert [d["sequence_number"] for _, d in evs] == list(range(len(evs)))
    idx = sorted({d["output_index"] for _, d in evs if "output_index" in d})
    assert idx == list(range(len(idx))) == [0, 1, 2, 3]
    # the web_search lifecycle, in order
    ws = [n for n in names if "web_search" in n]
    assert ws == [
        "response.web_search_call.in_progress",
        "response.web_search_call.searching",
        "response.web_search_call.completed",
    ]
    added = [d["item"]["type"] for n, d in evs if n == "response.output_item.added"]
    assert added == ["reasoning", "message", "web_search_call", "message"]
    # the swallowed function_call leaked no argument events
    assert not any("function_call_arguments" in n for n in names)
    # annotation event precedes the final output_text.done
    ai = names.index("response.output_text.annotation.added")
    ann = evs[ai][1]
    assert ann["annotation_index"] == 0 and ann["annotation"]["type"] == "url_citation"
    assert ann["output_index"] == 3 and ann["content_index"] == 0
    assert ai < len(names) - 1 - names[::-1].index("response.output_text.done")
    done = evs[-1][1]["response"]
    assert types(done) == ["reasoning", "message", "web_search_call", "message"]
    assert done["output"][3]["content"][0]["annotations"][0]["url"].startswith(
        "https://example.com"
    )
    assert done["usage"]["total_tokens"] == 30
    assert done["x_yunshu"]["server_tools"]["tools"][0]["ok"] is True
    part_done = [d for n, d in evs if n == "response.content_part.done"][-1]
    assert part_done["part"]["annotations"]


def test_web_search_no_provider_fails_item(make):
    c, inner = make([[call("web_search", {"query": "q"})], [text("I cannot search.")]])
    m = c.post("/v1/responses", json=body([WS])).json()
    assert m["output"][0]["type"] == "web_search_call"
    assert m["output"][0]["status"] == "failed"
    assert m["output"][0]["action"]["query"] == "q"
    tool_out = inner.requests[1].input[-1].output
    assert tool_out.startswith("Error: web search failed")
    assert m["status"] == "completed"
    assert "hint" in m["x_yunshu"]["server_tools"]["tools"][0]


def test_web_search_preview_alias_and_location(make):
    search.set_provider_for_tests(fake := FakeSearch())
    loc = {"type": "approximate", "country": "TW", "city": "Taipei"}
    c, _ = make([[call("web_search", {"query": "q"})], [text("x")]])
    r = c.post(
        "/v1/responses",
        json=body([{"type": "web_search_preview", "user_location": loc}]),
    )
    assert r.status_code == 200
    assert fake.calls[0]["loc"] == loc


# ── mcp ───────────────────────────────────────────────────────────────────────
def mcp_tool(srv, **kw):
    return {"type": "mcp", "server_label": "tiny", "server_url": srv.url, **kw}


def test_mcp_never_approval_flow(make):
    srv = FakeMcpHttp()
    try:
        c, inner = make([[call("tiny__add", {"a": 2, "b": 40})], [text("It is 42.")]])
        r = c.post(
            "/v1/responses",
            json=body([mcp_tool(srv, require_approval="never", headers={"X-A": "1"})]),
        )
        assert r.status_code == 200, r.text
        m = r.json()
        assert types(m) == ["mcp_list_tools", "mcp_call", "message"]
        lst, mc, _ = m["output"]
        assert lst["id"].startswith("mcpl_") and lst["server_label"] == "tiny"
        assert {t["name"] for t in lst["tools"]} == {"echo", "add"}
        assert all("input_schema" in t for t in lst["tools"])
        assert mc["id"].startswith("mcp_") and mc["name"] == "add"
        assert json.loads(mc["arguments"]) == {"a": 2, "b": 40}
        assert mc["output"] == "42" and mc["error"] is None
        assert mc["approval_request_id"] is None and mc["server_label"] == "tiny"
        # header/authorization are not echoed back
        assert "headers" not in m["tools"][0]
        assert srv.headers and any(h.get("X-A") == "1" for h in srv.headers)
        assert sorted(t.name for t in inner.requests[0].tools) == [
            "tiny__add",
            "tiny__echo",
        ]
    finally:
        srv.stop()


def test_mcp_stream_events(make):
    srv = FakeMcpHttp()
    try:
        c, _ = make([[call("tiny__echo", {"text": "hi"})], [text("done")]])
        r = c.post(
            "/v1/responses",
            json=body([mcp_tool(srv, require_approval="never")], stream=True),
        )
        evs = parse_sse(r.text)
        names = [n for n, _ in evs]
        sub = [n for n in names if "mcp" in n]
        assert sub == [
            "response.mcp_list_tools.in_progress",
            "response.mcp_list_tools.completed",
            "response.mcp_call.in_progress",
            "response.mcp_call_arguments.delta",
            "response.mcp_call_arguments.done",
            "response.mcp_call.completed",
        ]
        d = next(d for n, d in evs if n == "response.mcp_call_arguments.done")
        assert json.loads(d["arguments"]) == {"text": "hi"} and d["output_index"] == 1
        assert [d["sequence_number"] for _, d in evs] == list(range(len(evs)))
        assert not any("function_call_arguments" in n for n in names)
        assert names[-1] == "response.completed"
    finally:
        srv.stop()


def test_mcp_approval_two_step_chain(make):
    srv = FakeMcpHttp()
    try:
        c, inner = make(
            [[text("Adding."), call("tiny__add", {"a": 1, "b": 2})], [text("Three.")]]
        )
        tool = mcp_tool(srv)  # require_approval absent -> "always"
        m1 = c.post("/v1/responses", json=body([tool], store=True)).json()
        assert types(m1) == ["mcp_list_tools", "message", "mcp_approval_request"]
        ar = m1["output"][2]
        assert ar["id"].startswith("mcpr_") and ar["name"] == "add"
        assert json.loads(ar["arguments"]) == {"a": 1, "b": 2}
        assert m1["status"] == "completed"
        assert not any(r.get("method") == "tools/call" for r in srv.requests)
        assert len(inner.requests) == 1
        # chained follow-up approves; the list is not re-emitted
        m2 = c.post(
            "/v1/responses",
            json={
                "model": "m",
                "previous_response_id": m1["id"],
                "tools": [tool],
                "store": False,
                "input": [
                    {
                        "type": "mcp_approval_response",
                        "approval_request_id": ar["id"],
                        "approve": True,
                    }
                ],
            },
        ).json()
        assert types(m2) == ["mcp_call", "message"]
        assert m2["output"][0]["output"] == "3"
        assert m2["output"][0]["approval_request_id"] == ar["id"]
        # the model saw the approved call and its result
        items = inner.requests[1].input
        assert [i.type for i in items[-2:]] == ["function_call", "function_call_output"]
        assert items[-1].output == "3"
        # the merged first response is what the chain stores
        stored = responses._get_stored_response(m1["id"])
        assert [i["type"] for i in stored["output"]] == types(m1)
    finally:
        srv.stop()


def test_mcp_approval_input_replay_and_denial(make):
    srv = FakeMcpHttp()
    try:
        tool = mcp_tool(srv)
        replay = [
            {"type": "message", "role": "user", "content": "add"},
            {
                "type": "mcp_list_tools",
                "id": "mcpl_x",
                "server_label": "tiny",
                "tools": [],
            },
            {
                "type": "mcp_approval_request",
                "id": "mcpr_1",
                "server_label": "tiny",
                "name": "add",
                "arguments": '{"a": 5, "b": 6}',
            },
            {
                "type": "mcp_approval_response",
                "approval_request_id": "mcpr_1",
                "approve": True,
            },
        ]
        c, _ = make([[text("Eleven.")]])
        m = c.post("/v1/responses", json=body([tool], input=replay)).json()
        assert types(m) == [
            "mcp_call",
            "message",
        ]  # mcp_list_tools reused, not re-emitted
        assert m["output"][0]["output"] == "11"
        # denial: no execution, error "denied"
        replay[3]["approve"] = False
        replay[3]["reason"] = "no"
        c, inner = make([[text("ok")]])
        m = c.post("/v1/responses", json=body([tool], input=replay)).json()
        mc = m["output"][0]
        assert mc["type"] == "mcp_call" and mc["output"] is None
        assert "denied" in mc["error"]
        assert inner.requests[0].input[-1].output.startswith("Error:")
        assert not any(r.get("method") == "tools/call" for r in srv.requests[-3:])
        # a replayed history that already holds the executed mcp_call does not run it again
        n_calls = sum(r.get("method") == "tools/call" for r in srv.requests)
        replay[3]["approve"] = True
        replay.append(
            {
                "type": "mcp_call",
                "id": "mcp_1",
                "server_label": "tiny",
                "name": "add",
                "arguments": "{}",
                "output": "11",
                "approval_request_id": "mcpr_1",
            }
        )
        c, _ = make([[text("again")]])
        m = c.post("/v1/responses", json=body([tool], input=replay)).json()
        assert types(m) == ["message"]
        assert sum(r.get("method") == "tools/call" for r in srv.requests) == n_calls
    finally:
        srv.stop()


def test_mcp_require_approval_per_tool(make):
    srv = FakeMcpHttp()
    try:
        ra = {"never": {"tool_names": ["echo"]}}
        c, _ = make(
            [
                [
                    call("tiny__echo", {"text": "a"}, "c1"),
                    call("tiny__add", {"a": 1, "b": 1}, "c2"),
                ]
            ]
        )
        m = c.post(
            "/v1/responses", json=body([mcp_tool(srv, require_approval=ra)])
        ).json()
        assert types(m) == ["mcp_list_tools", "mcp_approval_request", "mcp_call"] or (
            sorted(types(m)) == ["mcp_approval_request", "mcp_call", "mcp_list_tools"]
        )
        call_item = next(i for i in m["output"] if i["type"] == "mcp_call")
        assert call_item["output"] == "a"
        assert m["status"] == "completed"
    finally:
        srv.stop()


def test_mcp_allowed_tools_filtering(make):
    srv = FakeMcpHttp()
    try:
        for allowed in (["echo"], {"tool_names": ["echo"]}):
            c, inner = make([[text("x")]])
            m = c.post(
                "/v1/responses",
                json=body(
                    [mcp_tool(srv, allowed_tools=allowed, require_approval="never")]
                ),
            ).json()
            assert [t["name"] for t in m["output"][0]["tools"]] == ["echo"]
            assert [t.name for t in inner.requests[0].tools] == ["tiny__echo"]
    finally:
        srv.stop()


def test_mcp_list_tools_reuse_from_chain(make):
    srv = FakeMcpHttp()
    try:
        c, _ = make([[text("a")], [text("b")]])
        tool = mcp_tool(srv, require_approval="never")
        m1 = c.post("/v1/responses", json=body([tool], store=True)).json()
        assert types(m1) == ["mcp_list_tools", "message"]
        m2 = c.post(
            "/v1/responses",
            json=body([tool], previous_response_id=m1["id"], input="more"),
        ).json()
        assert types(m2) == ["message"]
    finally:
        srv.stop()


def test_mcp_connection_failure(make):
    c, inner = make([[text("x")]])
    r = c.post(
        "/v1/responses",
        json=body(
            [
                {
                    "type": "mcp",
                    "server_label": "dead",
                    "server_url": "http://127.0.0.1:9/mcp",
                    "require_approval": "never",
                }
            ]
        ),
    )
    assert r.status_code == 424
    e = r.json()["error"]
    assert e["type"] == "invalid_request_error" and "dead" in e["message"]
    assert e["code"] == "external_connector_error"
    assert inner.requests == []


def test_mcp_connector_id_rejected(make):
    c, _ = make([])
    r = c.post(
        "/v1/responses",
        json=body(
            [{"type": "mcp", "server_label": "g", "connector_id": "connector_gmail"}]
        ),
    )
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert "connector_id" in r.json()["error"]["message"]


def test_background_with_server_tools_is_400(make):
    c, _ = make([])
    r = c.post("/v1/responses", json=body([WS], background=True))
    assert r.status_code == 400 and r.json()["error"]["param"] == "background"


# ── mixed calls, caps ─────────────────────────────────────────────────────────
def test_mixed_client_function_call_ends_response(make):
    search.set_provider_for_tests(FakeSearch())
    c, inner = make(
        [
            [
                call("web_search", {"query": "q"}, "c1"),
                call("shell", {"cmd": "ls"}, "c2"),
            ]
        ]
    )
    m = c.post(
        "/v1/responses",
        json=body(
            [
                WS,
                {
                    "type": "function",
                    "name": "shell",
                    "parameters": {"type": "object", "properties": {}},
                },
            ]
        ),
    ).json()
    assert sorted(types(m)) == ["function_call", "web_search_call"]
    fc = next(i for i in m["output"] if i["type"] == "function_call")
    assert fc["name"] == "shell" and fc["call_id"] == "c2"
    assert (
        next(i for i in m["output"] if i["type"] == "web_search_call")["status"]
        == "completed"
    )
    assert len(inner.requests) == 1
    assert [t.name for t in inner.requests[0].tools] == ["shell", "web_search"]


def test_max_iterations(make, monkeypatch):
    search.set_provider_for_tests(FakeSearch())
    monkeypatch.setenv("YUNSHU_SERVER_TOOL_MAX_ITERATIONS", "2")
    c, inner = make([[call("web_search", {"query": f"q{i}"})] for i in range(5)])
    m = c.post("/v1/responses", json=body([WS])).json()
    assert m["status"] == "incomplete"
    assert m["incomplete_details"] == {"reason": "max_tool_calls"}
    assert len(inner.requests) == 2
    assert types(m) == ["web_search_call", "web_search_call"]


def test_request_max_tool_calls(make):
    search.set_provider_for_tests(fake := FakeSearch())
    c, inner = make([[call("web_search", {"query": "a"})], [text("done")]])
    m = c.post("/v1/responses", json=body([WS], max_tool_calls=1)).json()
    assert m["status"] == "completed" and len(fake.calls) == 1
    # the cap was reached, so the second round no longer offered the server tool
    assert inner.requests[1].tools is None


def test_client_only_tools_untouched(make):
    # no server tools: not handled by the loop at all
    from yunshu_gateway.server_tools.responses_loop import has_server_tools_responses

    r = ResponsesRequest(
        model="m",
        input="x",
        tools=[{"type": "function", "name": "f", "parameters": {}}],
    )
    assert not has_server_tools_responses(r)


# ── request validation ────────────────────────────────────────────────────────
def test_tool_types_validate():
    r = ResponsesRequest(
        model="m",
        input="x",
        tools=[
            {"type": "web_search_preview"},
            {"type": "mcp", "server_label": "a", "server_url": "http://x"},
            {
                "type": "namespace",
                "name": "ns",
                "tools": [{"type": "function", "name": "f"}],
            },
            {"type": "custom", "name": "apply_patch", "format": {"type": "grammar"}},
        ],
    )
    assert [t.type for t in r.tools] == [
        "web_search_preview",
        "mcp",
        "namespace",
        "custom",
    ]
    from fastapi import HTTPException

    with pytest.raises(HTTPException, match="cannot be guaranteed"):
        function_tools(r.tools)
    r.tools[-1].format = {"type": "text"}
    fn = function_tools(r.tools)
    assert [t.name for t in fn] == ["f", "apply_patch"]


@pytest.mark.skipif(
    not CODEX_BODIES.exists(), reason="recorded Codex request not present"
)
def test_real_codex_body_validates():
    n = 0
    for line in CODEX_BODIES.read_text().splitlines():
        b = json.loads(line)["body"]
        b = json.loads(b) if isinstance(b, str) else b
        b["model"] = "qwen"
        req = ResponsesRequest(**b)
        assert req.store is False and req.include == ["reasoning.encrypted_content"]
        assert req.client_metadata and req.prompt_cache_key
        msgs = _convert_to_messages(req)
        assert msgs and msgs[0]["role"] == "system"
        assert all(m["role"] in ("system", "user", "assistant", "tool") for m in msgs)
        n += 1
    assert n


def test_unknown_input_items_do_not_422_and_convert():
    items = [
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "rules"}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hi"}],
        },
        {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "abc"},
        {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": [{"type": "input_text", "text": "r"}],
        },
        {
            "type": "custom_tool_call",
            "call_id": "c2",
            "name": "apply_patch",
            "input": "*** patch",
        },
        {"type": "custom_tool_call_output", "call_id": "c2", "output": "ok"},
        {
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {
                "type": "search",
                "query": "q",
                "sources": [{"type": "url", "url": "http://a"}],
            },
        },
        {"type": "web_search_call", "id": "ws_2", "status": "completed"},
        {
            "type": "mcp_call",
            "id": "mcp_1",
            "server_label": "tiny",
            "name": "add",
            "arguments": "{}",
            "output": "3",
        },
        {"type": "mcp_call", "id": "mcp_2", "server_label": "tiny"},
        {"type": "mcp_list_tools", "id": "mcpl_1", "server_label": "tiny", "tools": []},
        {
            "type": "mcp_approval_request",
            "id": "mcpr_1",
            "name": "n",
            "arguments": "{}",
        },
        {
            "type": "mcp_approval_response",
            "approval_request_id": "mcpr_1",
            "approve": True,
        },
        {"type": "compaction", "id": "cmp_1", "encrypted_content": "zzz"},
        {"type": "item_reference", "id": "msg_9"},
    ]
    req = ResponsesRequest(model="m", input=items)
    msgs = _convert_to_messages(req)
    roles = [m["role"] for m in msgs]
    assert (
        roles
        == [
            "system",
            "user",
            "assistant",  # function_call
            "tool",
            "assistant",  # custom_tool_call
            "tool",
            "assistant",  # web_search_call pair
            "tool",
            "assistant",  # mcp_call pair
            "tool",
        ]
        or roles[:2] == ["system", "user"]
    )
    ws = next(m for m in msgs if m["role"] == "tool" and "Sources" in m["content"])
    assert "http://a" in ws["content"]
    mc = next(
        m
        for m in msgs
        if m.get("tool_calls") and m["tool_calls"][0]["function"]["name"] == "tiny__add"
    )
    assert mc["tool_calls"][0]["id"] == "mcp_1"
    assert len(msgs) == 10


def test_annotations_for():
    src = [
        search.SearchResult("A", "http://a", "s"),
        search.SearchResult("B", "http://b", "s"),
    ]
    t = "x [1] y [1, 2] z [9]"
    a = annotations_for(t, src)
    assert [(x["url"], t[x["start_index"] : x["end_index"]]) for x in a] == [
        ("http://a", "[1]"),
        ("http://a", "[1, 2]"),
        ("http://b", "[1, 2]"),
    ]


# ── the real OpenAI SDK, when available ───────────────────────────────────────
def test_openai_sdk_parses_stream(make):
    openai = pytest.importorskip("openai")
    search.set_provider_for_tests(FakeSearch())
    c, _ = make([[call("web_search", {"query": "q"})], [text("Yes [1].")]])
    r = c.post("/v1/responses", json=body([WS], stream=True))
    from openai._streaming import Stream  # noqa: F401
    from openai.types.responses import Response

    final = parse_sse(r.text)[-1][1]["response"]
    parsed = Response.model_validate(final)
    assert parsed.output[0].type == "web_search_call"
    assert parsed.output[1].content[0].annotations[0].type == "url_citation"
    assert openai


def test_explicit_search_request_steers_only_the_first_round_responses():
    from yunshu_gateway.routers import responses
    from yunshu_gateway.server_tools import responses_loop
    from yunshu_gateway.server_tools.runtime import ServerToolDef

    defs = [ServerToolDef("web_search", "web_search", "", {})]
    tools = [
        {"name": "web_search", "description": "", "parameters": {"type": "object"}}
    ]
    req = responses.ResponsesRequest(
        model="m",
        input="Please search the web for yunshu",
        tools=[{"type": "web_search"}],
    )
    first = responses_loop._inner_request(req, [], tools, True, defs)
    assert first.tool_choice == {"type": "function", "name": "web_search"}
    later = responses_loop._inner_request(req, [], tools, False, defs)
    assert later.tool_choice == "auto"
    quiet = responses.ResponsesRequest(
        model="m", input="hello", tools=[{"type": "web_search"}]
    )
    assert (
        responses_loop._inner_request(quiet, [], tools, True, defs).tool_choice is None
    )
