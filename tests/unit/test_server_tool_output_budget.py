"""The caller's output-token cap covers the whole server-tool loop, not each round
(each scripted round spends 5 output tokens)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_gateway.routers import anthropic, responses
from yunshu_gateway.server_tools import search

from .server_tools_helpers import FakeSearch, ScriptedInner
from .server_tools_responses_helpers import ScriptedResponses, call, text


@pytest.fixture(autouse=True)
def _provider():
    search.set_provider_for_tests(FakeSearch())
    yield
    search.set_provider_for_tests(None)


def _responses(monkeypatch, rounds):
    inner = ScriptedResponses(rounds)
    monkeypatch.setattr(responses, "create_response", inner)
    app = FastAPI()
    app.include_router(responses.router, prefix="/v1")
    return TestClient(app), inner


def _anthropic(monkeypatch, rounds):
    inner = ScriptedInner(rounds)
    monkeypatch.setattr(anthropic, "create_message", inner)
    app = FastAPI()
    app.include_router(anthropic.router, prefix="/v1")
    return TestClient(app), inner


def _rbody(cap):
    return {
        "model": "m",
        "input": "find yunshu",
        "tools": [{"type": "web_search"}],
        "store": False,
        "max_output_tokens": cap,
    }


def _abody(cap):
    return {
        "model": "m",
        "max_tokens": cap,
        "messages": [{"role": "user", "content": "find yunshu"}],
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
    }


_ACALL = {
    "type": "tool_use",
    "id": "toolu_1",
    "name": "web_search",
    "input": {"query": "q"},
}


def test_responses_round_two_gets_remaining_budget(monkeypatch):
    c, inner = _responses(
        monkeypatch, [[call("web_search", {"query": "q"})], [text("done")]]
    )
    m = c.post("/v1/responses", json=_rbody(8)).json()
    assert m["status"] == "completed"
    assert [r.max_output_tokens for r in inner.requests] == [8, 3]


def test_responses_spent_budget_stops_incomplete(monkeypatch):
    c, inner = _responses(
        monkeypatch, [[call("web_search", {"query": "q"})], [text("done")]]
    )
    m = c.post("/v1/responses", json=_rbody(5)).json()
    assert len(inner.requests) == 1
    assert m["status"] == "incomplete"
    assert m["incomplete_details"] == {"reason": "max_output_tokens"}


def test_anthropic_round_two_gets_remaining_budget(monkeypatch):
    c, inner = _anthropic(
        monkeypatch, [([_ACALL], "tool_use"), ([text_block("done")], "end_turn")]
    )
    m = c.post("/v1/messages", json=_abody(8)).json()
    assert m["stop_reason"] == "end_turn"
    assert [r.max_tokens for r in inner.requests] == [8, 3]


def test_anthropic_spent_budget_stops_max_tokens(monkeypatch):
    c, inner = _anthropic(
        monkeypatch, [([_ACALL], "tool_use"), ([text_block("done")], "end_turn")]
    )
    m = c.post("/v1/messages", json=_abody(5)).json()
    assert len(inner.requests) == 1
    assert m["stop_reason"] == "max_tokens"


def text_block(t):
    return {"type": "text", "text": t}
