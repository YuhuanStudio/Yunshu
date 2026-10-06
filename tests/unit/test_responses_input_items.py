"""GET /v1/responses/{id}/input_items (OpenAI): the route was missing, the official SDK's
`client.responses.input_items.list` got a 404/405 (found by the route coverage gate)."""

from __future__ import annotations

import asyncio
import json
import types

import pytest

from yunshu_gateway.routers import responses as R  # noqa: N812


def _req(actor=""):
    r = types.SimpleNamespace()
    r.state = types.SimpleNamespace(role="")
    r.headers = {}
    r._actor = actor
    return r


def _call(rid="resp-1", actor="", **kw):
    kw.setdefault("limit", 20)
    kw.setdefault("order", "desc")
    kw.setdefault("after", None)
    kw.setdefault("before", None)
    resp = asyncio.run(R.list_response_input_items(rid, _req(actor), **kw))
    return resp.status_code, json.loads(bytes(resp.body).decode())


@pytest.fixture(autouse=True)
def _stub(monkeypatch):
    import yunshu_control.audit_log as al

    monkeypatch.setattr(al, "resolve_actor", lambda request: request._actor)
    monkeypatch.setattr(R, "_check_permission", lambda request, perm: None)
    monkeypatch.delenv("YUNSHU_AUTH_DISABLED", raising=False)
    R._response_store.clear()
    yield
    R._response_store.clear()


MSGS = [
    {"role": "user", "content": "one"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "c1", "function": {"name": "f", "arguments": '{"a": 1}'}}
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "42"},
    {"role": "user", "content": [{"type": "text", "text": "two"}]},
]


def _store(owner=""):
    R._store_response(
        "resp-1",
        {
            "id": "resp-1",
            "object": "response",
            "_input_messages": MSGS,
            "_owner": owner,
        },
    )


def test_lists_input_items_newest_first():
    _store()
    st, b = _call()
    assert st == 200 and b["object"] == "list" and b["has_more"] is False
    types_ = [i["type"] for i in b["data"]]
    assert types_ == ["message", "function_call_output", "function_call", "message"]
    assert b["data"][0]["content"] == [{"type": "input_text", "text": "two"}]
    assert b["data"][1]["call_id"] == "c1" and b["data"][1]["output"] == "42"
    assert b["data"][2]["name"] == "f" and b["data"][2]["arguments"] == '{"a": 1}'
    assert b["first_id"] == b["data"][0]["id"] and b["last_id"] == b["data"][-1]["id"]
    assert len({i["id"] for i in b["data"]}) == 4


def test_pagination_and_order():
    _store()
    _, asc = _call(order="asc")
    assert asc["data"][0]["content"][0]["text"] == "one"
    _, p1 = _call(order="asc", limit=2)
    assert len(p1["data"]) == 2 and p1["has_more"] is True
    _, p2 = _call(order="asc", limit=2, after=p1["last_id"])
    assert [i["id"] for i in p1["data"] + p2["data"]] == [i["id"] for i in asc["data"]]
    assert p2["has_more"] is False
    _, pb = _call(order="asc", before=asc["data"][2]["id"])
    assert [i["id"] for i in pb["data"]] == [i["id"] for i in asc["data"][:2]]
    st, _ = _call(after="nope")
    assert st == 404
    assert _call(limit=0)[0] == 400
    assert _call(order="sideways")[0] == 400


def test_unknown_and_foreign_response_are_404():
    assert _call("resp-nope")[0] == 404
    _store(owner="alice")
    assert _call(actor="bob")[0] == 404
    assert _call(actor="alice")[0] == 200
