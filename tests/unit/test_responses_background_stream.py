"""Responses `background: true` with `stream: true` and with server tools: the generation outlives
the HTTP request, its events are kept, and `GET /v1/responses/{id}?stream=true&starting_after=N`
resumes them."""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_gateway.routers import responses
from yunshu_gateway.server_tools import search

from .server_tools_helpers import FakeSearch
from .server_tools_responses_helpers import ScriptedResponses, call, parse_sse, text


@pytest.fixture
def make(monkeypatch):
    search.set_provider_for_tests(None)
    real = responses.create_response
    responses._response_store.clear()

    def _make(rounds):
        inner = ScriptedResponses(rounds)

        async def dispatch(req, request):
            # The request the client sent runs the real handler; the generation rounds it
            # issues (function tools only) are scripted.
            if any(t.type in ("web_search", "mcp") for t in (req.tools or [])):
                return await real(req, request)
            rid = getattr(request.state, "_forced_response_id", None)
            res = await inner(req, request)
            if rid is None:
                return res
            request.state._forced_response_id = None
            body = res.body_iterator

            async def storing():
                async for chunk in body:
                    chunk = chunk.replace(b"resp-inner", rid.encode())
                    yield chunk
                    for name, data in parse_sse(chunk.decode()):
                        if name == "response.completed":
                            queued = responses._get_stored_response(rid) or {}
                            responses._store_response(
                                rid,
                                {
                                    **data["response"],
                                    "id": rid,
                                    "_owner": queued.get("_owner"),
                                },
                            )

            res.body_iterator = storing()
            return res

        monkeypatch.setattr(responses, "create_response", dispatch)
        app = FastAPI()
        app.include_router(responses.router, prefix="/v1")
        return TestClient(app), inner

    yield _make
    search.set_provider_for_tests(None)


def _wait_status(c, rid, want, secs=5):
    end = time.time() + secs
    while time.time() < end:
        got = c.get(f"/v1/responses/{rid}").json()
        if got["status"] in want:
            return got
        time.sleep(0.02)
    raise AssertionError(f"{rid} never reached {want}: {got}")


def test_background_stream_streams_then_resumes(make):
    c, _ = make([[text("hello there")]])
    body = {"model": "m", "input": "hi", "background": True, "stream": True}
    r = c.post("/v1/responses", json=body)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    names = [n for n, _ in events]
    assert names[0] == "response.created" and names[-1] == "response.completed"
    rid = events[0][1]["response"]["id"]
    done = _wait_status(c, rid, ("completed",))
    assert done["output"][0]["content"][0]["text"] == "hello there"
    seqs = [d["sequence_number"] for _, d in events]
    cut = seqs[2]
    again = c.get(
        f"/v1/responses/{rid}", params={"stream": "true", "starting_after": cut}
    )
    assert again.status_code == 200
    rest = parse_sse(again.text)
    assert [d["sequence_number"] for _, d in rest] == [s for s in seqs if s > cut]
    full = parse_sse(c.get(f"/v1/responses/{rid}", params={"stream": "true"}).text)
    assert [d["sequence_number"] for _, d in full] == seqs


def test_stream_of_a_non_background_response_is_400(make):
    c, _ = make([[text("x")]])
    r = c.post("/v1/responses", json={"model": "m", "input": "hi", "background": True})
    rid = r.json()["id"]
    _wait_status(c, rid, ("completed",))
    got = c.get(f"/v1/responses/{rid}", params={"stream": "true"})
    assert got.status_code == 400
    assert got.json()["error"]["code"] == "response_not_streamable"


def test_background_with_web_search_runs_the_loop(make):
    search.set_provider_for_tests(FakeSearch())
    c, _ = make([[call("web_search", {"query": "yunshu"})], [text("found it")]])
    body = {
        "model": "m",
        "input": "find yunshu",
        "tools": [{"type": "web_search"}],
        "background": True,
    }
    r = c.post("/v1/responses", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "queued"
    done = _wait_status(c, r.json()["id"], ("completed", "failed"))
    assert done["status"] == "completed"
    assert [i["type"] for i in done["output"]] == ["web_search_call", "message"]


def test_background_stream_with_web_search_keeps_its_id(make):
    search.set_provider_for_tests(FakeSearch())
    c, _ = make([[call("web_search", {"query": "yunshu"})], [text("found it")]])
    body = {
        "model": "m",
        "input": "find yunshu",
        "tools": [{"type": "web_search"}],
        "background": True,
        "stream": True,
    }
    r = c.post("/v1/responses", json=body)
    assert r.status_code == 200, r.text
    events = parse_sse(r.text)
    rid = events[0][1]["response"]["id"]
    assert events[-1][0] == "response.completed"
    done = _wait_status(c, rid, ("completed", "failed"))
    assert done["status"] == "completed" and done["id"] == rid


def test_cancel_a_background_stream(make):
    c, _ = make([[text("x")]])
    r = c.post(
        "/v1/responses",
        json={"model": "m", "input": "hi", "background": True, "stream": True},
    )
    rid = parse_sse(r.text)[0][1]["response"]["id"]
    # finished already: cancel reports the terminal state, as the hosted API does
    got = c.post(f"/v1/responses/{rid}/cancel")
    assert got.status_code == 200
    assert got.json()["status"] in ("completed", "cancelled")
