"""Responses compaction: /v1/responses/compact, compaction items in input, auto compaction."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from yunshu_gateway import conversations_store as cs
from yunshu_gateway import responses_context as ctx
from yunshu_gateway.reasoning_token import seal_compaction, unseal_compaction
from yunshu_gateway.routers import conversations as conv_router
from yunshu_gateway.routers import responses as resp_mod
from yunshu_gateway.routers import responses_compact

SUMMARY = "GOAL: port the parser; files: a.py; open: tests"


class Inner:
    def __init__(self):
        self.calls: list = []
        self.stream = False

    async def __call__(self, req, request):
        self.calls.append(req)
        obj = {
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "model": "m",
            "status": "completed",
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                }
            ],
        }
        if not self.stream:
            return JSONResponse(obj)

        async def gen():
            ev = {"type": "response.completed", "response": obj}
            yield f"event: response.completed\ndata: {json.dumps(ev)}\n\n".encode()

        return StreamingResponse(gen(), media_type="text/event-stream")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("YUNSHU_CONVERSATIONS_DIR", str(tmp_path / "convs"))
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    cs.reset_store()
    inner = Inner()
    monkeypatch.setattr(resp_mod, "create_response", inner)
    summaries: list = []

    async def fake_summarize(request, model, messages, tools=None):
        summaries.append(
            {
                "model": model,
                "messages": messages,
                "tools": tools,
                "transcript": "\n".join(
                    ln for m in messages if (ln := ctx._transcript_line(m))
                ),
                "instr": messages[0]["content"]
                if messages[0]["role"] == "system"
                else None,
            }
        )
        return SUMMARY, {
            "input_tokens": 50,
            "output_tokens": 10,
            "total_tokens": 60,
        }

    monkeypatch.setattr(ctx, "summarize", fake_summarize)
    app = FastAPI()
    app.include_router(responses_compact.router, prefix="/v1")
    app.include_router(resp_mod.router, prefix="/v1")
    app.include_router(conv_router.router, prefix="/v1")
    yield TestClient(app), inner, summaries
    cs.reset_store()


def _u(text, role="user"):
    return {"type": "message", "role": role, "content": text}


def test_compact_endpoint_shape(env):
    client, _, summaries = env
    r = client.post(
        "/v1/responses/compact",
        json={
            "model": "m",
            "instructions": "keep file names",
            "input": [
                _u("build a parser"),
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "ok, plan: A then B",
                },
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "read",
                    "arguments": '{"p":"a.py"}',
                },
                {"type": "function_call_output", "call_id": "c1", "output": "print(1)"},
                _u("use rules", role="developer"),
                _u("thanks"),
            ],
        },
    )
    assert r.status_code == 200
    b = r.json()
    assert b["object"] == "response.compaction" and b["id"].startswith("resp_")
    assert isinstance(b["created_at"], int)
    assert b["usage"]["total_tokens"] == 60
    out = b["output"]
    assert [o["type"] for o in out] == ["message", "message", "message", "compaction"]
    assert [o["role"] for o in out[:3]] == ["user", "developer", "user"]
    assert out[0]["content"][0]["text"] == "build a parser"
    cmp_ = out[-1]
    assert cmp_["id"].startswith("cmp_")
    assert cmp_["encrypted_content"].startswith("yunshu1c:")
    assert unseal_compaction(cmp_["encrypted_content"]) == SUMMARY
    (call,) = summaries
    assert call["instr"] == "keep file names" and call["model"] == "m"
    t = call["transcript"]
    assert "assistant: ok, plan: A then B" in t
    assert "assistant called tool read" in t and "tool result: print(1)" in t


def test_compact_previous_response_id(env):
    client, _, summaries = env
    resp_mod._store_response(
        "resp_prev",
        {
            "id": "resp_prev",
            "output": [
                {
                    "role": "assistant",
                    "type": "message",
                    "content": [{"type": "output_text", "text": "old answer"}],
                }
            ],
            "_input_messages": [{"role": "user", "content": "old question"}],
            "_owner": "",
        },
    )
    r = client.post(
        "/v1/responses/compact",
        json={"model": "m", "previous_response_id": "resp_prev", "input": "new"},
    )
    out = r.json()["output"]
    assert [o["type"] for o in out] == ["message", "message", "compaction"]
    assert out[0]["content"][0]["text"] == "old question"
    assert "assistant: old answer" in summaries[0]["transcript"]
    r = client.post(
        "/v1/responses/compact", json={"model": "m", "previous_response_id": "resp_zz"}
    )
    assert r.status_code == 404


def test_compact_validation(env):
    client, _, _ = env
    assert client.post("/v1/responses/compact", json={"input": "x"}).status_code == 400
    assert client.post("/v1/responses/compact", json={"model": "m"}).status_code == 400
    assert (
        client.post(
            "/v1/responses/compact", json={"model": "m", "input": 3}
        ).status_code
        == 400
    )


def test_compaction_item_in_input(env):
    client, inner, _ = env
    token = seal_compaction("earlier we chose plan B")
    r = client.post(
        "/v1/responses",
        json={
            "model": "m",
            "input": [
                _u("kept user msg"),
                {"type": "compaction", "id": "cmp_1", "encrypted_content": token},
                {
                    "type": "compaction",
                    "id": "cmp_2",
                    "encrypted_content": "gAAAA-foreign",
                },
                _u("continue"),
            ],
        },
    )
    assert r.status_code == 200
    msgs = resp_mod._convert_to_messages(inner.calls[0])
    assert [m["role"] for m in msgs] == ["user", "user", "user"]
    assert (
        msgs[1]["content"]
        == "Summary of the earlier conversation:\nearlier we chose plan B"
    )
    assert msgs[2]["content"] == "continue"


def test_compaction_round_trip_through_endpoint(env):
    client, inner, _ = env
    out = client.post(
        "/v1/responses/compact", json={"model": "m", "input": [_u("q1"), _u("q2")]}
    ).json()["output"]
    client.post(
        "/v1/responses",
        json={"model": "m", "input": [*out, _u("q3")]},
    )
    msgs = resp_mod._convert_to_messages(inner.calls[0])
    assert [m["content"] for m in msgs][:2] == ["q1", "q2"]
    assert msgs[2]["content"].startswith("Summary of the earlier conversation:\n")
    assert SUMMARY in msgs[2]["content"] and msgs[3]["content"] == "q3"


def _threshold(n):
    return [{"type": "compaction", "compact_threshold": n}]


def test_auto_compaction_over_threshold(env, monkeypatch):
    client, inner, summaries = env
    monkeypatch.setattr(ctx, "_count_tokens", lambda msgs: 5000)
    r = client.post(
        "/v1/responses",
        json={
            "model": "m",
            "input": [
                _u("first"),
                {"type": "message", "role": "assistant", "content": "a"},
                _u("second"),
            ],
            "context_management": _threshold(1000),
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["output"][0]["type"] == "compaction"
    assert unseal_compaction(body["output"][0]["encrypted_content"]) == SUMMARY
    assert body["output"][1]["type"] == "message"
    assert len(summaries) == 1
    seen = inner.calls[0]
    assert seen.context_management is None
    msgs = resp_mod._convert_to_messages(seen)
    assert [m["content"] for m in msgs][:2] == ["first", "second"]
    assert msgs[2]["content"].startswith("Summary of the earlier conversation:")
    assert len(msgs) == 3


def test_auto_compaction_below_threshold(env, monkeypatch):
    client, inner, summaries = env
    monkeypatch.setattr(ctx, "_count_tokens", lambda msgs: 10)
    r = client.post(
        "/v1/responses",
        json={"model": "m", "input": "hi", "context_management": _threshold(1000)},
    )
    assert [o["type"] for o in r.json()["output"]] == ["message"]
    assert summaries == []
    assert resp_mod._convert_to_messages(inner.calls[0])[0]["content"] == "hi"


def test_auto_compaction_stream_and_conversation(env, monkeypatch):
    client, inner, _ = env
    inner.stream = True
    monkeypatch.setattr(ctx, "_count_tokens", lambda msgs: 5000)
    cid = client.post(
        "/v1/conversations", json={"items": [_u("old one"), _u("old two")]}
    ).json()["id"]
    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "m",
            "input": "new",
            "stream": True,
            "conversation": cid,
            "context_management": _threshold(10),
        },
    ) as r:
        raw = b"".join(r.iter_bytes()).decode()
    ev = json.loads(next(ln[6:] for ln in raw.split("\n") if ln.startswith("data: {")))
    out = ev["response"]["output"]
    assert out[0]["type"] == "compaction" and out[1]["type"] == "message"
    assert ev["response"]["conversation"] == {"id": cid}
    items = client.get(
        f"/v1/conversations/{cid}/items", params={"order": "asc"}
    ).json()["data"]
    # compaction item is stored; the next turn starts from it
    assert "compaction" in [i["type"] for i in items]
    inner.stream = False
    client.post(
        "/v1/responses", json={"model": "m", "input": "next", "conversation": cid}
    )
    msgs = resp_mod._convert_to_messages(inner.calls[-1])
    assert msgs[0]["content"].startswith("Summary of the earlier conversation:")
    assert [m["content"] for m in msgs][-2:] == ["done", "next"] or msgs[-1][
        "content"
    ] == "next"
    assert "old one" not in json.dumps(msgs)


def test_threshold_uses_previous_response_chain(env, monkeypatch):
    client, inner, summaries = env
    resp_mod._store_response(
        "resp_p",
        {
            "id": "resp_p",
            "output": [],
            "_input_messages": [{"role": "user", "content": "chain user"}],
            "_owner": "",
        },
    )
    seen_counts: list = []

    def count(msgs):
        seen_counts.append(len(msgs))
        return 999

    monkeypatch.setattr(ctx, "_count_tokens", count)
    client.post(
        "/v1/responses",
        json={
            "model": "m",
            "input": "now",
            "previous_response_id": "resp_p",
            "context_management": _threshold(5),
        },
    )
    assert seen_counts == [2]
    assert "chain user" in summaries[0]["transcript"]
    assert inner.calls[0].previous_response_id is None


async def _fake_chat(request):
    body = json.loads(request.content)
    assert request.url.path == "/v1/chat/completions"
    assert body["stream"] is False and body["model"] == "m"
    assert body["max_tokens"] == 321
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][1] == {"role": "user", "content": "hello"}
    assert body["messages"][-1]["content"].startswith(ctx.COMPACT_PROMPT)
    assert body["tools"][0]["function"]["name"] == "f"
    assert request.headers["authorization"] == "Bearer t"
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "  the summary  "}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        },
    )


@pytest.mark.asyncio
async def test_summarize_loopback(monkeypatch):
    monkeypatch.setenv("YUNSHU_COMPACT_MAX_TOKENS", "321")

    def fake_client(request):
        return httpx.AsyncClient(
            base_url="http://loop",
            headers={"Authorization": "Bearer t"},
            transport=httpx.MockTransport(_fake_chat),
        )

    from yunshu_gateway.routers import ollama

    monkeypatch.setattr(ollama, "_client", fake_client)
    msgs = ctx._summary_request_messages([{"role": "user", "content": "hello"}], "sys")
    text, usage = await ctx.summarize(
        None, "m", msgs, ctx.chat_tools([{"type": "function", "name": "f"}])
    )
    assert text == "the summary" and usage["total_tokens"] == 10


@pytest.mark.asyncio
async def test_summarize_error_status(monkeypatch):
    def fake_client(request):
        return httpx.AsyncClient(
            base_url="http://loop",
            transport=httpx.MockTransport(
                lambda r: httpx.Response(404, json={"error": {"message": "no model"}})
            ),
        )

    from yunshu_gateway.routers import ollama

    monkeypatch.setattr(ollama, "_client", fake_client)
    with pytest.raises(ctx.ContextError) as ei:
        await ctx.summarize(None, "m", [{"role": "user", "content": "x"}])
    assert ei.value.status == 404 and "no model" in ei.value.message


# ---- prefix reuse, multi-pass folding, streaming progress -------------------


def _msgs(*pairs):
    return [{"role": r, "content": c} for r, c in pairs]


def test_summary_request_keeps_history_as_prefix():
    hist = _msgs(("user", "a"), ("assistant", "b"))
    req = ctx._summary_request_messages(hist, "sys", "be brief")
    assert req[:3] == [{"role": "system", "content": "sys"}, *hist]
    assert req[-1]["role"] == "user" and "be brief" in req[-1]["content"]
    assert req[-1]["content"].startswith(ctx.COMPACT_PROMPT)


def test_safe_boundaries_never_split_tool_call_from_result():
    call = {"role": "assistant", "tool_calls": [{"id": "c1"}, {"id": "c2"}]}
    entries = [
        (None, [{"role": "user", "content": "q1"}]),
        (None, [call]),
        (None, [{"role": "tool", "tool_call_id": "c1", "content": "r"}]),
        (None, [{"role": "user", "content": "mid-call user"}]),
        (None, [{"role": "tool", "tool_call_id": "c2", "content": "r"}]),
        (None, [{"role": "assistant", "content": "done"}]),
        (None, [{"role": "user", "content": "q2"}]),
    ]
    assert ctx.safe_boundaries(entries) == [6]


@pytest.mark.asyncio
async def test_over_window_history_folds_in_passes(monkeypatch):
    # window of 40 "tokens" (1 per message): 3x the window must still summarize, every call
    # under the window, each later pass starting from the previous summary.
    monkeypatch.setattr(ctx, "_context_budget", lambda model: 40)
    monkeypatch.setattr(ctx, "_count_tokens", lambda msgs: len(msgs))
    entries = [
        (
            None,
            [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ],
        )
        for i in range(60)
    ]
    calls: list = []

    async def fake(request, model, messages, tools=None):
        calls.append(messages)
        return f"S{len(calls)}", {
            "input_tokens": len(messages),
            "output_tokens": 1,
            "total_tokens": len(messages) + 1,
        }

    monkeypatch.setattr(ctx, "summarize", fake)
    text, usage = await ctx.summarize_history(None, "m", entries)
    assert text == f"S{len(calls)}" and 2 <= len(calls) <= 8
    assert all(len(m) <= 40 for m in calls)
    assert ctx.SUMMARY_PREFIX + "S1" in calls[1][0]["content"]
    assert usage["input_tokens"] == sum(len(m) for m in calls)


@pytest.mark.asyncio
async def test_single_exchange_over_window_is_an_error(monkeypatch):
    monkeypatch.setattr(ctx, "_context_budget", lambda model: 3)
    monkeypatch.setattr(ctx, "_count_tokens", lambda msgs: len(msgs) * 10)
    with pytest.raises(ctx.ContextError) as ei:
        await ctx.summarize_history(
            None, "m", [(None, [{"role": "user", "content": "x"}])]
        )
    assert ei.value.status == 400


def test_streaming_auto_compaction_sends_bytes_before_summary_returns(env, monkeypatch):
    client, inner, summaries = env
    monkeypatch.setattr(ctx, "_count_tokens", lambda msgs: 999)
    monkeypatch.setattr(ctx, "_PROGRESS_SECONDS", 0.01)
    import asyncio

    orig = ctx.summarize_history
    gate: dict = {}

    async def slow(*a, **k):
        await asyncio.sleep(0.1)
        return await orig(*a, **k)

    monkeypatch.setattr(ctx, "summarize_history", slow)
    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "model": "m",
            "input": "hi",
            "stream": True,
            "context_management": _threshold(5),
        },
    ) as r:
        chunks = list(r.iter_raw())
    first = chunks[0]
    assert first.startswith(b": compacting")
    assert b"".join(chunks).count(b": compacting") >= 2
    assert inner.calls and inner.calls[0].context_management is None
    assert gate == {}
