"""Text WebSocket transport (/v1/stream, Responses WS mode) and Unix socket.

The engine is replaced by tiny fake SSE endpoints; everything else (in-process
ASGI dispatch, SSE parsing, multiplexing, cancel, backpressure) is real.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from yunshu_engine import settings
from yunshu_gateway.routers import stream_ws
from yunshu_gateway.streaming import ClosingStreamingResponse as StreamingResponse
from yunshu_gateway.ws_transport import SSEParser, is_delta

STATE = {"cancelled": 0, "seen_ids": [], "seen_auth": []}


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(stream_ws.router)

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        STATE["seen_ids"].append(request.headers.get("x-request-id"))
        STATE["seen_auth"].append(request.headers.get("authorization"))
        if body.get("model") == "bad":
            return JSONResponse(
                {
                    "error": {
                        "message": "no such model",
                        "type": "invalid_request_error",
                    }
                },
                status_code=404,
            )
        n = int(body.get("n_tokens", 5))
        delay = float(body.get("delay", 0.0))
        assert body["stream"] is True
        assert body["stream_options"]["include_usage"] is True

        async def gen():
            done = False
            try:
                yield ": keep-alive\n\n"
                if body.get("emit_x"):
                    yield ': yunshu-progress {"phase":"prefill","percent":50.0}\n\n'
                    yield ': yunshu-stats {"ttft_ms":12.5,"decode_tps":99.0}\n\n'
                for i in range(n):
                    chunk = {
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": f"t{i} "}}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(delay)
                yield "data: [DONE]\n\n"
                done = True
            finally:
                if not locals().get("done"):
                    STATE["cancelled"] += 1

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/v1/responses")
    async def responses(request: Request):
        body = await request.json()
        tag = body.get("tag", "x")
        delay = float(body.get("delay", 0.0))

        async def gen():
            yield f"event: response.created\ndata: {json.dumps({'type': 'response.created', 'tag': tag})}\n\n"
            for i in range(3):
                await asyncio.sleep(delay)
                d = {"type": "response.output_text.delta", "delta": f"{i}", "tag": tag}
                yield f"event: response.output_text.delta\ndata: {json.dumps(d)}\n\n"
            done = {"type": "response.completed", "tag": tag}
            yield f"event: response.completed\ndata: {json.dumps(done)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


@pytest.fixture()
def client():
    STATE.update(cancelled=0, seen_ids=[], seen_auth=[])
    with TestClient(make_app()) as c:
        yield c


def collect(ws, rid, until="done"):
    out = []
    while True:
        m = json.loads(ws.receive_text())
        if m.get("type") == "ping":
            continue
        if m.get("id") == rid or m.get("id") is None:
            out.append(m)
        if m.get("type") == until and m.get("id") in (rid, None):
            return out


def req(rid, **body):
    body.setdefault("model", "m")
    body.setdefault("messages", [{"role": "user", "content": "hi"}])
    return json.dumps(
        {"type": "request", "id": rid, "api": "chat.completions", "body": body}
    )


def test_sse_parser_and_delta():
    p = SSEParser()
    assert p.feed(b": hb\n\ndata: {") == []
    frames = p.feed(b'"a": 1}\n\nevent: x\ndata: [DONE]\n\n')
    assert frames == [(None, '{"a": 1}'), ("x", "[DONE]")]
    assert is_delta("chat.completions", {"choices": [{"delta": {"content": "a"}}]})
    assert not is_delta("chat.completions", {"choices": [{"delta": {}}]})
    assert is_delta("messages", {"type": "content_block_delta"})
    assert is_delta("responses", {"type": "response.output_text.delta"})


def test_stream_basic_and_request_id(client):
    with client.websocket_connect("/v1/stream") as ws:
        created = json.loads(ws.receive_text())
        assert created["type"] == "session.created"
        assert created["limits"]["max_inflight"] == 16
        ws.send_text(req("a", n_tokens=3))
        msgs = collect(ws, "a")
    events = [m for m in msgs if m["type"] == "event"]
    assert [e["data"]["choices"][0]["delta"]["content"] for e in events] == [
        "t0 ",
        "t1 ",
        "t2 ",
    ]
    done = msgs[-1]
    assert done["type"] == "done" and done["reason"] == "completed"
    assert done["stats"]["deltas"] == 3 and done["stats"]["ttft_ms"] is not None
    assert STATE["seen_ids"] == ["a"]  # client id is the HTTP request id


def test_progress_and_stats_comments_become_events(client):
    with client.websocket_connect("/v1/stream") as ws:
        ws.receive_text()
        ws.send_text(req("x1", n_tokens=2, emit_x=True))
        msgs = collect(ws, "x1")
    prog = [m for m in msgs if m["type"] == "progress"]
    stats = [m for m in msgs if m["type"] == "stats"]
    assert prog == [
        {"type": "progress", "id": "x1", "phase": "prefill", "percent": 50.0}
    ]
    assert stats[0]["decode_tps"] == 99.0 and stats[0]["id"] == "x1"
    # comments are never delivered as `event` messages
    assert all(m["type"] != "event" or isinstance(m["data"], dict) for m in msgs)


def test_unsafe_client_id_is_not_forwarded_as_header(client):
    with client.websocket_connect("/v1/stream") as ws:
        ws.receive_text()
        ws.send_text(req("has space/ok?", n_tokens=1))
        msgs = collect(ws, "has space/ok?")
    assert msgs[-1]["type"] == "done"
    assert STATE["seen_ids"][-1] is None  # handler generates its own id


def test_upstream_error_becomes_error_message(client):
    with client.websocket_connect("/v1/stream") as ws:
        ws.receive_text()
        ws.send_text(req("e", model="bad"))
        msgs = collect(ws, "e", until="done")
    err = [m for m in msgs if m["type"] == "error"][0]
    assert err["status"] == 404 and "no such model" in err["error"]["message"]
    assert msgs[-1]["type"] == "done" and msgs[-1]["reason"] == "error"


def test_protocol_errors(client):
    with client.websocket_connect("/v1/stream") as ws:
        ws.receive_text()
        ws.send_text("not json")
        assert (
            json.loads(ws.receive_text())["error"]["message"] == "Invalid JSON message"
        )
        ws.send_text(json.dumps({"type": "request", "api": "chat", "body": {}}))
        assert "'id'" in json.loads(ws.receive_text())["error"]["message"]
        ws.send_text(
            json.dumps({"type": "request", "id": "z", "api": "nope", "body": {}})
        )
        assert json.loads(ws.receive_text())["status"] == 400
        ws.send_text(json.dumps({"type": "cancel", "id": "ghost"}))
        assert json.loads(ws.receive_text())["status"] == 404
        ws.send_text(json.dumps({"type": "ping", "t": 5}))
        assert json.loads(ws.receive_text()) == {"type": "pong", "t": 5}


def test_multiplex_interleaves_and_cancel_by_id(client):
    with client.websocket_connect("/v1/stream") as ws:
        ws.receive_text()
        ws.send_text(req("slow", n_tokens=200, delay=0.02))
        ws.send_text(req("fast", n_tokens=3))
        seen, done = [], {}
        cancelled = False
        while len(done) < 2:
            m = json.loads(ws.receive_text())
            if m["type"] == "event":
                seen.append(m["id"])
                if m["id"] == "slow" and not cancelled:
                    ws.send_text(json.dumps({"type": "cancel", "id": "slow"}))
                    cancelled = True
            elif m["type"] == "done":
                done[m["id"]] = m
    assert done["fast"]["reason"] == "completed"
    assert done["slow"]["reason"] == "cancelled"
    assert done["slow"]["stats"]["deltas"] < 200
    assert "fast" in seen and "slow" in seen
    deadline = time.time() + 3
    while STATE["cancelled"] < 1 and time.time() < deadline:
        time.sleep(0.02)
    assert STATE["cancelled"] >= 1  # the generator was really stopped


def test_stop_and_max_tokens_update(client):
    with client.websocket_connect("/v1/stream") as ws:
        ws.receive_text()
        ws.send_text(req("s", n_tokens=500, delay=0.01))
        ws.send_text(json.dumps({"type": "update", "id": "s", "max_tokens": 4}))
        msgs = collect(ws, "s")
    assert any(m["type"] == "updated" and m["max_tokens"] == 4 for m in msgs)
    assert msgs[-1]["reason"] == "max_tokens"
    assert msgs[-1]["stats"]["deltas"] == 4

    with client.websocket_connect("/v1/stream") as ws:
        ws.receive_text()
        ws.send_text(req("t", n_tokens=500, delay=0.01))
        ws.send_text(json.dumps({"type": "stop", "id": "t"}))
        msgs = collect(ws, "t")
    assert msgs[-1]["reason"] == "stopped"


def test_inflight_limit_and_duplicate_id(client):
    settings.set_override("YUNSHU_WS_MAX_INFLIGHT", 2)
    try:
        with client.websocket_connect("/v1/stream") as ws:
            ws.receive_text()
            ws.send_text(req("1", n_tokens=100, delay=0.02))
            ws.send_text(req("2", n_tokens=100, delay=0.02))
            ws.send_text(req("3"))
            ws.send_text(req("1"))
            errs = {}
            while len(errs) < 2:
                m = json.loads(ws.receive_text())
                if m["type"] == "error":
                    errs[m["id"]] = m["status"]
            assert errs == {"3": 429, "1": 409}
            for rid in ("1", "2"):  # let cleanup finish before the harness exits
                ws.send_text(json.dumps({"type": "cancel", "id": rid}))
            done = 0
            while done < 2:
                done += json.loads(ws.receive_text())["type"] == "done"
    finally:
        settings.clear_overrides()


def test_heartbeat(client):
    settings.set_override("YUNSHU_WS_PING_INTERVAL", 0.05)
    try:
        with client.websocket_connect("/v1/stream") as ws:
            ws.receive_text()
            assert json.loads(ws.receive_text())["type"] == "ping"
    finally:
        settings.clear_overrides()


def test_backpressure_bounds_queue(client):
    settings.set_override("YUNSHU_WS_SEND_QUEUE", 4)
    try:
        with client.websocket_connect("/v1/stream") as ws:
            ws.receive_text()
            ws.send_text(req("big", n_tokens=5000))
            # Read slowly: the connection must not buffer 5000 events unboundedly;
            # all are still delivered in order.
            n = 0
            while True:
                m = json.loads(ws.receive_text())
                if m["type"] == "event":
                    n += 1
                if m["type"] == "done":
                    break
            assert n == 5000
    finally:
        settings.clear_overrides()


def test_responses_ws_mode_raw_events_and_stream_lanes(client):
    with client.websocket_connect("/v1/responses") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "response.create",
                    "model": "m",
                    "input": "x",
                    "tag": "A",
                    "stream_id": "s1",
                    "delay": 0.02,
                }
            )
        )
        ws.send_text(
            json.dumps(
                {
                    "type": "response.create",
                    "model": "m",
                    "input": "x",
                    "tag": "B",
                    "stream_id": "s1",
                }
            )
        )
        ws.send_text(
            json.dumps(
                {
                    "type": "response.create",
                    "model": "m",
                    "input": "x",
                    "tag": "C",
                    "stream_id": "s2",
                    "delay": 0.0,
                }
            )
        )
        completed, order = [], []
        while len(completed) < 3:
            m = json.loads(ws.receive_text())
            order.append((m["type"], m["tag"]))
            if m["type"] == "response.completed":
                completed.append(m["tag"])
    # same stream_id => FIFO (A fully before B); different lane C runs in parallel
    assert order.index(("response.completed", "A")) < order.index(
        ("response.created", "B")
    )
    assert completed.index("C") < completed.index("B")


def test_auth_rejected_before_upgrade():
    settings.set_override("YUNSHU_AUTH_TOKEN", "sekrit")
    settings.set_override("YUNSHU_AUTH_DISABLED", False)
    try:
        with TestClient(make_app()) as c:
            with pytest.raises(WebSocketDisconnect):
                with c.websocket_connect("/v1/stream"):
                    pass
            with pytest.raises(WebSocketDisconnect):
                with c.websocket_connect(
                    "/v1/stream", headers={"authorization": "Bearer no"}
                ):
                    pass
            with c.websocket_connect(
                "/v1/stream", headers={"authorization": "Bearer sekrit"}
            ) as ws:
                assert json.loads(ws.receive_text())["type"] == "session.created"
                ws.send_text(req("k", n_tokens=1))
                collect(ws, "k")
            assert (
                STATE["seen_auth"][-1] == "Bearer sekrit"
            )  # forwarded to the inner request
    finally:
        settings.clear_overrides()


# ── real server over a Unix domain socket: client helper, curl-style HTTP, openai SDK ──


@pytest.fixture()
def uds_server(tmp_path):
    import tempfile

    import uvicorn

    path = tempfile.mkdtemp(prefix="ys", dir="/tmp") + "/y.sock"  # AF_UNIX path limit
    server = uvicorn.Server(
        uvicorn.Config(make_app(), uds=path, log_level="error", lifespan="on")
    )
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started
    yield path
    server.should_exit = True
    th.join(timeout=10)


def test_uds_http_and_openai_sdk(uds_server):
    import httpx
    from openai import OpenAI

    transport = httpx.HTTPTransport(uds=uds_server)
    with httpx.Client(transport=transport, base_url="http://yunshu") as h:
        r = h.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [],
                "stream": True,
                "stream_options": {"include_usage": True},
                "n_tokens": 2,
            },
        )
        assert r.status_code == 200 and "t1 " in r.text
    oa = OpenAI(
        base_url="http://yunshu/v1",
        api_key="x",
        http_client=httpx.Client(transport=httpx.HTTPTransport(uds=uds_server)),
    )
    chunks = list(
        oa.chat.completions.create(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"n_tokens": 3},
        )
    )
    assert (
        "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
        == "t0 t1 t2 "
    )


def test_uds_websocket_client_helper(uds_server):
    from yunshu_client import StreamError, YunshuStream

    async def main():
        async with YunshuStream("ws://yunshu/v1/stream", uds=uds_server) as conn:
            assert conn.session["protocol"] == "yunshu.stream"

            async def one(name, n):
                txt = ""
                async for m in conn.chat(
                    {"model": "m", "messages": [], "n_tokens": n, "delay": 0.01},
                    id=name,
                ):
                    txt += m["data"]["choices"][0]["delta"]["content"]
                return txt

            a, b = await asyncio.gather(one("a", 2), one("b", 3))
            assert a == "t0 t1 " and b == "t0 t1 t2 "
            with pytest.raises(StreamError) as ei:
                async for _ in conn.chat({"model": "bad", "messages": []}):
                    pass
            assert ei.value.status == 404

            got = []
            async for m in conn.chat(
                {"model": "m", "messages": [], "n_tokens": 500, "delay": 0.01},
                id="c",
                with_done=True,
            ):
                got.append(m)
                if len(got) == 3:
                    await conn.cancel("c")
            assert got[-1]["reason"] == "cancelled"

    asyncio.run(main())
