"""R25 / F04: an engine failure mid-stream is one terminal error event in the dialect's own shape.

After the stream's headers are out the status line is spent: the failure must arrive as a
terminal event on the same response (never a second HTTP response, never a clean-looking finish),
and a request that fails before any output gets the dialect's error envelope with a 5xx.
Includes the WebSocket transports (/v1/stream, Responses WS mode, Realtime).
"""

from __future__ import annotations

import asyncio
import json
import time

import anthropic
import openai
import pytest

from yunshu_gateway.routers import realtime as rt

from .wire_clients import Clients
from .wire_harness import Script, install

USER = [{"role": "user", "content": "hi"}]
BOOM = "scripted engine failure"
BODIES = {
    "/v1/chat/completions": {
        "model": "m",
        "messages": USER,
        "stream_options": {"include_usage": True},
    },
    "/v1/completions": {
        "model": "m",
        "prompt": "hi",
        "stream_options": {"include_usage": True},
    },
    "/v1/messages": {"model": "m", "max_tokens": 32, "messages": USER},
    "/v1/responses": {"model": "m", "input": "hi"},
}


def _script(**kw):
    return Script(pieces=["a", "b", "c", "d"], error_after=2, error=BOOM, **kw)


def _asgi_post(app, path, body):
    """Every ASGI message the app sends for one POST (what a server turns into bytes)."""
    sent: list[dict] = []
    payload = json.dumps(body).encode()

    async def run():
        first = True

        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": payload, "more_body": False}
            await asyncio.sleep(30)
            return {"type": "http.disconnect"}

        async def send(msg):
            sent.append(msg)

        scope = {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [
                (b"host", b"testserver"),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode()),
            ],
            "client": ("127.0.0.1", 1),
            "server": ("testserver", 80),
            "scheme": "http",
            "app": app,
        }
        await asyncio.wait_for(app(scope, receive, send), 10)

    asyncio.run(run())
    return sent


@pytest.mark.parametrize("path", list(BODIES))
def test_one_response_and_a_terminal_error(monkeypatch, path):
    c, _ = install(monkeypatch, _script())
    sent = _asgi_post(c.app, path, {**BODIES[path], "stream": True})
    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert len(starts) == 1 and starts[0]["status"] == 200, (
        "a second response was started"
    )
    body = b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )
    assert not sent[-1].get("more_body"), "the response body was never closed"
    text = body.decode()
    assert BOOM in text
    # the first bytes of generation made it out before the failure
    assert text.index('"a"') < text.index(BOOM)
    # nothing that reads as success after the error: no finish chunk, completed, stop events
    tail = text[text.index(BOOM) :]
    for marker in (
        '"finish_reason": "stop"',
        "message_stop",
        "response.completed",
        '"usage": {"prompt_tokens"',
    ):
        assert marker not in tail, (path, marker)


def test_chat_stream_error_event_shape_and_sdk(monkeypatch):
    c, _ = install(monkeypatch, _script())
    r = c.post(
        "/v1/chat/completions", json={**BODIES["/v1/chat/completions"], "stream": True}
    )
    frames = [x[5:].strip() for x in r.text.splitlines() if x.startswith("data:")]
    assert frames[-1] == "[DONE]"
    err = json.loads(frames[-2])["error"]
    assert err["message"] == BOOM and err["type"] == "server_error"
    assert err["param"] is None and err["code"] == "internal_error"
    cl = Clients(c)
    with pytest.raises(openai.APIError) as e:
        for _ in cl.oa.chat.completions.create(model="m", messages=USER, stream=True):
            pass
    assert BOOM in str(e.value)


def test_completions_stream_error_is_not_a_clean_finish(monkeypatch):
    c, _ = install(monkeypatch, _script())
    cl = Clients(c)
    seen = []
    with pytest.raises(openai.APIError) as e:
        for ev in cl.oa.completions.create(model="m", prompt="hi", stream=True):
            seen.append(ev)
    assert BOOM in str(e.value)
    assert not any(ch.finish_reason for ev in seen for ch in ev.choices)


def test_messages_stream_error_is_an_error_event_without_message_delta(monkeypatch):
    c, _ = install(monkeypatch, _script())
    r = c.post("/v1/messages", json={**BODIES["/v1/messages"], "stream": True})
    events = [x[6:].strip() for x in r.text.splitlines() if x.startswith("event:")]
    assert events[-1] == "error"
    assert "message_delta" not in events and "message_stop" not in events
    # every opened block is closed before the error
    assert events.count("content_block_start") == events.count("content_block_stop")
    last = [x for x in r.text.splitlines() if x.startswith("data:")][-1]
    body = json.loads(last[5:])
    assert body == {"type": "error", "error": {"type": "api_error", "message": BOOM}}
    cl = Clients(c)
    with pytest.raises(anthropic.APIError) as e:
        for _ in cl.an.messages.create(
            model="m", max_tokens=9, messages=USER, stream=True
        ):
            pass
    assert BOOM in str(e.value)


def test_responses_stream_error_is_response_failed(monkeypatch):
    c, _ = install(monkeypatch, _script())
    cl = Clients(c)
    events = list(cl.oa.responses.create(model="m", input="hi", stream=True))
    assert events[-1].type == "response.failed"
    assert not any(e.type == "response.completed" for e in events)
    failed = events[-1].response
    assert failed.status == "failed"
    assert failed.error is not None and BOOM in failed.error.message
    assert failed.error.code


@pytest.mark.parametrize("path", ["/api/chat", "/api/generate"])
def test_ollama_stream_error_is_an_error_line(monkeypatch, path):
    c, _ = install(monkeypatch, _script())
    body = (
        {"model": "m", "prompt": "hi"}
        if "generate" in path
        else {"model": "m", "messages": USER}
    )
    r = c.post(path, json=body)
    lines = [json.loads(x) for x in r.text.splitlines() if x.strip()]
    assert lines[-1] == {"error": BOOM}
    assert not any(ln.get("done") for ln in lines)


@pytest.mark.parametrize(
    "path,shape",
    [
        ("/v1/chat/completions", "openai"),
        ("/v1/completions", "openai"),
        ("/v1/messages", "anthropic"),
        ("/v1/responses", "openai"),
        ("/api/chat", "ollama"),
        ("/api/generate", "ollama"),
    ],
)
def test_error_before_any_output_is_a_5xx_in_the_dialect_envelope(
    monkeypatch, path, shape
):
    c, _ = install(monkeypatch, Script(pieces=["a"], error_after=0, error=BOOM))
    body = BODIES.get(path) or (
        {"model": "m", "prompt": "hi"}
        if "generate" in path
        else {"model": "m", "messages": USER}
    )
    r = c.post(path, json={**body, "stream": False})
    assert r.status_code == 500
    j = r.json()
    if shape == "anthropic":
        assert j["type"] == "error" and j["error"]["type"] == "api_error"
    elif shape == "ollama":
        assert isinstance(j["error"], str)
    else:
        e = j["error"]
        assert e["type"] == "server_error" and e["param"] is None
        assert e["code"] == "internal_error" and e["message"]
    assert BOOM not in r.text  # an internal failure message is not echoed on a 500 body


# ── WebSocket transports ─────────────────────────────────────────────────────────
def _drain(ws, stop):
    out = []
    while True:
        m = json.loads(ws.receive_text())
        if m.get("type") == "ping":
            continue
        out.append(m)
        if stop(m):
            return out


@pytest.mark.parametrize(
    "api", ["chat.completions", "completions", "messages", "responses"]
)
def test_stream_ws_error_ends_with_done_error(monkeypatch, api):
    c, _ = install(monkeypatch, _script())
    body = {
        "chat.completions": {"model": "m", "messages": USER},
        "completions": {"model": "m", "prompt": "hi"},
        "messages": {"model": "m", "max_tokens": 9, "messages": USER},
        "responses": {"model": "m", "input": "hi"},
    }[api]
    with c.websocket_connect("/v1/stream") as ws:
        ws.receive_text()  # session.created
        ws.send_text(
            json.dumps({"type": "request", "id": "a", "api": api, "body": body})
        )
        msgs = _drain(ws, lambda m: m.get("type") == "done")
    assert msgs[-1]["reason"] == "error" and msgs[-1]["id"] == "a"
    assert sum(1 for m in msgs if m.get("type") == "done") == 1
    assert BOOM in json.dumps(msgs)
    # the failure event is forwarded, and nothing after it claims success
    kinds = [m.get("event") or m.get("type") for m in msgs]
    assert "response.completed" not in kinds and "message_stop" not in kinds


def test_responses_ws_error_is_response_failed(monkeypatch):
    c, _ = install(monkeypatch, _script())
    with c.websocket_connect("/v1/responses") as ws:
        ws.send_text(
            json.dumps({"type": "response.create", "model": "m", "input": "hi"})
        )
        msgs = _drain(
            ws,
            lambda m: (
                m.get("type") in ("response.failed", "response.completed", "error")
            ),
        )
        # Responses WS mode has no `done` message: let the server finish its own teardown
        # before the test client closes the socket (the close races it otherwise).
        time.sleep(0.2)
    assert msgs[-1]["type"] == "response.failed"
    assert msgs[-1]["response"]["status"] == "failed"
    assert [m["sequence_number"] for m in msgs if "sequence_number" in m] == sorted(
        m["sequence_number"] for m in msgs if "sequence_number" in m
    )


def _wait(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_stream_ws_cancel_stops_generation(monkeypatch):
    c, eng = install(
        monkeypatch, Script(pieces=[f"t{i}" for i in range(400)], delay=0.01)
    )
    with c.websocket_connect("/v1/stream") as ws:
        ws.receive_text()
        ws.send_text(
            json.dumps(
                {
                    "type": "request",
                    "id": "a",
                    "api": "chat.completions",
                    "body": {"model": "m", "messages": USER},
                }
            )
        )
        got = []
        while len(got) < 3:
            m = json.loads(ws.receive_text())
            if m.get("type") == "event":
                got.append(m)
        ws.send_text(json.dumps({"type": "cancel", "id": "a"}))
        msgs = _drain(ws, lambda m: m.get("type") == "done")
    assert msgs[-1]["reason"] == "cancelled"
    assert _wait(lambda: eng.stopped_early >= 1), (
        "the engine kept generating after cancel"
    )
    assert msgs[-1]["stats"]["deltas"] < 400


def test_responses_ws_cancel_stops_generation(monkeypatch):
    c, eng = install(
        monkeypatch, Script(pieces=[f"t{i}" for i in range(400)], delay=0.01)
    )
    with c.websocket_connect("/v1/responses") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "response.create",
                    "model": "m",
                    "input": "hi",
                    "client_request_id": "r1",
                }
            )
        )
        n = 0
        while n < 3:
            if (
                json.loads(ws.receive_text()).get("type")
                == "response.output_text.delta"
            ):
                n += 1
        ws.send_text(json.dumps({"type": "response.cancel", "client_request_id": "r1"}))
        stopped = _wait(lambda: eng.stopped_early >= 1)
        time.sleep(0.2)  # let the server finish its teardown before the client closes
    assert stopped, "the engine kept generating after cancel"


# ── Realtime ─────────────────────────────────────────────────────────────────────
def _realtime(monkeypatch, script):
    c, eng = install(monkeypatch, script)
    monkeypatch.setattr(rt.RealtimeSession, "_resolve_engine", lambda self: eng)
    return c, eng


def _rt_turn(ws):
    ws.send_text(
        json.dumps(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
            }
        )
    )
    ws.send_text(json.dumps({"type": "response.create"}))


def test_realtime_engine_error_is_error_event_and_failed_response(monkeypatch):
    c, _ = _realtime(monkeypatch, _script())
    with c.websocket_connect("/v1/realtime?model=m") as ws:
        _drain(ws, lambda m: m.get("type") == "session.created")
        _rt_turn(ws)
        msgs = _drain(ws, lambda m: m.get("type") == "response.done")
    kinds = [m["type"] for m in msgs]
    err = [m for m in msgs if m["type"] == "error"]
    assert len(err) == 1 and err[0]["error"]["message"] == BOOM
    assert kinds.index("error") < kinds.index("response.done")
    done = msgs[-1]["response"]
    assert done["status"] == "failed"
    assert done["status_details"]["type"] == "failed"
    assert done["status_details"]["error"]["type"] == "server_error"


def test_realtime_cancel_ends_response_cancelled(monkeypatch):
    c, eng = _realtime(
        monkeypatch, Script(pieces=[f"t{i} " for i in range(400)], delay=0.01)
    )
    with c.websocket_connect("/v1/realtime?model=m") as ws:
        _drain(ws, lambda m: m.get("type") == "session.created")
        _rt_turn(ws)
        seen = 0
        while seen < 3:
            if json.loads(ws.receive_text())["type"] == "response.output_text.delta":
                seen += 1
        ws.send_text(json.dumps({"type": "response.cancel"}))
        msgs = _drain(ws, lambda m: m.get("type") == "response.done")
    assert msgs[-1]["response"]["status"] == "cancelled"
    assert _wait(lambda: eng.stopped_early + eng.cancelled >= 1)
