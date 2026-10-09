"""The console process proxies the engine API stream-safely and answers its own history routes."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request, WebSocket
from starlette.responses import JSONResponse, StreamingResponse

from yunshu_console.app import create_app
from yunshu_console.poller import FIELDS
from yunshu_console.store import HistoryStore


def fake_engine() -> tuple[FastAPI, dict]:
    log: dict = {"finished_stream": False, "bodies": [], "headers": []}
    app = FastAPI()

    @app.get("/v1/echo")
    async def echo(request: Request):
        log["headers"].append(dict(request.headers))
        return {
            "auth": request.headers.get("authorization"),
            "key": request.headers.get("x-api-key"),
            "origin": request.headers.get("origin"),
            "referer": request.headers.get("referer"),
            "query": request.url.query,
            "xff": request.headers.get("x-forwarded-for"),
        }

    @app.post("/v1/upload")
    async def upload(request: Request):
        body = b"".join([chunk async for chunk in request.stream()])
        log["bodies"].append(len(body))
        return {"bytes": len(body), "type": request.headers.get("content-type")}

    @app.get("/v1/sse")
    async def sse():
        async def gen():
            for i in range(3):
                yield f"data: {i}\n\n".encode()
                await asyncio.sleep(0.15)
            log["finished_stream"] = True

        return StreamingResponse(
            gen(), media_type="text/event-stream", headers={"x-engine": "1"}
        )

    @app.get("/v1/missing")
    async def missing():
        return JSONResponse(
            {"error": "nope"}, status_code=404, headers={"WWW-Authenticate": "Bearer"}
        )

    @app.get("/v1/yunshu/status")
    async def status():
        return {"object": "yunshu.status", "uptime_s": 1}

    @app.websocket("/v1/realtime")
    async def realtime(ws: WebSocket):
        await ws.accept(
            subprotocol="yunshu.v1"
            if "yunshu.v1" in (ws.headers.get("sec-websocket-protocol") or "")
            else None
        )
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text") is not None:
                await ws.send_text("echo:" + msg["text"])
            else:
                await ws.send_bytes(msg["bytes"][::-1])

    return app, log


def console_over(engine_app: FastAPI, store=None, poll=False) -> FastAPI:
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=engine_app), base_url="http://engine"
    )
    return create_app(
        "http://engine", store=store, poll=poll, client=client, static_dir=None
    )


def asgi(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://console"
    )


def test_requests_pass_through_with_credentials_and_without_the_browsers_origin():
    engine, _ = fake_engine()

    async def go():
        async with asgi(console_over(engine)) as c:
            r = await c.get(
                "/v1/echo?a=1&b=two",
                headers={
                    "Authorization": "Bearer abc",
                    "x-api-key": "k",
                    "Origin": "http://console",
                    "Referer": "http://console/console/",
                },
            )
            return r.json()

    out = asyncio.run(go())
    assert out["auth"] == "Bearer abc" and out["key"] == "k"
    assert out["origin"] is None and out["referer"] is None
    assert out["query"] == "a=1&b=two"
    assert out["xff"]


def test_status_codes_and_headers_come_back_as_the_engine_sent_them():
    engine, _ = fake_engine()

    async def go():
        async with asgi(console_over(engine)) as c:
            return await c.get("/v1/missing"), await c.get("/v1/never-registered")

    missing, unknown = asyncio.run(go())
    assert (
        missing.status_code == 404 and missing.headers["www-authenticate"] == "Bearer"
    )
    assert missing.json() == {"error": "nope"}
    assert unknown.status_code == 404


def test_a_large_body_is_streamed_to_the_engine_intact():
    engine, log = fake_engine()
    payload = b"x" * (3 * 1024 * 1024)

    async def go():
        async with asgi(console_over(engine)) as c:
            return (
                await c.post(
                    "/v1/upload",
                    content=payload,
                    headers={"content-type": "application/octet-stream"},
                )
            ).json()

    out = asyncio.run(go())
    assert out == {"bytes": len(payload), "type": "application/octet-stream"}


def test_an_event_stream_reaches_the_browser_while_the_engine_is_still_producing_it():
    """Over real sockets (an in-process ASGI transport buffers a whole response)."""
    engine, log = fake_engine()
    eport, cport = free_port(), free_port()
    es, et = serve(engine, eport)
    console = create_app(f"http://127.0.0.1:{eport}", poll=False, static_dir=None)
    cs, ct = serve(console, cport)
    try:
        with (
            httpx.Client(timeout=10) as c,
            c.stream("GET", f"http://127.0.0.1:{cport}/v1/sse") as r,
        ):
            chunks = r.iter_raw()
            first = next(chunks)
            finished_before = log["finished_stream"]
            rest = b"".join(chunks)
            headers = r.headers
        assert (
            headers["content-type"].startswith("text/event-stream")
            and headers["x-engine"] == "1"
        )
        assert first.startswith(b"data: 0")
        assert finished_before is False, (
            "the first event arrived before the engine finished"
        )
        assert b"data: 2" in first + rest
    finally:
        cs.should_exit = es.should_exit = True
        ct.join(5)
        et.join(5)


def test_unreachable_engine_is_a_502_engine_unreachable_at_once():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(refuse), base_url="http://engine"
    )
    app = create_app("http://engine", poll=False, client=client)

    async def go():
        async with asgi(app) as c:
            t = time.perf_counter()
            r = await c.get("/v1/yunshu/status")
            return r, time.perf_counter() - t

    r, took = asyncio.run(go())
    assert r.status_code == 502 and took < 1.0
    body = r.json()
    assert (
        body["error"]["type"] == "engine_unreachable"
        and "not reachable" in body["error"]["message"]
    )


def test_history_is_answered_locally_even_when_the_engine_is_down(tmp_path):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    store = HistoryStore(tmp_path / "h.sqlite", FIELDS, flush_s=3600)
    now = time.time()
    for i in range(20):
        store.add_sample(now - 20 + i, dict.fromkeys(FIELDS, 1.0))
    store.add_event(now - 5, "engine_unreachable", {"reason": "x"})
    store.add_request(
        {"request_id": "req_a", "t": now - 3, "model": "m", "status": 200}
    )
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(refuse), base_url="http://engine"
    )
    app = create_app("http://engine", store=store, poll=False, client=client)

    async def go():
        async with asgi(app) as c:
            m = (
                await c.get("/v1/yunshu/metrics/history", params={"since": now - 60})
            ).json()
            r = (await c.get("/v1/yunshu/requests/history")).json()
            s = (await c.get("/v1/yunshu/console")).json()
            bad = await c.get(
                "/v1/yunshu/metrics/history", params={"since": 5, "until": 2}
            )
            lim = await c.get("/v1/yunshu/requests/history", params={"limit": 0})
            return m, r, s, bad, lim

    m, r, s, bad, lim = asyncio.run(go())
    assert (
        m["enabled"]
        and len(m["series"]["t"]) == 20
        and m["events"][0]["kind"] == "engine_unreachable"
    )
    assert [x["request_id"] for x in r["data"]] == ["req_a"]
    assert (
        s["object"] == "yunshu.console"
        and s["recording"] is True
        and s["store"]["bytes"] > 0
    )
    assert bad.status_code == 400 and lim.status_code == 400
    store.close()


def test_history_routes_need_the_token_when_one_is_configured(monkeypatch, tmp_path):
    from yunshu_engine import settings

    store = HistoryStore(tmp_path / "h.sqlite", FIELDS, flush_s=3600)
    monkeypatch.setattr(
        settings, "get", lambda key: "tok" if key == "YUNSHU_AUTH_TOKEN" else None
    )
    engine, _ = fake_engine()
    app = console_over(engine, store=store)

    async def go():
        async with asgi(app) as c:
            no = await c.get("/v1/yunshu/metrics/history")
            bad = await c.get(
                "/v1/yunshu/requests/history", headers={"Authorization": "Bearer nope"}
            )
            ok = await c.get(
                "/v1/yunshu/metrics/history", headers={"Authorization": "Bearer tok"}
            )
            return no.status_code, bad.status_code, ok.status_code

    assert asyncio.run(go()) == (401, 401, 200)
    store.close()


def test_without_a_store_history_says_disabled_and_the_console_still_proxies():
    engine, _ = fake_engine()

    async def go():
        async with asgi(console_over(engine)) as c:
            m = (await c.get("/v1/yunshu/metrics/history")).json()
            s = (await c.get("/v1/yunshu/status")).json()
            return m, s

    m, s = asyncio.run(go())
    assert (
        m["enabled"] is False
        and m["series"]["t"] == []
        and s["object"] == "yunshu.status"
    )


def test_root_goes_to_the_console():
    engine, _ = fake_engine()

    async def go():
        async with asgi(console_over(engine)) as c:
            return await c.get("/", follow_redirects=False)

    r = asyncio.run(go())
    assert r.status_code == 307 and r.headers["location"] == "/console/"


# ── WebSocket, over real sockets ─────────────────────────────────────────


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(app, port) -> tuple[uvicorn.Server, threading.Thread]:
    server = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="error", ws="websockets"
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    return server, thread


def test_a_websocket_is_piped_both_ways_with_its_subprotocol():
    import websockets

    engine, _ = fake_engine()
    eport, cport = free_port(), free_port()
    es, et = serve(engine, eport)
    console = create_app(f"http://127.0.0.1:{eport}", poll=False, static_dir=None)
    cs, ct = serve(console, cport)
    try:

        async def go():
            async with websockets.connect(
                f"ws://127.0.0.1:{cport}/v1/realtime", subprotocols=["yunshu.v1"]
            ) as ws:
                await ws.send("hello")
                text = await ws.recv()
                await ws.send(b"abc")
                blob = await ws.recv()
                return ws.subprotocol, text, blob

        sub, text, blob = asyncio.run(go())
        assert (sub, text, blob) == ("yunshu.v1", "echo:hello", b"cba")
    finally:
        cs.should_exit = es.should_exit = True
        ct.join(5)
        et.join(5)


def test_a_websocket_to_a_dead_engine_closes_instead_of_hanging():
    import websockets

    cport = free_port()
    console = create_app(f"http://127.0.0.1:{free_port()}", poll=False, static_dir=None)
    cs, ct = serve(console, cport)
    try:

        async def go():
            try:
                async with websockets.connect(
                    f"ws://127.0.0.1:{cport}/v1/realtime"
                ) as ws:
                    await asyncio.wait_for(ws.recv(), 3)
            except (websockets.ConnectionClosed, websockets.InvalidStatus, OSError):
                return "closed"
            return "open"

        assert asyncio.run(go()) == "closed"
    finally:
        cs.should_exit = True
        ct.join(5)


def test_json_dumps_roundtrip_of_a_history_payload(tmp_path):
    store = HistoryStore(tmp_path / "h.sqlite", FIELDS, flush_s=3600)
    store.add_sample(time.time() - 1, dict.fromkeys(FIELDS))
    json.dumps(store.read())
    store.close()


@pytest.mark.parametrize(
    "module",
    [
        "yunshu_console.app",
        "yunshu_console.poller",
        "yunshu_console.store",
        "yunshu_cli.console_cmd",
    ],
)
def test_nothing_in_the_console_process_loads_mlx(module):
    """A fresh interpreter imports the console process modules: no mlx, mlx-lm, mlx-vlm or
    mlx-audio module may be loaded, so it starts instantly and an engine fault cannot reach it."""
    import subprocess
    import sys

    code = (
        f"import importlib, sys; importlib.import_module({module!r});"
        "bad = sorted({m.split('.')[0] for m in sys.modules if m.split('.')[0] in "
        "('mlx', 'mlx_lm', 'mlx_vlm', 'mlx_audio', 'torch', 'transformers')});"
        "print('LOADED', bad); sys.exit(1 if bad else 0)"
    )
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert run.returncode == 0, run.stdout + run.stderr


def test_every_answer_is_stamped_as_coming_through_the_console_process():
    engine, _ = fake_engine()

    async def go():
        async with asgi(console_over(engine)) as c:
            proxied = await c.get("/v1/yunshu/status")
            local = await c.get("/v1/yunshu/metrics/history")
            missing = await c.get("/v1/never")
            return proxied, local, missing

    for response in asyncio.run(go()):
        assert response.headers["x-yunshu-console"] == "1"
