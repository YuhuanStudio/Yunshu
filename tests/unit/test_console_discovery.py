"""`yunshu console --engine auto`: discovery, sticky selection, switching, pinning, proxy follow."""

from __future__ import annotations

import asyncio
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, WebSocket
from starlette.responses import StreamingResponse

from yunshu_console.app import create_app
from yunshu_console.discovery import Discovery, is_yunshu_health, parse_ports
from yunshu_console.poller import FIELDS, Poller
from yunshu_console.store import HistoryStore


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Engine:
    """A tiny fake Yunshu engine on a real ephemeral port."""

    def __init__(self, name: str, uptime: float = 10.0, yunshu: bool = True) -> None:
        self.name, self.uptime, self.port = name, uptime, free_port()
        self.server = self.thread = None
        app = FastAPI()
        me = self

        @app.get("/health")
        async def health():
            if not yunshu:
                return {"status": "ok"}
            return {
                "status": "ok",
                "engine": {"loaded": True},
                "server_state": "ready",
                "uptime_seconds": me.uptime,
            }

        @app.get("/v1/models")
        async def models():
            return {"data": [{"id": f"model-{me.name}"}]}

        @app.get("/who")
        async def who():
            return {"engine": me.name}

        @app.get("/slow")
        async def slow():
            async def gen():
                yield b"a-"
                await asyncio.sleep(0.8)
                yield f"end-{me.name}".encode()

            return StreamingResponse(gen())

        @app.websocket("/ws")
        async def ws(sock: WebSocket):
            await sock.accept()
            await sock.send_text(me.name)
            await sock.close()

        @app.get("/v1/yunshu/status")
        async def status():
            return {"uptime_s": me.uptime, "pid": 1000 + me.port % 1000, "models": []}

        @app.get("/v1/yunshu/requests/recent")
        async def recent():
            return {
                "boot_id": me.name,
                "latest_seq": 1,
                "data": [
                    {"request_id": f"r-{me.name}", "t": time.time(), "status": 200}
                ],
            }

        self.app = app

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> Engine:
        self.server = uvicorn.Server(
            uvicorn.Config(
                self.app,
                host="127.0.0.1",
                port=self.port,
                log_level="error",
                ws="websockets",
            )
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                break
            time.sleep(0.05)
        return self

    def stop(self) -> None:
        if self.server is not None:
            self.server.should_exit = True
            self.thread.join(5)
            self.server = None


@pytest.fixture
def engines():
    made: list[Engine] = []

    def make(*a, **k) -> Engine:
        e = Engine(*a, **k).start()
        made.append(e)
        return e

    yield make
    for e in made:
        e.stop()


def discovery_for(*es: Engine, **kw) -> Discovery:
    return Discovery([e.url for e in es], **kw)


def test_parse_ports_and_health_shape():
    assert parse_ports("8000,18990-18992, 8000,x,9-7") == [
        8000,
        18990,
        18991,
        18992,
        7,
        8,
        9,
    ]
    assert is_yunshu_health(
        {
            "status": "ok",
            "engine": {"loaded": False},
            "server_state": "ready",
            "uptime_seconds": 1.5,
        }
    )
    assert not is_yunshu_health({"status": "ok"})
    assert not is_yunshu_health(
        {"status": "ok", "engine": {}, "server_state": "x", "uptime_seconds": True}
    )


def test_a_port_that_is_not_yunshu_is_ignored(engines):
    other = engines("other", yunshu=False)
    real = engines("real")
    closed = f"http://127.0.0.1:{free_port()}"

    async def go():
        d = Discovery([other.url, closed, real.url])
        await d.scan()
        await d.aclose()
        return d.snapshot()

    snap = asyncio.run(go())
    assert [e["url"] for e in snap["data"]] == [real.url]
    assert snap["selected"] == real.url and snap["waiting"] is False
    assert snap["data"][0]["model"] == "model-real"


def test_nothing_up_is_waiting_not_an_error(engines):
    async def go():
        d = Discovery([f"http://127.0.0.1:{free_port()}"])
        await d.scan()
        await d.aclose()
        return d.snapshot()

    snap = asyncio.run(go())
    assert snap["waiting"] is True and snap["selected"] is None and snap["data"] == []


def test_selection_is_sticky_then_switches_to_the_newest_when_it_disappears(engines):
    a = engines("a", uptime=500)
    b = engines("b", uptime=100)
    c = engines("c", uptime=5)

    async def go():
        d = Discovery([a.url, b.url, c.url][:1])
        await d.scan()
        assert d.selected == a.url
        d.candidates = [a.url, b.url, c.url]
        await d.scan()
        # b and c are newer, but a is healthy: stay
        assert d.selected == a.url
        a.stop()
        await d.scan()  # one miss is not enough
        assert d.selected == a.url
        await d.scan()
        picked = d.selected
        await d.aclose()
        return picked

    assert asyncio.run(go()) == c.url  # smallest uptime among the survivors


def test_pin_wins_while_live_and_unpin_returns_to_automatic(engines):
    a = engines("a", uptime=500)
    b = engines("b", uptime=5)

    async def go():
        d = discovery_for(a, b)
        await d.scan()
        assert d.selected == b.url  # newest first
        assert d.pin(a.url) is True and d.selected == a.url
        await d.scan()
        assert d.selected == a.url  # pinned beats newer
        assert d.pin("http://127.0.0.1:1") is False
        assert d.pin(None) is True and d.pinned is None
        await d.scan()
        await d.aclose()
        return d.selected

    assert asyncio.run(go()) == a.url  # sticky: unpinning does not move


def test_proxy_follows_the_switch_and_does_not_cut_an_inflight_stream(engines):
    a = engines("a", uptime=500)
    b = engines("b", uptime=5)

    async def go():
        d = discovery_for(a, b)
        d.candidates = [a.url]
        await d.scan()
        app = create_app(a.url, poll=False, static_dir=None, discovery=d)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://console"
        ) as c:
            first = (await c.get("/who")).json()["engine"]
            parts = []
            async with c.stream("GET", "/slow") as r:
                it = r.aiter_bytes()
                parts.append(await it.__anext__())
                # the followed engine switches to b mid-stream
                d.candidates = [a.url, b.url]
                await d.scan()
                d.pin(b.url)
                second = (await c.get("/who")).json()["engine"]
                async for chunk in it:
                    parts.append(chunk)
        await d.aclose()
        return first, second, b"".join(parts), app.state.engine_url

    first, second, body, url = asyncio.run(go())
    assert (first, second) == ("a", "b")
    assert body == b"a-end-a"  # the old stream finished on the old engine
    assert url == b.url


def test_websocket_connects_to_the_current_engine(engines):
    import websockets

    a = engines("a", uptime=500)
    b = engines("b", uptime=5)
    cport = free_port()

    async def go():
        d = discovery_for(a)
        await d.scan()
        app = create_app(a.url, poll=False, static_dir=None, discovery=d)
        server = uvicorn.Server(
            uvicorn.Config(
                app, host="127.0.0.1", port=cport, log_level="error", ws="websockets"
            )
        )
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        while not server.started:
            await asyncio.sleep(0.05)
        try:
            async with websockets.connect(f"ws://127.0.0.1:{cport}/ws") as ws:
                one = await ws.recv()
            d.candidates = [a.url, b.url]
            await d.scan()
            d.pin(b.url)
            async with websockets.connect(f"ws://127.0.0.1:{cport}/ws") as ws:
                two = await ws.recv()
            return one, two
        finally:
            server.should_exit = True
            t.join(5)
            await d.aclose()

    assert asyncio.run(go()) == ("a", "b")


def test_engines_api_lists_pins_and_unpins(engines):
    a = engines("a", uptime=500)
    b = engines("b", uptime=5)

    async def go():
        d = discovery_for(a, b)
        await d.scan()
        app = create_app(a.url, poll=False, static_dir=None, discovery=d)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://console"
        ) as c:
            listing = (await c.get("/console/api/engines")).json()
            pinned = (
                await c.post("/console/api/engines", json={"port": a.port})
            ).json()
            bad = await c.post(
                "/console/api/engines", json={"url": "http://127.0.0.1:1"}
            )
            unpinned = (await c.post("/console/api/engines", json={"url": None})).json()
        await d.aclose()
        return listing, pinned, bad.status_code, unpinned

    listing, pinned, bad, unpinned = asyncio.run(go())
    assert listing["mode"] == "auto" and listing["selected"] == b.url
    row = {e["url"]: e for e in listing["data"]}[b.url]
    assert row["selected"] and row["model"] == "model-b" and row["uptime_seconds"] == 5
    assert pinned["pinned"] == a.url and pinned["selected"] == a.url
    assert bad == 404
    assert unpinned["pinned"] is None


def test_fixed_mode_reports_one_engine_and_refuses_pins():
    app = create_app("http://127.0.0.1:1", poll=False, static_dir=None)

    async def go():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://console"
        ) as c:
            return (await c.get("/console/api/engines")).json(), (
                await c.post("/console/api/engines", json={"url": None})
            ).status_code

    listing, code = asyncio.run(go())
    assert listing["mode"] == "fixed" and len(listing["data"]) == 1 and code == 409


def test_history_keeps_each_engine_as_its_own_session(engines, tmp_path):
    a = engines("a", uptime=500)
    b = engines("b", uptime=5)
    store = HistoryStore(tmp_path / "h.sqlite", FIELDS)

    async def go():
        p = Poller(a.url, store, interval_s=1)
        await p.tick()
        p.retarget(b.url)
        await p.tick()
        await p.aclose()

    asyncio.run(go())
    rows = store.requests_page(10)["data"]
    store.close()
    by = {r["request_id"]: r for r in rows}
    assert by["r-a"]["engine_url"] == a.url and by["r-b"]["engine_url"] == b.url
    assert by["r-a"]["engine_started"] != by["r-b"]["engine_started"]


def test_every_port_mode_follows_the_listening_ports_and_skips_known_non_engines(
    engines,
):
    from yunshu_console import discovery as disc

    other = engines("other", yunshu=False)
    a = engines("a")
    listening = [other.port, a.port]
    now = [1000.0]
    probes: list[str] = []

    async def go():
        d = Discovery([], port_source=lambda: list(listening), clock=lambda: now[0])
        real_probe = d._probe

        async def counted(url):
            probes.append(url)
            return await real_probe(url)

        d._probe = counted
        await d.scan()
        first = d.snapshot()
        b = engines("b", uptime=1.0)
        listening.append(b.port)
        now[0] += 1.0  # before the port refresh: b not seen yet
        await d.scan()
        early = [e["url"] for e in d.snapshot()["data"]]
        now[0] += disc.PORT_REFRESH_S
        await d.scan()
        later = [e["url"] for e in d.snapshot()["data"]]
        await d.aclose()
        return first, early, later, b

    first, early, later, b = asyncio.run(go())
    assert [e["url"] for e in first["data"]] == [a.url]
    assert early == [a.url]
    assert sorted(later) == sorted([a.url, b.url])
    # the non-Yunshu port answered once and is left alone for NON_ENGINE_RECHECK_S
    assert probes.count(other.url) == 1


def test_one_engine_behind_a_proxy_is_listed_once_preferring_the_server_port(engines):
    from yunshu_console import discovery as disc

    proxy = engines("same", uptime=50.0)
    server = engines("same", uptime=50.0)
    disc.SERVER_PORTS.clear()
    disc.SERVER_PORTS.add(server.port)
    try:

        async def go():
            d = Discovery([], port_source=lambda: [proxy.port, server.port])
            await d.scan()
            await d.aclose()
            return d.snapshot()

        snap = asyncio.run(go())
    finally:
        disc.SERVER_PORTS.clear()
    assert [e["url"] for e in snap["data"]] == [server.url]
    assert snap["selected"] == server.url


def test_listening_ports_drops_consoles_and_marks_servers(monkeypatch):
    from yunshu_console import discovery as disc

    def fake_run(cmd):
        if cmd[0] == "lsof":
            return "p10\nf3\nn127.0.0.1:8100\np20\nf4\nn*:18990\np30\nf5\nn*:5000\n"
        return (
            "10 /x/python -m yunshu_console --engine auto --port 8100\n"
            "20 /x/python /x/bin/yunshu serve -m /m --port 18990\n"
            "30 /usr/bin/someserver\n"
        )

    monkeypatch.setattr(disc, "_run", fake_run)
    assert disc.listening_ports() == [5000, 18990]
    assert {18990} == disc.SERVER_PORTS
    disc.SERVER_PORTS.clear()
