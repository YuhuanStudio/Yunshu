"""Console change feed: change-only publishing, ordering, coalescing, slow tabs, auth, cleanup."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_gateway.event_hub import Hub, Topic
from yunshu_gateway.routers import events as events_router


def parse(frame: str) -> tuple[str, dict]:
    lines = frame.strip().split("\n")
    ev = next(line[7:] for line in lines if line.startswith("event: "))
    data = next(line[6:] for line in lines if line.startswith("data: "))
    return ev, json.loads(data)


def make_hub(values, interval=0.05, **kw):
    state = {"i": 0, "calls": 0}

    async def fetch(_req):
        state["calls"] += 1
        return values(state)

    hub = Hub()
    hub.register(Topic("t", fetch, lambda _p: interval, **kw))
    return hub, state


async def take(gen, n, timeout=3.0):
    out = []
    async with asyncio.timeout(timeout):
        while len(out) < n:
            out.append(await gen.__anext__())
    return out


@pytest.mark.asyncio
async def test_hello_snapshot_then_changes_only():
    hub, st = make_hub(lambda s: {"n": min(s["calls"], 3)}, heartbeat_s=999)
    gen = hub.subscribe(["t"], None, boot_id="B")
    frames = await take(gen, 4)
    assert parse(frames[0]) == ("hello", {"boot_id": "B", "topics": ["t"]})
    seen = [parse(f)[1]["d"]["n"] for f in frames[1:]]
    # strictly increasing: unchanged samples are never sent, order is preserved
    assert seen == sorted(set(seen)) and seen[0] == 1
    await gen.aclose()


@pytest.mark.asyncio
async def test_unchanged_is_silent_until_heartbeat():
    hub, st = make_hub(lambda s: {"n": 1}, interval=0.02, heartbeat_s=0.3)
    gen = hub.subscribe(["t"], None, boot_id="B", keepalive_s=99)
    frames = await take(gen, 3)  # hello, snapshot, heartbeat
    v = [parse(f)[1]["v"] for f in frames[1:]]
    assert v == [1, 2]
    assert st["calls"] > 5  # sampled often, sent rarely
    await gen.aclose()


@pytest.mark.asyncio
async def test_volatile_keys_do_not_count_as_change():
    hub, _ = make_hub(
        lambda s: {"uptime_s": s["calls"], "x": 1},
        interval=0.02,
        volatile=frozenset({"uptime_s"}),
        heartbeat_s=999,
    )
    gen = hub.subscribe(["t"], None, boot_id="B", keepalive_s=0.2)
    frames = await take(gen, 3)
    assert parse(frames[1])[1]["v"] == 1
    assert frames[2].startswith(": keepalive")
    await gen.aclose()


@pytest.mark.asyncio
async def test_slow_consumer_gets_latest_and_skipped_count_not_a_queue():
    hub, st = make_hub(lambda s: {"n": s["calls"]}, interval=0.01, heartbeat_s=999)
    gen = hub.subscribe(["t"], None, boot_id="B")
    await take(gen, 2)
    await asyncio.sleep(0.5)  # tab stalls; ~50 versions pass
    (frame,) = await take(gen, 1)
    d = parse(frame)[1]
    assert d["skipped"] >= 5
    assert d["d"]["n"] >= st["calls"] - 3  # the newest value, not the oldest backlog
    await gen.aclose()


@pytest.mark.asyncio
async def test_one_sampler_for_many_tabs_and_it_stops_with_the_last():
    hub, st = make_hub(lambda s: {"n": s["calls"]}, interval=0.02)
    gens = [hub.subscribe(["t"], None, boot_id="B") for _ in range(10)]
    for g in gens:
        await take(g, 2)
    assert hub.samplers_running() == 1
    for g in gens[:-1]:
        await g.aclose()
    assert hub.samplers_running() == 1
    await gens[-1].aclose()
    assert hub.samplers_running() == 0
    calls = st["calls"]
    await asyncio.sleep(0.2)
    assert st["calls"] == calls


@pytest.mark.asyncio
async def test_failing_sampler_does_not_kill_the_feed():
    def values(s):
        if s["calls"] == 1:
            raise RuntimeError("boom")
        return {"n": 1}

    hub, _ = make_hub(values, interval=0.02)
    gen = hub.subscribe(["t"], None, boot_id="B")
    frames = await take(gen, 2)
    assert parse(frames[1])[1]["d"] == {"n": 1}
    await gen.aclose()


@pytest.mark.asyncio
async def test_reconnect_starts_with_a_snapshot():
    hub, _ = make_hub(lambda s: {"n": 7}, heartbeat_s=999)
    g1 = hub.subscribe(["t"], None, boot_id="B")
    await take(g1, 2)
    await g1.aclose()
    g2 = hub.subscribe(["t"], None, boot_id="B")
    frames = await take(g2, 2)
    assert parse(frames[1])[1]["d"] == {"n": 7}
    await g2.aclose()


def app_with(hub, monkeypatch, allow=True):
    def check(request, perm):
        if not allow:
            from fastapi import HTTPException

            raise HTTPException(401, "no")

    monkeypatch.setattr(events_router, "_check_permission", check)
    monkeypatch.setattr(events_router, "_hub", hub)
    app = FastAPI()
    app.include_router(events_router.router, prefix="/v1")
    return TestClient(app)


def test_route_refuses_without_auth_and_unknown_topics(monkeypatch):
    hub, _ = make_hub(lambda s: {"n": 1})
    assert (
        app_with(hub, monkeypatch, allow=False).get("/v1/yunshu/events").status_code
        == 401
    )
    c = app_with(hub, monkeypatch)
    assert c.get("/v1/yunshu/events?topics=nope").status_code == 400


def test_default_hub_has_every_polled_topic():
    assert set(events_router.build_hub().topics) == {
        "status",
        "host",
        "memory",
        "requests",
        "downloads",
    }


@pytest.mark.asyncio
async def test_poke_samples_now_instead_of_waiting_for_the_tick():
    hub, st = make_hub(lambda s: {"n": s["calls"]}, interval=30.0, heartbeat_s=999)
    gen = hub.subscribe(["t"], None, boot_id="B")
    await take(gen, 2)  # hello + first snapshot; the next tick is 30 s away
    t0 = asyncio.get_running_loop().time()
    hub.poke("t")
    (frame,) = await take(gen, 1, timeout=2.0)
    assert asyncio.get_running_loop().time() - t0 < 0.5
    assert parse(frame)[1]["v"] == 2
    await gen.aclose()


@pytest.mark.asyncio
async def test_poke_from_another_thread_and_during_a_sample_is_not_lost():
    import threading

    gate = asyncio.Event()
    calls = {"n": 0}

    async def fetch(_req):
        calls["n"] += 1
        if calls["n"] == 1:
            await gate.wait()
        return {"n": calls["n"]}

    hub = Hub()
    hub.register(Topic("t", fetch, lambda _p: 30.0, heartbeat_s=999))
    gen = hub.subscribe(["t"], None, boot_id="B")
    first = asyncio.ensure_future(take(gen, 2))
    await asyncio.sleep(0.1)  # the first sample is in flight
    threading.Thread(target=hub.poke, args=("t",)).start()
    await asyncio.sleep(0.1)
    gate.set()
    frames = await first
    # the poke that landed during the first sample made the second one immediate (the tick is
    # 30 s): the subscriber, which was behind, got the newest payload straight away (coalesced).
    assert parse(frames[1])[1]["d"] == {"n": 2} and calls["n"] == 2
    await gen.aclose()


def test_registry_listener_fires_on_add_remove_and_record_done():
    from yunshu_gateway.x_yunshu import _Registry

    reg = _Registry()
    hits = []
    reg.add_listener(lambda: hits.append(1))

    class Info:
        request_id = "r1"
        engine_request_id = None

    info = Info()
    reg.add(info)  # type: ignore[arg-type]
    reg.record_done({"request_id": "r1"})
    reg.remove(info)  # type: ignore[arg-type]
    assert len(hits) == 3
    reg.add_listener(lambda: 1 / 0)  # a failing listener never breaks a request
    reg.remove(info)  # type: ignore[arg-type]


def test_model_state_change_notifies_listeners():
    from yunshu_engine import model_manager

    hits: list[int] = []
    model_manager.state_listeners.append(lambda: hits.append(1))
    model_manager.state_listeners.append(lambda: 1 / 0)  # never breaks a load
    try:
        model_manager.notify_state_change()
        assert hits == [1]
    finally:
        model_manager.state_listeners.clear()
