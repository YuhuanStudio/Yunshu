"""A request waiting behind a running one is visible as queued, can be cancelled without ever
reaching the engine, and does not disturb the one that is running."""

from __future__ import annotations

import asyncio

import pytest

from .harness import (
    FakeBatchedEngine,
    Script,
    asgi_call,
    assert_one_terminal,
    request_for,
    running_app,
)


async def _until(pred, what, tries=500):
    for _ in range(tries):
        if pred():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"never happened: {what}")


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("dialect", ["openai", "anthropic", "responses"])
async def test_queued_request_can_be_cancelled_and_never_runs(
    make_app, audit, dialect, stream
):
    eng = FakeBatchedEngine(Script.hang_after(1))
    eng.serial = asyncio.Semaphore(1)
    app = make_app(eng)
    path, body = request_for(dialect, stream)
    async with running_app(app):
        a = audit(eng)
        first = asyncio.create_task(
            asgi_call(app, "POST", path, body, headers={"X-Request-Id": "run-1"})
        )
        await _until(lambda: eng.inflight == 1, "first request running")
        second = asyncio.create_task(
            asgi_call(app, "POST", path, body, headers={"X-Request-Id": "wait-1"})
        )
        await _until(lambda: eng.waiting == 1, "second request queued")

        r = await asgi_call(app, "GET", "/v1/requests/wait-1")
        info = r.json()
        assert r.status == 200 and info["phase"] == "queued"
        assert info["queue_position"] == 1

        r = await asgi_call(app, "DELETE", "/v1/requests/wait-1")
        assert r.status == 200 and r.json()["status"] == "cancelled"
        rep = await asyncio.wait_for(second, 5)
        assert rep.status == 200
        if stream:
            assert_one_terminal(dialect, rep.text)
        assert len(eng.calls) == 1  # the cancelled request never reached the engine
        assert eng.inflight == 1 and eng.waiting == 0
        assert not first.done()  # the running one is untouched

        await asgi_call(app, "DELETE", "/v1/requests/run-1")
        await asyncio.wait_for(first, 5)
        await a.settle()
        a.assert_clean()
        assert (await asgi_call(app, "GET", "/v1/requests/wait-1")).status == 404


async def test_full_queue_refuses_the_next_request_and_a_cancel_makes_room(
    make_app, audit, monkeypatch
):
    monkeypatch.setenv("YUNSHU_QUEUE_LIMIT", "2")
    eng = FakeBatchedEngine(Script.hang_after(1))
    eng.serial = asyncio.Semaphore(1)
    app = make_app(eng)
    path, body = request_for("openai", True)
    async with running_app(app):
        a = audit(eng)
        t1 = asyncio.create_task(
            asgi_call(app, "POST", path, body, headers={"X-Request-Id": "r1"})
        )
        await _until(lambda: eng.inflight == 1, "r1 running")
        t2 = asyncio.create_task(
            asgi_call(app, "POST", path, body, headers={"X-Request-Id": "r2"})
        )
        await _until(lambda: eng.waiting == 1, "r2 queued")
        refused = await asgi_call(app, "POST", path, body)
        assert refused.status == 429 and int(refused.headers["retry-after"]) >= 1
        assert refused.headers["x-yunshu-queue-position"] == "2"
        # a client that gives up frees its place at once
        await asgi_call(app, "DELETE", "/v1/requests/r2")
        await asyncio.wait_for(t2, 5)
        t3 = asyncio.create_task(asgi_call(app, "POST", path, body))
        await _until(lambda: eng.waiting == 1, "a new request admitted")
        await asgi_call(app, "POST", "/v1/cancel", {"cancel_all": True})
        for t in (t1, t3):
            assert_one_terminal("openai", (await asyncio.wait_for(t, 5)).text)
        await a.settle()
        a.assert_clean()
