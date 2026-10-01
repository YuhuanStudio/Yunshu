"""Cancel: client disconnect, the cancel endpoints, before the engine starts, during prefill /
decode. One terminal per request, the engine call ends, nothing is left in the tracker."""

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

DIALECT_NAMES = ["openai", "anthropic", "responses"]


@pytest.mark.parametrize("tokens_before", [0, 2], ids=["prefill", "decode"])
@pytest.mark.parametrize("dialect", DIALECT_NAMES)
async def test_stream_client_disconnect_stops_the_engine(
    make_app, audit, dialect, tokens_before
):
    eng = FakeBatchedEngine(Script.hang_after(tokens_before))
    app = make_app(eng)
    path, body = request_for(dialect, True)
    async with running_app(app):
        a = audit(eng)
        # the keep-alive comment is the first chunk; leave right after it / after tokens
        r = await asgi_call(
            app, "POST", path, body, disconnect_after_chunks=1 + tokens_before
        )
        await a.settle()
        assert r.disconnected
        assert eng.cancel_events and eng.cancel_events[0].is_set()
        assert len(eng.finish_reasons) == 1  # one engine call, ended once
        a.assert_clean()


@pytest.mark.parametrize("dialect", DIALECT_NAMES)
async def test_nonstream_client_disconnect_stops_the_engine(make_app, audit, dialect):
    eng = FakeBatchedEngine(Script.hang_after(0))
    app = make_app(eng)
    path, body = request_for(dialect, False)
    async with running_app(app):
        a = audit(eng)
        await asgi_call(app, "POST", path, body, disconnect_after_s=0.2, timeout=10)
        await a.settle(4.0)
        assert eng.cancel_events[0].is_set()
        assert len(eng.finish_reasons) == 1
        a.assert_clean()


async def _start_and_cancel(app, dialect, stream, cancel, tokens=2):
    path, body = request_for(dialect, stream)
    task = asyncio.create_task(
        asgi_call(app, "POST", path, body, headers={"X-Request-Id": "job-1"})
    )
    for _ in range(200):  # until the request is in the tracker
        await asyncio.sleep(0.01)
        r = await cancel(probe=True)
        if r:
            break
    else:
        raise AssertionError("request never became visible")
    resp = await cancel(probe=False)
    return task, resp


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("dialect", DIALECT_NAMES)
async def test_delete_request_by_client_id_ends_the_request(
    make_app, audit, dialect, stream
):
    eng = FakeBatchedEngine(Script.hang_after(2))
    app = make_app(eng)
    async with running_app(app):
        a = audit(eng)

        async def cancel(probe):
            if probe:
                r = await asgi_call(app, "GET", "/v1/requests/job-1")
                return r.status == 200 and eng.inflight > 0
            return await asgi_call(app, "DELETE", "/v1/requests/job-1")

        task, resp = await _start_and_cancel(app, dialect, stream, cancel)
        assert resp.status == 200 and resp.json()["status"] == "cancelled"
        r = await task
        await a.settle()
        assert r.status == 200
        if stream:
            assert_one_terminal(dialect, r.text)
        assert eng.finish_reasons == ["cancel"]
        a.assert_clean()
        # a second cancel of a finished request is a clean 404, not a hang or a 500
        again = await asgi_call(app, "DELETE", "/v1/requests/job-1")
        assert again.status == 404


@pytest.mark.parametrize("dialect", DIALECT_NAMES)
async def test_cancel_endpoint_cancel_all(make_app, audit, dialect):
    eng = FakeBatchedEngine(Script.hang_after(1))
    app = make_app(eng)
    path, body = request_for(dialect, True)
    async with running_app(app):
        a = audit(eng)
        tasks = [
            asyncio.create_task(asgi_call(app, "POST", path, body)) for _ in range(3)
        ]
        for _ in range(300):
            await asyncio.sleep(0.01)
            if eng.inflight == 3:
                break
        assert eng.inflight == 3
        r = await asgi_call(app, "POST", "/v1/cancel", {"cancel_all": True})
        assert r.status == 200 and r.json()["count"] >= 3
        replies = await asyncio.gather(*tasks)
        await a.settle()
        for rep in replies:
            assert_one_terminal(dialect, rep.text)
        a.assert_clean()


async def test_cancel_unknown_request_is_404_in_both_endpoints(make_app):
    app = make_app(FakeBatchedEngine())
    async with running_app(app):
        r = await asgi_call(app, "POST", "/v1/cancel", {"request_id": "nope"})
        assert r.status == 404
        r = await asgi_call(app, "DELETE", "/v1/requests/nope")
        assert r.status == 404
        r = await asgi_call(app, "POST", "/v1/cancel", {})
        assert r.status == 400


@pytest.mark.parametrize("tokens_before", [0, 2], ids=["prefill", "decode"])
@pytest.mark.parametrize("dialect", DIALECT_NAMES)
async def test_send_failure_closes_the_stream_and_releases_the_request(
    make_app, audit, dialect, tokens_before
):
    """uvicorn raises from ``send`` when the socket is gone; Starlette then abandons the body
    generator mid-iteration. Its ``finally`` (tracker entry, LoRA lease) must still run: every
    generation stream closes its iterator, not only the VLM one."""
    eng = FakeBatchedEngine(Script.hang_after(tokens_before))
    app = make_app(eng)
    path, body = request_for(dialect, True)
    async with running_app(app):
        a = audit(eng)
        r = await asgi_call(
            app, "POST", path, body, fail_send_after_chunks=1 + tokens_before
        )
        assert r.disconnected
        await a.settle()
        assert eng.cancel_events[0].is_set()
        a.assert_clean()
