"""Overload and deadlines: a full or memory-starved server answers with a retryable error in the
dialect of the route instead of hanging; X-Yunshu-Deadline-Ms ends a request that outlives it."""

from __future__ import annotations

import asyncio
import json

import pytest

from yunshu_gateway import admission

from .harness import (
    FakeBatchedEngine,
    Script,
    asgi_call,
    assert_one_terminal,
    error_obj,
    parse_sse,
    request_for,
    running_app,
)

DIALECT_NAMES = ["openai", "anthropic", "responses"]


async def _fill(app, engine, n, dialect="openai"):
    """n requests that stay in flight until cancelled."""
    path, body = request_for(dialect, True)
    tasks = [asyncio.create_task(asgi_call(app, "POST", path, body)) for _ in range(n)]
    for _ in range(500):
        await asyncio.sleep(0.01)
        if engine.inflight == n:
            return tasks
    raise AssertionError("requests never reached the engine")


async def _drain(app, tasks):
    await asgi_call(app, "POST", "/v1/cancel", {"cancel_all": True})
    return await asyncio.gather(*tasks)


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("dialect", DIALECT_NAMES)
async def test_full_server_refuses_with_retry_after(
    make_app, audit, monkeypatch, dialect, stream
):
    monkeypatch.setenv("YUNSHU_QUEUE_LIMIT", "2")
    eng = FakeBatchedEngine(Script.hang_after(1))
    app = make_app(eng)
    async with running_app(app):
        a = audit(eng)
        tasks = await _fill(app, eng, 2)
        path, body = request_for(dialect, stream)
        t0 = asyncio.get_running_loop().time()
        r = await asgi_call(app, "POST", path, body, headers={"X-Request-Id": "late-1"})
        assert asyncio.get_running_loop().time() - t0 < 1.0  # refused, not queued
        assert r.status == 429
        assert r.headers["retry-after"].isdigit() and int(r.headers["retry-after"]) >= 1
        assert r.headers["x-request-id"] == "late-1"
        assert r.headers["content-type"].startswith("application/json")
        err = error_obj(dialect, r.json())
        if dialect == "anthropic":
            assert err["type"] == "rate_limit_error"
        else:
            assert err["type"] == "rate_limit_error" and err["code"] == "queue_full"
            assert err["x_yunshu"]["queue_depth"] == 2
            assert err["x_yunshu"]["queue_limit"] == 2
            assert err["x_yunshu"]["request_id"] == "late-1"
        assert len(eng.calls) == 2  # the refused request never reached the engine
        done = await _drain(app, tasks)
        for rep in done:
            assert_one_terminal("openai", rep.text)
        await a.settle()
        a.assert_clean()
        # room again: the same request is admitted (and, being a hang script, ends on cancel)
        eng.script = Script.ok("fine")
        assert (await asgi_call(app, "POST", path, body)).status == 200


async def test_queue_limit_zero_means_unlimited(make_app, monkeypatch):
    monkeypatch.setenv("YUNSHU_QUEUE_LIMIT", "0")
    eng = FakeBatchedEngine(Script.hang_after(0))
    app = make_app(eng)
    async with running_app(app):
        tasks = await _fill(app, eng, 5)
        await _drain(app, tasks)


@pytest.mark.parametrize("dialect", DIALECT_NAMES)
async def test_memory_pressure_refuses_503_only_while_busy(
    make_app, audit, monkeypatch, dialect
):
    monkeypatch.setattr(admission, "memory_pressure", lambda: 0.99)
    eng = FakeBatchedEngine(Script.ok("a"))
    app = make_app(eng)
    path, body = request_for(dialect, False)
    async with running_app(app):
        a = audit(eng)
        # idle server: waiting would not help, so it is served
        assert (await asgi_call(app, "POST", path, body)).status == 200
        eng.script = Script.hang_after(0)
        tasks = await _fill(app, eng, 1)
        r = await asgi_call(app, "POST", path, body)
        assert r.status == 503
        assert int(r.headers["retry-after"]) >= 1
        err = error_obj(dialect, r.json())
        if dialect == "anthropic":
            assert err["type"] == "overloaded_error"
        else:
            assert err["type"] == "server_error" and err["code"] == "memory_pressure"
            assert err["x_yunshu"]["memory_pressure"] == 0.99
        await _drain(app, tasks)
        await a.settle()
        a.assert_clean()


async def test_memory_pressure_off_when_zero(make_app, monkeypatch):
    monkeypatch.setattr(admission, "memory_pressure", lambda: 0.999)
    monkeypatch.setenv("YUNSHU_MEMORY_PRESSURE_REJECT", "0")
    eng = FakeBatchedEngine(Script.hang_after(0))
    app = make_app(eng)
    async with running_app(app):
        tasks = await _fill(app, eng, 2)
        await _drain(app, tasks)


def test_retry_after_follows_the_queue_estimate():
    assert admission.retry_after_s(None) == 5
    assert admission.retry_after_s(0) == 5
    assert admission.retry_after_s(400) == 1
    assert admission.retry_after_s(12_300) == 13
    assert admission.retry_after_s(10_000_000) == 60


@pytest.mark.parametrize("raw", ["abc", "0", "-5", "1.5", str(10**12)])
def test_bad_deadline_values_are_rejected(raw):
    with pytest.raises(ValueError):
        admission.parse_deadline_ms(raw)


def test_deadline_parsing():
    assert admission.parse_deadline_ms(None) is None
    assert admission.parse_deadline_ms("") is None
    assert admission.parse_deadline_ms(b" 1500 ") == 1500


@pytest.mark.parametrize("dialect", DIALECT_NAMES)
async def test_malformed_deadline_is_a_400_in_the_dialect(make_app, dialect):
    app = make_app(FakeBatchedEngine())
    path, body = request_for(dialect, False)
    async with running_app(app):
        r = await asgi_call(
            app, "POST", path, body, headers={"X-Yunshu-Deadline-Ms": "soon"}
        )
        assert r.status == 400
        err = error_obj(dialect, r.json())
        assert "Deadline" in err["message"]


@pytest.mark.parametrize("tokens_before", [0, 2], ids=["prefill", "decode"])
@pytest.mark.parametrize("dialect", DIALECT_NAMES)
async def test_deadline_ends_a_non_streaming_request_with_504(
    make_app, audit, dialect, tokens_before
):
    eng = FakeBatchedEngine(Script.hang_after(tokens_before))
    app = make_app(eng)
    path, body = request_for(dialect, False)
    async with running_app(app):
        a = audit(eng)
        r = await asgi_call(
            app, "POST", path, body, headers={"X-Yunshu-Deadline-Ms": "200"}
        )
        assert r.status == 504
        assert 0.15 <= r.elapsed < 3.0
        err = error_obj(dialect, r.json())
        if dialect == "anthropic":
            assert err["type"] == "timeout_error"
        else:
            assert err["code"] == "deadline_exceeded"
            assert err["x_yunshu"]["deadline_ms"] == 200
        assert "x-request-id" in r.headers
        await a.settle()
        assert eng.cancel_events[
            0
        ].is_set()  # the generation was stopped, not left running
        a.assert_clean()


@pytest.mark.parametrize("tokens_before", [0, 2], ids=["prefill", "decode"])
@pytest.mark.parametrize("dialect", DIALECT_NAMES)
async def test_deadline_ends_a_stream_with_exactly_one_terminal_error(
    make_app, audit, dialect, tokens_before
):
    eng = FakeBatchedEngine(Script.hang_after(tokens_before))
    app = make_app(eng)
    path, body = request_for(dialect, True)
    async with running_app(app):
        a = audit(eng)
        r = await asgi_call(
            app, "POST", path, body, headers={"X-Yunshu-Deadline-Ms": "250"}
        )
        assert r.status == 200  # the stream had started
        assert 0.2 <= r.elapsed < 3.0
        assert assert_one_terminal(dialect, r.text) == "error"
        events = parse_sse(r.text)
        err = json.loads(
            next(d for n, d in events if '"error"' in d and "choices" not in d)
        )
        if dialect == "openai":
            assert err["error"]["code"] == "deadline_exceeded"
            assert events[-1][1] == "[DONE]"
        elif dialect == "anthropic":
            assert err["error"]["type"] == "timeout_error"
        else:
            assert err["code"] == "deadline_exceeded"
        await a.settle()
        assert eng.cancel_events[0].is_set()
        a.assert_clean()


async def test_a_request_that_finishes_in_time_is_untouched(make_app, audit):
    eng = FakeBatchedEngine(Script.ok("a", "b"))
    app = make_app(eng)
    path, body = request_for("openai", False)
    async with running_app(app):
        a = audit(eng)
        r = await asgi_call(
            app, "POST", path, body, headers={"X-Yunshu-Deadline-Ms": "5000"}
        )
        assert r.status == 200
        assert r.json()["x_yunshu"]["deadline_ms"] == 5000
        assert "cancelled" not in r.json()["x_yunshu"]
        await asyncio.sleep(0.05)
        a.assert_clean()


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("dialect", ["openai", "anthropic"])
async def test_explicit_cancel_marks_the_cut_short_answer(make_app, dialect, stream):
    """A cancel from another connection used to end the original request as a normal
    completion (finish_reason stop / end_turn): the client could not tell it was cut short."""
    eng = FakeBatchedEngine(Script.hang_after(2))
    app = make_app(eng)
    path, body = request_for(dialect, stream)
    async with running_app(app):
        t = asyncio.create_task(
            asgi_call(app, "POST", path, body, headers={"X-Request-Id": "c-1"})
        )
        while eng.inflight == 0:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        assert (await asgi_call(app, "DELETE", "/v1/requests/c-1")).status == 200
        r = await t
        if stream:
            assert '"cancelled":true' in r.text.replace(" ", "")
        else:
            assert r.headers["x-yunshu-cancelled"] == "true"
            payload = r.json()
            x = (
                payload["x_yunshu"]
                if dialect == "openai"
                else payload["usage"]["x_yunshu"]
            )
            assert x["cancelled"] is True
