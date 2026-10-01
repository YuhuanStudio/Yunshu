"""What the client sees when generation goes wrong: an engine exception (mid-stream, before the
first token, out of memory, the engine stopped for a reload), a `timeout`, a prompt past the
context window. One terminal per stream, the dialect's error shape, no internals in the message,
nothing left behind."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

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
SECRET = "secret detail /Users/someone/model.safetensors"


class EngineStoppedError(RuntimeError):
    """What a model reload / unload looks like to a request that was running on it."""


FAILURES = {
    "mid-stream": Script.error_after(2, RuntimeError(SECRET)),
    "before-first-token": Script.error_after(0, RuntimeError(SECRET)),
    "engine-stopped": Script.error_after(1, EngineStoppedError(SECRET)),
}


@pytest.mark.parametrize("failure", list(FAILURES))
@pytest.mark.parametrize("dialect", DIALECT_NAMES)
async def test_stream_engine_exception_ends_in_one_error_terminal(
    make_app, audit, dialect, failure
):
    eng = FakeBatchedEngine(FAILURES[failure])
    app = make_app(eng)
    path, body = request_for(dialect, True)
    async with running_app(app):
        a = audit(eng)
        r = await asgi_call(app, "POST", path, body)
        assert r.status == 200  # the stream had begun; the error rides in the stream
        terminal = assert_one_terminal(dialect, r.text)
        assert terminal in ("error", "response.failed")
        assert SECRET not in r.text  # the exception text stays in the server log
        assert "Traceback" not in r.text
        if dialect == "openai":
            assert parse_sse(r.text)[-1][1] == "[DONE]"
        await a.settle()
        a.assert_clean()


@pytest.mark.parametrize("failure", list(FAILURES))
@pytest.mark.parametrize("dialect", DIALECT_NAMES)
async def test_nonstream_engine_exception_is_a_5xx_in_the_dialect(
    make_app, audit, dialect, failure
):
    eng = FakeBatchedEngine(FAILURES[failure])
    app = make_app(eng)
    path, body = request_for(dialect, False)
    async with running_app(app):
        a = audit(eng)
        r = await asgi_call(app, "POST", path, body)
        assert r.status == 500
        assert r.headers["content-type"].startswith("application/json")
        err = error_obj(dialect, r.json())
        assert SECRET not in r.text
        assert err["type"] in ("internal_error", "server_error", "api_error")
        assert "x-request-id" in r.headers
        await a.settle()
        a.assert_clean()


@pytest.mark.parametrize("dialect", DIALECT_NAMES)
async def test_stream_out_of_memory_has_one_terminal(make_app, audit, dialect):
    eng = FakeBatchedEngine(Script.error_after(1, MemoryError()))
    app = make_app(eng)
    path, body = request_for(dialect, True)
    async with running_app(app):
        a = audit(eng)
        r = await asgi_call(app, "POST", path, body)
        assert_one_terminal(dialect, r.text)
        assert "memory" in r.text.lower()
        await a.settle()
        a.assert_clean()


@pytest.mark.parametrize("dialect", DIALECT_NAMES)
async def test_nonstream_out_of_memory_is_507_with_a_dialect_shaped_error(
    make_app, dialect
):
    eng = FakeBatchedEngine(Script.error_after(0, MemoryError()))
    app = make_app(eng)
    path, body = request_for(dialect, False)
    async with running_app(app):
        r = await asgi_call(app, "POST", path, body)
        assert r.status == 507
        err = error_obj(dialect, r.json())
        assert "memory" in err["message"].lower()


async def test_engine_exception_does_not_poison_the_next_request(make_app, audit):
    eng = FakeBatchedEngine(Script.error_after(1, RuntimeError(SECRET)))
    app = make_app(eng)
    path, body = request_for("openai", True)
    async with running_app(app):
        a = audit(eng)
        await asgi_call(app, "POST", path, body)
        eng.script = Script.ok("fine", " again")
        r = await asgi_call(app, "POST", path, body)
        assert assert_one_terminal("openai", r.text) == "finish"
        await a.settle()
        a.assert_clean()


# ── timeout: a cut-short answer must not look finished ─────────────────────────────────────


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize(
    ("dialect", "marker"),
    [
        ("openai", '"finish_reason": "length"'),
        ("anthropic", '"stop_reason": "max_tokens"'),
        ("responses", '"status": "incomplete"'),
    ],
)
async def test_engine_timeout_is_reported_as_truncation(
    make_app, audit, dialect, marker, stream
):
    eng = FakeBatchedEngine(Script.hang_after(2))
    app = make_app(eng)
    path, body = request_for(dialect, stream, timeout=1)
    async with running_app(app):
        a = audit(eng)
        r = await asgi_call(app, "POST", path, body)
        assert r.status == 200
        assert 0.9 <= r.elapsed < 4.0
        text = r.text if stream else json.dumps(r.json())
        if not stream:
            text = text.replace('":"', '": "').replace('","', '", "')
        assert marker in text or marker.replace(": ", ":") in text
        if stream:
            assert assert_one_terminal(dialect, r.text) != "error"
        await a.settle()
        a.assert_clean()


# ── context overflow ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("dialect", DIALECT_NAMES)
async def test_prompt_past_the_context_window_is_a_400_before_the_engine(
    make_app, audit, dialect, stream
):
    eng = FakeBatchedEngine(Script.ok("x"))
    eng._model = SimpleNamespace(max_seq_len=64)
    app = make_app(eng)
    path, body = request_for(dialect, stream)
    big = "word " * 400
    if dialect == "responses":
        body["input"] = big
    else:
        body["messages"] = [{"role": "user", "content": big}]
    async with running_app(app):
        a = audit(eng)
        r = await asgi_call(app, "POST", path, body)
        assert r.status == 400
        assert r.headers["content-type"].startswith(
            "application/json"
        )  # not an SSE stream
        err = error_obj(dialect, r.json())
        assert "prompt is too long" in err["message"]  # the wording Claude Code parses
        if dialect != "anthropic":
            assert err["code"] == "context_length_exceeded"
        assert eng.calls == []  # never reached the engine
        a.assert_clean()
