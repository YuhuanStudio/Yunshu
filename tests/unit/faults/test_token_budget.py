"""One budget for prompt + thinking + answer inside the context window."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from yunshu_gateway import token_budget

from .harness import (
    FakeBatchedEngine,
    Script,
    asgi_call,
    error_obj,
    request_for,
    running_app,
)

# ── the pure planner ──────────────────────────────────────────────────────────────────────


def test_a_request_that_fits_is_untouched():
    b = token_budget.plan(32768, 1000, 4096, 2048)
    assert b is not None and not b.clamped
    assert (b.max_tokens_granted, b.thinking_budget_granted) == (4096, 2048)


def test_max_tokens_is_clamped_to_the_room_left():
    b = token_budget.plan(32768, 31000, 4096)
    assert b is not None and b.clamped
    assert b.max_tokens_granted == 1768
    rep = b.report()
    assert rep["max_tokens_requested"] == 4096 and rep["max_tokens_granted"] == 1768
    assert rep["context_window"] == 32768 and rep["prompt_tokens"] == 31000
    assert "thinking_budget_requested" not in rep


def test_thinking_is_a_part_of_the_answer_not_an_addend():
    # 4096 answer tokens already include the 2048 thinking tokens: no clamp, no double count
    assert not token_budget.plan(10_000, 1000, 4096, 2048).clamped
    # when the answer shrinks, thinking never exceeds what was granted
    b = token_budget.plan(10_000, 9_000, 4096, 3000)
    assert (b.max_tokens_granted, b.thinking_budget_granted) == (1000, 1000)
    assert b.report()["thinking_budget_requested"] == 3000


def test_a_full_window_is_an_error_not_a_zero_budget():
    with pytest.raises(ValueError, match="leave no room"):
        token_budget.plan(100, 100, 10)
    with pytest.raises(ValueError):
        token_budget.plan(100, 140, 10)


def test_unknown_window_means_no_budget():
    assert token_budget.plan(None, 5000, 4096) is None
    assert token_budget.plan(0, 5000, 4096) is None


# ── through the routes ────────────────────────────────────────────────────────────────────

CTX = 400
PROMPT = "x" * 300  # ~306 prompt tokens with the fake tokenizer (one per character)


def _engine():
    eng = FakeBatchedEngine(Script.ok("ok"))
    eng._model = SimpleNamespace(max_seq_len=CTX)
    return eng


def _with_prompt(dialect, body):
    if dialect == "responses":
        body["input"] = PROMPT
    else:
        body["messages"] = [{"role": "user", "content": PROMPT}]
    return body


def _find(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find(v, key)
            if r is not None:
                return r
    return None


@pytest.mark.parametrize("dialect", ["openai", "anthropic", "responses"])
async def test_max_tokens_is_clamped_to_the_window_and_reported(make_app, dialect):
    eng = _engine()
    app = make_app(eng)
    path, body = request_for(dialect, False)
    body = _with_prompt(dialect, body)
    body["max_output_tokens" if dialect == "responses" else "max_tokens"] = 1000
    async with running_app(app):
        r = await asgi_call(app, "POST", path, body)
        assert r.status == 200
        granted = eng.calls[0]["max_tokens"]
        assert 1 <= granted < 100  # CTX - prompt, not the 1000 asked for
        rep = _find(r.json(), "budget")
        assert rep["max_tokens_requested"] == 1000
        assert rep["max_tokens_granted"] == granted
        assert rep["context_window"] == CTX
        assert rep["clamped_by"] == "context_window"


async def test_thinking_budget_follows_the_clamp(make_app):
    eng = _engine()
    app = make_app(eng)
    path, body = request_for("openai", False, max_tokens=1000, thinking_budget=900)
    body["messages"] = [{"role": "user", "content": PROMPT}]
    async with running_app(app):
        r = await asgi_call(app, "POST", path, body)
        assert r.status == 200
        call = eng.calls[0]
        assert call["thinking_budget"] == call["max_tokens"] < 100
        rep = r.json()["x_yunshu"]["budget"]
        assert rep["thinking_budget_requested"] == 900


@pytest.mark.parametrize("dialect", ["openai", "anthropic", "responses"])
async def test_a_request_that_fits_reports_no_budget(make_app, dialect):
    eng = _engine()
    app = make_app(eng)
    path, body = request_for(dialect, False)
    body = _with_prompt(dialect, body)
    body["max_output_tokens" if dialect == "responses" else "max_tokens"] = 20
    async with running_app(app):
        r = await asgi_call(app, "POST", path, body)
        assert eng.calls[0]["max_tokens"] == 20
        assert _find(r.json(), "budget") is None


@pytest.mark.parametrize("dialect", ["openai", "anthropic", "responses"])
async def test_a_prompt_that_fills_the_window_is_a_400(make_app, dialect):
    eng = _engine()
    app = make_app(eng)
    path, body = request_for(dialect, False)
    big = "x" * (CTX - 6)  # exactly the window once the template tokens are added
    if dialect == "responses":
        body["input"] = big
    else:
        body["messages"] = [{"role": "user", "content": big}]
    async with running_app(app):
        r = await asgi_call(app, "POST", path, body)
        assert r.status == 400
        assert "context window" in error_obj(dialect, r.json())["message"]
        assert eng.calls == []


async def test_the_tokenizer_default_never_shrinks_a_request(make_app):
    """A stale ``model_max_length`` on the tokenizer is not a window: it may reject nothing and
    clamp nothing."""
    eng = FakeBatchedEngine(Script.ok("ok"))
    eng._tokenizer.model_max_length = 64  # absurdly small, from the tokenizer only
    app = make_app(eng)
    path, body = request_for("openai", False, max_tokens=500)
    async with running_app(app):
        r = await asgi_call(app, "POST", path, body)
        assert r.status == 200
        assert eng.calls[0]["max_tokens"] == 500
