"""When the context-window manager removes turns from a prompt, the response says so
(x_yunshu.context_policy, X-Yunshu-Context-Policy); when it does not, nothing is added."""

from __future__ import annotations

import json

import pytest

from .harness import (
    FakeBatchedEngine,
    Script,
    asgi_call,
    parse_sse,
    request_for,
    running_app,
)


def _history(turns: int) -> list[dict]:
    msgs = [{"role": "system", "content": "be brief"}]
    for i in range(turns):
        msgs.append(
            {"role": "user", "content": " ".join(f"q{i}w{j}" for j in range(20))}
        )
        msgs.append(
            {"role": "assistant", "content": " ".join(f"a{i}w{j}" for j in range(20))}
        )
    msgs.append({"role": "user", "content": "last question here"})
    return msgs


def _find(obj, key):
    """First value of ``key`` anywhere in a JSON structure."""
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
async def test_non_stream_reports_the_policy_in_body_and_header(make_app, dialect):
    eng = FakeBatchedEngine(Script.ok("ok"))
    eng.truncate = (_history(6), 120)
    app = make_app(eng)
    path, body = request_for(dialect, False)
    async with running_app(app):
        eng.truncate = (_history(6), 120)
        r = await asgi_call(app, "POST", path, body)
        assert r.status == 200
        cp = _find(r.json(), "context_policy")
        assert cp["policy"] == "truncate_oldest"
        assert cp["tokens_before"] > cp["tokens_after"] > 0
        assert cp["tokens_after"] <= 120 and cp["budget_tokens"] == 120
        assert cp["messages_removed"] > 0
        assert cp["removed_roles"]["user"] >= 1  # user turns are counted, not hidden
        header = r.headers["x-yunshu-context-policy"]
        assert header.startswith("truncate_oldest; removed=")
        assert f"tokens={cp['tokens_before']}->{cp['tokens_after']}" in header


@pytest.mark.parametrize("dialect", ["openai", "anthropic", "responses"])
async def test_stream_reports_the_policy_in_the_stats(make_app, dialect):
    eng = FakeBatchedEngine(Script.ok("ok"))
    app = make_app(eng)
    path, body = request_for(dialect, True)
    async with running_app(app):
        eng.truncate = (_history(6), 120)
        r = await asgi_call(app, "POST", path, body)
        assert r.status == 200
        found = None
        for line in r.text.split("\n"):
            if line.startswith(": yunshu-stats "):
                found = json.loads(line[len(": yunshu-stats ") :]).get("context_policy")
        for _name, data in parse_sse(r.text):
            if data != "[DONE]":
                found = _find(json.loads(data), "context_policy") or found
        assert found and found["policy"] == "truncate_oldest"
        assert found["messages_removed"] > 0


@pytest.mark.parametrize("dialect", ["openai", "anthropic", "responses"])
async def test_no_truncation_no_report(make_app, dialect):
    eng = FakeBatchedEngine(Script.ok("ok"))
    app = make_app(eng)
    path, body = request_for(dialect, False)
    async with running_app(app):
        eng.truncate = (_history(1), 10_000)
        r = await asgi_call(app, "POST", path, body)
        assert "x-yunshu-context-policy" not in r.headers
        assert _find(r.json(), "context_policy") is None


async def test_responses_truncation_auto_reports_what_it_dropped(make_app):
    from types import SimpleNamespace

    eng = FakeBatchedEngine(Script.ok("ok"))
    eng._model = SimpleNamespace(max_seq_len=400)
    app = make_app(eng)
    items = [{"role": "user", "content": f"turn{i} " + "x" * 140} for i in range(5)]
    body = {"model": "m", "input": items, "max_output_tokens": 20, "stream": False}
    async with running_app(app):
        # default (truncation disabled): too long is a 400, nothing is dropped behind its back
        r = await asgi_call(app, "POST", "/v1/responses", body)
        assert r.status == 400
        assert eng.calls == []
        r = await asgi_call(
            app, "POST", "/v1/responses", {**body, "truncation": "auto"}
        )
        assert r.status == 200
        cp = _find(r.json(), "context_policy")
        assert cp["policy"] == "truncation_auto"
        assert cp["messages_before"] == 5 and cp["messages_removed"] >= 3
        assert cp["removed_roles"]["user"] == cp["messages_removed"]
        assert cp["tokens_before"] > cp["budget_tokens"] >= cp["tokens_after"] > 0
        assert r.headers["x-yunshu-context-policy"].startswith(
            "truncation_auto; removed="
        )
