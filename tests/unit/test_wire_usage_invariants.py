"""R26: invariants every dialect keeps whatever the engine reports (details inside totals,
prompt split adds up, stream usage exact when asked, n choices billed once for the prompt)."""

from __future__ import annotations

from .wire_clients import DIALECTS, Clients, run
from .wire_harness import Script, install

USER = [{"role": "user", "content": "hi"}]


def _serve(monkeypatch, **script):
    c, eng = install(monkeypatch, Script(**script))
    return Clients(c), eng


def test_reasoning_and_cached_never_exceed_their_totals(monkeypatch):
    # An engine that over-reports its details (cached > prompt, reasoning > completion).
    pieces = [
        ("<think>", "reasoning"),
        ("x", "reasoning"),
        ("</think>", "reasoning"),
        "a",
    ]
    cl, _ = _serve(monkeypatch, pieces=pieces, prompt_tokens=5, cached_tokens=50)
    for d in DIALECTS:
        for stream in (False, True):
            o = run(cl, d, stream=stream, max_tokens=2)
            assert (o.cached or 0) <= (o.prompt or 0), (d, stream, o.cached, o.prompt)
            assert (o.reasoning or 0) <= (o.completion or 0), (d, stream)


def test_anthropic_prompt_split_adds_up(monkeypatch):
    cl, _ = _serve(monkeypatch, pieces=["a", "b"], prompt_tokens=11, cached_tokens=6)
    for stream in (False, True):
        o = run(cl, "messages", stream=stream)
        assert o.prompt == 11 and o.cached == 6, (stream, o.prompt, o.cached)


def test_anthropic_message_delta_restates_the_prompt_split(monkeypatch):
    cl, _ = _serve(monkeypatch, pieces=["a", "b"], prompt_tokens=11, cached_tokens=6)
    o = run(cl, "messages", stream=True)
    d = o.delta_usage
    assert (
        d.input_tokens,
        d.cache_read_input_tokens,
        d.cache_creation_input_tokens,
    ) == (5, 6, 0)
    assert d.output_tokens == 2


def test_stream_usage_is_optional_and_exact_when_asked(monkeypatch):
    cl, _ = _serve(monkeypatch, pieces=["a", "b", "c"], prompt_tokens=4)
    for d in ("chat", "completions"):
        on = run(cl, d, stream=True, include_usage=True)
        off = run(cl, d, stream=True, include_usage=False)
        assert on.events[-1] == "usage" and "usage" not in off.events, d
        assert (on.prompt, on.completion) == (4, 3)
        assert off.completion is None


def test_n_choices_bill_the_prompt_once_and_sum_the_completions(monkeypatch):
    cl, _ = _serve(monkeypatch, pieces=["a", "b", "c"], prompt_tokens=4)
    r = cl.oa.chat.completions.create(model="m", messages=USER, n=2)
    assert len(r.choices) == 2
    assert (r.usage.prompt_tokens, r.usage.completion_tokens) == (4, 6)
    assert r.usage.total_tokens == 10
    r = cl.oa.completions.create(model="m", prompt="hi", n=2)
    assert (r.usage.prompt_tokens, r.usage.completion_tokens) == (4, 6)


def test_thinking_controls_reach_the_engine_the_same_way(monkeypatch):
    cl, eng = _serve(monkeypatch, pieces=["a"])
    cl.oa.chat.completions.create(
        model="m",
        messages=USER,
        extra_body={"thinking_budget": 77, "enable_thinking": True},
    )
    cl.an.messages.create(
        model="m",
        max_tokens=200,
        messages=USER,
        thinking={"type": "enabled", "budget_tokens": 77},
    )
    cl.oa.responses.create(
        model="m",
        input="hi",
        extra_body={"thinking_budget": 77, "enable_thinking": True},
    )
    budgets = [c.get("thinking_budget") for c in eng.calls]
    assert budgets == [77, 77, 77], budgets
    assert all(c.get("enable_thinking") for c in eng.calls)


def test_zero_max_tokens_reports_the_prompt_and_no_output(monkeypatch):
    cl, eng = _serve(monkeypatch, pieces=["a", "b"], prompt_tokens=4)
    for create in (
        lambda: cl.oa.chat.completions.create(model="m", messages=USER, max_tokens=0),
        lambda: cl.oa.completions.create(model="m", prompt="hi", max_tokens=0),
    ):
        r = create()
        assert r.usage.completion_tokens == 0
        assert r.usage.total_tokens == r.usage.prompt_tokens
        assert r.choices[0].finish_reason == "length"
    assert not eng.calls, "max_tokens=0 must not generate"
