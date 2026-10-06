"""A forced tool_choice is a guarantee: always constrained by the tool grammar, flag or not.

OpenAI (required / named) and Anthropic (any / tool) both promise a tool call. Found by the M3
sweep: without YUNSHU_TOOL_GRAMMAR a small model answered a forced choice with prose; the flag
now governs only auto. Usage reports what the client sent (the server's own prefill is not
counted), and /v1/completions text agrees between stream and non-stream.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

from yunshu_engine import tool_call_grammar as tcg
from yunshu_gateway import usage_shapes
from yunshu_gateway.routers import anthropic, completions, responses
from yunshu_gateway.routers.chat import _append_tool_prefill


def _req(choice, tools=True):
    return SimpleNamespace(
        tool_choice=choice,
        parallel_tool_calls=False,
        _native_tools=[{"type": "function"}] if tools else None,
    )


@pytest.fixture(autouse=True)
def _flag_off(monkeypatch):
    from yunshu_engine import settings

    orig = settings.get_bool
    monkeypatch.setattr(
        settings,
        "get_bool",
        lambda k, *a, **kw: False if k == "YUNSHU_TOOL_GRAMMAR" else orig(k, *a, **kw),
    )


def test_is_forced():
    for c in (
        "required",
        "any",
        {"type": "any"},
        {"type": "tool", "name": "x"},
        {"type": "function", "function": {"name": "x"}},
        {"name": "x"},
    ):
        assert tcg.is_forced(c), c
    for c in (None, "auto", "none", {"type": "auto"}, {"type": "none"}):
        assert not tcg.is_forced(c), c


def test_responses_forced_native_without_flag():
    kw = responses._native_kw(_req("required"))
    assert kw["tool_choice"] == "required" and kw["parallel_tool_calls"] is False
    named = {"type": "function", "name": "get_weather"}
    assert responses._native_kw(_req(named))["tool_choice"] == named
    assert responses._native_kw(_req("auto")) == {"tools": [{"type": "function"}]}
    assert responses._native_kw(_req("required", tools=False)) == {}


def test_messages_forced_without_flag():
    assert anthropic._forced_by_grammar({"type": "any"})
    assert anthropic._forced_by_grammar({"type": "tool", "name": "x"})
    assert not anthropic._forced_by_grammar({"type": "auto"})
    assert not anthropic._forced_by_grammar(None)


def test_engine_constrains_forced_but_not_auto_without_flag():
    """engine_fast._tool_call_processor: forced compiles a grammar, auto stays free."""
    from yunshu_engine import batched_engine as be

    called = []
    orig = tcg.compile_tool_grammar
    tcg.compile_tool_grammar = lambda *a, **k: (
        called.append(k.get("tool_choice")) or None
    )
    try:
        eng = SimpleNamespace(
            _tokenizer=[0] * 8,
            _model=SimpleNamespace(args=SimpleNamespace(vocab_size=8)),
        )
        eng.__dict__["_tool_grammars"] = {}
        tools = [
            {
                "type": "function",
                "function": {"name": "f", "parameters": {"type": "object"}},
            }
        ]
        for choice, want in (
            ({"tool_choice": "required"}, 1),
            ({"tool_choice": "auto"}, 0),
        ):
            called.clear()
            t1 = be._REQUEST_TOOLS.set(tools)
            t2 = be._REQUEST_TOOL_USE.set({**choice, "parallel": True})
            try:
                with contextlib.suppress(Exception):
                    be.BatchedEngine._tool_call_processor(eng, [1, 2])
            finally:
                be._REQUEST_TOOLS.reset(t1)
                be._REQUEST_TOOL_USE.reset(t2)
            assert len(called) == want, (choice, called)
    finally:
        tcg.compile_tool_grammar = orig


def test_usage_excludes_server_prefill():
    class Tok:
        def encode(self, text, add_special_tokens=True):
            return text.split()

    ctx_token = usage_shapes._PREFILL_TOKENS.set(0)
    try:
        eng = SimpleNamespace(_tokenizer=Tok())
        msgs = _append_tool_prefill(
            [{"role": "user", "content": "hi"}], "<tool_call> \n", eng
        )
        assert msgs[-1]["content"].startswith("<tool_call>")
        assert usage_shapes.openai_usage(10, 5)["prompt_tokens"] == 9
        assert usage_shapes.responses_usage(10, 5)["input_tokens"] == 9
        assert anthropic._anthropic_cache_usage(10, 0)[0] == 9
    finally:
        usage_shapes._PREFILL_TOKENS.reset(ctx_token)
    assert usage_shapes.openai_usage(10, 5)["prompt_tokens"] == 10


def test_completions_untag_reasoning():
    f = completions._untag_reasoning
    assert f("<think>why</think>answer", 3) == "whyanswer"
    assert f("<think>cut off", 3) == "cut off"
    assert f("<think>literal</think>x", 0) == "<think>literal</think>x"
    assert f("plain", 3) == "plain"
