"""Forced tool_choice on Responses / Messages reaches the tool-call grammar (YUNSHU_TOOL_GRAMMAR=1).

Found by the M3 sweep: with the grammar on, a forced tool_choice on /v1/responses and
/v1/messages was routed through the injected prompt (never native), so the engine never saw
the forced choice and a streamed answer was prose.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from yunshu_gateway.routers import anthropic, responses


@pytest.fixture
def grammar(monkeypatch):
    def set_flag(on):
        orig = responses.settings.get_bool
        monkeypatch.setattr(
            responses.settings,
            "get_bool",
            lambda k, *a, **kw: on if k == "YUNSHU_TOOL_GRAMMAR" else orig(k, *a, **kw),
        )

    return set_flag


def _req(choice, tools=True):
    return SimpleNamespace(
        tool_choice=choice,
        parallel_tool_calls=False,
        _native_tools=[{"type": "function"}] if tools else None,
    )


def test_responses_forced_native_kw(grammar):
    grammar(True)
    kw = responses._native_kw(_req("required"))
    assert kw["tool_choice"] == "required" and kw["parallel_tool_calls"] is False
    named = {"type": "function", "name": "get_weather"}
    assert responses._native_kw(_req(named))["tool_choice"] == named
    assert responses._native_kw(_req("auto")) == {"tools": [{"type": "function"}]}
    assert responses._native_kw(_req("required", tools=False)) == {}


def test_responses_forced_stays_advisory_without_grammar(grammar):
    grammar(False)
    assert responses._native_kw(_req("required")) == {"tools": [{"type": "function"}]}
    assert not responses._forced_by_grammar(_req("required"))


def test_messages_forced_by_grammar(grammar):
    grammar(True)
    assert anthropic._forced_by_grammar({"type": "any"})
    assert anthropic._forced_by_grammar({"type": "tool", "name": "x"})
    assert not anthropic._forced_by_grammar({"type": "auto"})
    assert not anthropic._forced_by_grammar({"type": "none"})
    assert not anthropic._forced_by_grammar(None)
    grammar(False)
    assert not anthropic._forced_by_grammar({"type": "any"})
