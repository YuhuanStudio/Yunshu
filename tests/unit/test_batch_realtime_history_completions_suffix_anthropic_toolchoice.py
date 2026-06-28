"""Waves 942-946: realtime tool-call/think history, completions suffix, Anthropic tool_choice.

W942 (HIGH): realtime stored raw full_text (with <tool_call> markup) as the assistant message
  on a tool-calling turn → markup leaked into the next turn's prompt + duplicate assistant
  turn. Store the markup-stripped text (empty → _build_messages skips the message).
W943 (HIGH): completions appended req.suffix to choices[].text — but `suffix` is FIM context,
  never returned. Stop appending (no FIM engine support).
W945: Anthropic forced tool_choice {"type":"tool","name":X} was never enforced post-generation
  (parity gap with chat's _enforce_tool_choice). Drop wrong-named/surplus calls.
W946: realtime streaming never separated reasoning → <think> leaked into stored history.
  Strip it.
"""
from __future__ import annotations

import inspect


def test_w942_realtime_stores_cleaned_text():
    from yunshu_gateway.routers import realtime
    src = inspect.getsource(realtime)
    assert "_visible_text = clean_tool_call_markup(_visible_text).strip()" in src
    # the assistant content/transcript/synth use _visible_text, not raw full_text
    assert 'content_parts = [{"type": "text", "text": _visible_text}]' in src
    assert "self._synthesize_audio_response(_visible_text" in src


def test_w946_realtime_strips_think():
    from yunshu_gateway.routers import realtime
    src = inspect.getsource(realtime)
    assert '_, _visible_text = extract_thinking(full_text, _snap_model)' in src


def test_w943_completions_no_suffix_append():
    from yunshu_gateway.routers import completions
    src = inspect.getsource(completions)
    # the wrong appends are gone
    assert "text = text + req.suffix" not in src
    assert "text=req.suffix," not in src


def test_w945_anthropic_enforces_forced_tool():
    from yunshu_gateway.routers.anthropic import _enforce_anthropic_tool_choice

    class _TC:
        def __init__(self, name):
            self.name = name

    calls = [_TC("get_weather"), _TC("get_time")]
    # forced "get_time" → only that call survives
    out = _enforce_anthropic_tool_choice(calls, {"type": "tool", "name": "get_time"})
    assert [c.name for c in out] == ["get_time"]
    # auto (no dict / string) → unchanged
    assert _enforce_anthropic_tool_choice(calls, "auto") == calls
    assert _enforce_anthropic_tool_choice(calls, {"type": "auto"}) == calls
    # disable_parallel_tool_use → cap to 1
    out2 = _enforce_anthropic_tool_choice(calls, {"type": "any", "disable_parallel_tool_use": True})
    assert len(out2) == 1


def test_w945_anthropic_streamer_threads_forced_name_and_none_cleanup():
    from yunshu_gateway.routers import anthropic
    src = inspect.getsource(anthropic)
    assert "forced_tool_name=_forced_name" in src
    assert "allow_parallel=_allow_parallel" in src
    # none-cleanup: under suppression the visible text is still cleaned
    assert "if _suppress_tool_extraction and req.tools:" in src
    assert "_enforce_anthropic_tool_choice(tool_calls, req.tool_choice)" in src
