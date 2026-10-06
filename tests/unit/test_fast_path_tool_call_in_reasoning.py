"""Text-only fast path: a tool call opened inside unclosed reasoning ends the reasoning
(same behaviour as VLMEngine; tests/unit/test_tool_call_in_reasoning.py)."""

import inspect

from yunshu_engine import engine_fast, engine_stream
from yunshu_engine.reasoning_parser import (
    close_reasoning_at_tool_call,
    get_reasoning_parser,
    tool_call_marker_id,
)

CALL = '<tool_call>{"name": "Read", "arguments": {}}</tool_call>'


def _split(text, prompt_open):
    out = get_reasoning_parser("Qwen3.5").parse(
        close_reasoning_at_tool_call(text, prompt_open)
    )
    return out.reasoning, out.content


def test_templated_think_unclosed_call_becomes_content():
    r, c = _split("I will read it. " + CALL, True)
    assert r == "I will read it." and c == CALL


def test_explicit_open_unclosed_call_becomes_content():
    r, c = _split("<think>plan " + CALL, False)
    assert r == "plan" and c == CALL


def test_closed_reasoning_unchanged():
    t = "plan</think>" + CALL
    assert close_reasoning_at_tool_call(t, True) == t
    assert _split(t, True) == ("plan", CALL)


def test_no_reasoning_context_unchanged():
    t = "hello " + CALL
    assert close_reasoning_at_tool_call(t, False) == t
    assert close_reasoning_at_tool_call("no call here", True) == "no call here"


def test_marker_id_single_token_only():
    class Tok:
        def __init__(self, ids):
            self.ids = ids

        def encode(self, s, add_special_tokens=False):
            return self.ids

    assert tool_call_marker_id(Tok([7])) == 7
    assert tool_call_marker_id(Tok([1, 2])) is None


def test_both_fast_paths_are_wired():
    assert "close_reasoning_at_tool_call" in inspect.getsource(engine_fast)
    src = inspect.getsource(engine_stream)
    assert "_tool_marker" in src and "_in_thinking = False" in src
