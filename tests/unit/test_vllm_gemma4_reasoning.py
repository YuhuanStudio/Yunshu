"""Gemma 4 reasoning split cases from vLLM tests/reasoning/test_gemma4_reasoning_parser.py (Apache-2.0),
non-streaming ones whose semantics match Yunshu's text parser."""

from __future__ import annotations

import pytest

from yunshu_engine.reasoning_parser import GemmaReasoningParser

CASES = {
    "channel": (
        "<|channel>This is a reasoning section<channel|>This is the rest",
        "This is a reasoning section",
        "This is the rest",
    ),
    "complete": (
        "<|channel>This is a reasoning section<channel|>",
        "This is a reasoning section",
        "",
    ),
    "multiline": (
        "<|channel>This\nThat<channel|>This is the rest\nThat",
        "This\nThat",
        "This is the rest\nThat",
    ),
    "no_end": (
        "<|channel>This is a reasoning section",
        "This is a reasoning section",
        "",
    ),
    "thought_prefix": (
        "<|channel>thought\nActual reasoning here<channel|>Final answer",
        "Actual reasoning here",
        "Final answer",
    ),
    "thought_multiline": (
        "<|channel>thought\nLine1\nLine2<channel|>Answer",
        "Line1\nLine2",
        "Answer",
    ),
    "thousand_not_thought": (
        "<|channel>thousand reasons<channel|>Done",
        "thousand reasons",
        "Done",
    ),
    "no_start_token": (
        "This is a reasoning section<channel|>This is the rest",
        "This is a reasoning section",
        "This is the rest",
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_split(name):
    out, reasoning, content = CASES[name]
    r = GemmaReasoningParser().parse(out)
    assert (r.reasoning, r.content) == (reasoning, content)


def test_plain_content_and_tool_call_untouched():
    for t in ("This is content", "<|tool_call>call:f{}<tool_call|>"):
        r = GemmaReasoningParser().parse(t)
        assert r.reasoning is None and r.content == t


def test_legacy_tags_still_work():
    r = GemmaReasoningParser().parse("<start_think>a</end_think>b")
    assert (r.reasoning, r.content) == ("a", "b")
