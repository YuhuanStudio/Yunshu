"""Qwen3 reasoning split cases ported from vLLM tests/reasoning/test_qwen3_reasoning_parser.py
(Apache-2.0, vllm-project/vllm). Yunshu's parser sees no 'thinking disabled' signal, so the
'no tags at all' case is content here (the engine decides from the prompt); the rest must agree."""

from __future__ import annotations

import pytest

from yunshu_engine.reasoning_parser import QwenReasoningParser

BODY = "<tool_call>\n<function=bash>\n<parameter=command>ls</parameter>\n</function>\n</tool_call>"

CASES = {
    "without_start_token": (
        "This is a reasoning section</think>This is the rest",
        "This is a reasoning section",
        "This is the rest",
    ),
    "without_start_complete": (
        "This is a reasoning section</think>",
        "This is a reasoning section",
        "",
    ),
    "with_think": (
        "<think>This is a reasoning section</think>This is the rest",
        "This is a reasoning section",
        "This is the rest",
    ),
    "complete_reasoning": (
        "<think>This is a reasoning section</think>",
        "This is a reasoning section",
        "",
    ),
    "multiline": (
        "<think>This is a reasoning\nsection</think>This is the rest\nThat",
        "This is a reasoning\nsection",
        "This is the rest\nThat",
    ),
    "only_open_tag": (
        "<think>This is a reasoning section",
        "This is a reasoning section",
        "",
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_split(name):
    out, reasoning, content = CASES[name]
    r = QwenReasoningParser().parse(out)
    assert (r.reasoning, r.content) == (reasoning, content)


def test_tool_call_inside_unclosed_reasoning_ends_reasoning():
    from yunshu_engine.reasoning_parser import close_reasoning_at_tool_call

    for prompt_opens, text in (
        (True, "I need to read the file.\n\n" + BODY),
        (False, "<think>I need to read the file.\n\n" + BODY),
    ):
        fixed = close_reasoning_at_tool_call(text, prompt_opens)
        r = QwenReasoningParser().parse(fixed)
        assert r.reasoning == "I need to read the file."
        assert r.content == BODY
