"""(HIGH): non-streaming tool-call extraction dropped + LEAKED calls when a
composite/distill model name misroutes the family detector.

detect_tool_call_format(name) returns the FIRST family substring found, in dict order
(qwen, qwq, deepseek, glm, ...). A merged/distill name like "Qwen-DeepSeek-merge" or
"DeepSeek-R1-Distill-Llama" therefore dispatches to the wrong family parser. If the model
then emits a DIFFERENT family's DISTINCTIVE markers (DeepSeek fullwidth tokens, GLM blocks,
Llama [TOOL_CALL]), the detected parser returns [] and extract_tool_calls_v2 has no rule for
those markers → [] → the call is DROPPED and, because the chat assembly's `if _raw_calls:`
gate is empty, clean_tool_call_markup is SKIPPED and the raw markup leaks into
message.content (finish_reason stuck at "stop").

W1040 adds a last-resort sweep through the OTHER distinctive-marker parsers (NOT the generic
bare-JSON parser) without touching the fragile detection precedence — so correctly-routed
names (incl. DeepSeek-R1-Distill-Qwen, which genuinely IS Qwen-format) are unaffected.
"""
from __future__ import annotations

import json

from yunshu_engine.tool_call_parsers import ToolCallFormat, detect_tool_call_format
from yunshu_gateway.streaming import extract_tool_calls_model_aware

# DeepSeek-V3 fullwidth on-the-wire form: <｜tool▁call▁begin｜>name<｜tool▁sep｜>{args}<｜tool▁call▁end｜>
_DEEPSEEK_WIRE = (
    "Sure.<｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>"
    '{"city": "Paris"}<｜tool▁call▁end｜>'
)


def test_misrouted_name_recovers_deepseek_markup_via_sweep():
    # name contains "qwen" FIRST → detector picks QWEN, but the text is DeepSeek markup.
    name = "Qwen-DeepSeek-merge-v1"
    assert detect_tool_call_format(name) is ToolCallFormat.QWEN  # the misroute
    calls = extract_tool_calls_model_aware(_DEEPSEEK_WIRE, name)
    assert len(calls) == 1, calls
    assert calls[0]["name"] == "get_weather"
    assert json.loads(calls[0]["arguments"]) == {"city": "Paris"}


def test_ordinary_json_completion_is_not_a_phantom_tool_call():
    # No tool markers anywhere → the marker guard must keep the sweep off → [].
    text = 'Here is the data you asked for: {"city": "Paris", "temp": 21}.'
    assert extract_tool_calls_model_aware(text, "Qwen-DeepSeek-merge-v2") == []


def test_correctly_routed_qwen_still_parsed_by_detected_parser():
    # Regression: a real Qwen name + Qwen markup is handled by the detected parser,
    # before the sweep ever runs (path unchanged).
    text = '<tool_call>{"name": "get_time", "arguments": {"tz": "UTC"}}</tool_call>'
    calls = extract_tool_calls_model_aware(text, "Qwen2.5-7B-Instruct")
    assert len(calls) == 1 and calls[0]["name"] == "get_time"


def test_distill_qwen_name_stays_qwen_format_unchanged():
    # DeepSeek-R1-Distill-Qwen genuinely IS Qwen-format (Qwen base, Hermes <tool_call>).
    # The detector picks QWEN (correct); the fix must NOT change that, and Qwen markup
    # must still parse via the detected parser.
    name = "DeepSeek-R1-Distill-Qwen-7B"
    assert detect_tool_call_format(name) is ToolCallFormat.QWEN
    text = '<tool_call>{"name": "search", "arguments": {"q": "x"}}</tool_call>'
    calls = extract_tool_calls_model_aware(text, name)
    assert len(calls) == 1 and calls[0]["name"] == "search"
