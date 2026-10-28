"""Pins for tool-call bugs peers fixed that Yunshu's readers already handle (mlx-swift-lm 0754e10, a1f575d)."""

import json
import time

import pytest

from yunshu_engine import tool_format as tf
from yunshu_engine.tool_call_streamer import ToolCallStreamer

Q = '<|"|>'
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fhir_search",
            "parameters": {
                "type": "object",
                "properties": {
                    "resourceType": {"type": "string"},
                    "searchParams": {"type": "object"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    }
]


def _gemma(body):
    fmt = tf._upstream("gemma4")
    if fmt is None:
        pytest.skip("no gemma4 tool parser in this mlx-vlm")
    (call,) = fmt.parse(body, TOOLS)
    assert call["name"] == "fhir_search"
    return json.loads(call["arguments"])


def test_gemma_marker_quoted_strings_inside_a_nested_object():
    got = _gemma(
        f"call:fhir_search{{resourceType:{Q}Observation{Q},"
        f"searchParams:{{_count:{Q}1{Q},_sort:{Q}effective_datetime-1{Q}}}}}"
    )
    assert got == {
        "resourceType": "Observation",
        "searchParams": {"_count": "1", "_sort": "effective_datetime-1"},
    }


def test_gemma_single_key_nested_object():
    got = _gemma(f"call:fhir_search{{searchParams:{{name:{Q}Maria Schneider{Q}}}}}")
    assert got == {"searchParams": {"name": "Maria Schneider"}}


def test_gemma_marker_quoted_keys_inside_nested_objects():
    got = _gemma(
        f"call:fhir_search{{searchParams:{{{Q}given name{Q}:{Q}Maria{Q},"
        f"filter:{{{Q}a b{Q}:1,{Q}code{Q}:{Q}8867-4{Q}}}}}}}"
    )
    assert got == {
        "searchParams": {"given name": "Maria", "filter": {"a b": 1, "code": "8867-4"}}
    }


def test_gemma_array_of_marker_strings_and_a_quoted_top_level_key():
    assert _gemma(
        f"call:fhir_search{{tags:[{Q}a, b{Q},{Q}c{Q}],resourceType:{Q}X{Q}}}"
    ) == {
        "tags": ["a, b", "c"],
        "resourceType": "X",
    }
    assert _gemma(f"call:fhir_search{{{Q}resourceType{Q}:{Q}X{Q}}}") == {
        "resourceType": "X"
    }


def test_long_call_streamed_one_character_at_a_time_stays_cheap():
    fmt = tf._upstream("qwen3_coder")
    if fmt is None:
        pytest.skip("no qwen3_coder tool parser in this mlx-lm")
    write = [
        {
            "type": "function",
            "function": {
                "name": "Write",
                "parameters": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                },
            },
        }
    ]
    streamer = ToolCallStreamer([fmt], tools=write)
    text = (
        "<tool_call>\n<function=Write>\n<parameter=content>\n"
        + "x" * 40_000
        + "\n</parameter>\n</function>\n</tool_call>"
    )
    start = time.perf_counter()
    outs = []
    for ch in text:
        outs += streamer.process_token(ch)
    outs += streamer.flush()
    elapsed = time.perf_counter() - start
    calls = [o.tool_call for o in outs if o.tool_call]
    assert len(calls) == 1 and len(json.loads(calls[0].arguments)["content"]) >= 40_000
    assert (
        elapsed < 10
    )  # 0.3 s measured; the bound only catches an accidental O(n^2) in Python
