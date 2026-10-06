"""Streaming tool-call delta cases from vLLM tests/tool_parsers/test_qwen3coder_tool_parser.py (:858-1350,
Apache-2.0): a speculative step can deliver any slice of the text, so the streamed result under ANY
chunking must equal the one-shot result (calls, arguments, content)."""

from __future__ import annotations

import json
import random

import pytest

from tests.unit.test_tool_format import _formats
from yunshu_engine.tool_call_streamer import ToolCallStreamer


def _tool(name, props):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": props},
        },
    }


TOOLS = [
    _tool(
        "get_current_weather",
        {
            "city": {"type": "string"},
            "state": {"type": "string"},
            "n": {"type": "integer"},
        },
    )
]


def _blk(city, state="TX", n="3"):
    return (
        "<tool_call>\n<function=get_current_weather>\n"
        f"<parameter=city>\n{city}\n</parameter>\n<parameter=state>\n{state}\n</parameter>\n"
        f"<parameter=n>\n{n}\n</parameter>\n</function>\n</tool_call>"
    )


OUTPUTS = {
    "single": _blk("Dallas"),
    "content_then_call": "Sure! Let me check." + _blk("Dallas"),
    "two_calls": _blk("Dallas") + "\n" + _blk("Orlando", "FL", "5"),
    "call_then_content": _blk("Dallas") + "\nDone.",
    "dropped_close_parameter": (
        "<tool_call>\n<function=get_current_weather>\n<parameter=city>\nDallas\n"
        "<parameter=state>\nTX\n</parameter>\n</function>\n</tool_call>"
    ),
    "no_call": "Just text with <b>html</b> and a < sign.",
    "whitespace": "  lead\n\n" + _blk("Dallas") + "\n\ntail  ",
}


def _run(chunks):
    s = ToolCallStreamer(_formats("qwen3_coder"), tools=TOOLS)
    content, calls = "", []
    outs = []
    for c in chunks:
        outs += s.process_token(c)
    outs += s.flush()
    for o in outs:
        content += o.text or ""
        if o.tool_call is not None:
            calls.append((o.tool_call.name, json.loads(o.tool_call.arguments)))
    return content, calls


def _splits(text):
    yield [text]
    yield list(text)  # one char per delta
    for i in range(1, len(text)):
        yield [text[:i], text[i:]]
    rng = random.Random(7)
    for _ in range(40):
        cuts = sorted(
            rng.sample(range(1, len(text)), min(len(text) - 1, rng.randint(2, 9)))
        )
        yield [text[a:b] for a, b in zip([0, *cuts], [*cuts, len(text)], strict=True)]


@pytest.mark.parametrize("name", list(OUTPUTS))
def test_any_chunking_equals_one_shot(name):
    text = OUTPUTS[name]
    want = _run([text])
    for chunks in _splits(text):
        assert _run(chunks) == want, chunks[:6]


def test_one_shot_results():
    c, calls = _run([OUTPUTS["two_calls"]])
    assert [x[1]["city"] for x in calls] == ["Dallas", "Orlando"]
    assert calls[1][1]["n"] == 5 and c.strip() == ""
    c, calls = _run([OUTPUTS["content_then_call"]])
    assert c == "Sure! Let me check." and len(calls) == 1
    c, calls = _run([OUTPUTS["dropped_close_parameter"]])
    assert calls[0][1] == {"city": "Dallas", "state": "TX"}
    c, calls = _run([OUTPUTS["no_call"]])
    assert c == OUTPUTS["no_call"] and calls == []


def test_start_precedes_arguments_and_markup_never_in_content():
    s = ToolCallStreamer(_formats("qwen3_coder"), tools=TOOLS)
    text = OUTPUTS["content_then_call"]
    outs = []
    for ch in text:
        outs += s.process_token(ch)
    outs += s.flush()
    content = "".join(o.text for o in outs)
    assert "<tool_call" not in content and "<function" not in content
    kinds = [
        "start" if o.tool_call_start else "call" if o.tool_call else "x" for o in outs
    ]
    assert kinds.index("start") < kinds.index("call")
