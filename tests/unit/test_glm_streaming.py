"""GLM-4.x streaming tool calls were silently dropped / leaked as text.

The ToolCallStreamer defaults to StreamFormat.XML. GLM-4.6/4.7 emit
`<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` — key/value
PAIRS, not a JSON body — so the streamer entered TOOL_JSON via the shared `<tool_call`
opener, _parse_tool_json (JSON-only) returned None, and the whole tool call was DROPPED.
The older block form `<|tool_call_block_begin|>name```json{...}``` ` had no handler and
LEAKED raw into delta.content as text. Both are now handled: the block form is a
BUFFER_ALL marker, the key/value form falls back to the model-aware GLM parser at the
close tag, and a GLMToolCallParser was added to the flush registry (it had drifted from
the non-streaming registry which already had one). Streaming now matches non-streaming.
"""
from __future__ import annotations

from yunshu_engine.tool_call_streamer import ToolCallStreamer


def _run(tokens, model="glm-4.6"):
    s = ToolCallStreamer(model_name=model)
    outs = []
    for t in tokens:
        outs.extend(s.process_token(t))
    outs.extend(s.flush())
    calls = [(o.tool_call.name, o.tool_call.arguments) for o in outs if o.tool_call]
    text = "".join(o.text for o in outs)
    return calls, text


def test_glm46_keyvalue_form_surfaces_call_no_leak():
    calls, text = _run([
        "Let me check.", "<tool_call>", "get_weather", "<arg_key>", "location",
        "</arg_key>", "<arg_value>", "San Francisco", "</arg_value>", "</tool_call>",
    ])
    assert len(calls) == 1
    assert calls[0][0] == "get_weather"
    assert "San Francisco" in calls[0][1]
    # only the legitimate preamble leaks — NOT the tool markup
    assert text == "Let me check."
    assert "<tool_call>" not in text and "<arg_key>" not in text


def test_glm_block_form_surfaces_call_no_leak():
    calls, text = _run([
        "Sure.", "<|tool_call_block_begin|>", "get_weather", "\n```json\n",
        '{"location": "SF"}', "\n```", "<|tool_call_block_end|>",
    ])
    assert len(calls) == 1
    assert calls[0][0] == "get_weather"
    assert "SF" in calls[0][1]
    assert text == "Sure."
    assert "tool_call_block_begin" not in text


def test_glm_parser_in_flush_registry():
    from yunshu_engine.tool_call_parser import _MODEL_HINTS, _REGISTRY, parse_tool_calls
    assert "glm" in _REGISTRY
    assert any(p.search("glm-4.6") for p, _ in _MODEL_HINTS)
    # the registry parser handles GLM-4.6 markup end-to-end
    calls = parse_tool_calls(
        "<tool_call>get_weather<arg_key>location</arg_key><arg_value>SF</arg_value></tool_call>",
        model_name="glm-4.6",
    )
    assert len(calls) == 1 and calls[0].name == "get_weather"


def test_no_regression_hermes_mistral_plain():
    # Hermes JSON body still parses
    c, _ = _run(['<tool_call>', '{"name": "f", "arguments": {"a": 1}}', '</tool_call>'], "qwen2.5")
    assert c and c[0][0] == "f"
    # Mistral [TOOL_CALLS] still parses, preamble preserved
    c, t = _run(["Hi ", "[TOOL_CALLS]", '[{"name": "f", "arguments": {"a": 1}}]'], "mistral")
    assert c and c[0][0] == "f" and t == "Hi "
    # plain text untouched
    c, t = _run(["Hello ", "world", "!"], "qwen2.5")
    assert not c and t == "Hello world!"
