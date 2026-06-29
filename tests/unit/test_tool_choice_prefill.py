"""Prefill-forced tool_choice.

`_inject_tool_system_prompt` only ADVISES the model to call a tool ("You MUST call
the tool…"); a model can ignore that and emit plain text, and "required"/named-forced
then can't be honored post-hoc (we can't fabricate arguments). The fix PREFILLS the
assistant turn with the opening `<tool_call>` marker (via the engine's
continue_final_message chat template) so the model is STRUCTURALLY committed to
continuing a tool call. The gateway prepends the prefill back onto the generated text
(or seeds the streamer with it) before tool-call parsing.

These tests exercise the helpers in isolation plus the prefill→parse pipeline with a
stubbed model output (no real model), mirroring the existing test_tool_choice_enforce
pattern.
"""

from __future__ import annotations

import inspect

from yunshu_engine.tool_call_streamer import ToolCallStreamer
from yunshu_gateway.routers.chat import (
    ToolChoiceFunction,
    _append_tool_prefill,
    _enforce_tool_choice,
    _tool_choice_prefill,
)
from yunshu_gateway.streaming import (
    clean_tool_call_markup,
    extract_tool_calls_model_aware,
)

MODEL = "qwen2.5"


# ── _tool_choice_prefill ──


def test_required_prefills_opening_marker():
    assert _tool_choice_prefill("required") == "<tool_call>\n"


def test_named_function_prefills_name_and_args_open():
    tc = ToolChoiceFunction(type="function", function={"name": "get_weather"})
    assert (
        _tool_choice_prefill(tc)
        == '<tool_call>\n{"name": "get_weather", "arguments": {'
    )


def test_named_function_json_escapes_exotic_name():
    # A name with a quote can't break out of the JSON shape.
    tc = ToolChoiceFunction(type="function", function={"name": 'we"ird'})
    pre = _tool_choice_prefill(tc)
    assert pre == '<tool_call>\n{"name": "we\\"ird", "arguments": {'


def test_auto_none_and_default_apply_no_prefill():
    assert _tool_choice_prefill("auto") == ""
    assert _tool_choice_prefill("none") == ""
    assert _tool_choice_prefill(None) == ""


# ── _append_tool_prefill ──


def test_append_prefill_adds_assistant_turn():
    msgs = [{"role": "user", "content": "weather?"}]
    out = _append_tool_prefill(msgs, "<tool_call>\n")
    assert out[-1] == {"role": "assistant", "content": "<tool_call>\n"}
    # input not mutated
    assert msgs == [{"role": "user", "content": "weather?"}]


def test_append_prefill_merges_into_trailing_assistant():
    msgs = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "Sure, "},
    ]
    out = _append_tool_prefill(msgs, "<tool_call>\n")
    assert out[-1]["content"] == "Sure, <tool_call>\n"
    assert len(out) == 2
    # input not mutated
    assert msgs[-1]["content"] == "Sure, "


def test_append_prefill_empty_is_noop():
    msgs = [{"role": "user", "content": "hi"}]
    assert _append_tool_prefill(msgs, "") is msgs


# ── prefill → parse pipeline (the non-streaming path's core) ──


def _parse_with_prefill(prefill: str, model_completion: str, tool_choice):
    """Mirror the non-streaming gateway path: the engine returns only the
    CONTINUATION (model_completion); prepend the prefill, parse, enforce."""
    raw_text = prefill + model_completion
    raw_calls = extract_tool_calls_model_aware(raw_text, MODEL)
    calls = _enforce_tool_choice(raw_calls, tool_choice, True)
    return raw_text, raw_calls, calls


def test_required_parses_call_from_completion():
    # Model, structurally committed by the prefill, emits name + arguments + close.
    prefill = _tool_choice_prefill("required")
    completion = '{"name": "get_weather", "arguments": {"city": "SF"}}\n</tool_call>'
    _raw, raw_calls, calls = _parse_with_prefill(prefill, completion, "required")
    assert raw_calls, "prefilled <tool_call> marker must let the parser see a tool call"
    assert len(calls) == 1
    assert calls[0]["name"] == "get_weather"


def test_named_function_parses_forced_name_from_args_only_completion():
    # With a named choice the NAME is prefilled, so the model returns ONLY the
    # arguments completion — yet the forced name must still appear in the call.
    tc = ToolChoiceFunction(type="function", function={"name": "get_weather"})
    prefill = _tool_choice_prefill(tc)
    completion = '"city": "SF"}}\n</tool_call>'
    _raw, raw_calls, calls = _parse_with_prefill(prefill, completion, tc)
    assert len(calls) == 1
    assert calls[0]["name"] == "get_weather"
    # arguments come back as a JSON string (or dict, depending on parser) — normalize.
    import json as _json

    args = calls[0]["arguments"]
    if isinstance(args, str):
        args = _json.loads(args)
    assert args == {"city": "SF"}


def test_prefill_markup_does_not_leak_into_content():
    # Whatever the model returns, the prefilled <tool_call> marker must be stripped
    # from user-visible content (the gateway cleans when a prefill was applied).
    prefill = _tool_choice_prefill("required")
    completion = '{"name": "get_weather", "arguments": {"city": "SF"}}\n</tool_call>'
    raw_text = prefill + completion
    cleaned = clean_tool_call_markup(raw_text)
    assert "<tool_call>" not in cleaned
    assert "get_weather" not in cleaned


# ── streaming: seeding the ToolCallStreamer with the prefill ──


def test_streamer_seeded_with_prefill_surfaces_call():
    # The opening marker lives in the prompt, so the streamer must be seeded with it
    # before the model's continuation or it never enters tool-call state.
    streamer = ToolCallStreamer(model_name=MODEL)
    prefill = _tool_choice_prefill("required")
    completion = '{"name": "get_weather", "arguments": {"city": "SF"}}</tool_call>'

    surfaced = []
    # Seed the prefill onto the first fed token (what the gateway streaming path does).
    for out in streamer.process_token(prefill + completion):
        if out.tool_call is not None:
            surfaced.append(out.tool_call)
    for out in streamer.flush():
        if out.tool_call is not None:
            surfaced.append(out.tool_call)

    assert len(surfaced) == 1
    assert surfaced[0].name == "get_weather"


def test_streamer_named_force_seeded_with_args_only_completion():
    tc = ToolChoiceFunction(type="function", function={"name": "get_weather"})
    streamer = ToolCallStreamer(model_name=MODEL, forced_tool_name="get_weather")
    prefill = _tool_choice_prefill(tc)
    completion = '"city": "SF"}}</tool_call>'

    surfaced = []
    for out in streamer.process_token(prefill + completion):
        if out.tool_call is not None:
            surfaced.append(out.tool_call)
    for out in streamer.flush():
        if out.tool_call is not None:
            surfaced.append(out.tool_call)

    assert len(surfaced) == 1
    assert surfaced[0].name == "get_weather"


# ── wiring: the endpoint applies prefill only for forced choices on BatchedEngine ──


def test_prefill_gated_on_batched_engine_and_forced_choice():
    from yunshu_gateway.routers import chat

    src = inspect.getsource(chat.create_chat_completion)
    # prefill is computed only when tools are present AND the engine is batched
    # (only BatchedEngine's chat template honors a trailing-assistant prefill).
    assert "if req.tools and is_batched:" in src
    assert "_tool_choice_prefill(req.tool_choice)" in src
    assert "_append_tool_prefill(messages, _tool_prefill)" in src
