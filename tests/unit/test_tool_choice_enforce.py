"""OpenAI tool_choice / parallel_tool_calls were ADVISORY ONLY — injected as
natural-language system prompt by `_inject_tool_system_prompt` but never enforced on the
extracted call list. A model that ignored the prompt could emit tool calls under
tool_choice="none", emit a different tool than a forced named choice, or emit multiple
calls under parallel_tool_calls=false. `_enforce_tool_choice` now applies the hard
deterministic guarantees post-extraction across all 3 non-streaming response paths, and
the streamer is gated off under tool_choice="none"."""

from __future__ import annotations

from yunshu_gateway.routers.chat import ToolChoiceFunction, _enforce_tool_choice


def _calls():
    return [
        {"name": "get_weather", "arguments": {"city": "SF"}},
        {"name": "get_time", "arguments": {"tz": "PST"}},
    ]


def test_none_suppresses_all_calls():
    assert _enforce_tool_choice(_calls(), "none", True) == []
    # None convention preserved when input is None
    assert _enforce_tool_choice(None, "none", True) is None


def test_auto_passes_through():
    out = _enforce_tool_choice(_calls(), "auto", True)
    assert len(out) == 2
    out2 = _enforce_tool_choice(_calls(), None, True)
    assert len(out2) == 2


def test_named_function_drops_mismatched():
    tc = ToolChoiceFunction(type="function", function={"name": "get_time"})
    out = _enforce_tool_choice(_calls(), tc, True)
    assert len(out) == 1
    assert out[0]["name"] == "get_time"


def test_named_function_zero_match_returns_empty():
    tc = ToolChoiceFunction(type="function", function={"name": "no_such_tool"})
    assert _enforce_tool_choice(_calls(), tc, True) == []


def test_parallel_false_truncates_to_first():
    out = _enforce_tool_choice(_calls(), "auto", False)
    assert len(out) == 1
    assert out[0]["name"] == "get_weather"


def test_parallel_false_named_compose():
    # named filter first, then single-call truncation (already 1 here)
    tc = ToolChoiceFunction(type="function", function={"name": "get_time"})
    out = _enforce_tool_choice(_calls(), tc, False)
    assert len(out) == 1
    assert out[0]["name"] == "get_time"


def test_empty_input_unchanged():
    assert _enforce_tool_choice([], "none", True) == []
    assert _enforce_tool_choice(None, "auto", True) is None


def test_all_three_nonstream_paths_call_enforce():
    import inspect

    from yunshu_gateway.routers import chat

    src = inspect.getsource(chat)
    # the enforcement helper is invoked right after every model-aware extraction.
    # renamed the extraction var to _raw_calls (cleanup now gates on the raw
    # parse, not the enforced result) but enforcement is still called on all 3 paths.
    assert (
        src.count(
            "_enforce_tool_choice(_raw_calls, req.tool_choice, req.parallel_tool_calls)"
        )
        >= 3
    )


def test_streamer_gated_on_not_none():
    import inspect

    from yunshu_gateway.routers import chat

    src = inspect.getsource(chat)
    assert src.count('len(req.tools) > 0 and req.tool_choice != "none"') >= 2
