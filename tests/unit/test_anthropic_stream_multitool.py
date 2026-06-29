"""the multi-tool-call fix was never propagated to the DEFAULT Anthropic
streaming path.

Default serving uses the legacy Engine (is_batched=False). That branch still had a `break`
after emitting a complete tool_call, so when a single token carried multiple
<tool_call>…</tool_call> blocks (forced-grammar / one-shot parallel output) every call after
the first — and any trailing text — was silently dropped. The batched path removed this
break; the non-batched path must match.
"""

from __future__ import annotations

import inspect

from yunshu_gateway.routers import anthropic


def test_non_batched_streaming_does_not_break_after_first_tool_call():
    src = inspect.getsource(anthropic)
    # the offending lone `break` after tool_use block close + tool_use_block_started=False
    # must be gone from the non-batched streaming path
    assert (
        "tool_use_block_started = False\n                                break"
        not in src
    )
    # both paths now carry the no-break rationale
    assert src.count("do NOT break") >= 2


def test_streamer_yields_two_tool_calls_for_doubled_markup():
    """The streamer surfaces BOTH calls from one doubled-markup token — proving the drop was
    real once the consuming loop stops breaking after the first."""
    from yunshu_engine.tool_call_streamer import ToolCallStreamer

    streamer = ToolCallStreamer()
    doubled = (
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
    )
    names = []
    for ch in doubled:
        for so in streamer.process_token(ch):
            if so.tool_call is not None:
                names.append(so.tool_call.name)
    for so in streamer.flush():
        if so.tool_call is not None:
            names.append(so.tool_call.name)
    assert names == ["a", "b"], f"expected both calls, got {names}"
