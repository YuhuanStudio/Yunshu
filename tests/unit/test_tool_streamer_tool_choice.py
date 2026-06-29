"""the non-streaming chat path enforces the OpenAI tool_choice /
parallel_tool_calls contract via _enforce_tool_choice, but the STREAMING generators
emitted whatever the model produced — a named tool_choice did NOT filter a call to a
different function and parallel_tool_calls=False did NOT cap to one call. (Only the
injected system prompt advised the model, which it can ignore.)

Fix: ToolCallStreamer enforces the constraint at the source (forced_tool_name /
allow_parallel ctor args), so every streaming consumer — single-choice and per-choice
n>1 — honours it. A suppressed call has its start, every args delta, and its final
tool_call dropped. Unconstrained → pure pass-through (no behaviour change).
"""

from __future__ import annotations

from yunshu_engine.tool_call_streamer import ToolCallStreamer

_TWO = [
    "<tool_call>",
    '{"name":"alpha","arguments":{"x":1}}',
    "</tool_call>",
    "<tool_call>",
    '{"name":"beta","arguments":{"y":2}}',
    "</tool_call>",
]


def _run(streamer, tokens):
    starts, finals, args = [], [], []
    content = ""

    def consume(outs):
        nonlocal content
        for o in outs:
            if getattr(o, "text", None):
                content += o.text
            if o.tool_call_start:
                starts.append(o.tool_call_start.name)
            if o.tool_call_args_delta:
                args.append(o.tool_call_args_delta)
            if o.tool_call:
                finals.append(o.tool_call.name)

    for t in tokens:
        consume(streamer.process_token(t))
    consume(streamer.flush())
    return starts, finals, args, content


def test_unconstrained_passthrough_surfaces_all():
    starts, finals, _, _ = _run(ToolCallStreamer(), _TWO)
    assert starts == ["alpha", "beta"]
    assert finals == ["alpha", "beta"]


def test_parallel_false_keeps_only_first():
    starts, finals, _, _ = _run(ToolCallStreamer(allow_parallel=False), _TWO)
    assert starts == ["alpha"]
    assert finals == ["alpha"]


def test_forced_name_keeps_only_match():
    starts, finals, _, _ = _run(ToolCallStreamer(forced_tool_name="beta"), _TWO)
    assert starts == ["beta"]
    assert finals == ["beta"]


def test_forced_name_no_match_suppresses_all():
    starts, finals, args, _ = _run(ToolCallStreamer(forced_tool_name="zeta"), _TWO)
    assert starts == []
    assert finals == []
    # the suppressed calls' argument fragments must NOT leak either
    assert args == []


def test_reset_clears_enforcement_counters():
    s = ToolCallStreamer(allow_parallel=False)
    starts1, _, _, _ = _run(s, _TWO)
    assert starts1 == ["alpha"]
    s.reset()
    # after reset the per-request accepted counter is cleared → first call surfaces again
    starts2, _, _, _ = _run(s, _TWO)
    assert starts2 == ["alpha"]
