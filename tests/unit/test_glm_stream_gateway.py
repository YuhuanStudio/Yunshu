"""Gateway SSE must emit a tool_call_start (id+name) for a single-shot tool_call.

An earlier fix handled the STREAMER half of GLM streaming tool calls, but an adversarial
self-review found the GATEWAY half still dropped them: for GLM-4.x the name is bare text so the
streamer never emits a `tool_call_start`, producing a single complete `out.tool_call`. The chat
SSE `out.tool_call` branch emitted only an arguments delta — with NO id and NO function.name —
so an OpenAI-compatible streaming client (which keys the first delta per tool-call index on
id+name) couldn't use the call and dropped it. That fix's test only checked the streamer level,
so it passed while end-to-end SSE was still broken.

Fix: track `_choice_tc_start_emitted`; in the `out.tool_call` branch, synthesize the start
chunk (id+name) before the args delta when no start was emitted. (Both streaming choice blocks.)
"""

from __future__ import annotations

import inspect

from yunshu_gateway.routers import chat


def test_start_chunk_synthesized_for_single_shot_tool_call():
    src = inspect.getsource(chat)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    # the tracker flag exists and is initialized
    assert "_choice_tc_start_emitted = False" in code
    # the out.tool_call branch synthesizes the start chunk when none was emitted
    assert "if not _choice_tc_start_emitted:" in code
    # it carries the complete call's id + name (not just arguments)
    assert "tc_id=out.tool_call.id" in code
    assert "tc_name=out.tool_call.name" in code
    # the flag is set when a real tool_call_start IS emitted (so we don't double-send)
    assert "_choice_tc_start_emitted = True" in code


def test_both_streaming_blocks_fixed():
    """There are two streaming choice blocks (single + multi-choice); both must synthesize."""
    src = inspect.getsource(chat)
    # the start-synthesis appears in BOTH blocks (replace_all applied)
    assert src.count("tc_id=out.tool_call.id") == 2
    assert (
        src.count("_choice_tc_start_emitted = False  # next index needs its own start")
        == 2
    )


def test_start_synthesis_precedes_args_in_source_order():
    """The start chunk must be yielded BEFORE the args-delta in the out.tool_call branch."""
    src = inspect.getsource(chat)
    # within the branch, the start-synthesis guard appears before the args-streamed guard
    start_idx = src.index("if not _choice_tc_start_emitted:")
    # the next args guard after it
    args_idx = src.index("if not _choice_tc_args_streamed:", start_idx)
    assert start_idx < args_idx
