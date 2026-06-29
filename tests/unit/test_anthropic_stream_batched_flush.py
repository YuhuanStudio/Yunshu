"""(HIGH): the Anthropic /v1/messages STREAMING batched path
(is_batched=True, opt-in YUNSHU_ENGINE_LOOP=1) never called ToolCallStreamer.flush().

In BUFFER_ALL mode (— Mistral [TOOL_CALLS], Qwen, GLM block-form) and for tool calls
truncated at max_tokens, the streamer HOLDS the whole tool-call buffer during process_token
(yields nothing) and only parses + emits the calls at flush(). The flush loop was nested
inside the non-batched `else:` branch (indent 12), so the batched branch fell straight
through to finalization with no flush → the SSE stream emitted no tool_use content_block at
all, _has_tool_calls stayed False, and stop_reason was end_turn instead of tool_use (the
agent loop breaks). Fix: dedent the flush block to run after BOTH branches.

The streaming generator is a deeply nested closure, so this is verified structurally
(indent of the flush guard) plus the existing non-batched test still passing.
"""
from __future__ import annotations

import inspect

from yunshu_gateway.routers import anthropic


def test_flush_guard_runs_outside_the_batched_else_branch():
    src = inspect.getsource(anthropic)
    lines = src.splitlines()
    flush_idx = next(i for i, ln in enumerate(lines)
                     if "for _tc_out in _tool_streamer.flush():" in ln)
    guard_idx = next(i for i in range(flush_idx, -1, -1)
                     if "if has_tools and _tool_streamer:" in lines[i])
    guard_indent = len(lines[guard_idx]) - len(lines[guard_idx].lstrip())
    # 8 = function-body level (after the if/else); 12 would mean still nested in `else`.
    assert guard_indent == 8, (
        f"flush guard at indent {guard_indent}; must be 8 so the batched branch flushes too")


def test_only_one_flush_block_exists():
    # the dedent must MOVE the flush, not duplicate it (a second flush would re-emit
    # tool_use blocks / double-increment block_index).
    src = inspect.getsource(anthropic)
    assert src.count("for _tc_out in _tool_streamer.flush():") == 1
