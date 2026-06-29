"""The Anthropic streaming tool path drained the tool-call streamer's outputs
for each token with `for _tc_out in _tool_streamer.process_token(...)`, but the
`elif _tc_out.tool_call:` branch ended with a `break`. A single token can carry MORE than
one complete tool call (forced-grammar / one-shot output of
<tool_call>…</tool_call><tool_call>…</tool_call>) — the streamer emits one tool_call output
per call, so breaking after the first DROPPED every subsequent call (and trailing text) in
that token. Anthropic supports parallel tool_use blocks and the OpenAI chat path has no such
break. Fix: remove the break so the loop drains all of the token's outputs.
"""

from __future__ import annotations

import inspect
import re

from yunshu_engine.tool_call_streamer import ToolCallStreamer
from yunshu_gateway.routers import anthropic


def test_one_token_emits_multiple_tool_calls_premise():
    """The data the loop must not drop: one token → two complete tool_call outputs."""
    s = ToolCallStreamer()
    one_token = (
        '<tool_call>{"name":"a","arguments":{"x":1}}</tool_call>'
        '<tool_call>{"name":"b","arguments":{"y":2}}</tool_call>'
    )
    names = [o.tool_call.name for o in s.process_token(one_token) if o.tool_call]
    assert names == ["a", "b"], f"streamer should emit both calls, got {names}"


def test_anthropic_tool_loop_no_longer_breaks_after_first_call():
    """Structural guard: the `elif _tc_out.tool_call:` branch closed the tool_use block
    (tool_use_block_started=False; _tc_args_streamed=False) and then `break`-ed out of the
    per-token drain loop. That exact `_tc_args_streamed = False` → `break` sequence is the
    bug and must be gone (legitimate stop-sequence breaks inside `for seq in stop:` stay)."""
    src = inspect.getsource(anthropic._stream_anthropic)
    # the buggy pattern: the tool_call branch's `_tc_args_streamed = False` followed
    # (next non-comment, non-blank line) by a bare `break`.
    assert not re.search(r"_tc_args_streamed = False\s*\n\s*break\b", src), (
        "the tool_call branch still breaks out of the drain loop, dropping later calls"
    )
    # and the rationale comment is present at the fix site
    assert "do NOT break" in src
