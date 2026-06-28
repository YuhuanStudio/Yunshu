"""(HIGH): Responses previous_response_id chaining dropped a client-supplied
assistant tool_call turn while keeping its tool result → orphaned tool message on replay.

create_response snapshots this hop's own input as `_own_input_messages` for later chained
replay. The old filter stripped BOTH `assistant` and `system`. So a tool loop supplied in
the `input` list — user → assistant(tool_calls) → tool(result) — was stored as
[user, tool], and when a later request chained off it (previous_response_id), the bare
`tool` message (no preceding assistant tool_calls) was injected into the prompt and fed to
apply_chat_template, which requires a tool message to follow its assistant tool_calls →
template raise / plaintext fallback / model answering a call it never saw. The filter now
keeps assistant/tool turns and strips only system. (Assistant INPUT turns are distinct from
this response's generated `output`, replayed separately, so no double-count.)
"""
from __future__ import annotations

import inspect

from yunshu_gateway.routers import (
    responses as R,  # noqa: N812  # intentional short module alias
)
from yunshu_gateway.routers.responses import ResponsesRequest, _convert_to_messages


def _snapshot(messages):
    """Mirror the production _own_input_messages filter (single source: create_response)."""
    return [m for m in messages if m.get("role") != "system"]


def test_tool_loop_input_keeps_assistant_toolcall_anchor():
    req = ResponsesRequest(model="m", instructions="be terse", input=[
        {"role": "user", "content": "weather in SF?"},
        {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
    ])
    messages = _convert_to_messages(req)
    snap = _snapshot(messages)
    roles = [m["role"] for m in snap]
    # system/instructions stripped from the replay snapshot
    assert "system" not in roles
    # the assistant tool_call ANCHOR is preserved immediately before its tool result
    assert roles == ["user", "assistant", "tool"]
    asst = snap[1]
    assert asst.get("tool_calls") and asst["tool_calls"][0]["function"]["name"] == "get_weather"
    tool = snap[2]
    assert tool["role"] == "tool" and tool["tool_call_id"] == "c1"
    # the tool result is anchored: the message right before it carries the matching call id
    assert snap[1]["tool_calls"][0]["id"] == tool["tool_call_id"]


def test_plain_multiturn_assistant_text_preserved():
    req = ResponsesRequest(model="m", input=[
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "how are you"},
    ])
    snap = _snapshot(_convert_to_messages(req))
    assert [m["role"] for m in snap] == ["user", "assistant", "user"]
    assert snap[1]["content"] == "hello"  # the assistant turn is NOT dropped


def test_production_filter_strips_only_system():
    src = inspect.getsource(R.create_response)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    # the fix: snapshot strips only system, not assistant
    assert 'if m.get("role") != "system"' in code
    assert 'm.get("role") not in ("assistant", "system")' not in code
