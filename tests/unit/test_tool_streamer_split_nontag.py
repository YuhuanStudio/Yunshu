"""the streaming tool-call accumulator (ToolCallStreamer, the DEFAULT batched
streaming path) silently DROPPED content when a "<tool_call"-prefixed token that is NOT a
real tool-call tag (<tool_calls>, <tool_call_id>, <tool_callback>) arrived SPLIT across
token deltas. The inline detector in _handle_text_state validates the delimiter char after
"<tool_call" (must be space/>/\\//), rejecting such non-tags — but a split token reaches
TAG_START instead, which accepted ANY "<tool_call...>" and consumed the surrounding text as
a bogus tool call. Verified: ['Use ','<tool','_calls','>',' here'] streamed 'Use  here'
(the <tool_calls> and its neighbours lost).

Fix: TAG_START now applies the same delimiter validation when it completes the tag; a
non-tag reverts to TEXT and is re-scanned as plain content (nothing lost). Real tool calls
— including ones whose open tag is split across deltas — still parse.
"""

from __future__ import annotations

from yunshu_engine.tool_call_streamer import ToolCallStreamer


def _run(tokens):
    s = ToolCallStreamer()
    content = ""
    calls = 0
    for t in tokens:
        for o in s.process_token(t):
            if getattr(o, "text", None):
                content += o.text
            if getattr(o, "tool_call_start", None) or getattr(o, "tool_call_id", None):
                calls += 1
    for o in s.flush():
        if getattr(o, "text", None):
            content += o.text
    return content, calls


def test_split_tool_calls_nontag_preserves_content():
    content, calls = _run(["Use ", "<tool", "_calls", ">", " here"])
    assert content == "Use <tool_calls> here"
    assert calls == 0


def test_split_tool_call_id_nontag_preserves_content():
    content, calls = _run(["ref ", "<tool", "_call", "_id", ">", "x"])
    assert content == "ref <tool_call_id>x"
    assert calls == 0


def test_split_tool_callback_nontag_preserves_content():
    content, calls = _run(["a ", "<tool", "_callback", ">", " b"])
    assert content == "a <tool_callback> b"
    assert calls == 0


def test_real_tool_call_still_parses_unsplit():
    content, calls = _run(
        ["<tool_call>", '{"name": "search", "arguments": {"q": "hi"}}', "</tool_call>"]
    )
    assert calls == 1
    assert content == ""


def test_real_tool_call_with_split_open_tag_and_preceding_text():
    # the open tag is split across deltas AND has preceding text — must still be detected
    content, calls = _run(
        ["Call ", "<tool", "_call", ">", '{"name":"f","arguments":{}}', "</tool_call>"]
    )
    assert calls == 1
    assert content == "Call "
