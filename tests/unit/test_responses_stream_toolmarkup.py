"""the Responses streaming path leaked tool-call markup into output_text.delta.

Every non-reasoning token was emitted verbatim as a response.output_text.delta, so raw
<tool_call>…</tool_call> / ChatML markup appeared as visible assistant text in the stream,
while only the terminal .done carried the markup-stripped text. The chat router already
routes streamed tokens through a ToolCallStreamer; the Responses streaming generator now
does the same — emitting only the clean-text chunks as deltas and surfacing tool calls as
function_call items at end-of-stream.
"""

from __future__ import annotations

import inspect

from yunshu_engine.tool_call_streamer import ToolCallStreamer


def test_responses_streaming_routes_text_through_tool_streamer():
    from yunshu_gateway.routers import responses

    src = inspect.getsource(responses)
    # the streaming generator builds a ToolCallStreamer when tools are active
    assert "_resp_tool_streamer = ToolCallStreamer(" in src
    assert 'req.tools and req.tool_choice != "none"' in src
    # text deltas are gated on the streamer's clean-text chunks, not raw tokens
    assert "_resp_tool_streamer.process_token(token_text)" in src
    assert "_resp_tool_streamer.flush()" in src


def test_tool_streamer_holds_back_markup_from_text():
    """Mechanism check: feeding tool-call markup yields NO markup in the .text chunks."""
    streamer = ToolCallStreamer()
    markup = (
        '<tool_call>{"name": "get_weather", "arguments": {"city": "SF"}}</tool_call>'
    )
    emitted_text = []
    saw_tool = False
    # feed char-by-char to simulate token streaming
    for ch in markup:
        for so in streamer.process_token(ch):
            if so.text:
                emitted_text.append(so.text)
            if so.tool_call is not None or so.tool_call_start is not None:
                saw_tool = True
    for so in streamer.flush():
        if so.text:
            emitted_text.append(so.text)
        if so.tool_call is not None:
            saw_tool = True
    joined = "".join(emitted_text)
    assert "<tool_call>" not in joined
    assert "get_weather" not in joined
    assert saw_tool, "the tool call should be surfaced structurally, not as text"


def test_tool_streamer_passes_plain_text_through():
    streamer = ToolCallStreamer()
    out = []
    for ch in "Hello, world!":
        for so in streamer.process_token(ch):
            if so.text:
                out.append(so.text)
    for so in streamer.flush():
        if so.text:
            out.append(so.text)
    assert "".join(out) == "Hello, world!"
