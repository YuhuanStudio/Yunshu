"""Waves 919-920: tool-call streamer format-coverage + DeepSeek double-text fixes.

W919 (HIGH): the DeepSeek ✿FUNCTION✿ transition pre-assigned self._buffer = buffer+token in
  process_token, then the handler appended token AGAIN → the transition token was buffered
  twice and the doubled text later leaked to the user.
W920 (HIGH): the streamer only understood XML <tool_call> + DeepSeek, so streaming a Mistral
  [TOOL_CALLS] or Qwen <function=name> response dropped ALL tool calls and leaked the markup
  as text (the non-streaming parse_tool_calls handled them). Now: detect those markers and
  buffer-all + parse at flush, plus a flush-time recovery net (re-parse the full text when no
  call was surfaced and a structural marker is present) for markers split across an emit
  boundary.
"""
from __future__ import annotations

from yunshu_engine.tool_call_streamer import ToolCallStreamer


def _stream(text, **kw):
    s = ToolCallStreamer(**kw)
    txt, calls = [], []
    for ch in text:
        for o in s.process_token(ch):
            if o.text:
                txt.append(o.text)
            if o.tool_call:
                calls.append((o.tool_call.name, o.tool_call.arguments))
    for o in s.flush():
        if o.text:
            txt.append(o.text)
        if o.tool_call:
            calls.append((o.tool_call.name, o.tool_call.arguments))
    return "".join(txt), calls


def test_w919_deepseek_transition_text_not_doubled():
    text, _ = _stream("hello ✿FUNCTION✿ never fences so this flushes as text")
    # the leading text must appear exactly once (was doubled)
    assert text.count("hello") == 1


def test_w920_mistral_streaming_recovers_call_no_leak():
    text, calls = _stream(
        'Let me check the weather for you right now. '
        '[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "SF"}}]')
    assert [c[0] for c in calls] == ["get_weather"]
    assert "[TOOL_CALLS]" not in text  # markup not leaked as content


def test_w920_qwen_function_xml_streaming():
    text, calls = _stream('<function=get_time>{"tz": "PST"}</function>')
    assert [c[0] for c in calls] == ["get_time"]
    assert "<function=" not in text


def test_w920_no_false_recovery_on_plain_json():
    # plain content that merely looks like {"name": ...} must NOT become a tool call
    _, calls = _stream('here is some data {"name": "Alice", "age": 30} end')
    assert calls == []


def test_w920_hermes_xml_unaffected():
    text, calls = _stream('<tool_call>{"name": "f", "arguments": {}}</tool_call>')
    assert [c[0] for c in calls] == ["f"]
    assert "<tool_call>" not in text


def test_w920_plain_text_unaffected():
    text, calls = _stream("just some normal assistant text with no tools")
    assert calls == []
    assert text == "just some normal assistant text with no tools"
