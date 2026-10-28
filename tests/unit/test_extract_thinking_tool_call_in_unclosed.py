"""A complete <tool_call> block inside reasoning that never closes is a tool call, not thinking
text (Qwen3.x opens calls before </think> sometimes); everything else keeps its old split."""

from yunshu_gateway.streaming import extract_thinking

CALL = '<tool_call>\n{"name": "Read", "arguments": {"path": "/a"}}\n</tool_call>'
XML_CALL = (
    "<tool_call>\n<function=Read>\n<parameter=path>\n/a\n</parameter>\n</function>\n"
    "</tool_call>"
)


def test_complete_call_in_unclosed_think_is_content():
    thinking, content = extract_thinking("<think>I should read it.\n" + CALL)
    assert thinking == "I should read it."
    assert content == CALL


def test_xml_format_and_trailing_text_follow_the_call():
    thinking, content = extract_thinking("<think>plan " + XML_CALL + "\ndone")
    assert thinking == "plan"
    assert content == XML_CALL + "\ndone"


def test_text_before_the_open_tag_stays_content():
    thinking, content = extract_thinking("Sure. <think>plan " + CALL)
    assert (thinking, content) == ("plan", "Sure. " + CALL)


def test_incomplete_call_stays_in_thinking():
    cut = '<think>plan <tool_call>\n{"name": "Read"'
    thinking, content = extract_thinking(cut)
    assert thinking == cut[len("<think>") :].strip() and content == ""


def test_unclosed_think_without_call_is_unchanged():
    assert extract_thinking("<think>still thinking about <tool_call> syntax") == (
        "still thinking about <tool_call> syntax",
        "",
    )


def test_closed_think_is_unchanged():
    assert extract_thinking("<think>plan</think>" + CALL) == ("plan", CALL)
    assert extract_thinking("<think>mention " + CALL + "</think>ok") == (
        "mention " + CALL,
        "ok",
    )
