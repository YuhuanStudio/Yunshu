"""A forced Anthropic tool_choice turns thinking off unless the request enables it."""

from yunshu_gateway.routers.anthropic import (
    AnthropicMessagesRequest,
    _thinking_switches,
)


def _req(**kw):
    return AnthropicMessagesRequest(
        model="m", max_tokens=2048, messages=[{"role": "user", "content": "x"}], **kw
    )


def test_forced_tool_choice_disables_default_thinking():
    for tc in ({"type": "tool", "name": "web_fetch"}, {"type": "any"}):
        assert _thinking_switches(_req(tool_choice=tc)) == (False, None)
        assert _thinking_switches(
            _req(tool_choice=tc, thinking={"type": "adaptive"})
        ) == (False, None)


def test_unforced_and_explicit_thinking_unchanged():
    assert _thinking_switches(_req()) == (None, None)
    assert _thinking_switches(_req(tool_choice={"type": "auto"})) == (None, None)
    assert _thinking_switches(
        _req(
            tool_choice={"type": "any"},
            thinking={"type": "enabled", "budget_tokens": 64},
        )
    ) == (True, 64)
    assert _thinking_switches(_req(thinking={"type": "disabled"})) == (False, None)
