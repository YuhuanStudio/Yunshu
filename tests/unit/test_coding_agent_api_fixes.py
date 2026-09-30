"""API gaps found by driving real coding agents (Claude Code and Codex CLI) at the server."""

from yunshu_engine.message_adapter import QwenMessageAdapter
from yunshu_gateway.routers.anthropic import (
    AnthropicMessage,
    AnthropicMessagesRequest,
    _thinking_flag,
)


def _req(thinking):
    return AnthropicMessagesRequest(
        model="m",
        messages=[AnthropicMessage(role="user", content="hi")],
        max_tokens=100,
        thinking=thinking,
    )


def test_adaptive_thinking_is_accepted():
    # Claude Code 2.x sends {"type": "adaptive"} on every request; it was a 400.
    _req({"type": "adaptive"}).validate_request()


def test_unknown_thinking_type_still_rejected():
    import pytest

    with pytest.raises(ValueError, match="adaptive"):
        _req({"type": "sometimes"}).validate_request()


def test_thinking_flag():
    assert _thinking_flag({"type": "enabled", "budget_tokens": 5}) is True
    assert _thinking_flag({"type": "disabled"}) is False
    assert _thinking_flag({"type": "adaptive"}) is None  # model default
    assert _thinking_flag(None) is None


def test_qwen_merges_mid_conversation_developer_messages():
    # Codex sends `developer` items after the first user turn even when a system message
    # leads; the Qwen template raises "System message must be at the beginning".
    msgs = [
        {"role": "system", "content": "base"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "permissions"},  # a remapped developer item
        {"role": "assistant", "content": "ok"},
        {"role": "system", "content": "env"},
    ]
    out = QwenMessageAdapter().adapt(msgs)
    assert [m["role"] for m in out] == ["system", "user", "assistant"]
    assert out[0]["content"] == "base\n\npermissions\n\nenv"


def test_qwen_single_leading_system_unchanged():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert QwenMessageAdapter().adapt(msgs) == msgs
