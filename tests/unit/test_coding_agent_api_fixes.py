"""API gaps found by driving real coding agents (Claude Code and Codex CLI) at the server."""

from types import SimpleNamespace

import pytest

from yunshu_engine.message_adapter import QwenMessageAdapter
from yunshu_gateway.routers.anthropic import (
    AnthropicMessage,
    AnthropicMessagesRequest,
    _thinking_switches,
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
    with pytest.raises(ValueError, match="adaptive"):
        _req({"type": "sometimes"}).validate_request()


def test_thinking_flag():
    def sw(thinking):
        return _thinking_switches(SimpleNamespace(thinking=thinking))

    assert sw({"type": "enabled", "budget_tokens": 5}) == (True, 5)
    assert sw({"type": "disabled"}) == (False, None)
    assert sw({"type": "adaptive"}) == (None, None)  # model default
    assert sw(None) == (None, None)


def test_qwen_mid_conversation_system_messages_stay_in_place():
    # Codex sends `developer` items and Claude Code sends per-turn notes after the first
    # user turn; the Qwen template rejects them as system messages. They must not be
    # hoisted (that rewrites the prompt start every turn and kills prefix caching).
    msgs = [
        {"role": "system", "content": "base"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "permissions"},
        {"role": "assistant", "content": "ok"},
        {"role": "system", "content": "env"},
    ]
    out = QwenMessageAdapter().adapt(msgs)
    assert [m["role"] for m in out] == ["system", "user", "user", "assistant", "user"]
    assert [m["content"] for m in out] == ["base", "hi", "permissions", "ok", "env"]


def test_qwen_system_after_user_first_message_not_hoisted():
    msgs = [
        {"role": "user", "content": "u"},
        {"role": "system", "content": "note"},
    ]
    out = QwenMessageAdapter().adapt(msgs)
    assert [m["role"] for m in out] == ["user", "user"]


def test_qwen_leading_system_messages_merge():
    msgs = [
        {"role": "system", "content": "a"},
        {"role": "system", "content": "b"},
        {"role": "user", "content": "u"},
    ]
    out = QwenMessageAdapter().adapt(msgs)
    assert out[0] == {"role": "system", "content": "a\n\nb"} and len(out) == 2
