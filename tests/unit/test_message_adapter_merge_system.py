"""Qwen's template accepts one system message, first. Codex sends ``instructions`` plus ``developer``
messages (both become ``system``): they must be merged before the template renders."""

from __future__ import annotations

from yunshu_engine.message_adapter import QwenMessageAdapter


def roles(msgs):
    return [m["role"] for m in msgs]


def test_consecutive_leading_system_messages_are_merged():
    out = QwenMessageAdapter().adapt(
        [
            {"role": "system", "content": "instructions"},
            {"role": "system", "content": "developer: permissions"},
            {"role": "user", "content": "hi"},
        ]
    )
    assert roles(out) == ["system", "user"]
    assert out[0]["content"] == "instructions\n\ndeveloper: permissions"


def test_hoisted_system_messages_are_merged_too():
    out = QwenMessageAdapter().adapt(
        [
            {"role": "system", "content": "a"},
            {"role": "user", "content": "u1"},
            {"role": "system", "content": "b"},
            {"role": "user", "content": "u2"},
        ]
    )
    # the first is already leading (no hoist); a later one stays where it is
    assert roles(out)[0] == "system"
    out = QwenMessageAdapter().adapt(
        [
            {"role": "user", "content": "u"},
            {"role": "system", "content": "a"},
            {"role": "system", "content": "b"},
        ]
    )
    assert roles(out) == ["system", "user"] and out[0]["content"] == "a\n\nb"


def test_list_content_and_single_system_untouched():
    out = QwenMessageAdapter().adapt(
        [
            {"role": "system", "content": [{"type": "text", "text": "x"}]},
            {"role": "system", "content": "y"},
            {"role": "user", "content": "u"},
        ]
    )
    assert out[0]["content"] == "x\n\ny"
    one = QwenMessageAdapter().adapt(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    )
    assert one[0]["content"] == "s"
