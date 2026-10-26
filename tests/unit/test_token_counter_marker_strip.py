"""Claude Code's ``<total_tokens>N tokens left</total_tokens>`` budget marker never reaches the prompt, whether it
arrives as a lone system message, as the tail of the system text or as an extra system block (omlx 38f420d9):
its number changes every request, so any prompt that carries it diverges there and nothing after it is reused."""

import pytest

from yunshu_engine.message_adapter import adapt_messages, strip_token_counter


def _marker(n):
    return f"<total_tokens>{n} tokens left</total_tokens>"


def _render(system, n):
    msgs = [
        {"role": "system", "content": system(n)},
        {"role": "user", "content": "run the tests"},
        {"role": "assistant", "content": "ok"},
        {"role": "system", "content": _marker(n)},
        {"role": "user", "content": "and now?"},
    ]
    return adapt_messages(msgs, "Qwen3.5-27B")


@pytest.mark.parametrize(
    "system",
    [
        lambda n: f"You are a coding agent.\n\n{_marker(n)}",
        lambda n: [
            {"type": "text", "text": "You are a coding agent."},
            {"type": "text", "text": _marker(n)},
        ],
        lambda n: [{"type": "text", "text": f"You are a coding agent.\n{_marker(n)}"}],
        lambda n: "You are a coding agent.",
    ],
    ids=["suffix", "extra-block", "suffix-in-block", "none"],
)
def test_prompt_is_identical_whatever_the_counter_says(system):
    first, later = _render(system, 190_000), _render(system, 187_321)
    assert first == later
    assert "total_tokens" not in repr(first)
    assert [m["role"] for m in first] == ["system", "user", "assistant", "user"]


def test_message_without_a_marker_is_the_same_object():
    msg = {"role": "system", "content": "plain"}
    assert strip_token_counter(msg) is msg
    blocks = {"role": "system", "content": [{"type": "text", "text": "plain"}]}
    assert strip_token_counter(blocks) is blocks


def test_marker_only_message_is_dropped_and_other_text_is_kept():
    assert strip_token_counter({"role": "system", "content": _marker(5)}) is None
    kept = strip_token_counter({"role": "system", "content": f"rules\n\n{_marker(5)}"})
    assert kept == {"role": "system", "content": "rules"}


def test_user_text_that_mentions_the_marker_is_untouched():
    msgs = [{"role": "user", "content": f"what is {_marker(5)} for?"}]
    assert adapt_messages(msgs, None) == msgs
