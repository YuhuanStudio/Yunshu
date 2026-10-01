"""The context-window manager says what it removed: policy, turns, tokens before / after."""

from __future__ import annotations

import pytest

from yunshu_engine.context_window import ContextWindowManager


def _counter(text: str) -> int:
    return len(text.split())


def _convo(turns: int, words: int = 20) -> list[dict]:
    msgs = [{"role": "system", "content": "be brief"}]
    for i in range(turns):
        msgs.append(
            {"role": "user", "content": " ".join(f"q{i}w{j}" for j in range(words))}
        )
        msgs.append(
            {
                "role": "assistant",
                "content": " ".join(f"a{i}w{j}" for j in range(words)),
            }
        )
    msgs.append({"role": "user", "content": "last question here"})
    return msgs


def test_report_counts_turns_tokens_and_policy():
    mgr = ContextWindowManager(token_counter=_counter)
    msgs = _convo(6)
    before = mgr.count_messages_tokens(msgs)
    res = mgr.compute_truncation(msgs, before // 2, "truncate_oldest")
    rep = res.report()
    assert rep["policy"] == "truncate_oldest"
    assert rep["tokens_before"] == before
    assert rep["tokens_after"] == mgr.count_messages_tokens(res.messages) <= before // 2
    assert rep["tokens_removed"] == before - rep["tokens_after"] > 0
    assert rep["budget_tokens"] == before // 2
    assert rep["messages_before"] == len(msgs)
    assert rep["messages_after"] == len(res.messages)
    assert rep["messages_removed"] == len(msgs) - len(res.messages) > 0
    # which roles lost messages: only user / assistant turns, never the system prompt
    removed = rep["removed_roles"]
    assert "system" not in removed
    assert removed["user"] + removed["assistant"] == rep["messages_removed"]
    # the latest user question is never silently dropped
    assert res.messages[-1] == msgs[-1]
    assert rep["cannot_fit"] is False


@pytest.mark.parametrize(
    "policy",
    ["truncate_oldest", "sliding_window", "importance_aware", "summary_compression"],
)
def test_every_policy_reports_itself_and_keeps_the_last_question(policy):
    mgr = ContextWindowManager(token_counter=_counter)
    msgs = _convo(8)
    res = mgr.compute_truncation(msgs, 60, policy)
    rep = res.report()
    assert rep["policy"] == policy
    assert rep["tokens_after"] < rep["tokens_before"]
    assert rep["messages_before"] == len(msgs)
    assert res.messages[-1] == msgs[-1]


def test_nothing_removed_publishes_nothing():
    mgr = ContextWindowManager(token_counter=_counter)
    res = mgr.compute_truncation(_convo(1), 10_000, "truncate_oldest")
    assert not res.truncated
    assert res.report()["messages_removed"] == 0

    class Box:
        context_policy = None

    from yunshu_engine.request_tracker import current_request_info

    box = Box()
    tok = current_request_info.set(box)
    try:
        res.publish()
        assert box.context_policy is None  # no truncation: no report
        big = mgr.compute_truncation(_convo(6), 50, "truncate_oldest")
        big.publish()
        assert box.context_policy == big.report()
    finally:
        current_request_info.reset(tok)


def test_cannot_fit_is_reported_not_hidden():
    mgr = ContextWindowManager(token_counter=_counter)
    msgs = [
        {"role": "system", "content": "s " * 50},
        {"role": "user", "content": "q " * 50},
    ]
    res = mgr.compute_truncation(msgs, 10, "truncate_oldest")
    assert res.cannot_fit
    assert res.report()["cannot_fit"] is True
    assert res.messages_removed == 0  # nothing was dropped to pretend it fits


def test_publish_outside_a_request_is_harmless():
    mgr = ContextWindowManager(token_counter=_counter)
    mgr.compute_truncation(_convo(6), 50, "truncate_oldest").publish()
