"""previous_response_id: unknown id is a 404, long chains are not silently truncated."""

from __future__ import annotations

import pytest

from yunshu_gateway.responses_context import ContextError
from yunshu_gateway.routers import responses as resp_mod


def _req(prev):
    return resp_mod.ResponsesRequest(model="m", input="hi", previous_response_id=prev)


def _store(monkeypatch, store):
    monkeypatch.setattr(resp_mod, "_get_stored_response", lambda i: store.get(i))
    monkeypatch.setattr(resp_mod, "_owns_stored", lambda request, p: True)


def test_unknown_previous_response_id_is_404(monkeypatch):
    _store(monkeypatch, {})
    with pytest.raises(ContextError) as ei:
        resp_mod._apply_previous_response_chain(_req("resp_nope"), [], None)
    assert ei.value.status == 404
    assert ei.value.code == "previous_response_not_found"


def test_long_chain_replays_all_hops(monkeypatch):
    n = 40
    store = {
        f"r{i}": {
            "previous_response_id": f"r{i - 1}" if i else None,
            "_input_messages": [{"role": "user", "content": f"u{i}"}],
            "output": [],
        }
        for i in range(n)
    }
    _store(monkeypatch, store)
    out = resp_mod._apply_previous_response_chain(_req(f"r{n - 1}"), [], None)
    assert [m["content"] for m in out] == [f"u{i}" for i in range(n)]


def test_chain_over_cap_errors_instead_of_truncating(monkeypatch):
    monkeypatch.setattr(resp_mod, "_CHAIN_MAX_HOPS", 3)
    store = {
        f"r{i}": {
            "previous_response_id": f"r{i - 1}" if i else None,
            "_input_messages": [],
            "output": [],
        }
        for i in range(6)
    }
    _store(monkeypatch, store)
    with pytest.raises(ContextError) as ei:
        resp_mod._apply_previous_response_chain(_req("r5"), [], None)
    assert ei.value.status == 400


def test_missing_middle_parent_is_404_naming_the_missing_id(monkeypatch):
    store = {
        "r2": {
            "previous_response_id": "r1",
            "_input_messages": [{"role": "user", "content": "u2"}],
            "output": [],
        },
        # r1 was evicted from the LRU / deleted mid-chain
    }
    _store(monkeypatch, store)
    with pytest.raises(ContextError) as ei:
        resp_mod._apply_previous_response_chain(_req("r2"), [], None)
    assert ei.value.status == 404
    assert ei.value.code == "previous_response_not_found"
    assert "'r1'" in str(ei.value.message)
    # actionable: says it was an ancestor, the store limit, and how to continue
    assert "chain of 'r2'" in ei.value.message
    assert "1024" in ei.value.message and "without `previous_response_id`" in ei.value.message


def test_cancelled_and_failed_parents_replay_input_but_no_assistant_turn(monkeypatch):
    def parent(status, pid=None):
        return {
            "status": status,
            "previous_response_id": pid,
            "_input_messages": [{"role": "user", "content": f"u-{status}"}],
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": f"a-{status}"}],
                }
            ],
        }

    store = {
        "c": parent("cancelled"),
        "i": parent("incomplete", "c"),
        "f": parent("failed", "i"),
    }
    _store(monkeypatch, store)
    out = resp_mod._apply_previous_response_chain(_req("f"), [], None)
    assert [m["content"] for m in out] == [
        "u-cancelled",
        "u-incomplete",
        "a-incomplete",
        "u-failed",
    ]


def test_client_supplied_history_is_only_normalised_to_the_shape_the_model_produced():
    """_coalesce_assistant_turns also runs on client `input`: plain valid histories pass
    unchanged, only the model's own parallel-tool-call output shape is merged."""
    from yunshu_gateway.routers.responses import _coalesce_assistant_turns as co

    plain = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "again"},
        {"role": "assistant", "content": "sure"},
    ]
    assert co(list(plain)) == plain
    # a tool call with a user turn in between is never merged into the earlier answer
    sep = [
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "now call"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
    ]
    assert co(list(sep)) == sep
    # text + the tool call it announced in the same generation: one assistant turn
    joined = co(
        [
            {"role": "assistant", "content": "checking"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        ]
    )
    assert joined == [
        {"role": "assistant", "content": "checking", "tool_calls": [{"id": "c1"}]}
    ]
    # a lone empty assistant message with reasoning, or the last message, is kept
    keep = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "reasoning_content": "thought"},
        {"role": "user", "content": "q2"},
    ]
    assert co(list(keep)) == keep
    tail = [{"role": "user", "content": "q"}, {"role": "assistant", "content": ""}]
    assert co(list(tail)) == tail
