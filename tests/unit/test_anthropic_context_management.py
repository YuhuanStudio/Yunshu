"""Messages `context_management`: clear_tool_uses / clear_thinking edits are applied and reported."""

import json

import pytest

from yunshu_gateway.anthropic_context_edit import (
    CLEARED_PLACEHOLDER,
    ContextEditError,
    apply_edits,
)

from .test_anthropic_version_2026 import server  # noqa: F401


def _turn(i, think=False):
    a = [{"type": "tool_use", "id": f"t{i}", "name": "read", "input": {"p": "x" * 200}}]
    if think:
        a.insert(0, {"type": "thinking", "thinking": "hmm " * 50, "signature": "s"})
    return [
        {"role": "assistant", "content": a},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": f"t{i}", "content": "R" * 400}
            ],
        },
    ]


def _convo(n, think=False):
    msgs = [{"role": "user", "content": "go"}]
    for i in range(n):
        msgs += _turn(i, think)
    return msgs


TOOL = {"type": "clear_tool_uses_20250919"}


def test_clear_tool_uses_keeps_recent_and_reports():
    edit = dict(TOOL, trigger={"type": "tool_uses", "value": 2}, keep={"value": 1})
    out, applied = apply_edits(_convo(4), {"edits": [edit]})
    results = [
        b["content"]
        for m in out
        if isinstance(m["content"], list)
        for b in m["content"]
        if b["type"] == "tool_result"
    ]
    assert results[:3] == [CLEARED_PLACEHOLDER] * 3 and results[3] == "R" * 400
    assert applied[0]["type"] == "clear_tool_uses_20250919"
    assert applied[0]["cleared_tool_uses"] == 3
    assert applied[0]["cleared_input_tokens"] > 0


def test_below_trigger_clear_at_least_and_exclude():
    msgs = _convo(4)
    assert apply_edits(msgs, {"edits": [TOOL]})[1] == []  # default trigger 100000
    big = dict(TOOL, trigger={"type": "tool_uses", "value": 1}, keep={"value": 0})
    least = dict(big, clear_at_least={"type": "input_tokens", "value": 10**6})
    assert apply_edits(msgs, {"edits": [least]})[1] == []
    out, applied = apply_edits(msgs, {"edits": [dict(big, exclude_tools=["read"])]})
    assert applied == [] and out == msgs
    out, applied = apply_edits(msgs, {"edits": [dict(big, clear_tool_inputs=True)]})
    assert out[1]["content"][0]["input"] == {}


def test_clear_thinking_keeps_last_turns():
    think = {"type": "clear_thinking_20251015"}
    out, applied = apply_edits(_convo(3, think=True), {"edits": [think]})
    kinds = [b["type"] for m in out if m["role"] == "assistant" for b in m["content"]]
    assert kinds.count("thinking") == 1  # default keep: 1 turn
    assert applied[0]["cleared_thinking_turns"] == 2
    _, applied = apply_edits(
        _convo(3, think=True), {"edits": [dict(think, keep="all")]}
    )
    assert applied == []


def test_invalid_edits_rejected():
    with pytest.raises(ContextEditError):
        apply_edits([], {"edits": [{"type": "nope"}]})
    with pytest.raises(ContextEditError):  # clear_thinking must come first
        apply_edits([], {"edits": [TOOL, {"type": "clear_thinking_20251015"}]})


def _body(**extra):
    edit = dict(TOOL, trigger={"type": "tool_uses", "value": 2}, keep={"value": 1})
    return {
        "model": "m",
        "max_tokens": 8,
        "messages": _convo(4),
        "context_management": {"edits": [edit]},
        **extra,
    }


def test_endpoint_applies_edits_and_reports(server):  # noqa: F811
    client, captured = server
    r = client.post("/v1/messages", json=_body())
    assert r.status_code == 200, r.text
    applied = r.json()["context_management"]["applied_edits"]
    assert applied[0]["cleared_tool_uses"] == 3
    assert CLEARED_PLACEHOLDER in json.dumps(captured["messages"])


def test_endpoint_stream_reports_in_message_delta(server):  # noqa: F811
    from yunshu_engine.batched_engine import GenerationOutput
    from yunshu_gateway.engine import get_engine

    async def _stream(*args, **kwargs):
        yield GenerationOutput(
            text="ok",
            new_text="ok",
            prompt_tokens=1,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
        )

    get_engine().stream_chat = _stream
    client, _ = server
    r = client.post("/v1/messages", json=_body(stream=True))
    assert r.status_code == 200, r.text
    deltas = [
        json.loads(ln[5:])
        for ln in r.text.splitlines()
        if ln.startswith("data:") and '"message_delta"' in ln
    ]
    edits = deltas[0]["context_management"]["applied_edits"]
    assert edits[0]["cleared_tool_uses"] == 3


def test_endpoint_bad_edit_is_400(server):  # noqa: F811
    client, _ = server
    body = _body()
    body["context_management"] = {"edits": [{"type": "bogus"}]}
    r = client.post("/v1/messages", json=body)
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"


# ── compact_20260112: server-side compaction ─────────────────────────────


def _big(**extra):
    big = "word " * 60000  # about 75K tokens by the estimator
    msgs = [
        {"role": "user", "content": "OLDTURN " + big},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "LASTTURN"},
    ]
    edit = {"type": "compact_20260112", **extra}
    return {
        "model": "m",
        "max_tokens": 8,
        "messages": msgs,
        "context_management": {"edits": [edit]},
    }


@pytest.fixture
def summarizer(monkeypatch):
    from yunshu_gateway import responses_context

    calls = []

    async def fake(request, model, entries, instructions=None, guidance=None, *a, **k):
        calls.append({"n": len(entries), "guidance": guidance})
        return "SUMMARY-TEXT", {"input_tokens": 70000, "output_tokens": 12}

    monkeypatch.setattr(responses_context, "summarize_history", fake)
    return calls


def test_compaction_summarizes_and_prepends_block(server, summarizer):  # noqa: F811
    client, captured = server
    r = client.post(
        "/v1/messages",
        json=_big(
            trigger={"type": "input_tokens", "value": 50000}, instructions="be brief"
        ),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["content"][0] == {"type": "compaction", "content": "SUMMARY-TEXT"}
    assert body["content"][1]["type"] == "text"
    its = body["usage"]["iterations"]
    assert [i["type"] for i in its] == ["compaction", "message"]
    assert its[0]["input_tokens"] == 70000 and its[0]["output_tokens"] == 12
    sent = json.dumps(captured["messages"])
    assert "SUMMARY-TEXT" in sent and "LASTTURN" in sent and "OLDTURN" not in sent
    assert summarizer[0]["guidance"] == "be brief"


def test_compaction_not_triggered_below_threshold(server, summarizer):  # noqa: F811
    client, _ = server
    r = client.post(
        "/v1/messages",
        json=_big(trigger={"type": "input_tokens", "value": 500000}),
    )
    assert r.status_code == 200
    assert all(b["type"] != "compaction" for b in r.json()["content"])
    assert not summarizer


def test_compaction_pause_returns_only_the_block(server, summarizer):  # noqa: F811
    client, captured = server
    r = client.post(
        "/v1/messages",
        json=_big(
            trigger={"type": "input_tokens", "value": 50000},
            pause_after_compaction=True,
        ),
    )
    body = r.json()
    assert body["stop_reason"] == "compaction"
    assert [b["type"] for b in body["content"]] == ["compaction"]
    assert "messages" not in captured  # nothing was generated


def test_compaction_stream_shifts_block_indexes(server, summarizer):  # noqa: F811
    from yunshu_engine.batched_engine import GenerationOutput
    from yunshu_gateway.engine import get_engine

    async def _stream(*args, **kwargs):
        yield GenerationOutput(
            text="ok",
            new_text="ok",
            prompt_tokens=1,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
        )

    get_engine().stream_chat = _stream
    client, _ = server
    r = client.post(
        "/v1/messages",
        json={**_big(trigger={"type": "input_tokens", "value": 50000}), "stream": True},
    )
    ev = [json.loads(ln[5:]) for ln in r.text.splitlines() if ln.startswith("data:")]
    starts = [e for e in ev if e["type"] == "content_block_start"]
    assert (
        starts[0]["index"] == 0 and starts[0]["content_block"]["type"] == "compaction"
    )
    assert starts[1]["index"] == 1 and starts[1]["content_block"]["type"] == "text"
    delta = next(
        e for e in ev if e["type"] == "content_block_delta" and e["index"] == 0
    )
    assert delta["delta"] == {"type": "compaction_delta", "content": "SUMMARY-TEXT"}
    last = next(e for e in ev if e["type"] == "message_delta")
    assert [i["type"] for i in last["usage"]["iterations"]] == ["compaction", "message"]


def test_round_tripped_compaction_block_replaces_earlier_turns(server):  # noqa: F811
    client, captured = server
    body = {
        "model": "m",
        "max_tokens": 8,
        "messages": [
            {"role": "user", "content": "ANCIENT"},
            {
                "role": "assistant",
                "content": [
                    {"type": "compaction", "content": "EARLIER-SUMMARY"},
                    {"type": "text", "text": "reply"},
                ],
            },
            {"role": "user", "content": "NEXT"},
        ],
    }
    r = client.post("/v1/messages", json=body)
    assert r.status_code == 200, r.text
    sent = json.dumps(captured["messages"])
    assert "EARLIER-SUMMARY" in sent and "NEXT" in sent and "ANCIENT" not in sent
    body["messages"][1]["content"][0]["content"] = (
        None  # a failed compaction is a no-op
    )
    r = client.post("/v1/messages", json=body)
    assert "ANCIENT" in json.dumps(captured["messages"])


def test_compaction_trigger_below_minimum_is_400(server):  # noqa: F811
    client, _ = server
    r = client.post(
        "/v1/messages", json=_big(trigger={"type": "input_tokens", "value": 1000})
    )
    assert r.status_code == 400
