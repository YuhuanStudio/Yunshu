"""Reasoning items round-trip for stateless Responses clients (Codex: store=false, include encrypted_content)."""

from __future__ import annotations

from yunshu_gateway.reasoning_token import (
    reasoning_text_of,
    seal_reasoning,
    unseal_reasoning,
)
from yunshu_gateway.routers import responses


def test_seal_round_trip_and_rejects_foreign_tokens():
    tok = seal_reasoning("step 1: read the file\nstep 2: 日本語")
    assert (
        tok.startswith("yunshu1:")
        and unseal_reasoning(tok) == "step 1: read the file\nstep 2: 日本語"
    )
    assert unseal_reasoning("gAAAA-openai-ciphertext") is None
    assert unseal_reasoning(None) is None
    assert unseal_reasoning("yunshu1:!!!not base64!!!") is None


def test_reasoning_text_of_prefers_sealed_then_summary_then_content():
    assert (
        reasoning_text_of(
            {
                "encrypted_content": seal_reasoning("sealed"),
                "summary": [{"text": "sum"}],
            }
        )
        == "sealed"
    )
    assert (
        reasoning_text_of(
            {
                "summary": [
                    {"type": "summary_text", "text": "a"},
                    {"type": "summary_text", "text": "b"},
                ]
            }
        )
        == "a\nb"
    )
    assert (
        reasoning_text_of({"content": [{"type": "reasoning_text", "text": "raw"}]})
        == "raw"
    )
    assert reasoning_text_of({"encrypted_content": "opaque-from-openai"}) == ""


def _req(**kw):
    return responses.ResponsesRequest(model="m", **kw)


def test_reasoning_items_ride_on_the_next_assistant_turn():
    req = _req(
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "list files"}],
            },
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [],
                "encrypted_content": seal_reasoning("I should run ls"),
            },
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "exec_command",
                "arguments": '{"cmd":"ls"}',
            },
            {"type": "function_call_output", "call_id": "c1", "output": "a.txt"},
        ]
    )
    msgs = responses._convert_to_messages(req)
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool"]
    assert msgs[1]["reasoning_content"] == "I should run ls"
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "exec_command"
    assert "reasoning_content" not in msgs[0]


def test_reasoning_before_an_assistant_message():
    req = _req(
        input=[
            {"type": "message", "role": "user", "content": "q"},
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "thought"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "a"}],
            },
            {"type": "message", "role": "user", "content": "next"},
        ]
    )
    msgs = responses._convert_to_messages(req)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["reasoning_content"] == "thought" and msgs[1]["content"] == "a"


def test_seal_extra_only_when_included():
    assert responses._seal_extra(_req(input="x"), "text") == {}
    r = _req(input="x", include=["reasoning.encrypted_content"])
    assert (
        unseal_reasoning(responses._seal_extra(r, "text")["encrypted_content"])
        == "text"
    )
    assert responses._seal_extra(r, "") == {}
