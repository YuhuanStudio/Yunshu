"""Responses streaming event sequence, checked like vLLM tests/entrypoints/openai/responses/
test_streaming_events.py (Apache-2.0) and against the official SDK's ResponseStreamEvent models."""

from __future__ import annotations

import json

import pytest
from openai.types.responses import ResponseStreamEvent
from pydantic import TypeAdapter

from .wire_harness import Script, install

ADAPTER = TypeAdapter(ResponseStreamEvent)


def _events(c, **body):
    r = c.post(
        "/v1/responses",
        json={"model": "m", "input": "hi", "stream": True, **body},
    )
    assert r.status_code == 200, r.text
    evs, name = [], None
    for line in r.text.splitlines():
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:") and line[5:].strip() != "[DONE]":
            d = json.loads(line[5:])
            assert d["type"] == name, (name, d["type"])  # event: line == data.type
            evs.append(d)
    return evs


def test_every_event_parses_with_the_sdk_models(monkeypatch):
    c, _ = install(monkeypatch)
    for e in _events(c):
        ADAPTER.validate_python(e)


def test_sequence_numbers_increase_and_lifecycle_order(monkeypatch):
    c, _ = install(monkeypatch)
    evs = _events(c)
    seq = [e["sequence_number"] for e in evs]
    assert seq == sorted(set(seq)) and len(seq) == len(set(seq)), seq
    types = [e["type"] for e in evs]
    assert types[0] == "response.created"
    assert types[-1] == "response.completed"
    order = [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    pos = [types.index(t) for t in order]
    assert pos == sorted(pos), types
    assert types.count("response.completed") == 1


def test_done_text_equals_deltas_and_final_output(monkeypatch):
    c, _ = install(monkeypatch)
    evs = _events(c)
    deltas = "".join(
        e["delta"] for e in evs if e["type"] == "response.output_text.delta"
    )
    done = next(e for e in evs if e["type"] == "response.output_text.done")
    final = evs[-1]["response"]
    assert deltas == done["text"] == "Hello there!"
    msg = next(o for o in final["output"] if o["type"] == "message")
    assert msg["content"][0]["text"] == deltas
    item_id = next(e for e in evs if e["type"] == "response.output_item.added")["item"][
        "id"
    ]
    assert done["item_id"] == item_id == msg["id"]
    assert final["status"] == "completed" and final["usage"]["output_tokens"] > 0


def test_reasoning_item_precedes_message_with_consistent_indices(monkeypatch):
    c, _ = install(
        monkeypatch,
        Script(pieces=[("think", "reasoning"), ("ing", "reasoning"), "Answer"]),
    )
    evs = _events(c, reasoning={"effort": "low"})
    for e in evs:
        ADAPTER.validate_python(e)
    added = [e for e in evs if e["type"] == "response.output_item.added"]
    kinds = [a["item"]["type"] for a in added]
    if "reasoning" in kinds:
        assert kinds.index("reasoning") < kinds.index("message")
    assert [a["output_index"] for a in added] == list(range(len(added)))


@pytest.mark.parametrize("finish", ["length"])
def test_length_stop_is_incomplete(monkeypatch, finish):
    c, _ = install(monkeypatch, Script(finish_reason=finish))
    evs = _events(c, max_output_tokens=16)
    last = evs[-1]
    assert last["type"] in ("response.incomplete", "response.completed")
    if last["type"] == "response.incomplete":
        assert last["response"]["incomplete_details"]["reason"] == "max_output_tokens"


def _sse(c, path, body):
    r = c.post(path, json=body)
    assert r.status_code == 200, r.text
    return [
        json.loads(line[5:])
        for line in r.text.splitlines()
        if line.startswith("data:") and line[5:].strip() != "[DONE]"
    ]


def test_chat_chunks_validate_with_the_sdk_models(monkeypatch):
    from openai.types.chat import ChatCompletionChunk

    c, _ = install(monkeypatch)
    chunks = _sse(
        c,
        "/v1/chat/completions",
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    parsed = [ChatCompletionChunk.model_validate(x) for x in chunks]
    assert parsed[0].choices[0].delta.role == "assistant"
    assert parsed[-1].choices == [] and parsed[-1].usage.total_tokens > 0
    assert sum(1 for p in parsed if p.choices and p.choices[0].finish_reason) == 1


def test_anthropic_events_validate_with_the_sdk_models(monkeypatch):
    from anthropic.types import RawMessageStreamEvent

    c, _ = install(monkeypatch)
    evs = _sse(
        c,
        "/v1/messages",
        {
            "model": "m",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    ad = TypeAdapter(RawMessageStreamEvent)
    for e in evs:
        ad.validate_python(e)
    types = [e["type"] for e in evs]
    assert types[0] == "message_start" and types[-1] == "message_stop"
    assert types.index("content_block_start") < types.index("content_block_delta")
    assert types.index("content_block_stop") < types.index("message_delta")
