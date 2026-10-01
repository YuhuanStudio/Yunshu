"""F04: exact stream event ordering and field shapes against the official API references.

Raw HTTP, so the assertions see what any client would: OpenAI chat / completions chunk
streams, Anthropic Messages events, Responses events, Ollama NDJSON.
"""

from __future__ import annotations

import json

import pytest

from .wire_clients import chunks, tool_text
from .wire_harness import Script, install

THINK = [
    ("<think>", "reasoning"),
    ("plan", "reasoning"),
    ("</think>", "reasoning"),
]
USER = [{"role": "user", "content": "hi"}]


def sse(c, path, body):
    r = c.post(path, json={**body, "stream": True})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/event-stream")
    frames, event, data = [], None, []
    for line in r.text.split("\n"):
        if line.startswith(":"):
            continue
        if line == "":
            if data:
                frames.append((event, "\n".join(data)))
            event, data = None, []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    return frames


def parse(frames):
    return [(e, d if d == "[DONE]" else json.loads(d)) for e, d in frames]


# ── OpenAI chat completions ──────────────────────────────────────────────────────
def test_chat_stream_wire_shape(monkeypatch):
    c, _ = install(monkeypatch, Script(pieces=["Hel", "lo", "!"], prompt_tokens=7))
    ev = parse(
        sse(
            c,
            "/v1/chat/completions",
            {"model": "m", "messages": USER, "stream_options": {"include_usage": True}},
        )
    )
    assert ev[-1][1] == "[DONE]"
    chunks_ = [d for _, d in ev[:-1]]
    ids = {d["id"] for d in chunks_}
    assert len(ids) == 1 and next(iter(ids)).startswith("chatcmpl-")
    for d in chunks_:
        assert d["object"] == "chat.completion.chunk"
        assert isinstance(d["created"], int) and d["model"] == "m"
    first = chunks_[0]["choices"][0]
    assert first["delta"]["role"] == "assistant" and first["index"] == 0
    assert all("role" not in d["choices"][0]["delta"] for d in chunks_[1:-2])
    content = [d["choices"][0]["delta"].get("content") for d in chunks_ if d["choices"]]
    assert "".join(x for x in content if x) == "Hello!"
    # exactly one finish chunk, then the usage-only chunk (choices == [])
    fin = [d for d in chunks_ if d["choices"] and d["choices"][0]["finish_reason"]]
    assert len(fin) == 1 and fin[0]["choices"][0]["finish_reason"] == "stop"
    assert chunks_.index(fin[0]) == len(chunks_) - 2
    usage = chunks_[-1]
    assert usage["choices"] == []
    assert usage["usage"]["prompt_tokens"] == 7
    assert usage["usage"]["completion_tokens"] == 3
    assert usage["usage"]["total_tokens"] == 10
    assert all(not d.get("usage") for d in chunks_[:-1])


def test_chat_stream_without_include_usage_has_no_usage_chunk(monkeypatch):
    c, _ = install(monkeypatch, Script(pieces=["a", "b"]))
    ev = parse(sse(c, "/v1/chat/completions", {"model": "m", "messages": USER}))
    assert all(not d.get("usage") for _, d in ev[:-1])
    assert all(d["choices"] for _, d in ev[:-1])


def test_chat_nonstream_wire_shape(monkeypatch):
    c, _ = install(monkeypatch, Script(pieces=["Hi", "!"], prompt_tokens=4))
    r = c.post("/v1/chat/completions", json={"model": "m", "messages": USER}).json()
    assert r["object"] == "chat.completion" and r["id"].startswith("chatcmpl-")
    ch = r["choices"][0]
    assert ch["index"] == 0 and ch["finish_reason"] == "stop"
    assert ch["message"]["role"] == "assistant" and ch["message"]["content"] == "Hi!"
    u = r["usage"]
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"] == 6


def test_chat_stream_reasoning_precedes_content(monkeypatch):
    c, _ = install(monkeypatch, Script(pieces=[*THINK, "Ans"]))
    ev = parse(sse(c, "/v1/chat/completions", {"model": "m", "messages": USER}))
    seq = []
    for _, d in ev[:-1]:
        for ch in d["choices"]:
            dl = ch["delta"]
            if dl.get("reasoning_content"):
                seq.append("r")
            if dl.get("content"):
                seq.append("c")
    assert "".join(seq).strip("rc") == "" and seq.index("c") > max(
        i for i, x in enumerate(seq) if x == "r"
    )


def test_chat_stream_length_finish_reason(monkeypatch):
    c, _ = install(monkeypatch, Script(pieces=list("abcdef")))
    ev = parse(
        sse(
            c, "/v1/chat/completions", {"model": "m", "messages": USER, "max_tokens": 2}
        )
    )
    fins = [
        d["choices"][0]["finish_reason"]
        for _, d in ev[:-1]
        if d["choices"] and d["choices"][0]["finish_reason"]
    ]
    assert fins == ["length"]


def test_chat_stream_tool_call_chunks(monkeypatch):
    text = tool_text("get_weather", {"city": "Paris"})
    c, _ = install(monkeypatch, Script(pieces=chunks(text, 5)))
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
    ev = parse(
        sse(
            c, "/v1/chat/completions", {"model": "m", "messages": USER, "tools": [tool]}
        )
    )
    calls = [
        tc
        for _, d in ev[:-1]
        for ch in d["choices"]
        for tc in ch["delta"].get("tool_calls") or []
    ]
    assert calls[0]["index"] == 0 and calls[0]["type"] == "function"
    assert (
        calls[0]["id"].startswith("call_")
        and calls[0]["function"]["name"] == "get_weather"
    )
    # later deltas of the same call carry only arguments, with no new id
    assert all("id" not in tc for tc in calls[1:])
    assert json.loads("".join(tc["function"].get("arguments", "") for tc in calls)) == {
        "city": "Paris"
    }
    fins = [
        d["choices"][0]["finish_reason"]
        for _, d in ev[:-1]
        if d["choices"] and d["choices"][0]["finish_reason"]
    ]
    assert fins == ["tool_calls"]


# ── OpenAI completions ───────────────────────────────────────────────────────────
def test_completions_stream_wire_shape(monkeypatch):
    c, _ = install(monkeypatch, Script(pieces=["a", "b", "c"], prompt_tokens=3))
    ev = parse(
        sse(
            c,
            "/v1/completions",
            {"model": "m", "prompt": "hi", "stream_options": {"include_usage": True}},
        )
    )
    assert ev[-1][1] == "[DONE]"
    body = [d for _, d in ev[:-1]]
    assert all(d["object"] == "text_completion" for d in body)
    assert "".join(ch["text"] for d in body for ch in d["choices"]) == "abc"
    fin = [ch for d in body for ch in d["choices"] if ch["finish_reason"]]
    assert [f["finish_reason"] for f in fin] == ["stop"]
    assert body[-1]["choices"] == [] and body[-1]["usage"]["total_tokens"] == 6


# ── Anthropic Messages ───────────────────────────────────────────────────────────
def _anth(c, **extra):
    body = {"model": "m", "max_tokens": 64, "messages": USER, **extra}
    return parse(sse(c, "/v1/messages", body))


def test_messages_stream_event_order_and_shapes(monkeypatch):
    c, _ = install(
        monkeypatch, Script(pieces=["Hel", "lo"], prompt_tokens=7, cached_tokens=4)
    )
    ev = _anth(c)
    names = [e for e, _ in ev]
    assert names == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert all(e == d["type"] for e, d in ev)  # the `event:` line names the data type
    ms = ev[0][1]["message"]
    assert ms["id"].startswith("msg_") and ms["type"] == "message"
    assert ms["role"] == "assistant" and ms["content"] == [] and ms["model"] == "m"
    assert ms["stop_reason"] is None and ms["stop_sequence"] is None
    assert ms["usage"]["output_tokens"] == 0
    cbs = ev[1][1]
    assert cbs["index"] == 0 and cbs["content_block"] == {"type": "text", "text": ""}
    deltas = [d["delta"] for e, d in ev if e == "content_block_delta"]
    assert all(x["type"] == "text_delta" for x in deltas)
    assert "".join(x["text"] for x in deltas) == "Hello"
    md = ev[5][1]
    assert md["delta"] == {"stop_reason": "end_turn", "stop_sequence": None}
    # message_delta usage is cumulative and restates the prompt split of message_start
    assert md["usage"]["output_tokens"] == 2
    for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
        assert md["usage"][k] == ms["usage"][k], k
    assert ms["usage"]["input_tokens"] + ms["usage"]["cache_read_input_tokens"] == 7


def test_messages_stream_thinking_block_precedes_text(monkeypatch):
    c, _ = install(monkeypatch, Script(pieces=[*THINK, "Ans"]))
    ev = _anth(c)
    starts = [
        (d["index"], d["content_block"]["type"])
        for e, d in ev
        if e == "content_block_start"
    ]
    assert starts == [(0, "thinking"), (1, "text")]
    stops = [d["index"] for e, d in ev if e == "content_block_stop"]
    assert stops == [0, 1]
    # every block is closed before the next opens, and before message_delta
    order = [e for e, _ in ev]
    assert order.index("content_block_stop") < order.index("content_block_start", 2)
    assert max(
        i for i, e in enumerate(order) if e == "content_block_stop"
    ) < order.index("message_delta")


@pytest.mark.parametrize(
    "script,req,reason,seq",
    [
        (dict(pieces=list("abcdef")), dict(max_tokens=2), "max_tokens", None),
        (
            dict(pieces=["ab", "EN", "D", "x"]),
            dict(stop_sequences=["END"]),
            "stop_sequence",
            "END",
        ),
    ],
)
def test_messages_stop_reasons(monkeypatch, script, req, reason, seq):
    c, _ = install(monkeypatch, Script(**script))
    ev = _anth(c, **req)
    md = [d for e, d in ev if e == "message_delta"][0]
    assert md["delta"]["stop_reason"] == reason
    assert md["delta"]["stop_sequence"] == seq


def test_messages_stream_tool_use_blocks(monkeypatch):
    text = tool_text("get_weather", {"city": "Paris"})
    c, _ = install(monkeypatch, Script(pieces=chunks(text, 5)))
    tool = {
        "name": "get_weather",
        "description": "w",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
    ev = _anth(c, tools=[tool])
    starts = [d for e, d in ev if e == "content_block_start"]
    tu = [s for s in starts if s["content_block"]["type"] == "tool_use"]
    assert len(tu) == 1
    blk = tu[0]["content_block"]
    assert blk["id"].startswith("toolu_") and blk["name"] == "get_weather"
    assert blk["input"] == {}
    parts = [
        d["delta"]["partial_json"]
        for e, d in ev
        if e == "content_block_delta" and d["delta"]["type"] == "input_json_delta"
    ]
    assert json.loads("".join(parts)) == {"city": "Paris"}
    md = [d for e, d in ev if e == "message_delta"][0]
    assert md["delta"]["stop_reason"] == "tool_use"
    assert [e for e, _ in ev][-1] == "message_stop"


def test_messages_zero_output_still_has_one_content_block(monkeypatch):
    c, _ = install(monkeypatch, Script(pieces=[], prompt_tokens=5))
    ev = _anth(c)
    names = [e for e, _ in ev]
    assert names == [
        "message_start",
        "content_block_start",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert [d for e, d in ev if e == "message_delta"][0]["usage"]["output_tokens"] == 0


# ── OpenAI Responses ─────────────────────────────────────────────────────────────
def _resp(c, **extra):
    return parse(sse(c, "/v1/responses", {"model": "m", "input": "hi", **extra}))


def test_responses_stream_event_order_and_shapes(monkeypatch):
    c, _ = install(monkeypatch, Script(pieces=["Hel", "lo"], prompt_tokens=7))
    ev = [(e, d) for e, d in _resp(c) if d != "[DONE]"]
    names = [e for e, _ in ev]
    assert names == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert all(e == d["type"] for e, d in ev)
    seqs = [d["sequence_number"] for _, d in ev if "sequence_number" in d]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    created = ev[0][1]["response"]
    assert created["object"] == "response" and created["status"] in (
        "in_progress",
        "created",
    )
    rid = created["id"]
    done = ev[-1][1]["response"]
    assert done["id"] == rid and done["status"] == "completed"
    item = done["output"][0]
    assert item["type"] == "message" and item["role"] == "assistant"
    assert item["content"][0]["type"] == "output_text"
    assert item["content"][0]["text"] == "Hello"
    assert done["usage"]["input_tokens"] == 7 and done["usage"]["output_tokens"] == 2
    assert done["usage"]["total_tokens"] == 9
    assert (
        "output_tokens_details" in done["usage"]
        and "input_tokens_details" in done["usage"]
    )
    td = [d for e, d in ev if e == "response.output_text.done"][0]
    assert td["text"] == "Hello" and td["item_id"] == item["id"]


def test_responses_stream_incomplete_on_max_output_tokens(monkeypatch):
    c, _ = install(monkeypatch, Script(pieces=list("abcdef")))
    ev = [(e, d) for e, d in _resp(c, max_output_tokens=2) if d != "[DONE]"]
    assert ev[-1][0] == "response.incomplete"
    r = ev[-1][1]["response"]
    assert r["status"] == "incomplete"
    assert r["incomplete_details"] == {"reason": "max_output_tokens"}
    assert r["usage"]["output_tokens"] == 2


def test_responses_stream_function_call_items(monkeypatch):
    text = tool_text("get_weather", {"city": "Paris"})
    c, _ = install(monkeypatch, Script(pieces=chunks(text, 5)))
    tool = {
        "type": "function",
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
    ev = [(e, d) for e, d in _resp(c, tools=[tool]) if d != "[DONE]"]
    names = [e for e, _ in ev]
    # The call is delivered whole (added carries it, .done restates it) after the text item
    # closes; clients that accumulate argument deltas see none, so nothing is double-counted.
    for n in (
        "response.output_item.added",
        "response.function_call_arguments.done",
        "response.output_item.done",
    ):
        assert n in names, names
    assert "response.function_call_arguments.delta" not in names
    added = [
        d["item"]
        for e, d in ev
        if e == "response.output_item.added" and d["item"]["type"] == "function_call"
    ]
    assert added and added[0]["name"] == "get_weather"
    assert added[0]["call_id"].startswith("call_")
    fin = [d for e, d in ev if e == "response.function_call_arguments.done"][0]
    assert json.loads(fin["arguments"]) == {"city": "Paris"}
    done = ev[-1][1]["response"]
    assert [i["type"] for i in done["output"]].count("function_call") == 1


def test_responses_stream_reasoning_item_precedes_message(monkeypatch):
    c, _ = install(monkeypatch, Script(pieces=[*THINK, "Ans"]))
    ev = [(e, d) for e, d in _resp(c) if d != "[DONE]"]
    added = [
        (d["output_index"], d["item"]["type"])
        for e, d in ev
        if e == "response.output_item.added"
    ]
    assert added == [(0, "reasoning"), (1, "message")]
    done = ev[-1][1]["response"]
    assert [i["type"] for i in done["output"]] == ["reasoning", "message"]


# ── Ollama NDJSON ────────────────────────────────────────────────────────────────
def _ndjson(c, path, body):
    r = c.post(path, json=body)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(x) for x in r.text.splitlines() if x.strip()]


def test_ollama_chat_stream_shape(monkeypatch):
    c, _ = install(monkeypatch, Script(pieces=["Hel", "lo"], prompt_tokens=7))
    lines = _ndjson(c, "/api/chat", {"model": "m", "messages": USER})
    assert [ln["done"] for ln in lines] == [False, False, True]
    for ln in lines:
        assert ln["model"] == "m" and ln["created_at"].endswith("Z")
        assert ln["message"]["role"] == "assistant"
    assert "".join(ln["message"]["content"] for ln in lines) == "Hello"
    last = lines[-1]
    assert last["done_reason"] == "stop"
    for k in (
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    ):
        assert isinstance(last[k], int), k
    assert last["prompt_eval_count"] == 7 and last["eval_count"] == 2
    assert all("done_reason" not in ln for ln in lines[:-1])


def test_ollama_generate_stream_shape(monkeypatch):
    c, _ = install(monkeypatch, Script(pieces=["a", "b"], prompt_tokens=3))
    lines = _ndjson(c, "/api/generate", {"model": "m", "prompt": "hi"})
    assert [ln["done"] for ln in lines] == [False, False, True]
    assert "".join(ln["response"] for ln in lines) == "ab"
    assert lines[-1]["done_reason"] == "stop" and lines[-1]["eval_count"] == 2


def test_ollama_nonstream_is_one_json_object(monkeypatch):
    c, _ = install(monkeypatch, Script(pieces=["a", "b"], prompt_tokens=3))
    r = c.post("/api/chat", json={"model": "m", "messages": USER, "stream": False})
    body = r.json()
    assert body["done"] is True and body["message"]["content"] == "ab"
    assert body["done_reason"] == "stop" and body["eval_count"] == 2


def test_ollama_length_done_reason(monkeypatch):
    c, _ = install(monkeypatch, Script(pieces=list("abcdef")))
    lines = _ndjson(
        c,
        "/api/chat",
        {"model": "m", "messages": USER, "options": {"num_predict": 2}},
    )
    assert lines[-1]["done_reason"] == "length" and lines[-1]["eval_count"] == 2
