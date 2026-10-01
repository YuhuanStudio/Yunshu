"""F04: tool calls (single, parallel, tool_choice) and structured output on every dialect.

The scripted engine emits Hermes-style ``<tool_call>`` markup; a trailing assistant turn (the
forced-choice prefill) is continued, not repeated, like a real engine. Each dialect's SDK must
see the same calls, stream and non-stream.
"""

from __future__ import annotations

import json

import pytest

from .wire_clients import Clients, chunks, run
from .wire_harness import Script, install

TOOL_DIALECTS = ["chat", "messages", "responses", "ollama_chat"]
OA_DIALECTS = ["chat", "messages", "responses"]  # have tool_choice


def call(city: str) -> str:
    body = json.dumps({"name": "get_weather", "arguments": {"city": city}})
    return f"<tool_call>\n{body}\n</tool_call>"


def _serve(monkeypatch, text: str):
    c, eng = install(monkeypatch, Script(pieces=chunks(text, 7), prompt_tokens=9))
    return Clients(c), eng


@pytest.mark.parametrize("dialect", TOOL_DIALECTS)
@pytest.mark.parametrize("stream", [False, True])
def test_single_tool_call(monkeypatch, dialect, stream):
    cl, _ = _serve(monkeypatch, call("Paris"))
    o = run(cl, dialect, stream=stream, tools=True)
    assert o.tools == [("get_weather", {"city": "Paris"})]
    assert o.finish == "tool_calls"
    assert o.text.strip() == ""


@pytest.mark.parametrize("dialect", TOOL_DIALECTS)
@pytest.mark.parametrize("stream", [False, True])
def test_parallel_tool_calls(monkeypatch, dialect, stream):
    cl, _ = _serve(monkeypatch, call("Paris") + "\n" + call("Rome"))
    o = run(cl, dialect, stream=stream, tools=True)
    assert o.tools == [
        ("get_weather", {"city": "Paris"}),
        ("get_weather", {"city": "Rome"}),
    ]
    assert o.finish == "tool_calls"


@pytest.mark.parametrize("dialect", OA_DIALECTS)
@pytest.mark.parametrize("stream", [False, True])
def test_parallel_disabled_keeps_the_first_call(monkeypatch, dialect, stream):
    cl, _ = _serve(monkeypatch, call("Paris") + "\n" + call("Rome"))
    o = run(cl, dialect, stream=stream, tools=True, parallel=False)
    assert o.tools == [("get_weather", {"city": "Paris"})]


@pytest.mark.parametrize("dialect", OA_DIALECTS)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("choice", ["required", "get_weather"])
def test_forced_tool_choice(monkeypatch, dialect, stream, choice):
    cl, eng = _serve(monkeypatch, call("Paris"))
    o = run(cl, dialect, stream=stream, tools=True, tool_choice=choice)
    assert o.tools == [("get_weather", {"city": "Paris"})], (dialect, stream, choice)
    assert o.finish == "tool_calls"


@pytest.mark.parametrize("dialect", OA_DIALECTS)
def test_named_choice_drops_other_functions(monkeypatch, dialect):
    other = call("Paris").replace("get_weather", "other_fn")
    cl, _ = _serve(monkeypatch, other)
    o = run(cl, dialect, stream=False, tools=True, tool_choice="get_weather")
    assert o.tools == [], (dialect, o.tools)
    assert o.finish != "tool_calls"


@pytest.mark.parametrize("dialect", OA_DIALECTS)
def test_tool_choice_none_surfaces_no_call(monkeypatch, dialect):
    cl, _ = _serve(monkeypatch, call("Paris"))
    for stream in (False, True):
        o = run(cl, dialect, stream=stream, tools=True, tool_choice="none")
        assert o.tools == [], (dialect, stream)
        assert o.finish == "stop", (dialect, stream)


@pytest.mark.parametrize("dialect", TOOL_DIALECTS)
def test_tools_declared_but_plain_answer(monkeypatch, dialect):
    cl, _ = _serve(monkeypatch, "It is sunny.")
    for stream in (False, True):
        o = run(cl, dialect, stream=stream, tools=True)
        assert o.tools == [] and o.text.strip() == "It is sunny."
        assert o.finish == "stop"


# ── structured output reaches the engine the same way on every dialect ───────────
@pytest.mark.parametrize(
    "dialect", ["chat", "responses", "ollama_chat", "ollama_generate"]
)
@pytest.mark.parametrize("stream", [False, True])
def test_json_schema_is_forwarded_and_answer_parses(monkeypatch, dialect, stream):
    cl, eng = _serve(monkeypatch, '{"a": 3}')
    o = run(cl, dialect, stream=stream, schema=True)
    assert json.loads(o.text) == {"a": 3}
    schemas = [c.get("json_schema") for c in eng.calls]
    assert schemas and all(s for s in schemas), (dialect, stream, eng.calls)
    assert schemas[-1].get("properties", {}).get("a") == {
        "type": "integer"
    } or "a" in str(schemas[-1])
