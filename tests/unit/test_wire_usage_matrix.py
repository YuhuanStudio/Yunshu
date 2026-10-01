"""R26: one scripted generation reports the same token counts on every dialect, stream and not.

The scripted engine replays the same tokens for every route, so any difference between two
dialects (or between a dialect's stream and non-stream answer) is the gateway's own accounting.
"""

from __future__ import annotations

import pytest

from .wire_clients import DIALECTS, Clients, chunks, run, tool_text
from .wire_harness import Script, install

THINK = [
    ("<think>", "reasoning"),
    ("think", "reasoning"),
    ("ing", "reasoning"),
    ("</think>", "reasoning"),
]
TEXT_DIALECTS = [
    d for d in DIALECTS if d != "completions"
]  # completions never splits thinking


def _serve(monkeypatch, **script):
    c, eng = install(monkeypatch, Script(**script))
    return Clients(c), eng


def _counts(o):
    return (o.prompt, o.completion)


SCENARIOS = {
    "plain": (dict(pieces=["Hello", " there", "!"], prompt_tokens=7), {}),
    "cached": (
        dict(pieces=["Hello", " there", "!"], prompt_tokens=9, cached_tokens=4),
        {},
    ),
    "thinking": (
        dict(pieces=[*THINK, "Hello", " there", "!"], prompt_tokens=7, cached_tokens=4),
        {},
    ),
    "truncated": (
        dict(pieces=["a", "b", "c", "d", "e", "f"], prompt_tokens=5),
        dict(max_tokens=3),
    ),
    "stop_string": (
        dict(pieces=["Hello", " wor", "ld", " END", " more"], prompt_tokens=5),
        dict(stop=["END"]),
    ),
    "zero_output": (dict(pieces=[], prompt_tokens=6), {}),
    "cut_off_while_thinking": (
        dict(pieces=[*THINK, "Hello"], prompt_tokens=6),
        dict(max_tokens=3),
    ),
}


@pytest.mark.parametrize("dialect", DIALECTS)
@pytest.mark.parametrize("name", SCENARIOS)
def test_stream_and_nonstream_agree(monkeypatch, dialect, name):
    script, req = SCENARIOS[name]
    cl, _ = _serve(monkeypatch, **script)
    a = run(cl, dialect, stream=False, **req)
    b = run(cl, dialect, stream=True, **req)
    assert (b.text.strip(), b.finish) == (a.text.strip(), a.finish), (dialect, name)
    assert _counts(b) == _counts(a), (dialect, name, _counts(a), _counts(b))
    if a.reasoning is not None or b.reasoning is not None:
        assert (b.reasoning or 0) == (a.reasoning or 0), (dialect, name)
    if a.cached is not None and b.cached is not None:
        assert b.cached == a.cached, (dialect, name)


@pytest.mark.parametrize("name", SCENARIOS)
def test_dialects_agree(monkeypatch, name):
    script, req = SCENARIOS[name]
    cl, _ = _serve(monkeypatch, **script)
    outs = {d: run(cl, d, stream=False, **req) for d in DIALECTS}
    ref = outs["chat"]
    for d, o in outs.items():
        if d == "completions" and "thinking" in name:
            continue  # raw completions keep the thinking in the text
        assert o.text.strip() == ref.text.strip(), (d, name)
        assert o.prompt == ref.prompt, (d, name, o.prompt, ref.prompt)
        assert o.completion == ref.completion, (d, name, o.completion, ref.completion)
        if o.reasoning is not None:
            assert o.reasoning == ref.reasoning, (d, name)
        if o.cached is not None:
            assert o.cached == ref.cached, (d, name)


def test_tool_call_markers_count_as_completion_tokens(monkeypatch):
    pieces = chunks(tool_text("get_weather", {"city": "Paris"}), 6)
    cl, _ = _serve(monkeypatch, pieces=pieces, prompt_tokens=11)
    for d in ("chat", "messages", "responses", "ollama_chat"):
        for stream in (False, True):
            o = run(cl, d, stream=stream, tools=True)
            assert o.tools == [("get_weather", {"city": "Paris"})], (d, stream)
            assert o.completion == len(pieces), (d, stream, o.completion)
            assert o.prompt == 11, (d, stream, o.prompt)
