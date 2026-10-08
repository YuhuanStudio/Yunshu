"""Forced-tool compilation failures are HTTP errors before streaming headers."""

from types import SimpleNamespace

import pytest

from yunshu_engine import tool_call_grammar as tcg

from .wire_harness import Script, install


@pytest.mark.parametrize("choice", ["required", {"name": "run"}])
def test_uncompiled_forced_grammar_rejected(choice):
    with pytest.raises(ValueError, match="Cannot guarantee forced tool_choice"):
        tcg.require_tool_grammar(None, choice)
    assert tcg.require_tool_grammar(None, "auto") is None


def test_preflight_caches_and_rejects_unsupported_markers(monkeypatch):
    class Tokenizer:
        def __len__(self):
            return 100

    eng = SimpleNamespace(_tokenizer=Tokenizer())

    def compile_(*a, **k):
        return None

    monkeypatch.setattr(tcg, "compile_tool_grammar", compile_)
    with pytest.raises(ValueError, match="multi-token markers"):
        tcg.validate_forced_tools(eng, [{"name": "run"}], "required")
    assert len(eng._tool_grammars) == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "path,body",
    [
        (
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "run"}],
                "tools": [
                    {"type": "function", "function": {"name": "run", "parameters": {}}}
                ],
                "tool_choice": "required",
            },
        ),
        (
            "/v1/responses",
            {
                "input": "run",
                "tools": [{"type": "function", "name": "run", "parameters": {}}],
                "tool_choice": "required",
            },
        ),
        (
            "/v1/messages",
            {
                "max_tokens": 20,
                "messages": [{"role": "user", "content": "run"}],
                "tools": [{"name": "run", "input_schema": {}}],
                "tool_choice": {"type": "any"},
            },
        ),
    ],
)
def test_dialects_reject_before_generation(monkeypatch, stream, path, body):
    client, eng = install(monkeypatch, Script(pieces=["text"]))

    def reject(*args):
        raise ValueError("Cannot guarantee forced tool_choice: recursive schema")

    monkeypatch.setattr(eng, "validate_forced_tools", reject)
    response = client.post(path, json={"model": "scripted", "stream": stream, **body})
    assert response.status_code == 400, response.text
    assert "Cannot guarantee forced tool_choice" in response.text
    assert not eng.calls
