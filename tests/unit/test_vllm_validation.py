"""Request validation ported from vLLM's entrypoint tests (see route_checks_vllm.py). Each case is a
request the gateway answered 200 to before; a strict OpenAI server answers 400 in the OpenAI shape."""

from __future__ import annotations

import pytest

from .wire_harness import install

MSG = [{"role": "user", "content": "hi"}]
TOOL = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}

BAD_CHAT = {
    "stream_options_without_stream": {"stream_options": {"include_usage": True}},
    "tool_choice_unknown_function": {
        "tools": [TOOL],
        "tool_choice": {"type": "function", "function": {"name": "nope"}},
    },
    "tool_choice_required_no_tools": {"tool_choice": "required"},
    "prompt_logprobs_negative": {"prompt_logprobs": -1},
    "structured_regex_invalid": {"structured_outputs": {"regex": "[.*"}},
    "structured_grammar_empty": {"structured_outputs": {"grammar": ""}},
    "json_schema_without_schema": {"response_format": {"type": "json_schema"}},
    "structured_two_kinds": {"structured_outputs": {"regex": "a", "choice": ["a"]}},
    "structured_unknown_key": {"structured_outputs": {"nope": 1}},
}


@pytest.mark.parametrize("name", list(BAD_CHAT))
def test_chat_rejects(name, monkeypatch):
    http, _ = install(monkeypatch)
    r = http.post(
        "/v1/chat/completions",
        json={"model": "scripted", "messages": MSG, "max_tokens": 4, **BAD_CHAT[name]},
    )
    assert 400 <= r.status_code < 500, (name, r.status_code, r.text[:200])
    assert r.json()["error"]["message"]


@pytest.mark.parametrize(
    "body",
    [
        {"prompt": [-5, 3]},
        {"prompt": "hi", "stream_options": {"include_usage": True}},
        {"prompt": "hi", "prompt_logprobs": -1},
    ],
)
def test_completion_rejects(body, monkeypatch):
    http, _ = install(monkeypatch)
    r = http.post(
        "/v1/completions", json={"model": "scripted", "max_tokens": 4, **body}
    )
    assert 400 <= r.status_code < 500, (r.status_code, r.text[:200])


def test_structured_outputs_folds_into_native_fields():
    from yunshu_gateway.routers.chat import ChatCompletionRequest

    base = {"model": "m", "messages": MSG}
    r = ChatCompletionRequest.model_validate(
        {**base, "structured_outputs": {"regex": "ab+"}}
    )
    assert r.grammar == {"type": "regex", "pattern": "ab+"}
    r = ChatCompletionRequest.model_validate(
        {**base, "structured_outputs": {"choice": ["yes", "no"]}}
    )
    assert r.grammar == {"type": "choice", "choices": ["yes", "no"]}
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    r = ChatCompletionRequest.model_validate(
        {**base, "structured_outputs": {"json": schema}}
    )
    assert r.response_format["json_schema"]["schema"] == schema
    # a native field the caller also sent wins
    r = ChatCompletionRequest.model_validate(
        {
            **base,
            "grammar": {"type": "regex", "pattern": "x"},
            "structured_outputs": {"regex": "y"},
        }
    )
    assert r.grammar["pattern"] == "x"


def test_stream_options_with_stream_still_valid():
    from yunshu_gateway.routers.chat import ChatCompletionRequest

    ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": MSG,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    )
