"""Lazy constraints preserve reasoning and constrain every triggered structure."""

import json

import pytest
from llguidance import LLMatcher, LLTokenizer, TokenizerWrapper

from yunshu_gateway.schemas.structured_outputs import (
    fold_structured_outputs,
    structural_tag_grammar,
)

SPEC = {
    "type": "structural_tag",
    "structures": [
        {
            "begin": "<tool=weather>",
            "end": "</tool>",
            "schema": {
                "type": "object",
                "properties": {"city": {"const": "Taipei"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        }
    ],
    "triggers": ["<tool="],
}


class ByteTokenizer:
    tokens = [bytes([i]) for i in range(256)] + [b"<eos>"]
    eos_token_id = 256
    bos_token_id = None

    def __call__(self, text):
        return list(text if isinstance(text, bytes) else text.encode())


def matcher():
    lark = structural_tag_grammar(SPEC)
    return LLMatcher(
        LLTokenizer(TokenizerWrapper(ByteTokenizer())),
        LLMatcher.grammar_from_lark(lark),
    )


@pytest.mark.parametrize(
    "text",
    [
        "<think>任意推理 {broken json}</think> plain prose",
        'reasoning <tool=weather>{"city":"Taipei"}</tool> more prose',
        '<tool=weather>{"city":"Taipei"}</tool><tool=weather>{"city":"Taipei"}</tool>',
    ],
)
def test_reasoning_free_and_repeated_tags(text):
    m = matcher()
    for byte in text.encode():
        assert m.consume_token(byte), m.get_error()
    assert m.is_accepting()


@pytest.mark.parametrize(
    "text",
    [
        "<tool=wrong>{}",
        '<tool=weather>{"city":"Tokyo"}',
        '<tool=weather>{"unexpected":1}',
        "<tool=weather>not json",
    ],
)
def test_trigger_constrains_and_blocks_early_eos(text):
    m = matcher()
    prefix = "<think>free</think><tool=weather>"
    for byte in prefix.encode():
        assert m.consume_token(byte)
    assert not m.is_accepting()
    m = matcher()
    assert not all(m.consume_token(byte) for byte in text.encode())


def test_api_folds_to_cfg_on_both_serving_paths():
    from yunshu_gateway.routers.chat import (
        ChatCompletionRequest,
        _parse_response_format,
    )
    from yunshu_gateway.routers.completions import CompletionRequest

    for cls in (ChatCompletionRequest, CompletionRequest):
        kwargs = (
            {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
            if cls is ChatCompletionRequest
            else {"model": "test", "prompt": "hi"}
        )
        req = cls(**kwargs, response_format=SPEC)
        spec = _parse_response_format(req.response_format, None)
        assert spec["type"] == "cfg"
    folded = fold_structured_outputs(
        {"structured_outputs": {"structural_tag": json.dumps(SPEC)}}
    )
    assert folded["response_format"] == SPEC


@pytest.mark.parametrize(
    "update",
    [
        {"triggers": []},
        {"structures": []},
        {"triggers": ["bad"]},
        {"triggers": ["<tool=", "<tool=weather>"]},
    ],
)
def test_invalid_spec_rejected(update):
    with pytest.raises(ValueError):
        structural_tag_grammar({**SPEC, **update})


def test_actual_qwen_reasoning_special_tokens_remain_free():
    from pathlib import Path

    from transformers import AutoTokenizer

    from yunshu_engine.grammar_constraint import CfgGrammarConstraint

    path = Path("/Volumes/P5Plus/models/Qwen3.5-0.8B-MLX-bf16")
    if not path.exists():
        pytest.skip("local Qwen tokenizer unavailable")
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    c = CfgGrammarConstraint(structural_tag_grammar(SPEC), tokenizer=tok)
    text = '<think>自由推理</think><tool=weather>{"city":"Taipei"}</tool>'
    for tid in tok.encode(text, add_special_tokens=False):
        c.advance_token(tid)
        assert not c._dead, c._matcher.get_error()
    assert c._matcher.is_accepting()


def test_special_trigger_and_schema_marker_const():
    from pathlib import Path

    from transformers import AutoTokenizer

    from yunshu_engine.grammar_constraint import CfgGrammarConstraint

    path = Path("/Volumes/P5Plus/models/Qwen3.5-0.8B-MLX-bf16")
    if not path.exists():
        pytest.skip("local Qwen tokenizer unavailable")
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    spec = {
        "structures": [
            {
                "begin": "<tool_call>",
                "end": "</tool_call>",
                "schema": {"const": "<think>"},
            }
        ],
        "triggers": ["<tool_call>"],
    }
    c = CfgGrammarConstraint(structural_tag_grammar(spec), tokenizer=tok)
    ids = tok.encode("free </think><tool_call>", add_special_tokens=False)
    for piece in ['"', "<", "think", ">", '"']:
        ids += tok.encode(piece, add_special_tokens=False)
    ids += tok.encode("</tool_call>", add_special_tokens=False)
    for tid in ids:
        c.advance_token(tid)
        assert not c._dead, c._matcher.get_error()
    assert c._matcher.is_accepting()


def test_responses_structural_tag_maps_to_same_cfg():
    from yunshu_gateway.routers.responses import (
        ResponsesRequest,
        _parse_response_format_unchecked,
    )

    req = ResponsesRequest(model="test", input="hi", response_format=SPEC)
    assert _parse_response_format_unchecked(req.response_format)["type"] == "cfg"


def test_empty_arguments_stream_is_valid_json_on_actual_chat_route(monkeypatch):
    from .wire_harness import Script, install

    http, _ = install(
        monkeypatch,
        Script(pieces=['<tool_call>{"name":"ping","arguments":{}}</tool_call>']),
    )
    reply = http.post(
        "/v1/chat/completions",
        json={
            "model": "scripted",
            "messages": [{"role": "user", "content": "ping"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "ping", "parameters": {"type": "object"}},
                }
            ],
            "stream": True,
        },
    )
    assert reply.status_code == 200
    fragments = []
    for line in reply.text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            row = json.loads(line[6:])
            for choice in row.get("choices", []):
                for call in choice.get("delta", {}).get("tool_calls", []):
                    fragments.append(call.get("function", {}).get("arguments", ""))
    assert json.loads("".join(fragments)) == {}
