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


def test_token_id_rollback_does_not_reuse_other_branch_mask(monkeypatch):
    from types import SimpleNamespace

    from yunshu_engine import tool_call_grammar as tcg
    from yunshu_engine.grammar_constraint import CfgGrammarConstraint

    llt = LLTokenizer(TokenizerWrapper(ByteTokenizer()))
    monkeypatch.setattr(tcg, "llg_tokenizer", lambda *_: llt)
    hf = SimpleNamespace(vocab_size=257, eos_token_id=256)
    c = CfgGrammarConstraint('start: "ab" | "xy"', tokenizer=hf)
    cp = c.checkpoint()
    c.advance_token(ord("a"))
    assert c.get_allowed_tokens(hf, []) == [ord("b")]
    c.rollback(cp)
    c.advance_token(ord("x"))
    assert c.get_allowed_tokens(hf, []) == [ord("y")]


def test_vlm_auto_tools_can_use_lazy_tags_but_forced_contract_is_preserved():
    from yunshu_gateway.routers.chat import (
        ChatCompletionRequest,
        _parse_response_format,
        _vlm_tool_schema_conflict,
    )

    tools = [
        {
            "type": "function",
            "function": {"name": "weather", "parameters": {"type": "object"}},
        }
    ]
    req = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        response_format=SPEC,
        tool_choice="auto",
    )
    parsed = _parse_response_format(req.response_format, None)
    assert not _vlm_tool_schema_conflict(req, parsed)
    forced = req.model_copy(update={"tool_choice": "required"})
    assert _vlm_tool_schema_conflict(forced, parsed)
    assert _vlm_tool_schema_conflict(req, {"type": "object"})


def test_lazy_defaults_and_native_auto_guide_do_not_conflict(monkeypatch):
    from yunshu_engine import settings
    from yunshu_engine.structural_tag import constrains_initial_output
    from yunshu_engine.vlm_engine import VLMEngine

    spec = {"type": "cfg", "grammar": structural_tag_grammar(SPEC)}
    engine = VLMEngine.__new__(VLMEngine)
    engine._config = {"model_type": "qwen3_5"}
    engine._model_path = "qwen"
    assert (
        engine._default_enable_thinking(
            None, constrained=constrains_initial_output(spec)
        )
        is None
    )
    assert (
        engine._default_enable_thinking(
            None, constrained=constrains_initial_output({"type": "object"})
        )
        is False
    )
    monkeypatch.setattr(settings, "get_bool", lambda key: key == "YUNSHU_TOOL_GRAMMAR")
    tools = [{"name": "weather", "parameters": {"type": "object"}}]
    kwargs = {"tools": tools, "tool_choice": "auto", "json_schema": spec}
    assert engine._request_template_extra(kwargs)["tools"] == tools
    assert "_tool_spec" not in kwargs
    kwargs = {"tools": tools, "tool_choice": "auto"}
    engine._request_template_extra(kwargs)
    assert "_tool_spec" in kwargs
    with pytest.raises(ValueError, match="forced tool_choice"):
        engine._request_template_extra(
            {"tools": tools, "tool_choice": "required", "json_schema": spec}
        )


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
def test_lazy_forced_tool_contract_rejected_before_generation(endpoint):
    from yunshu_gateway.routers.chat import ChatCompletionRequest
    from yunshu_gateway.routers.responses import ResponsesRequest

    if endpoint == "chat":
        cls = ChatCompletionRequest
        body = {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "weather", "parameters": {"type": "object"}},
                }
            ],
        }
    else:
        cls = ResponsesRequest
        body = {
            "model": "test",
            "input": "hi",
            "tools": [
                {
                    "type": "function",
                    "name": "weather",
                    "parameters": {"type": "object"},
                }
            ],
        }
    with pytest.raises(ValueError, match="forced tool_choice"):
        cls(**body, response_format=SPEC, tool_choice="required")


def test_bitmask_includes_added_reasoning_tokens():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from yunshu_engine.grammar_bitmask import GrammarBitmaskEngine, TokenStringTable
    from yunshu_engine.grammar_constraint import CfgGrammarConstraint

    tok = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"a": 0, "<eos>": 1}, unk_token="a")),
        eos_token="<eos>",
    )
    from tokenizers.decoders import ByteLevel

    tok.backend_tokenizer.decoder = ByteLevel()
    tok.add_special_tokens({"additional_special_tokens": ["<think>"]})
    assert tok.vocab_size == 2 and len(tok) == 3
    table = TokenStringTable.get(tok)
    assert table.vocab_size == 3
    c = CfgGrammarConstraint("start: <[2]>", tokenizer=tok)
    mask = GrammarBitmaskEngine(c).compute_bitmask(tok)
    assert mask.shape[0] == 3 and bool(mask[2].item())


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
async def test_text_engine_preserves_thinking_budget_only_for_lazy_tags(
    streaming, lazy
):
    from unittest.mock import AsyncMock

    from yunshu_engine.batched_engine import GenerationOutput

    from .test_batched_engine_spec_paths import _make_batched_engine

    eng = _make_batched_engine()
    calls = []
    out = GenerationOutput(
        text="ok", new_text="ok", finished=True, finish_reason="stop"
    )
    eng._generate_fast = AsyncMock(return_value=out)

    async def fake_stream(*args, **kwargs):
        calls.append(kwargs)
        yield out

    eng._stream_generate_fast = fake_stream
    schema = (
        {"type": "cfg", "grammar": structural_tag_grammar(SPEC)}
        if lazy
        else {"type": "object"}
    )
    if streaming:
        result = [
            o
            async for o in eng.stream_generate(
                "hi", json_schema=schema, thinking_budget=96, use_engine_loop=False
            )
        ]
        assert result
        passed = calls[0]
    else:
        await eng.generate(
            "hi", json_schema=schema, thinking_budget=96, use_engine_loop=False
        )
        passed = eng._generate_fast.call_args.kwargs
    assert passed["thinking_budget"] == (96 if lazy else None)


@pytest.mark.parametrize("lazy", [False, True])
def test_vlm_runner_preserves_thinking_budget_only_for_lazy_tags(lazy):
    from types import SimpleNamespace

    import numpy as np

    from yunshu_engine.vlm_engine import VLMEngine

    eng = VLMEngine.__new__(VLMEngine)
    eng._config = {"vocab_size": 257}
    eng._tokenizer = SimpleNamespace(
        detokenizer=SimpleNamespace(
            reset=lambda: None, finalize=lambda: None, last_segment=""
        )
    )
    eng._reasoning_markers = lambda: (1, 2, "<think>", "</think>", False)
    eng._get_eos_ids = lambda: [256]
    eng._tool_guide = lambda *_: None
    eng._build_text_constraint = lambda _: SimpleNamespace(rollback=lambda: None)
    calls = []

    def iterate(*args, **kwargs):
        calls.append(kwargs)
        return iter([])

    eng._batch_runner = SimpleNamespace(iter_tokens=iterate)
    schema = (
        {"type": "cfg", "grammar": structural_tag_grammar(SPEC)}
        if lazy
        else {"type": "object"}
    )
    list(
        eng._runner_events_impl(
            np.array([1]),
            max_tokens=16,
            temperature=0,
            top_p=1,
            top_k=0,
            min_p=0,
            seed=0,
            stop=None,
            stop_token_ids=None,
            repetition_penalty=1,
            frequency_penalty=0,
            presence_penalty=0,
            logit_bias=None,
            json_schema=schema,
            enable_thinking=True,
            thinking_budget=96,
            cancel_event=None,
            stats=SimpleNamespace(finish_reason="stop"),
        )
    )
    assert calls[0]["thinking_budget"] == (96 if lazy else None)


def test_partial_trigger_matches_special_and_byte_spellings():
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
                "schema": {"const": {"city": "Taipei"}},
            }
        ],
        "triggers": ["<tool"],
    }
    source = structural_tag_grammar(spec)
    for special in [True, False]:
        c = CfgGrammarConstraint(source, tokenizer=tok)
        text = '<tool_call>{"city":"Taipei"}</tool_call>'
        ids = (
            tok.encode(text, add_special_tokens=False)
            if special
            else [i for ch in text for i in tok.encode(ch, add_special_tokens=False)]
        )
        for tid in ids:
            c.advance_token(tid)
            assert not c._dead, c._matcher.get_error()
        assert c._matcher.is_accepting()
    c = CfgGrammarConstraint(source, tokenizer=tok)
    for tid in tok.encode('<tool_call>{"city":"Tokyo"}', add_special_tokens=False):
        c.advance_token(tid)
    assert c._dead
