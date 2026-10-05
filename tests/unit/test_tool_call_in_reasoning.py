"""A tool call opened inside an unclosed reasoning block still reaches the client.

Qwen3.5-family models sometimes write `<tool_call>` before closing `</think>`. vLLM's
Qwen3ReasoningParser and SGLang's Qwen3 reasoning detector treat the call marker as the implicit end
of reasoning; Yunshu delivered the whole call as reasoning text (found by the coverage audit on
/v1/responses with Qwen3.5-9B).
"""

from unittest.mock import MagicMock

import mlx.core as mx

from yunshu_engine.vlm_engine import VLMEngine

THINK, END, TOOL, EOS = 5, 6, 7, 1
PIECES = {10: "I will read it. ", TOOL: "<tool_call>", 11: "Read{}", 12: "</tool_call>"}


class _Detok:
    last_segment = ""

    def reset(self):
        self.last_segment = ""

    def add_token(self, t):
        self.last_segment = PIECES[t]

    def finalize(self):
        self.last_segment = ""


def _events(tokens, *, tool_spec=None, tools_declared=False, prompt_opens_think=True):
    def iter_tokens(ids, **kw):
        return iter(tokens)

    eng = MagicMock()
    eng._build_text_constraint.return_value = None
    eng._get_eos_ids.return_value = [EOS]
    eng._config = {"vocab_size": 20}
    eng._tokenizer = MagicMock()
    eng._tokenizer.detokenizer = _Detok()
    eng._tool_guide.return_value = None
    eng._reasoning_markers.return_value = (THINK, END, "<think>", "</think>", False)
    eng._tool_call_marker_id.return_value = TOOL
    eng._batch_runner.iter_tokens = iter_tokens
    ids = [2, THINK] if prompt_opens_think else [2, 3]
    return list(
        VLMEngine._runner_events_impl(
            eng,
            mx.array(ids),
            max_tokens=50,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            seed=None,
            stop=None,
            stop_token_ids=None,
            repetition_penalty=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            logit_bias=None,
            json_schema=None,
            enable_thinking=True,
            thinking_budget=None,
            cancel_event=None,
            stats=MagicMock(),
            tool_spec=tool_spec,
            tools_declared=tools_declared,
        )
    )


def _split(events):
    reasoning = "".join(e[0] for e in events if e[2] == "reasoning")
    content = "".join(e[0] for e in events if e[2] == "normal")
    return reasoning, content


SPEC = {"tools": [], "tool_choice": "auto", "parallel": True}


def test_tool_call_marker_ends_unclosed_reasoning():
    reasoning, content = _split(_events([10, TOOL, 11, 12, EOS], tool_spec=SPEC))
    assert reasoning == "I will read it. "
    assert content == "<tool_call>Read{}</tool_call>"


def test_closed_reasoning_is_unchanged():
    reasoning, content = _split(_events([10, END, TOOL, 11, 12, EOS], tool_spec=SPEC))
    assert reasoning == "I will read it. "
    assert content == "<tool_call>Read{}</tool_call>"


def test_without_declared_tools_reasoning_keeps_the_marker():
    reasoning, content = _split(_events([10, TOOL, 11, 12, EOS], tool_spec=None))
    assert "<tool_call>" in reasoning and content == ""


def test_tools_declared_without_grammar_spec_also_ends_reasoning():
    # YUNSHU_TOOL_GRAMMAR off (or tool_choice none handled elsewhere): no _tool_spec,
    # but the request still declares tools and its calls are parsed from normal text.
    reasoning, content = _split(_events([10, TOOL, 11, 12, EOS], tools_declared=True))
    assert reasoning == "I will read it. "
    assert content == "<tool_call>Read{}</tool_call>"
