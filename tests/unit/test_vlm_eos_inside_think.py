"""EOS (or the end of the stream) inside an open <think> block: the final text is reasoning."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from yunshu_engine.vlm_engine import VLMEngine

THINK, END_THINK, EOS = 20, 21, 2


class Detok:
    last_segment = ""

    def reset(self):
        pass

    def add_token(self, t):
        self.last_segment = "thinking "

    def finalize(self):
        self.last_segment = "tail"  # bytes the detokenizer still held


class Tok:
    detokenizer = Detok()


def _events(tokens):
    def iter_tokens(ids, **kw):
        yield from tokens

    eng = SimpleNamespace(
        _tokenizer=Tok(),
        _config={},
        _batch_runner=SimpleNamespace(iter_tokens=iter_tokens),
        _get_eos_ids=lambda: [EOS],
        _generation_suppress_ids=lambda: [],
        _reasoning_markers=lambda: (THINK, END_THINK, "<think>", "</think>", False),
        _tool_call_marker_id=lambda: None,
        _tool_guide=lambda spec, in_think: None,
        _build_text_constraint=lambda schema: None,
    )
    return list(
        VLMEngine._runner_events_impl(
            eng,
            mx.array([1]),
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
            stats=SimpleNamespace(finish_reason=None, stop_string_hit=False),
        )
    )


def test_eos_inside_think_keeps_the_tail_in_reasoning():
    last = _events([THINK, 10, EOS])[-1]
    assert last[0] == "tail" and last[2] == "reasoning" and last[3] == "stop"


def test_eos_after_think_closed_is_content():
    last = _events([THINK, 10, END_THINK, 11, EOS])[-1]
    assert last[0] == "tail" and last[2] == "normal"


def test_runner_end_inside_think_keeps_the_tail_in_reasoning():
    last = _events([THINK, 10])[-1]
    assert last[0] == "tail" and last[2] == "reasoning"
