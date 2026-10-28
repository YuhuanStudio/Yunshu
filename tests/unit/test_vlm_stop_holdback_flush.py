"""Text held back as a possible stop-string prefix is delivered when the stream ends without
a stop match (EOS, length, or the runner simply finishing)."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from yunshu_engine.vlm_engine import VLMEngine

TEXT = {10: "ok ", 11: "ST", 12: "O"}  # "STO" is a prefix of the stop string "STOP"


class Detok:
    last_segment = ""

    def reset(self):
        pass

    def add_token(self, t):
        self.last_segment = TEXT.get(t, "")

    def finalize(self):
        self.last_segment = ""


class Tok:
    detokenizer = Detok()


def _events(tokens, max_tokens, eos=2):
    def iter_tokens(ids, **kw):
        yield from tokens

    eng = SimpleNamespace(
        _tokenizer=Tok(),
        _config={},
        _batch_runner=SimpleNamespace(iter_tokens=iter_tokens),
        _get_eos_ids=lambda: [eos],
        _generation_suppress_ids=lambda: [],
        _reasoning_markers=lambda: (None, None, None, None, None),
        _tool_call_marker_id=lambda: None,
        _tool_guide=lambda spec, in_think: None,
        _build_text_constraint=lambda schema: None,
    )
    out = list(
        VLMEngine._runner_events_impl(
            eng,
            mx.array([1]),
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            seed=None,
            stop=["STOP"],
            stop_token_ids=None,
            repetition_penalty=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            logit_bias=None,
            json_schema=None,
            enable_thinking=None,
            thinking_budget=None,
            cancel_event=None,
            stats=SimpleNamespace(finish_reason=None, stop_string_hit=False),
        )
    )
    return "".join(e[0] for e in out), out[-1][3]


def test_held_prefix_is_flushed_at_eos():
    text, finish = _events([10, 11, 12, 2], max_tokens=50)
    assert (text, finish) == ("ok STO", "stop")


def test_held_prefix_is_flushed_at_length():
    text, finish = _events([10, 11, 12], max_tokens=3)
    assert (text, finish) == ("ok STO", "length")


def test_held_prefix_is_flushed_when_the_runner_ends():
    text, finish = _events([10, 11, 12], max_tokens=50)
    assert text == "ok STO" and finish == "stop"
