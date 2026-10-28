"""A stop string that completes only in the final detokenizer flush is still a stop:
trimmed from the text, reported as stop_string_hit and finish_reason 'stop'."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from yunshu_engine.text_utils import StopHoldbackBuffer

mx = pytest.importorskip("mlx.core")

from yunshu_engine.vlm_engine import VLMEngine  # noqa: E402

EOS = 2


class Detok:
    last_segment = ""

    def reset(self):
        pass

    def add_token(self, t):
        self.last_segment = "EN"  # the first two characters of the stop string

    def finalize(self):
        self.last_segment = "D"  # the detokenizer's final flush completes "END"


class Tok:
    detokenizer = Detok()


def _run(tokens, max_tokens=50):
    def iter_tokens(ids, **kw):
        yield from tokens

    eng = SimpleNamespace(
        _tokenizer=Tok(),
        _config={},
        _batch_runner=SimpleNamespace(iter_tokens=iter_tokens),
        _get_eos_ids=lambda: [EOS],
        _generation_suppress_ids=lambda: [],
        _reasoning_markers=lambda: (None, None, "", "", False),
        _tool_call_marker_id=lambda: None,
        _tool_guide=lambda spec, in_think: None,
        _build_text_constraint=lambda schema: None,
    )
    stats = SimpleNamespace(finish_reason=None, stop_string_hit=False)
    events = list(
        VLMEngine._runner_events_impl(
            eng,
            mx.array([1]),
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            seed=None,
            stop=["END"],
            stop_token_ids=None,
            repetition_penalty=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            logit_bias=None,
            json_schema=None,
            enable_thinking=False,
            thinking_budget=None,
            cancel_event=None,
            stats=stats,
        )
    )
    return events, stats


def test_buffer_reports_a_stop_dropped_by_flush():
    hb = StopHoldbackBuffer(["END"])
    assert hb.feed("hi EN") == "hi "
    assert hb.feed("D") == ""  # complete stop is held, never emitted
    assert hb.flush() == "" and hb.stop_hit
    clean = StopHoldbackBuffer(["END"])
    clean.feed("EN")
    assert (
        clean.flush() == "EN" and not clean.stop_hit
    )  # a partial prefix is plain text


def test_stop_completing_in_flush_at_end_of_stream():
    events, stats = _run([5])
    assert "".join(e[0] for e in events).endswith("") and "END" not in "".join(
        e[0] for e in events
    )
    assert events[-1][3] == "stop" and stats.stop_string_hit


def test_stop_completing_in_flush_wins_over_length():
    events, stats = _run([5], max_tokens=1)
    assert events[-1][3] == "stop" and stats.stop_string_hit


def test_stop_completing_in_flush_at_eos():
    events, stats = _run([5, EOS])
    assert "END" not in "".join(e[0] for e in events)
    assert events[-1][3] == "stop" and stats.stop_string_hit
