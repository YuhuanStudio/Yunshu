"""Streamed usage counts the tokens the consumer took (not the runner's lead), and the first
token is flagged for TTFT even when the runner is already past it."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from yunshu_engine.vlm_engine import VLMEngine

TEXT = {10: "ok ", 11: "ST", 12: "OP", 13: " after"}
LEAD = 5  # tokens the runner has produced beyond what the consumer has read


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


class Q:
    def __init__(self):
        self.items = []

    def put_nowait(self, o):
        self.items.append(o)


def _stream(tokens, stop=("STOP",)):
    holder = {}

    def runner_events(ids, stats=None, **kw):
        holder["stats"] = stats
        return VLMEngine._runner_events_impl(eng, ids, stats=stats, **kw)

    def iter_tokens(ids, **kw):
        stats = holder["stats"]
        stats.first_token_s = 0.25
        for i, token in enumerate(tokens, 1):
            stats.generated = i + LEAD  # the runner thread is ahead of this consumer
            yield token

    eng = SimpleNamespace(
        _tokenizer=Tok(),
        _config={},
        _get_eos_ids=lambda: [2],
        _generation_suppress_ids=lambda: [],
        _reasoning_markers=lambda: (None, None, None, None, None),
        _tool_call_marker_id=lambda: None,
        _tool_guide=lambda spec, in_think: None,
        _build_text_constraint=lambda schema: None,
        _batch_runner=SimpleNamespace(iter_tokens=iter_tokens),
        _runner_events=runner_events,
    )
    q = Q()
    VLMEngine._stream_vlm_runner_text(
        eng,
        mx.array([1, 2, 3]),
        "r",
        q,
        max_tokens=50,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        seed=None,
        stop=list(stop),
        stop_token_ids=None,
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        logit_bias=None,
        json_schema=None,
        enable_thinking=None,
        thinking_budget=None,
        cancel_event=None,
    )
    return q.items


def test_usage_counts_consumed_tokens_not_the_runner_lead():
    outs = _stream(
        [10, 11, 12, 13]
    )  # "ok " + "ST" + "OP" completes the stop at token 3
    last = outs[-1]
    assert last.finished and last.stopped_by_stop_sequence
    assert last.completion_tokens == 3  # the runner had produced 3 + LEAD


def test_first_token_event_is_flagged_once_even_if_runner_is_ahead():
    outs = _stream([10, 10, 10], stop=())
    flagged = [o for o in outs if o.ttft_ms > 0]
    assert len(flagged) == 1 and flagged[0] is outs[0]
