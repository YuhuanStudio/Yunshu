"""generation_config.json ``suppress_tokens`` is the VLM path's default mask; a request's own list wins."""

from __future__ import annotations

import json
from types import SimpleNamespace

import mlx.core as mx

from yunshu_engine.vlm_batch_runner import TokenMaskProcessor
from yunshu_engine.vlm_engine import VLMEngine


class Detok:
    def reset(self):
        pass

    def add_token(self, t):
        self.last_segment = f"<{t}>"

    def finalize(self):
        self.last_segment = ""


class Tok:
    detokenizer = Detok()

    def __len__(self):
        return 100


def _engine(tmp_path, suppress):
    if suppress is not None:
        (tmp_path / "generation_config.json").write_text(
            json.dumps({"suppress_tokens": suppress})
        )
    seen = {}

    def iter_tokens(ids, **kw):
        seen.update(kw)
        yield 7

    eng = SimpleNamespace(
        _model_path=str(tmp_path),
        _tokenizer=Tok(),
        _config={},
        _batch_runner=SimpleNamespace(iter_tokens=iter_tokens),
        _get_eos_ids=lambda: [2],
        _reasoning_markers=lambda: (None, None, None, None, None),
        _tool_call_marker_id=lambda: None,
        _tool_guide=lambda spec, in_think: None,
        _build_text_constraint=lambda schema: None,
    )
    eng._generation_suppress_ids = lambda: VLMEngine._generation_suppress_ids(eng)
    return eng, seen


def _run(eng, **over):
    kw = dict(
        max_tokens=1,
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
        enable_thinking=None,
        thinking_budget=None,
        cancel_event=None,
        stats=SimpleNamespace(),
    )
    kw.update(over)
    return list(VLMEngine._runner_events_impl(eng, mx.array([1, 2, 3]), **kw))


def _masked(seen):
    (proc,) = [
        p for p in seen["logits_processors"] if isinstance(p, TokenMaskProcessor)
    ]
    return set(proc._suppress.tolist())


def test_model_default_suppress_tokens_are_masked(tmp_path):
    eng, seen = _engine(tmp_path, [5, 9, -1, 5000])  # -1 and 5000 are not valid ids
    _run(eng)
    assert _masked(seen) == {5, 9}


def test_request_suppress_tokens_win(tmp_path):
    eng, seen = _engine(tmp_path, [5, 9])
    _run(eng, suppress_tokens=[3])
    assert _masked(seen) == {3}


def test_no_generation_config_adds_no_processor(tmp_path):
    eng, seen = _engine(tmp_path, None)
    _run(eng)
    assert not any(isinstance(p, TokenMaskProcessor) for p in seen["logits_processors"])
