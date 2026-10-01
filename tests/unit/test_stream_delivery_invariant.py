"""Generated == delivered: every generated token's text reaches the consumer.

The 2026-10-01 soak saw chat answers 'J' for 'JADE' (2 completion tokens). The
capture showed the engine had generated only J and EOS -- nothing was dropped on the
way out -- and these tests pin that the runner event stream never loses a piece:
the joined event text equals the detokenized generated tokens for every ending
(EOS, length, stop string) and the debug capture records the same.
"""

import json
from types import SimpleNamespace

import mlx.core as mx
import pytest

from yunshu_engine.vlm_batch_runner import RunStats
from yunshu_engine.vlm_engine import VLMEngine

EOS = 2
PIECES = {1: "J", 3: "ADE", 4: " code", 5: "!", 6: "é"}


class _Detok:
    def __init__(self):
        self.text, self.last_segment = "", ""

    def reset(self):
        self.text = self.last_segment = ""

    def add_token(self, t):
        self.last_segment = PIECES[t]
        self.text += self.last_segment

    def finalize(self):
        self.last_segment = ""


class _Tok:
    detokenizer = _Detok()

    def encode(self, text, add_special_tokens=False):
        return []


def _events(generated, *, max_tokens=32, stop=None):
    eng = VLMEngine.__new__(VLMEngine)
    eng._tokenizer = _Tok()
    eng._get_eos_ids = lambda: [EOS]

    def iter_tokens(input_ids, stats, **_):
        for t in generated:
            stats.generated += 1
            yield t

    eng._batch_runner = SimpleNamespace(iter_tokens=iter_tokens)
    return list(
        eng._runner_events(
            mx.array([5, 6, 7]),
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            seed=None,
            stop=stop,
            stop_token_ids=None,
            repetition_penalty=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            logit_bias=None,
            json_schema=None,
            enable_thinking=False,
            thinking_budget=None,
            cancel_event=None,
            stats=RunStats(),
        )
    )


def _delivered(events):
    return "".join(e[0] for e in events)


@pytest.mark.parametrize(
    "generated, expected, finish",
    [
        ([1, 3, EOS], "JADE", "stop"),
        ([1, EOS], "J", "stop"),
        ([1, 3, 4, 5], "JADE code!", "stop"),
        ([1, 6, 3, EOS], "JéADE", "stop"),
    ],
)
def test_every_generated_piece_is_delivered(generated, expected, finish):
    ev = _events(generated)
    assert _delivered(ev) == expected
    assert ev[-1][3] == finish
    # the finish never precedes content: only the last event carries a reason
    assert all(e[3] is None for e in ev[:-1])


def test_length_finish_keeps_the_last_piece():
    ev = _events([1, 3, 4], max_tokens=2)
    assert _delivered(ev) == "JADE"
    assert ev[-1][3] == "length"


def test_stop_string_delivers_text_before_the_match_only():
    ev = _events([1, 3, 4, 5], stop=["!"])
    assert _delivered(ev) == "JADE code"
    assert ev[-1][3] == "stop"


def test_stop_string_holdback_is_flushed_when_generation_ends_first():
    # "AD" could start the stop string "ADX": held back, then released at EOS
    ev = _events([1, 3, EOS], stop=["ADEX"])
    assert _delivered(ev) == "JADE"


def test_debug_capture_matches_delivery(monkeypatch, tmp_path):
    path = tmp_path / "capture.jsonl"
    monkeypatch.setenv("YUNSHU_DEBUG_STREAM_CAPTURE", str(path))
    ev = _events([1, 3, EOS])
    row = json.loads(path.read_text().splitlines()[-1])
    assert row["token_ids"] == [1, 3, EOS]
    assert row["text"] == _delivered(ev) == "JADE"
    assert row["finish"] == "stop"
    assert row["prompt_tokens"] == 3
