"""Runner event stream: reasoning state when the prompt pre-opens <think>."""

from types import SimpleNamespace

import mlx.core as mx

from yunshu_engine.vlm_batch_runner import RunStats
from yunshu_engine.vlm_engine import VLMEngine

THINK, END_THINK, EOS = 900, 901, 2
PIECES = {1: "a", 3: "b", 4: " answer", THINK: "<think>", END_THINK: "</think>"}


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
        return {"<think>": [THINK], "</think>": [END_THINK]}[text]


def _events(prompt_tail, generated):
    eng = VLMEngine.__new__(VLMEngine)
    eng._tokenizer = _Tok()
    eng._get_eos_ids = lambda: [EOS]

    def iter_tokens(input_ids, stats, **_):
        stats.prompt_tokens = int(input_ids.shape[0])
        for t in generated:
            stats.generated += 1
            yield t

    eng._batch_runner = SimpleNamespace(iter_tokens=iter_tokens)
    return list(
        eng._runner_events(
            mx.array([5, 6] + prompt_tail),
            max_tokens=32, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None,
            stop=None, stop_token_ids=None, repetition_penalty=1.0,
            frequency_penalty=0.0, presence_penalty=0.0, logit_bias=None,
            json_schema=None, enable_thinking=True, thinking_budget=None,
            cancel_event=None, stats=RunStats(),
        )
    )


def test_prompt_opened_think_starts_in_reasoning_and_hides_tags():
    ev = _events([THINK], [1, 3, END_THINK, 4, EOS])
    shown = [(e[0], e[2]) for e in ev if e[0]]
    assert shown == [("a", "reasoning"), ("b", "reasoning"), (" answer", "normal")]
    assert all("think>" not in e[0] for e in ev)
    assert ev[-1][3] == "stop"


def test_closed_think_in_prompt_starts_normal():
    ev = _events([THINK, END_THINK], [4, EOS])
    assert [(e[0], e[2]) for e in ev if e[0]] == [(" answer", "normal")]
