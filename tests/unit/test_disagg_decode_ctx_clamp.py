"""(MED): the disagg KV-reuse decode path had no context-window clamp.

_decode_from_kv_sync loops range(max_tokens) with the prompt already prefilled at offset
len(token_ids). The normal/streaming generate paths clamp max_tokens to the model's context
window, but this reuse path bypassed it — a near-context-length prefilled prompt + a
large max_tokens would decode past max_position_embeddings into RoPE-extrapolation garbage.
Now clamped to max(0, max_ctx - len(token_ids)).
"""

from __future__ import annotations

import inspect

import mlx.core as mx

from yunshu_engine import batched_engine
from yunshu_engine.batched_engine import BatchedEngine


class _FakeModel:
    def __init__(self, max_ctx):
        self.max_position_embeddings = max_ctx

    def __call__(self, x, cache=None):
        return mx.zeros((1, 1, 4))  # greedy argmax → token 0 every step


class _FakeTok:
    eos_token_id = 99  # never emitted (argmax of zeros = 0)

    def decode(self, ids):
        return "".join(chr(97 + (i % 26)) for i in ids) if ids else ""


def _engine(max_ctx):
    e = BatchedEngine.__new__(BatchedEngine)
    e._model = _FakeModel(max_ctx)
    e._tokenizer = _FakeTok()
    return e


def _decode(e, n_prompt, max_tokens):
    return e._decode_from_kv_sync(
        kv_cache=None,
        token_ids=[1] * n_prompt,
        first_logits=mx.zeros((1, 4)),
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        seed=None,
        stop_token_ids=None,
        stop=None,
    )


def test_clamped_to_remaining_window():
    # ctx=15, prompt=10 → at most 5 tokens may be generated regardless of max_tokens=100
    out = _decode(_engine(15), n_prompt=10, max_tokens=100)
    assert out.completion_tokens == 5


def test_clamped_to_zero_when_prompt_fills_context():
    out = _decode(_engine(10), n_prompt=10, max_tokens=100)
    assert out.completion_tokens == 0
    assert out.text == ""


def test_not_clamped_when_room_available():
    # ctx=50, prompt=2, max_tokens=3 → 3 < 48 room → unchanged
    out = _decode(_engine(50), n_prompt=2, max_tokens=3)
    assert out.completion_tokens == 3


def test_no_clamp_when_context_undeterminable():
    # max_ctx 0 (no attr) → no clamp → generates the full request
    e = BatchedEngine.__new__(BatchedEngine)

    class _NoCtxModel:
        def __call__(self, x, cache=None):
            return mx.zeros((1, 1, 4))

    e._model = _NoCtxModel()
    e._tokenizer = _FakeTok()
    out = _decode(e, n_prompt=10, max_tokens=4)
    assert out.completion_tokens == 4


def test_source_uses_resolve_model_max_ctx():
    src = inspect.getsource(batched_engine.BatchedEngine._decode_from_kv_sync)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    assert "_resolve_model_max_ctx(model)" in code
    assert "_max_ctx - len(token_ids)" in code
