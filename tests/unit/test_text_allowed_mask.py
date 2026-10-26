"""The text path's ConstrainedSampler uses a constraint's ``allowed_mask`` when it has one instead of
expanding the allowed set into an id list and scattering it back every token (port of 490898d; ~17 ms
per token at a 248K vocab for llguidance and the JSON string states)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from yunshu_engine.constraint_eos import ConstrainedDecodingError
from yunshu_engine.json_schema import ConstrainedSampler

V = 16


class _Tok:
    eos_token_id = 1
    eos_token_ids = {1}

    def decode(self, ids, **kw):
        return "x"


class _MaskConstraint:
    def __init__(self, allowed, mask_ok=True):
        self._allowed = allowed
        self._mask_ok = mask_ok
        self.list_calls = 0
        self.advanced = []

    def allowed_mask(self, tokenizer, vocab_size):
        if not self._mask_ok:
            return None
        m = [False] * vocab_size
        for t in self._allowed:
            m[t] = True
        return mx.array(m)

    def get_allowed_tokens(self, tokenizer, generated_ids):
        self.list_calls += 1
        return list(self._allowed)

    def advance(self, text):
        self.advanced.append(text)


def _logits():
    base = mx.arange(V, dtype=mx.float32)  # token 15 is the unconstrained argmax
    return base.reshape(1, V)


def _argmax(x):
    return mx.argmax(x, axis=-1)


def test_mask_path_masks_without_the_id_list():
    c = _MaskConstraint([3, 4, 7])
    tok = ConstrainedSampler(_argmax, c, _Tok())(_logits())
    assert int(tok.item()) == 7  # best allowed
    assert c.list_calls == 0


def test_no_mask_falls_back_to_the_id_list():
    c = _MaskConstraint([3, 4, 7], mask_ok=False)
    tok = ConstrainedSampler(_argmax, c, _Tok())(_logits())
    assert int(tok.item()) == 7
    assert c.list_calls == 1


def test_mask_over_neg_inf_logits_is_a_dead_end():
    c = _MaskConstraint([3])
    logits = mx.full((1, V), -float("inf"))
    with pytest.raises(ConstrainedDecodingError):
        ConstrainedSampler(_argmax, c, _Tok())(logits)
