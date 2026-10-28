"""ConstrainedSampler normalises the tokenizer's EOS ids once, not per token; behaviour unchanged."""

import mlx.core as mx

import yunshu_engine.json_schema as js


class _Tok:
    eos_token_ids = [2, 3]
    eos_token_id = 2


class _Constraint:
    def __init__(self):
        self.advanced = []

    def allowed_mask(self, tokenizer, vocab):
        return mx.array([True] * vocab)

    def get_allowed_tokens(self, tokenizer, generated):
        return [0, 1, 2, 3]

    def advance_token(self, tid):
        self.advanced.append(tid)


def test_eos_ids_normalised_once_and_eos_not_advanced(monkeypatch):
    calls = []
    real = js.normalize_eos_ids
    monkeypatch.setattr(js, "normalize_eos_ids", lambda t: calls.append(1) or real(t))
    seq = iter([0, 1, 2, 3, 1])
    constraint = _Constraint()
    sampler = js.ConstrainedSampler(lambda lg: mx.array(next(seq)), constraint, _Tok())
    for _ in range(5):
        sampler(mx.zeros((1, 4)))
    assert len(calls) == 1
    # EOS tokens (2, 3) never advance the constraint; the others do
    assert constraint.advanced == [0, 1, 1]
