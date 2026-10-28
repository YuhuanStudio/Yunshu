"""B06 (EOS shapes) and B07 (dead end must not fall back to argmax)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from yunshu_engine.constraint_eos import ConstrainedDecodingError, normalize_eos_ids
from yunshu_engine.grammar_bitmask import BitmaskApplicator
from yunshu_engine.grammar_constraint import ChoiceConstraint, RegexConstraint
from yunshu_engine.json_schema import apply_json_constraint


class _Tok:
    def __init__(self, **kw):
        self.vocab = {"a": 0, "b": 1, "yes": 2, "<eos>": 3}
        for k, v in kw.items():
            setattr(self, k, v)

    def get_vocab(self):
        return self.vocab

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv[i] for i in ids)


SHAPES = [
    ({"eos_token_ids": 3}, [3]),
    ({"eos_token_ids": [3]}, [3]),
    ({"eos_token_ids": {3}}, [3]),
    ({"eos_token_ids": (3, 3)}, [3]),
    ({"eos_token_ids": None, "eos_token_id": 3}, [3]),
    ({"eos_token_id": 3}, [3]),
    ({"eos_token_ids": []}, []),
    ({}, []),
]


@pytest.mark.parametrize("attrs,expected", SHAPES)
def test_normalize_eos_ids(attrs, expected):
    assert normalize_eos_ids(_Tok(**attrs)) == expected


@pytest.mark.parametrize("attrs,expected", SHAPES)
def test_choice_done_allows_eos_for_every_shape(attrs, expected):
    c = ChoiceConstraint(["yes"])
    c.advance("yes")
    assert c.get_allowed_tokens(_Tok(**attrs), []) == expected


def test_regex_done_with_int_eos():
    c = RegexConstraint("yes")
    c.advance("yes")
    assert c.get_allowed_tokens(_Tok(eos_token_ids=3), []) == [3]


def test_apply_json_constraint_empty_raises():
    with pytest.raises(ConstrainedDecodingError):
        apply_json_constraint(mx.array([[1.0, 5.0, 3.0]]), [])


def test_apply_json_constraint_all_allowed_neg_inf_raises():
    neg = float("-inf")
    with pytest.raises(ConstrainedDecodingError):
        apply_json_constraint(mx.array([[neg, 5.0, 3.0]]), [0])


def test_apply_json_constraint_out_of_vocab_only_raises():
    with pytest.raises(ConstrainedDecodingError):
        apply_json_constraint(mx.array([[1.0, 5.0]]), [99])


def test_bitmask_empty_raises_not_argmax():
    app = BitmaskApplicator(4)
    with pytest.raises(ConstrainedDecodingError):
        app.apply(mx.array([1.0, 9.0, 2.0, 3.0]), mx.zeros((4,), dtype=mx.bool_))


def test_bitmask_allowed_but_neg_inf_raises():
    app = BitmaskApplicator(3)
    mask = mx.array([True, False, False])
    with pytest.raises(ConstrainedDecodingError):
        app.apply(mx.array([float("-inf"), 9.0, 2.0]), mask)


def test_bitmask_padding_positions_stay_blocked():
    app = BitmaskApplicator(2)
    out = app.apply(mx.array([1.0, 2.0, 99.0]), mx.array([True, True]))
    assert float(out[2]) == float("-inf")


class _MaskConstraint:
    """A constraint that hands out a boolean mask (llguidance-style) next to its allowed list."""

    def __init__(self, allowed, width):
        self.allowed = allowed
        self.width = width

    def allowed_mask(self, tokenizer, vocab):
        return mx.array([i in self.allowed for i in range(vocab)])

    def get_allowed_tokens(self, tokenizer, generated):
        return list(self.allowed)


@pytest.mark.parametrize(
    "allowed, logits",
    [
        ([], [1.0, 5.0, 3.0]),  # no allowed token
        ([0], [float("-inf"), 5.0, 3.0]),  # every allowed token banned by a processor
        ([1], [1.0, 5.0, 3.0]),  # fine
        ([0, 2], [1.0, 5.0, 3.0]),  # fine
    ],
)
def test_mask_path_is_the_same_dead_end_rule_as_the_allowlist_path(allowed, logits):
    """The mask fast path (ConstrainedSampler) must fail exactly where the allowlist path
    (apply_json_constraint, unchanged since before the mask path) fails, and mask the same."""
    from yunshu_engine.json_schema import apply_allowed_mask_from

    x = mx.array([logits])
    try:
        want = apply_json_constraint(x, allowed)
    except ConstrainedDecodingError:
        want = None
    try:
        got = apply_allowed_mask_from(_MaskConstraint(allowed, 3), None, x)
    except ConstrainedDecodingError:
        got = None
    assert (got is None) == (want is None)
    if want is not None:
        assert got.tolist() == want.tolist()
