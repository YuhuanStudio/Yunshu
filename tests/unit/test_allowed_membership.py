"""``allowed_membership`` answers exactly ``tid in get_allowed_tokens(...)``.

The n-gram draft filter built ``set(get_allowed_tokens(...))`` from a ~240K-id list for
every draft position; the predicate reads the constraint's mask instead, and must give
the same answer for every id at every step (llguidance and the in-house JSON engine).
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("llguidance")

from yunshu_engine.json_schema import allowed_membership  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    corpus = [
        json.dumps({"name": "Ada Lovelace", "age": 36, "tags": ["math", "poetry"]}),
        "The quick brown fox jumps over the lazy dog 0123456789",
    ] * 20
    t.train_from_iterator(
        corpus,
        trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=["<eos>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    return PreTrainedTokenizerFast(tokenizer_object=t, eos_token="<eos>")


SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
    "additionalProperties": False,
}


def _walk(tok, c, text):
    steps = 0
    for tid in tok.encode(text):
        allowed = set(c.get_allowed_tokens(tok, []))
        member = allowed_membership(c, tok, [])
        assert member is not None
        assert {i for i in range(len(tok) + 5) if member(i)} == allowed
        assert member(tid)
        c.advance(tok.decode([tid]))
        steps += 1
    return steps


def test_llguidance_membership_matches_the_allowlist(tok):
    from yunshu_engine.grammar_constraint import LlgJsonSchemaConstraint

    c = LlgJsonSchemaConstraint(SCHEMA, tokenizer=tok)
    assert _walk(tok, c, '{"name": "Ada", "age": 36}') > 3


def test_inhouse_membership_matches_the_allowlist(tok):
    from yunshu_engine.json_schema import JsonSchemaConstraint

    c = JsonSchemaConstraint(SCHEMA)
    assert _walk(tok, c, '{"name": "Ada", "age": 36}') > 3


def test_constraint_without_a_mask_uses_the_list():
    class ListOnly:
        def get_allowed_tokens(self, tokenizer, generated):
            return [2, 5]

    member = allowed_membership(ListOnly(), object(), [])
    assert member(2) and member(5) and not member(3)

    class Nothing:
        def get_allowed_tokens(self, tokenizer, generated):
            return []

    assert allowed_membership(Nothing(), object(), []) is None
