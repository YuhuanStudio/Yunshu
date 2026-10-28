"""llguidance constraints expose ``allowed_mask``, identical to ``get_allowed_tokens``.

Without it the VLM runner expanded the bitmask into a ~240K-id Python list and scattered it
back every constrained token (~17 ms/token at a 248K vocab). The mask must be the same
set: grammar tokens, EOS once accepting, padded head rows off, None when finished.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("llguidance")


@pytest.fixture(scope="module")
def tok():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    corpus = [
        json.dumps({"name": "Ada Lovelace", "age": 36, "tags": ["math", "poetry"]}),
        json.dumps({"city": "Taipei", "ok": True, "n": [1, 2, 3]}),
        "The quick brown fox jumps over the lazy dog 0123456789",
    ] * 20
    t.train_from_iterator(
        corpus,
        trainers.BpeTrainer(
            vocab_size=400,
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


def _walk(tok, c, text, pad):
    vocab = len(tok) + pad
    steps = 0
    for tid in tok.encode(text):
        allowed = set(c.get_allowed_tokens(tok, []))
        mask = c.allowed_mask(tok, vocab)
        assert mask is not None and mask.shape == (vocab,)
        got = set(np.flatnonzero(np.array(mask)).tolist())
        assert got == allowed
        assert tid in got
        c.advance(tok.decode([tid]))
        steps += 1
    # complete: the allowlist is EOS; the mask either carries it or (grammar stopped)
    # returns None so the caller takes that short allowlist
    eos = tok.eos_token_id
    assert eos in c.get_allowed_tokens(tok, [])
    mask = c.allowed_mask(tok, vocab)
    assert mask is None or bool(np.array(mask)[eos])
    return steps


@pytest.mark.parametrize("pad", [0, 7])
def test_json_schema_mask_matches_allowlist(tok, pad):
    from yunshu_engine.grammar_constraint import LlgJsonSchemaConstraint

    c = LlgJsonSchemaConstraint(SCHEMA, tokenizer=tok)
    assert _walk(tok, c, '{"name": "Ada", "age": 36}', pad) > 3


def test_cfg_mask_matches_allowlist(tok):
    from yunshu_engine.grammar_constraint import CfgGrammarConstraint

    c = CfgGrammarConstraint('start: "yes" | "no" | "maybe"', tokenizer=tok)
    _walk(tok, c, "maybe", 3)


def test_finished_grammar_falls_back_to_the_allowlist(tok):
    from yunshu_engine.grammar_constraint import CfgGrammarConstraint

    c = CfgGrammarConstraint('start: "no"', tokenizer=tok)
    for tid in tok.encode("no"):
        c.advance(tok.decode([tid]))
    assert c.allowed_mask(tok, len(tok)) is None
    assert c.get_allowed_tokens(tok, []) == [tok.eos_token_id]


def test_accepting_but_open_grammar_adds_eos(tok):
    from yunshu_engine.grammar_constraint import CfgGrammarConstraint

    # after "yes" the grammar may stop or continue with "!": EOS joins the mask
    c = CfgGrammarConstraint('start: "yes" | "yes!"', tokenizer=tok)
    for tid in tok.encode("yes"):
        c.advance(tok.decode([tid]))
    mask = c.allowed_mask(tok, len(tok))
    assert mask is not None
    got = set(np.flatnonzero(np.array(mask)).tolist())
    assert got == set(c.get_allowed_tokens(tok, []))
    assert tok.eos_token_id in got and len(got) > 1
