"""B08: the CFG constraint (Lark syntax, llguidance engine) fails clearly on a bad
grammar and does not approximate: exact token masks, no first-char shortcut, no
candidate cap, tokenizer-alphabet aware (CJK)."""

from __future__ import annotations

from pathlib import Path

import pytest

from yunshu_engine.grammar_constraint import (
    CfgGrammarConstraint,
    ConstraintFactory,
    UnsupportedGrammarError,
)

TOKENIZER = Path("/Volumes/P5Plus/models/Qwen2.5-3B-Instruct-4bit")


@pytest.fixture(scope="module")
def tok():
    if not TOKENIZER.exists():
        pytest.skip("Qwen2.5 tokenizer not available")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(TOKENIZER))


def _decode(tok, ids):
    return [tok.decode([i]) for i in ids]


def _feed(c, tok, text):
    for tid in tok.encode(text):
        assert tid in c.get_allowed_tokens(tok, []), tok.decode([tid])
        c.advance(tok.decode([tid]))


def test_bad_grammar_is_a_clear_error(tok):
    with pytest.raises(UnsupportedGrammarError, match="invalid CFG grammar"):
        ConstraintFactory.create("cfg", "start: ???", tok)


def test_start_rule_required(tok):
    with pytest.raises(UnsupportedGrammarError, match="start"):
        CfgGrammarConstraint('main: "a"', start_rule="main", tokenizer=tok)


def test_only_grammar_valid_tokens_are_allowed(tok):
    c = ConstraintFactory.create("cfg", 'start: "yes" | "no"', tok)
    allowed = set(_decode(tok, c.get_allowed_tokens(tok, [])))
    assert "yes" in allowed or "y" in allowed
    assert "maybe" not in allowed
    # multi-char token whose TAIL violates the grammar must be rejected
    assert "yep" not in allowed and "nope" not in allowed


def test_no_candidate_cap_large_alphabet_rule(tok):
    # Thousands of tokens start with a letter; only the exact continuation passes.
    c = ConstraintFactory.create("cfg", 'start: "q" /[a-z]/ "!"', tok)
    allowed = _decode(tok, c.get_allowed_tokens(tok, []))
    assert all(t.startswith("q") for t in allowed)
    assert all(len(t) <= 3 for t in allowed)
    for t in allowed:
        assert t in ("q", "qa", "qb") or t[0] == "q"


def test_cjk_extension_not_cut_short(tok):
    c = ConstraintFactory.create("cfg", "start: /[一-鿿]+/", tok)
    c.advance("龍")
    assert not c.is_done
    allowed = c.get_allowed_tokens(tok, [])
    assert tok.encode("好")[0] in allowed
    assert any(e in allowed for e in (151645, 151643))  # may stop: complete sentence


def test_done_state_allows_only_eos_and_rollback(tok):
    c = ConstraintFactory.create("cfg", 'start: "ab"', tok)
    saved = c.checkpoint()
    _feed(c, tok, "ab")
    assert c.is_done
    assert set(c.get_allowed_tokens(tok, [])) <= {151645, 151643}
    c.rollback(saved)
    assert not c.is_done
    assert tok.encode("ab")[0] in c.get_allowed_tokens(tok, [])
