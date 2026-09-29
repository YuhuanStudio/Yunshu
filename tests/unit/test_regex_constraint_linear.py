"""RegexConstraint must cost O(token) per step, not O(buffer) x vocab.

The Anthropic router forces ``<think>...</think>`` on text-only models with the pattern
below. The old constraint re-walked the whole buffer for every candidate token and ran a
fresh reachability BFS per candidate, so a 1500-token generation on a real vocabulary never
finished (CPU-bound Python, GPU idle).
"""

import time

from yunshu_engine.grammar_constraint import RegexConstraint

PATTERN = r"<think>[\s\S]+?</think>[\s\S]*"


class FakeTokenizer:
    """Vocab of single chars, common words and the tag pieces, plus padding tokens."""

    def __init__(self, extra: int = 4000):
        words = [
            "<think>",
            "</think>",
            "<",
            ">",
            "think",
            "/",
            " ",
            "\n",
            "a",
            "b",
            "x",
        ]
        words += [f"w{i}" for i in range(extra)]
        words.append("<eos>")
        self._vocab = {w: i for i, w in enumerate(words)}
        self._inv = {i: w for w, i in self._vocab.items()}
        self.eos_token_id = len(words) - 1

    def get_vocab(self):
        return dict(self._vocab)

    def decode(self, ids):
        return "".join(self._inv.get(i, "") for i in ids)


def _walk(constraint, tok, text_tokens):
    for t in text_tokens:
        allowed = set(constraint.get_allowed_tokens(tok, []))
        tid = tok._vocab[t]
        assert tid in allowed, (t, sorted(tok._inv[i] for i in allowed)[:10])
        constraint.advance(t)


def test_forced_thinking_pattern_semantics():
    tok = FakeTokenizer(50)
    c = RegexConstraint(PATTERN)
    first = {tok._inv[i] for i in c.get_allowed_tokens(tok, [])}
    assert "<think>" in first and "a" not in first and "</think>" not in first
    _walk(c, tok, ["<think>", "a", "b", "</think>", "x", "w7", "\n"])
    # After the closing tag the rest is free text and EOS is legal (full match).
    allowed = set(c.get_allowed_tokens(tok, []))
    assert tok.eos_token_id in allowed and tok._vocab["w3"] in allowed


def test_eos_allowed_once_the_block_is_closed():
    tok = FakeTokenizer(20)
    c = RegexConstraint(PATTERN)
    _walk(c, tok, ["<think>", "a", "</think>", "b"])
    assert tok.eos_token_id in set(c.get_allowed_tokens(tok, []))


def test_whitespace_is_inside_any_char_class():
    tok = FakeTokenizer(20)
    c = RegexConstraint(PATTERN)
    _walk(c, tok, ["<think>", "a", " ", "\n", "b"])


def test_checkpoint_rollback():
    tok = FakeTokenizer(20)
    c = RegexConstraint(PATTERN)
    _walk(c, tok, ["<think>", "a"])
    saved = c.checkpoint()
    _walk(c, tok, ["</think>", "x"])
    c.rollback(saved)
    assert tok._vocab["</think>"] in set(c.get_allowed_tokens(tok, []))
    c.reset()
    assert tok._vocab["<think>"] in set(c.get_allowed_tokens(tok, []))


def test_long_generation_stays_fast():
    """1500 steps over a 4k vocab: linear in steps, so well under the budget."""
    tok = FakeTokenizer(4000)
    c = RegexConstraint(PATTERN)
    body = ["<think>"] + ["a", " ", "b"] * 400 + ["</think>"] + ["w1", " "] * 150
    t0 = time.perf_counter()
    for t in body:
        assert c.get_allowed_tokens(tok, [])
        c.advance(t)
    assert time.perf_counter() - t0 < 5.0
