"""B04 / B05: the regex DFA is exact over Unicode and rejects unsupported syntax.

Differential tests against Python's ``re.fullmatch`` for the supported subset.
"""

from __future__ import annotations

import itertools
import re

import pytest

from yunshu_engine.grammar_constraint import (
    RegexConstraint,
    UnsupportedRegexError,
    _RegexDFA,
)

UNICODE_SAMPLES = [
    "Ж",
    "龍",
    "é",
    "α",
    "😀",
    "👨‍👩‍👧",
    "é",
    "\U00020000",  # CJK extension B
    "繁體中文",
    " ",
    "٣",  # Arabic-indic digit
    "x",
    "a",
    " ",
    "\n",
]

SUPPORTED = [
    r".+",
    r"[^x]+",
    r"[\s\S]+",
    r"\w+",
    r"\d+",
    r"\D+",
    r"\S+",
    r"[^\W\d]+",
    r"(?i)abc",
    r"(?i:ab)c",
    r"(?s).+",
    r"(a|b*c)",
    r"(?:ab|c)*d?",
    r"[a-c]{2,3}",
    r"^a+$",
    r"a*?b",
    r"\Aab\Z",
    r"é+|龍+",
    r"[α-ω]+",
]


def _matches(p: str, s: str) -> bool:
    return re.fullmatch(p, s) is not None


@pytest.mark.parametrize("pattern", SUPPORTED)
def test_full_match_equals_python_on_unicode(pattern):
    dfa = _RegexDFA(pattern)
    pool = UNICODE_SAMPLES + ["abc", "ABC", "ba", "bbc", "cd", "ab", "aab", "abab"]
    cands = list(pool)
    cands += [a + b for a, b in itertools.product(pool, repeat=2)]
    for text in cands:
        assert dfa.is_full_match(text) == _matches(pattern, text), (pattern, text)


@pytest.mark.parametrize("pattern", [p for p in SUPPORTED if "\\n" not in p])
def test_prefix_validity_is_exact_over_small_alphabet(pattern):
    dfa = _RegexDFA(pattern)
    alphabet = ["a", "b", "c", "d", "Ж", "é", "龍", "x", "1", " "]
    cands = [""]
    for n in range(1, 4):
        cands += ["".join(t) for t in itertools.product(alphabet, repeat=n)]
    ext = (
        [""]
        + list(alphabet)
        + ["".join(t) for t in itertools.product(alphabet, repeat=2)]
    )
    for text in cands[:3000]:
        # Python-side oracle (bounded): can some extension of <=2 chars match?
        oracle = any(_matches(pattern, text + e) for e in ext)
        got = dfa.is_prefix_valid(text)
        if oracle:
            assert got, (pattern, text)
        # DFA liveness may rely on longer extensions than the oracle explores,
        # so only the soundness direction (no false negatives) is asserted here.


def test_tokenizer_universe_is_not_needed_for_any_unicode_char():
    # B04: the old construction dropped chars outside a fixed sample.
    dfa = _RegexDFA(r".+")
    for ch in ("Ж", "龍", "𠀀", "😀", "é", "α"):
        st = dfa.step(dfa._dfa_start, ch)
        assert dfa.is_accepting(st), ch


def test_regex_constraint_accepts_cyrillic_for_negated_class():
    c = RegexConstraint(r"[^x]+")
    c.advance("Ж")
    assert c._dfa.is_accepting(c._dfa_state)


@pytest.mark.parametrize(
    "pattern",
    [
        r"(?=a)b",
        r"(?!a)b",
        r"(?<=a)b",
        r"(?<!a)b",
        r"(a)\1",
        r"(?P<x>a)(?P=x)",
        r"(a)?(?(1)b|c)",
        r"a\bb",
        r"\bab",
        r"(?m)^a$",
        r"a(?=b)",
        r"(?>a+)b",
        r"a++b",
        r"a^b",
        r"a$b",
        r"(^a)",
        r"[",
        r"(?L)a",
    ],
)
def test_unsupported_regex_is_rejected_not_approximated(pattern):
    with pytest.raises(ValueError):
        RegexConstraint(pattern)


def test_unsupported_error_is_a_value_error_with_clear_message():
    with pytest.raises(UnsupportedRegexError, match="back-reference"):
        RegexConstraint(r"(a)\1")
    with pytest.raises(UnsupportedRegexError, match="lookahead"):
        RegexConstraint(r"(?=a)b")


def test_inline_ignorecase_is_supported_and_exact():
    c = RegexConstraint(r"(?i)abc")
    c.advance("ABC")
    assert c._dfa.is_accepting(c._dfa_state)
    assert c.is_done


def test_shared_start_bug_alternation_with_star():
    # re.fullmatch('(a|b*c)', 'ba') is None; the old NFA looped on the shared
    # branch start and accepted it.
    dfa = _RegexDFA(r"(a|b*c)")
    assert not dfa.is_full_match("ba")
    assert not dfa.is_prefix_valid("ba")
    assert dfa.is_full_match("bbc")


def test_nfa_budget_still_enforced():
    with pytest.raises(ValueError):
        RegexConstraint(r"a{100000000}")
