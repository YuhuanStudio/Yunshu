from __future__ import annotations

"""Structured output constraints beyond JSON Schema — regex and CFG support.

Provides token-level constraint masking for:
1. RegexConstraint — regex pattern enforcement (e.g., date, email, phone)
2. CfgGrammarConstraint — context-free grammar (Lark syntax) via llguidance
3. ChoiceConstraint — enumeration from a fixed list of strings
4. ConstraintFactory — unified interface matching grammar_type to constraint

All constraints implement the same interface as JsonSchemaConstraint:
- advance(token_text) — update state after each token
- get_allowed_tokens(tokenizer, generated_ids) — return valid next tokens
- is_done — whether generation is complete
- reset() — reset to initial state

Integration: plug into ConstrainedSampler alongside JsonSchemaConstraint.
"""

import logging
import re
import re._parser as _sre_parse
import weakref
from typing import Any

from yunshu_engine.constraint_eos import normalize_eos_ids

logger = logging.getLogger(__name__)


# ── Regex DFA for prefix matching ────────────────────────────────────────────


class UnsupportedRegexError(ValueError):
    """The regex uses syntax outside the supported subset (or is invalid).

    Supported subset (everything is matched with ``re.fullmatch`` semantics):
    literals, ``.``, character classes (ranges, negation, ``\\d \\w \\s`` and their
    negations with Python's Unicode semantics), alternation, capturing /
    non-capturing groups, scoped and global ``i`` / ``s`` / ``a`` / ``u`` / ``x``
    flags, the quantifiers ``* + ? {n} {n,} {n,m}`` (greedy or lazy), and the
    anchors ``^`` / ``\\A`` at the very start and ``$`` / ``\\Z`` at the very end.
    Rejected, never approximated: back-references, lookahead / lookbehind,
    conditionals, atomic groups, possessive quantifiers, word-boundary and
    multiline anchors, anchors in the middle of the pattern, the ``m`` / ``L`` flags.
    """


_REGEX_SUPPORTED_FLAGS = re.IGNORECASE | re.DOTALL | re.ASCII | re.UNICODE | re.VERBOSE


class _CharLabel:
    """A symbolic single-character predicate: no alphabet is ever enumerated."""

    __slots__ = ("ranges", "negate", "pred", "cache")

    def __init__(
        self,
        ranges: list[tuple[int, int]] | None = None,
        negate: bool = False,
        pred: Any = None,
    ) -> None:
        self.ranges = ranges or []
        self.negate = negate
        self.pred = pred  # compiled single-char ``re`` pattern (flags / categories)
        self.cache: dict[int, bool] = {}

    def matches(self, cp: int) -> bool:
        if self.pred is not None:
            hit = self.cache.get(cp)
            if hit is None:
                hit = self.pred.fullmatch(chr(cp)) is not None
                self.cache[cp] = hit
            return hit
        inside = False
        for lo, hi in self.ranges:
            if lo <= cp <= hi:
                inside = True
                break
        return inside != self.negate

    def satisfiable(self) -> bool:
        if self.pred is not None:
            return True
        return bool(self.ranges) or self.negate


class _RegexDFA:
    """Exact, lazily built DFA over a symbolic-alphabet NFA for a regex.

    Single-character transitions are predicates (code-point ranges, negation,
    or Python's own ``re`` for categories / flags), so the automaton is exact
    for every Unicode character, not for a sampled subset.  DFA states are
    subsets of NFA states created on demand while stepping.
    """

    # SECURITY (ReDoS/DoS): a user-supplied regex like "a{100000000}" or
    # nested "(a{5000}){5000}" materializes one NFA copy per repeat; cap states.
    _MAX_NFA_STATES = 200_000

    def __init__(self, pattern: str) -> None:
        self._pattern = pattern
        try:
            self._compiled = re.compile(pattern)
            parsed = _sre_parse.parse(pattern)
        except (re.error, RecursionError, OverflowError) as exc:
            raise UnsupportedRegexError(f"invalid regex: {exc}") from exc
        self._nfa_transitions: dict[int, list[tuple[_CharLabel, int]]] = {}
        self._nfa_epsilon: dict[int, list[int]] = {}
        self._nfa_counter = 0
        self._closure_cache: dict[frozenset[int], frozenset[int]] = {}
        self._trans: dict[frozenset[int], dict[int, frozenset[int] | None]] = {}
        flags = int(parsed.state.flags)
        if flags & re.LOCALE:
            raise UnsupportedRegexError("regex flag L (locale) is not supported")
        if flags & re.MULTILINE:
            raise UnsupportedRegexError("regex flag m (multiline) is not supported")
        s, e = self._seq(list(parsed), flags, parsed.state, top=True)
        self._nfa_accept = e
        self._dfa_start = self._closure(frozenset({s}))
        self._live_nfa = self._compute_live()
        if not (self._dfa_start & self._live_nfa):
            raise UnsupportedRegexError("regex matches no string")

    # ── NFA construction (Thompson, fresh states per fragment) ─────────────

    def _new_state(self) -> int:
        if self._nfa_counter >= self._MAX_NFA_STATES:
            raise UnsupportedRegexError(
                f"regex too complex: NFA exceeds {self._MAX_NFA_STATES} states "
                "(repeat counts too large)"
            )
        s = self._nfa_counter
        self._nfa_counter += 1
        return s

    def _eps(self, a: int, b: int) -> None:
        self._nfa_epsilon.setdefault(a, []).append(b)

    def _edge(self, a: int, label: _CharLabel, b: int) -> None:
        self._nfa_transitions.setdefault(a, []).append((label, b))

    def _pred_label(self, state: Any, item: tuple, flags: int) -> _CharLabel:
        import re._compiler as _sre_compile

        sub = _sre_parse.SubPattern(state, [item])
        try:
            pred = _sre_compile.compile(sub, flags)
        except re.error as exc:  # pragma: no cover - defensive
            raise UnsupportedRegexError(f"invalid regex: {exc}") from exc
        return _CharLabel(pred=pred)

    def _label_for(self, op: Any, av: Any, flags: int, state: Any) -> _CharLabel:
        icase = bool(flags & re.IGNORECASE)
        if op == _sre_parse.ANY:
            if flags & re.DOTALL:
                return _CharLabel(negate=True)
            return _CharLabel([(10, 10)], negate=True)
        if icase:
            return self._pred_label(state, (op, av), flags)
        if op == _sre_parse.LITERAL:
            return _CharLabel([(av, av)])
        if op == _sre_parse.NOT_LITERAL:
            return _CharLabel([(av, av)], negate=True)
        # IN
        ranges: list[tuple[int, int]] = []
        negate = False
        for iop, iav in av:
            if iop == _sre_parse.NEGATE:
                negate = True
            elif iop == _sre_parse.LITERAL:
                ranges.append((iav, iav))
            elif iop == _sre_parse.RANGE:
                ranges.append((iav[0], iav[1]))
            elif iop == _sre_parse.CATEGORY:
                return self._pred_label(state, (op, av), flags)
            else:
                raise UnsupportedRegexError(
                    f"unsupported character-class element {iop}"
                )
        return _CharLabel(ranges, negate)

    def _seq(
        self, items: list, flags: int, state: Any, top: bool = False
    ) -> tuple[int, int]:
        s = self._new_state()
        cur = s
        n = len(items)
        for idx, (op, av) in enumerate(items):
            if op == _sre_parse.AT:
                begin = av in (_sre_parse.AT_BEGINNING, _sre_parse.AT_BEGINNING_STRING)
                end = av in (_sre_parse.AT_END, _sre_parse.AT_END_STRING)
                if (
                    top
                    and begin
                    and all(
                        i[0] == _sre_parse.AT
                        and i[1]
                        in (_sre_parse.AT_BEGINNING, _sre_parse.AT_BEGINNING_STRING)
                        for i in items[:idx]
                    )
                ):
                    continue
                if (
                    top
                    and end
                    and all(
                        i[0] == _sre_parse.AT
                        and i[1] in (_sre_parse.AT_END, _sre_parse.AT_END_STRING)
                        for i in items[idx + 1 : n]
                    )
                ):
                    continue
                raise UnsupportedRegexError(
                    "unsupported regex anchor (only ^/\\A at the very start and "
                    "$/\\Z at the very end are supported)"
                )
            fs, fe = self._item(op, av, flags, state)
            self._eps(cur, fs)
            cur = fe
        return s, cur

    def _item(self, op: Any, av: Any, flags: int, state: Any) -> tuple[int, int]:
        if op in (
            _sre_parse.LITERAL,
            _sre_parse.NOT_LITERAL,
            _sre_parse.ANY,
            _sre_parse.IN,
        ):
            label = self._label_for(op, av, flags, state)
            s, e = self._new_state(), self._new_state()
            if label.satisfiable():
                self._edge(s, label, e)
            return s, e
        if op == _sre_parse.BRANCH:
            s, e = self._new_state(), self._new_state()
            for branch in av[1]:
                fs, fe = self._seq(list(branch), flags, state)
                self._eps(s, fs)
                self._eps(fe, e)
            return s, e
        if op == _sre_parse.SUBPATTERN:
            _group, add, delete, sub = av
            new_flags = (flags | add) & ~delete
            if new_flags & re.LOCALE or new_flags & re.MULTILINE:
                raise UnsupportedRegexError("regex flag L/m is not supported")
            return self._seq(list(sub), new_flags, state)
        if op in (_sre_parse.MAX_REPEAT, _sre_parse.MIN_REPEAT):
            lo, hi, sub = av
            sub_items = list(sub)
            s = self._new_state()
            cur = s
            for _ in range(lo):
                fs, fe = self._seq(sub_items, flags, state)
                self._eps(cur, fs)
                cur = fe
            if hi == _sre_parse.MAXREPEAT:
                fs, fe = self._seq(sub_items, flags, state)
                self._eps(cur, fs)
                self._eps(fe, cur)
                return s, cur
            e = self._new_state()
            for _ in range(hi - lo):
                self._eps(cur, e)
                fs, fe = self._seq(sub_items, flags, state)
                self._eps(cur, fs)
                cur = fe
            self._eps(cur, e)
            return s, e
        names = {
            getattr(_sre_parse, n): n
            for n in (
                "ASSERT",
                "ASSERT_NOT",
                "GROUPREF",
                "GROUPREF_EXISTS",
                "ATOMIC_GROUP",
                "POSSESSIVE_REPEAT",
            )
            if hasattr(_sre_parse, n)
        }
        what = {
            "ASSERT": "lookahead/lookbehind",
            "ASSERT_NOT": "negative lookahead/lookbehind",
            "GROUPREF": "back-reference",
            "GROUPREF_EXISTS": "conditional group",
            "ATOMIC_GROUP": "atomic group",
            "POSSESSIVE_REPEAT": "possessive quantifier",
        }.get(names.get(op, ""), f"construct {op}")
        raise UnsupportedRegexError(f"unsupported regex syntax: {what}")

    def _compute_live(self) -> frozenset[int]:
        reverse: dict[int, list[int]] = {}
        for a, targets in self._nfa_epsilon.items():
            for b in targets:
                reverse.setdefault(b, []).append(a)
        for a, edges in self._nfa_transitions.items():
            for _label, b in edges:
                reverse.setdefault(b, []).append(a)
        live = {self._nfa_accept}
        stack = [self._nfa_accept]
        while stack:
            node = stack.pop()
            for prev in reverse.get(node, ()):
                if prev not in live:
                    live.add(prev)
                    stack.append(prev)
        return frozenset(live)

    # ── lazy DFA ────────────────────────────────────────────────────────────

    def _closure(self, states: frozenset[int]) -> frozenset[int]:
        cached = self._closure_cache.get(states)
        if cached is not None:
            return cached
        closure = set(states)
        stack = list(states)
        while stack:
            s = stack.pop()
            for ns in self._nfa_epsilon.get(s, ()):
                if ns not in closure:
                    closure.add(ns)
                    stack.append(ns)
        out = frozenset(closure)
        if len(self._closure_cache) < 100_000:
            self._closure_cache[states] = out
        return out

    def _next(self, state: frozenset[int], cp: int) -> frozenset[int] | None:
        targets: set[int] = set()
        for n in state:
            for label, t in self._nfa_transitions.get(n, ()):
                if t in self._live_nfa and label.matches(cp):
                    targets.add(t)
        if not targets:
            return None
        return self._closure(frozenset(targets))

    @property
    def has_dfa(self) -> bool:
        return True

    def step(self, state: frozenset[int] | None, text: str) -> frozenset[int] | None:
        """Advance ``state`` over ``text``; None when a character has no transition."""
        for ch in text:
            if state is None:
                return None
            cp = ord(ch)
            memo = self._trans.get(state)
            if memo is None:
                memo = self._trans[state] = {}
            if cp in memo:
                state = memo[cp]
            else:
                nxt = self._next(state, cp)
                if len(memo) < 4096:
                    memo[cp] = nxt
                state = nxt
        return state

    def is_accepting(self, state: frozenset[int] | None) -> bool:
        return state is not None and self._nfa_accept in state

    def is_live(self, state: frozenset[int] | None) -> bool:
        """True when an accept state is reachable from ``state``."""
        return state is not None and not self._live_nfa.isdisjoint(state)

    def can_extend(self, state: frozenset[int] | None) -> bool:
        """Exact: can ANY character continue toward an accepting string?"""
        if state is None:
            return False
        for n in state:
            for _label, t in self._nfa_transitions.get(n, ()):
                if t in self._live_nfa:
                    return True
        return False

    def valid_next_chars_from(
        self, state: frozenset[int] | None, char_range: Any
    ) -> set[int]:
        if state is None:
            return set()
        edges = [
            label
            for n in state
            for label, t in self._nfa_transitions.get(n, ())
            if t in self._live_nfa
        ]
        if not edges:
            return set()
        return {cp for cp in char_range if any(lb.matches(cp) for lb in edges)}

    def is_prefix_valid(self, text: str) -> bool:
        """True when ``text`` can still be extended to a full match."""
        return self.is_live(self.step(self._dfa_start, text))

    def is_full_match(self, text: str) -> bool:
        return self.is_accepting(self.step(self._dfa_start, text))

    def valid_next_chars(self, text: str, char_range: Any) -> set[int]:
        return self.valid_next_chars_from(self.step(self._dfa_start, text), char_range)


class RegexConstraint:
    """Constrains output to match a regex pattern.

    Uses a DFA (Deterministic Finite Automaton) built from the regex
    pattern to perform correct prefix-validity checks. The DFA approach
    fixes the fundamental flaw in using re.match() for prefix checking:
    patterns like \\d{3}, \\d+-\\d+, or a+bc would fail because re.match()
    requires the entire pattern to consume the string from the start,
    while a partial string (e.g., '1' for \\d{3}) cannot be matched by
    a pattern that has a minimum length requirement.

    The DFA is built once at construction time and reused for every
    character validation, making per-step cost O(alphabet_size) instead
    of O(alphabet_size * pattern_complexity).

    Supports checkpoint/rollback for speculative decoding.
    """

    def __init__(self, pattern: str) -> None:
        self._pattern = pattern
        # Build the DFA first: it validates the pattern and rejects anything
        # outside the supported subset (UnsupportedRegexError, a ValueError).
        from .grammar_compile import regex_dfa

        self._dfa = regex_dfa(pattern)
        self._compiled = self._dfa._compiled
        self._text_buffer = ""
        self._done = False
        self._valid_chars_cache: dict[str, set[str] | None] = {}
        # the codepoints the DFA probe should test. Lazily set to
        # the served tokenizer's actual first-character universe in get_allowed_tokens,
        # so EVERY script present in the vocab (Cyrillic/Greek/Hebrew/Armenian/…) is
        # covered. The old hardcoded char_range sampled only ASCII/CJK/Hangul/kana/
        # Arabic/Thai/Devanagari/Latin-ext/emoji, so a literal pattern in any other
        # script (e.g. "Привет|Пока") was never probed → empty allow-set → premature
        # EOS → empty output.
        self._query_codepoints: set[int] | None = None
        # Cache: buffer_text -> list of token IDs whose full decoded text
        # keeps the buffer on a valid DFA path. This is the correct prefix
        # check for multi-character tokens (single-char check via
        # _valid_next_chars is insufficient — see Bug-1 fix).
        self._valid_tokens_cache: dict[Any, list[int]] = {}
        # Incremental DFA position: every step costs O(len(token)), never O(len(buffer)),
        # and the caches are keyed by DFA state so a long free-text tail (``[\\s\\S]*``)
        # stays one cache entry instead of a miss (and a full vocab scan) per token.
        self._dfa_state = self._dfa._dfa_start if self._dfa.has_dfa else None

    def _cache_key(self) -> Any:
        return self._dfa_state if self._dfa.has_dfa else self._text_buffer

    @property
    def state(self) -> str:
        return "done" if self._done else "active"

    @property
    def is_done(self) -> bool:
        return self._done

    def advance(self, token_text: str) -> None:
        if self._done:
            return
        self._text_buffer += token_text
        if self._dfa.has_dfa:
            self._dfa_state = self._dfa.step(self._dfa_state, token_text)
            if self._dfa.is_accepting(self._dfa_state) and not self._dfa.can_extend(
                self._dfa_state
            ):
                self._done = True
            return
        # Only mark done if the buffer is a full match AND no further
        # characters can extend the match. For unbounded patterns (e.g.,
        # \d+, a*, [a-z]+) a partial buffer like "1" already fully matches,
        # but the model should keep generating — setting _done=True here
        # would prematurely terminate the output.
        if self._dfa.is_full_match(self._text_buffer):
            # Check if ANY character can extend the match — use the same
            # vocab-derived char range as _valid_next_chars() to avoid premature
            # termination for patterns involving non-ASCII characters.
            char_range = self._probe_codepoints()
            extendable = self._dfa.valid_next_chars(self._text_buffer, char_range)
            if not extendable:
                self._done = True

    def checkpoint(self) -> dict[str, Any]:
        """Save current state for rollback (speculative decoding support)."""
        return {
            "text_buffer": self._text_buffer,
            "done": self._done,
            "dfa_state": self._dfa_state,
        }

    def rollback(self, saved: dict[str, Any]) -> None:
        """Restore state from a checkpoint."""
        self._text_buffer = saved["text_buffer"]
        self._done = saved["done"]
        self._dfa_state = saved.get("dfa_state", self._dfa_state)
        # Clear cache since text_buffer changed — old cache entries are stale
        self._valid_chars_cache.clear()

    def _probe_codepoints(self) -> list[int]:
        """Codepoints the DFA next-char probe should test.

        Prefers the served tokenizer's actual first-character universe (set lazily in
        get_allowed_tokens) so every script in the vocab is covered. Falls back to a
        broad hardcoded sample (now incl. Cyrillic/Greek/Hebrew/Armenian/Georgian +
        wider CJK) when called before any tokenizer is known.
        """
        base = list(range(32, 127))  # printable ASCII
        base.extend([ord("\n"), ord("\t"), ord("\r")])
        if self._query_codepoints:
            base.extend(self._query_codepoints)
            return base
        base.extend(range(0x4E00, 0x4E00 + 1000))  # CJK Unified (broader sample)
        base.extend(range(0xAC00, 0xAC00 + 100))  # Hangul
        base.extend(range(0x3040, 0x30FF))  # Hiragana + Katakana
        base.extend(range(0x0400, 0x0500))  # Cyrillic
        base.extend(range(0x0370, 0x0400))  # Greek + Coptic
        base.extend(range(0x0590, 0x0600))  # Hebrew
        base.extend(range(0x0531, 0x0590))  # Armenian
        base.extend(range(0x10A0, 0x1100))  # Georgian
        base.extend(range(0x0600, 0x0660))  # Arabic
        base.extend(range(0x0E00, 0x0E50))  # Thai
        base.extend(range(0x0900, 0x0970))  # Devanagari
        base.extend(range(0x1F600, 0x1F6C8))  # Common Emoji
        base.extend(range(0x00C0, 0x0250))  # Latin Extended
        return base

    def _valid_next_chars(self) -> set[str] | None:
        """Compute characters that can follow the current partial match.

        Uses the DFA to determine which characters lead to a valid state
        (either an accept state or a state from which an accept state is
        reachable). This is O(alphabet_size) per call with caching.

        Returns None if any character is valid (permissive pattern),
        or a set of valid characters otherwise.
        """
        if self._done:
            return set()

        # Cache hit: same accumulated text always produces the same valid set.
        # Cap cache size to prevent unbounded growth during long generations.
        _MAX_CACHE_SIZE = 256
        cache_key = self._cache_key()
        if cache_key in self._valid_chars_cache:
            return self._valid_chars_cache[cache_key]
        if len(self._valid_chars_cache) >= _MAX_CACHE_SIZE:
            self._valid_chars_cache.clear()

        # Build the set of character codepoints to test (vocab-derived when available).
        char_range = self._probe_codepoints()
        total_tested = len(char_range)

        # Use DFA to find valid next character codepoints
        if self._dfa.has_dfa:
            valid_codepoints = self._dfa.valid_next_chars_from(
                self._dfa_state, char_range
            )
        else:
            valid_codepoints = self._dfa.valid_next_chars(self._text_buffer, char_range)

        # Convert codepoints back to characters
        valid = {chr(cp) for cp in valid_codepoints}

        # If >90% of tested chars are valid, treat as unrestricted.
        # This avoids false negatives for permissive patterns like ".*".
        result = None if len(valid) > total_tested * 0.9 else valid
        self._valid_chars_cache[cache_key] = result
        return result

    def get_allowed_tokens(
        self, tokenizer: Any, generated_token_ids: list[int]
    ) -> list[int]:
        if self._done:
            eos_ids = normalize_eos_ids(tokenizer)
            return eos_ids

        # seed the DFA probe with the tokenizer's actual first-char
        # universe so non-Latin scripts aren't silently un-probed (→ empty allow-set →
        # premature EOS). Done once per (cached) tokenizer char-map.
        if self._query_codepoints is None:
            if not hasattr(self.__class__, "_token_char_cache"):
                self.__class__._token_char_cache = weakref.WeakKeyDictionary()
            if tokenizer not in self.__class__._token_char_cache:
                self.__class__._token_char_cache[tokenizer] = _build_token_char_map(
                    tokenizer
                )
            _cmap = self.__class__._token_char_cache[tokenizer]
            self._query_codepoints = {ord(c) for c in _cmap if c}
            self._valid_chars_cache.clear()  # any pre-tokenizer cached probe is incomplete

        valid_chars = self._valid_next_chars()
        if valid_chars is None:
            everything = self._get_all_token_ids(tokenizer)
            # Special tokens (EOS included) are excluded from the permissive vocab, so a
            # pattern with an unbounded tail (``[\\s\\S]*``) could never stop before
            # max_tokens: add EOS back whenever the buffer already fully matches.
            if (
                self._dfa.is_accepting(self._dfa_state)
                if self._dfa.has_dfa
                else (self._dfa.is_full_match(self._text_buffer))
            ):
                eos = normalize_eos_ids(tokenizer)
                everything = list(everything) + [
                    e for e in eos if e not in set(everything)
                ]
            return everything
        if not valid_chars:
            return []

        # First-char prefilter (cheap), then full-text DFA validation per
        # candidate (correct for multi-character tokens, e.g. "<!DOCTYPE"
        # cannot extend a buffer of "<" under regex "<tool_call>...").
        candidates = self._find_tokens_for_chars(tokenizer, valid_chars)
        allowed = self._filter_tokens_by_dfa(tokenizer, candidates)

        # EOS-when-already-matched: for an UNBOUNDED pattern (e.g. `\d+`,
        # `[a-z]+`, an email `[a-z]+@[a-z]+\.[a-z]+`) the buffer can be a
        # COMPLETE valid match while still extendable, so `advance()` correctly
        # leaves `_done` False (the model MAY keep going). But without EOS in the
        # allow-set at such a state the model is FORCED to keep emitting matching
        # characters until max_tokens — a runaway (same class as the JSON-number
        # bug). When the current buffer already fully matches, EOS is a
        # legitimate stop, so add it alongside the continuation tokens. (The
        # `valid_chars is None` permissive branch above already returns the whole
        # vocab incl. EOS, so it is unaffected.)
        _full = (
            self._dfa.is_accepting(self._dfa_state)
            if self._dfa.has_dfa
            else self._dfa.is_full_match(self._text_buffer)
        )
        if _full:
            eos_ids = normalize_eos_ids(tokenizer)
            if eos_ids:
                # _filter_tokens_by_dfa returns a cached list; build a new list so
                # we never mutate the cached entry with EOS ids.
                allowed = list(allowed) + [e for e in eos_ids if e not in allowed]
        return allowed

    def _get_all_token_ids(self, tokenizer: Any) -> list[int]:
        from .json_schema import without_special_ids

        if hasattr(tokenizer, "get_vocab"):
            return without_special_ids(tokenizer, tokenizer.get_vocab().values())
        if hasattr(tokenizer, "vocab") and isinstance(tokenizer.vocab, dict):
            return without_special_ids(tokenizer, tokenizer.vocab.values())
        vocab_size = getattr(tokenizer, "vocab_size", 32000)
        return list(range(vocab_size))

    def _find_tokens_for_chars(self, tokenizer: Any, chars: set[str]) -> list[int]:
        if not hasattr(self.__class__, "_token_char_cache"):
            self.__class__._token_char_cache = weakref.WeakKeyDictionary()
        if tokenizer not in self.__class__._token_char_cache:
            self.__class__._token_char_cache[tokenizer] = _build_token_char_map(
                tokenizer
            )

        char_map = self.__class__._token_char_cache[tokenizer]
        allowed = set()
        for ch in chars:
            if ch in char_map:
                allowed.update(char_map[ch])
        return list(allowed)

    def _filter_tokens_by_dfa(self, tokenizer: Any, candidates: list[int]) -> list[int]:
        """Filter candidate tokens so that only those whose full decoded
        text keeps the DFA on a valid (accept-reachable) path remain.

        This is the correct prefix check for multi-character tokens:
        the single-char filter in get_allowed_tokens is necessary (cheap
        prefilter) but not sufficient — e.g. with regex "<tool_call>{..."
        and buffer "<", first-char check allows ANY token starting with
        '<' (including "<!DOCTYPE"), but only tokens whose text begins
        with "<t" are actually valid extensions.
        """
        # Cache per-buffer state. _text_buffer is the key — same buffer
        # always yields the same allowed-token set.
        cache_key = self._cache_key()
        cached = self._valid_tokens_cache.get(cache_key)
        if cached is not None:
            return cached

        # Cap cache to avoid unbounded growth on long generations.
        if len(self._valid_tokens_cache) >= 64:
            self._valid_tokens_cache.clear()

        # Build/lookup id->decoded-text map for this tokenizer.
        if not hasattr(self.__class__, "_token_text_cache"):
            self.__class__._token_text_cache = weakref.WeakKeyDictionary()
        if tokenizer not in self.__class__._token_text_cache:
            self.__class__._token_text_cache[tokenizer] = _build_token_text_map(
                tokenizer
            )
        text_map = self.__class__._token_text_cache[tokenizer]

        filtered: list[int] = []
        buf = self._text_buffer
        dfa = self._dfa
        fast = dfa.has_dfa
        state = self._dfa_state
        for tid in candidates:
            text = text_map.get(tid)
            if not text:
                # Unknown / undecodable — keep (will be re-checked on advance)
                filtered.append(tid)
                continue
            try:
                if fast:
                    if dfa.is_live(dfa.step(state, text)):
                        filtered.append(tid)
                elif dfa.is_prefix_valid(buf + text):
                    filtered.append(tid)
            except Exception:
                # On any DFA error, fall back to keeping the candidate
                filtered.append(tid)

        self._valid_tokens_cache[cache_key] = filtered
        return filtered

    def reset(self) -> None:
        self._text_buffer = ""
        self._done = False
        self._dfa_state = self._dfa._dfa_start if self._dfa.has_dfa else None
        self._valid_chars_cache.clear()
        self._valid_tokens_cache.clear()

    def get_stats(self) -> dict[str, Any]:
        return {
            "type": "regex",
            "pattern": self._pattern,
            "buffer_len": len(self._text_buffer),
            "is_done": self._done,
            "cache_size": len(self._valid_chars_cache),
        }


class ChoiceConstraint:
    """Constrains output to one of a fixed set of choices.

    Efficiently prunes tokens by maintaining a trie of remaining
    valid completions.
    """

    def __init__(self, choices: list[str], case_sensitive: bool = True) -> None:
        self._choices = choices
        self._case_sensitive = case_sensitive
        self._original_choices = choices
        self._text_buffer = ""
        self._done = False
        self._matched_choice: str | None = None
        self._has_partial_match = (
            False  # True when matched text is also a prefix of a longer choice
        )
        self._failed = False  # True when an invalid path was encountered
        # Build prefix trie using appropriate case form
        self._trie: dict[str, Any] = {}
        trie_choices = choices if case_sensitive else [c.lower() for c in choices]
        for idx, choice in enumerate(trie_choices):
            node = self._trie
            for ch in choice:
                node = node.setdefault(ch, {})
            node["__end__"] = choices[idx]  # store the original (cased) choice

    @property
    def state(self) -> str:
        return "done" if self._done else "active"

    @property
    def is_done(self) -> bool:
        return self._done

    def advance(self, token_text: str) -> None:
        if self._done:
            return
        self._text_buffer += token_text
        buf = self._text_buffer if self._case_sensitive else self._text_buffer.lower()

        # Check if we have an exact match
        node = self._trie
        for ch in buf:
            if ch not in node:
                # Invalid path — mark failed, stop generation
                self._failed = True
                self._done = True
                return
            node = node[ch]

        if "__end__" in node:
            self._matched_choice = node["__end__"]
            # Check if there are longer choices still possible
            if any(k != "__end__" for k in node):
                # The matched text is a prefix of a longer choice — don't set done
                self._has_partial_match = True
            else:
                # No longer choices possible — generation is complete
                self._done = True

    def get_allowed_tokens(
        self, tokenizer: Any, generated_token_ids: list[int]
    ) -> list[int]:
        if self._done:
            eos_ids = normalize_eos_ids(tokenizer)
            return eos_ids

        buf = self._text_buffer if self._case_sensitive else self._text_buffer.lower()
        node = self._trie
        for ch in buf:
            if ch not in node:
                return []
            node = node[ch]

        # Valid next chars are the keys in this trie node
        valid_chars = set()
        for key in node:
            if key == "__end__":
                # EOS is valid — we've completed a choice
                valid_chars.add("__eos__")
            else:
                valid_chars.add(key)
                # For case-insensitive matching, also allow the
                # opposite-case version of this character so that
                # the model can generate e.g. 'H' to match 'h' in
                # the trie.
                if not self._case_sensitive:
                    swapped = key.swapcase()
                    if swapped != key:
                        valid_chars.add(swapped)

        # If only __eos__ is valid, return EOS tokens
        if valid_chars == {"__eos__"}:
            eos_ids = normalize_eos_ids(tokenizer)
            return eos_ids

        # Re-check whether the current buffer position is at a valid
        # end-of-choice in the trie. The advance() method sets
        # _has_partial_match optimistically but never resets it, so we
        # must verify at query time whether EOS is truly valid here.
        has_eos = "__end__" in node

        # Build allowed tokens. Candidates start with a valid next char, but we
        # MUST filter by the token's FULL decoded text — a first-char allowlist
        # alone admits tokens like "boom" (starts with 'b', like "blue") which
        # immediately diverge from every choice, so the model can emit
        # off-grammar output that advance() only detects after the fact.
        if not hasattr(self.__class__, "_token_char_cache"):
            self.__class__._token_char_cache = weakref.WeakKeyDictionary()
        if tokenizer not in self.__class__._token_char_cache:
            self.__class__._token_char_cache[tokenizer] = _build_token_char_map(
                tokenizer
            )
        if not hasattr(self.__class__, "_token_text_cache"):
            self.__class__._token_text_cache = weakref.WeakKeyDictionary()
        if tokenizer not in self.__class__._token_text_cache:
            self.__class__._token_text_cache[tokenizer] = _build_token_text_map(
                tokenizer
            )
        char_map = self.__class__._token_char_cache[tokenizer]
        text_map = self.__class__._token_text_cache[tokenizer]

        # Remaining continuations from the current buffer toward each choice.
        choices_cmp = (
            self._choices
            if self._case_sensitive
            else [c.lower() for c in self._choices]
        )
        continuations = [c[len(buf) :] for c in choices_cmp if c.startswith(buf)]

        candidates: set[int] = set()
        for ch in valid_chars:
            if ch == "__eos__":
                continue
            candidates.update(char_map.get(ch, ()))

        allowed = set()
        for tid in candidates:
            ttext = text_map.get(tid, "")
            if not ttext:
                continue
            t_cmp = ttext if self._case_sensitive else ttext.lower()
            # Keep the token only if its full text is a non-empty prefix of some
            # remaining continuation (stays on a valid choice path, possibly
            # completing it). This rejects over-running / diverging tokens.
            for cont in continuations:
                if cont.startswith(t_cmp):
                    allowed.add(tid)
                    break
        # Include EOS tokens only when the current position is a valid
        # choice end (__end__ marker present in the trie node).
        if has_eos:
            eos_ids = normalize_eos_ids(tokenizer)
            allowed.update(eos_ids)
        return list(allowed)

    def checkpoint(self) -> dict[str, Any]:
        """Save current state for rollback (speculative decoding support)."""
        return {
            "text_buffer": self._text_buffer,
            "done": self._done,
            "matched_choice": self._matched_choice,
            "has_partial_match": self._has_partial_match,
            "failed": self._failed,
        }

    def rollback(self, saved: dict[str, Any]) -> None:
        """Restore state from a checkpoint."""
        self._text_buffer = saved["text_buffer"]
        self._done = saved["done"]
        self._matched_choice = saved["matched_choice"]
        self._has_partial_match = saved["has_partial_match"]
        self._failed = saved["failed"]

    def reset(self) -> None:
        self._text_buffer = ""
        self._done = False
        self._matched_choice = None
        self._has_partial_match = False
        self._failed = False

    def get_stats(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "num_choices": len(self._choices),
            "buffer_len": len(self._text_buffer),
            "is_done": self._done,
            "matched": self._matched_choice,
        }


class UnsupportedGrammarError(ValueError):
    """The CFG grammar cannot be served (invalid grammar or no engine/tokenizer)."""


class _LlgConstraint:
    """Token-mask constraint backed by an llguidance grammar (base class).

    llguidance is an Earley engine that computes the exact set of tokens that
    keep the input a viable prefix of the grammar, so there is no first-character
    shortcut, no candidate cap and no fixed probe alphabet.  The grammar is
    validated at construction (:class:`UnsupportedGrammarError` with the engine's
    message); the start rule is ``start`` as in Lark.  The advance interface is
    text-based like the other constraints: the text is re-tokenized into
    vocabulary ids that llguidance consumes (the grammar state depends on the
    bytes, not on the token boundaries).
    """

    _KIND = "llguidance"

    def __init__(self, grammar: Any, tokenizer: Any = None) -> None:
        self._grammar_text = grammar
        self._text_buffer = ""
        self._done = False
        self._dead = False
        self._consumed = 0
        self._llt: Any = None
        self._matcher: Any = None
        self._cache_text = ""
        self._allowed_cache: dict[int, list[int]] = {}
        self._ckpt_stack: list[dict[str, Any]] = []
        try:
            from llguidance import LLMatcher
        except ImportError as exc:  # pragma: no cover - llguidance is a dependency
            raise UnsupportedGrammarError(
                "CFG grammar constraints require llguidance"
            ) from exc
        self._LLMatcher = LLMatcher
        if tokenizer is not None:
            self._bind(tokenizer)

    def _make_grammar(self) -> str:
        raise NotImplementedError

    def _error(self, message: str) -> Exception:
        return UnsupportedGrammarError(f"invalid grammar: {message[:400]}")

    # ── engine binding ──────────────────────────────────────────────────────

    @staticmethod
    def _vocab_width(tokenizer: Any) -> int:
        from yunshu_engine.tool_call_grammar import _hf_tokenizer

        hf = _hf_tokenizer(tokenizer)
        for probe in (
            lambda: len(hf),
            lambda: int(hf.vocab_size),
            lambda: len(hf.get_vocab()),
        ):
            try:
                n = int(probe())
                if n > 0:
                    return n
            except Exception:  # noqa: BLE001 - try the next shape
                continue
        raise UnsupportedGrammarError("cannot determine the tokenizer vocabulary size")

    def _bind(self, tokenizer: Any) -> None:
        from yunshu_engine.tool_call_grammar import llg_tokenizer

        try:
            llt = llg_tokenizer(tokenizer, self._vocab_width(tokenizer))
        except UnsupportedGrammarError:
            raise
        except Exception as exc:
            raise UnsupportedGrammarError(
                f"CFG constraints need a Hugging Face tokenizer: {exc}"
            ) from exc
        grammar = self._make_grammar()
        from .grammar_compile import new_llg_matcher

        try:
            matcher = new_llg_matcher(llt, grammar)
        except Exception as exc:
            raise self._error(str(exc)) from exc
        self._llt = llt
        self._matcher = matcher
        self._words = (llt.vocab_size + 31) // 32

    # ── constraint interface ────────────────────────────────────────────────

    @property
    def state(self) -> str:
        return "done" if self._done else "active"

    @property
    def is_done(self) -> bool:
        return self._done

    def advance(self, token_text: str) -> None:
        if self._done or self._dead or self._matcher is None or not token_text:
            return
        self._text_buffer += token_text
        for tid in self._llt.tokenize_str(token_text):
            if not self._matcher.consume_token(tid):
                # The model emitted text the mask forbade: fail closed.
                self._dead = True
                return
            self._consumed += 1
        if self._matcher.is_stopped():
            self._done = True

    def get_allowed_tokens(
        self, tokenizer: Any, generated_token_ids: list[int]
    ) -> list[int]:
        if self._matcher is None:
            self._bind(tokenizer)
        if self._dead:
            return []
        if self._done or self._matcher.is_stopped():
            return normalize_eos_ids(tokenizer)
        cached = self._allowed_cache.get(self._consumed)
        if cached is not None and self._cache_text == self._text_buffer:
            return cached
        import llguidance.numpy as lnp
        import numpy as np

        row = np.zeros((1, self._words), dtype=np.int32)
        lnp.fill_next_token_bitmask(self._matcher, row, 0)
        bits = np.unpackbits(row.view(np.uint8), bitorder="little")
        allowed = np.flatnonzero(bits[: self._llt.vocab_size]).tolist()
        if self._matcher.is_accepting():
            allowed += [e for e in normalize_eos_ids(tokenizer) if e not in allowed]
        if len(self._allowed_cache) >= 64:
            self._allowed_cache.clear()
        self._allowed_cache[self._consumed] = allowed
        self._cache_text = self._text_buffer
        return allowed

    def checkpoint(self) -> dict[str, Any]:
        """Save current state; returns it and also pushes it on a stack so the
        no-argument ``rollback()`` / ``discard_checkpoint()`` of the JSON-schema
        constraint interface work too (speculative decoding)."""
        saved = {
            "text_buffer": self._text_buffer,
            "done": self._done,
            "dead": self._dead,
            "consumed": self._consumed,
        }
        self._ckpt_stack.append(saved)
        if len(self._ckpt_stack) > 64:
            del self._ckpt_stack[:32]
        return saved

    def rollback(self, saved: dict[str, Any] | None = None) -> None:
        """Restore a checkpoint (the given one, else the most recent)."""
        if saved is None:
            if not self._ckpt_stack:
                return
            saved = self._ckpt_stack.pop()
        if self._matcher is not None and self._consumed > saved["consumed"]:
            self._matcher.rollback(self._consumed - saved["consumed"])
        self._text_buffer = saved["text_buffer"]
        self._done = saved["done"]
        self._dead = saved["dead"]
        self._consumed = saved["consumed"]

    def discard_checkpoint(self) -> None:
        if self._ckpt_stack:
            self._ckpt_stack.pop()

    def reset(self) -> None:
        if self._matcher is not None:
            self._matcher.reset()
        self._text_buffer = ""
        self._done = False
        self._dead = False
        self._consumed = 0
        self._cache_text = ""
        self._allowed_cache.clear()

    def get_stats(self) -> dict[str, Any]:
        return {
            "type": self._KIND,
            "has_parser": self._matcher is not None,
            "buffer_len": len(self._text_buffer),
            "is_done": self._done,
        }


class CfgGrammarConstraint(_LlgConstraint):
    """Context-free grammar constraint (Lark syntax) backed by llguidance."""

    _KIND = "cfg"

    def __init__(
        self, grammar: str, start_rule: str = "start", tokenizer: Any = None
    ) -> None:
        if start_rule != "start":
            raise UnsupportedGrammarError("CFG grammars must define a 'start' rule")
        super().__init__(grammar, tokenizer)

    def _make_grammar(self) -> str:
        from .grammar_compile import llg_grammar

        try:
            return llg_grammar("cfg", self._grammar_text)
        except Exception as exc:
            raise self._error(str(exc)) from exc

    def _error(self, message: str) -> Exception:
        return UnsupportedGrammarError(f"invalid CFG grammar: {message[:400]}")


#: Structural whitespace llguidance may emit between JSON tokens: nothing, one space,
#: or a newline plus indentation (the in-house engine's run; llguidance's default is
#: any run of blanks, which lets a greedy model spin on whitespace forever).
_LLG_WHITESPACE: Any = {"whitespace_pattern": r"[ ]?|\n[ ]{0,16}"}
_LLG_COMPACT: Any = {"whitespace_flexible": False}

_SCHEMA_MAP_KEYS = ("properties", "patternProperties", "$defs", "definitions")
_SCHEMA_LIST_KEYS = ("anyOf", "oneOf", "allOf", "prefixItems")
_SCHEMA_ONE_KEYS = (
    "items",
    "additionalProperties",
    "not",
    "if",
    "then",
    "else",
    "contains",
    "propertyNames",
)
_MERGEABLE_KEYS = frozenset(
    {
        "type",
        "properties",
        "required",
        "title",
        "description",
        "additionalProperties",
        "$comment",
        "default",
        "examples",
    }
)


def _merge_all_of_objects(node: dict) -> None:
    """Fold ``allOf`` branches that are plain object schemas into ``node`` so the
    merged object can be closed once (closing each branch would make the
    intersection unsatisfiable).  Anything else stays an ``allOf``."""
    branches = node.get("allOf")
    if not isinstance(branches, list) or not branches:
        return
    props: dict[str, Any] = dict(node.get("properties") or {})
    required: list[str] = list(node.get("required") or [])
    for b in branches:
        if not (
            isinstance(b, dict)
            and set(b) <= _MERGEABLE_KEYS
            and b.get("type") in (None, "object")
            and "additionalProperties" not in b
            and isinstance(b.get("properties", {}), dict)
        ):
            return
        for k, v in (b.get("properties") or {}).items():
            if k in props and props[k] != v:
                return
            props[k] = v
        required += [r for r in (b.get("required") or []) if r not in required]
    del node["allOf"]
    node["type"] = "object"
    node["properties"] = props
    if required:
        node["required"] = required


def _normalize_for_llg(node: Any, *, closing: bool = True) -> Any:
    """Our documented JSON-schema conventions, applied before llguidance sees the
    schema (it follows plain JSON Schema): an object that declares ``properties``
    and omits ``additionalProperties`` is closed (the OpenAI structured-output
    convention) and ``properties`` without a ``type`` mean an object."""
    if isinstance(node, list):
        return [_normalize_for_llg(v, closing=closing) for v in node]
    if not isinstance(node, dict):
        return node
    out = dict(node)
    _merge_all_of_objects(out)
    if closing and isinstance(out.get("properties"), dict) and out["properties"]:
        out.setdefault("type", "object")
        if out.get("type") == "object":
            out.setdefault("additionalProperties", False)
    child_closing = closing and "allOf" not in out
    for k in _SCHEMA_MAP_KEYS:
        if isinstance(out.get(k), dict):
            out[k] = {
                name: _normalize_for_llg(sub, closing=child_closing)
                for name, sub in out[k].items()
            }
    for k in _SCHEMA_LIST_KEYS:
        if isinstance(out.get(k), list):
            out[k] = [
                _normalize_for_llg(sub, closing=child_closing and k != "allOf")
                for sub in out[k]
            ]
    for k in _SCHEMA_ONE_KEYS:
        if isinstance(out.get(k), dict):
            out[k] = _normalize_for_llg(out[k], closing=child_closing)
        elif isinstance(out.get(k), list):
            out[k] = [_normalize_for_llg(v, closing=child_closing) for v in out[k]]
    return out


def _llg_schema_json(schema: Any) -> str:
    """The schema as JSON for llguidance: our conventions applied
    (:func:`_normalize_for_llg`) and without ``format`` values it does not know
    (``format`` is an annotation in JSON Schema, so an unknown one is not an error)."""
    import json as _json

    from .json_schema import LLG_ENFORCED_FORMATS

    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                k: clean(v)
                for k, v in node.items()
                if not (k == "format" and v not in LLG_ENFORCED_FORMATS)
            }
        if isinstance(node, list):
            return [clean(v) for v in node]
        return node

    return _json.dumps(clean(_normalize_for_llg(schema)), ensure_ascii=False)


class LlgJsonSchemaConstraint(_LlgConstraint):
    """JSON Schema constraint backed by llguidance: value constraints (pattern,
    lengths, ranges, multipleOf, item counts), prefixItems, recursive ``$ref`` and
    common ``format`` values are enforced token by token.

    Where it differs from the in-house engine (measured on the Qwen3.5/3.8
    tokenizer, ``scripts/research/c06_json_ab.py``): properties come in schema order
    (OpenAI's documented key order), literal text the schema forces (keys, ``true``)
    is only allowed in the tokenizer's canonical tokenization, ``\\uXXXX`` escapes
    are limited to control characters, and no whitespace precedes the first value.
    Closed objects, whitespace runs, number and integer literals match."""

    _KIND = "json_schema"

    def __init__(
        self, schema: dict, tokenizer: Any = None, *, compact: bool = False
    ) -> None:
        """``compact``: no structural whitespace at all (``{"a":1}``), so every
        key, separator and brace is forced text; jump-forward decoding uses it."""
        from .grammar_compile import schema_source

        self._schema_json = schema_source(schema)
        self._compact = compact
        super().__init__(schema, tokenizer)

    def _make_grammar(self) -> str:
        from .grammar_compile import llg_grammar

        try:
            return llg_grammar("json_schema", self._schema_json, compact=self._compact)
        except Exception as exc:
            raise self._error(str(exc)) from exc

    def _error(self, message: str) -> Exception:
        from .json_schema import UnsupportedSchemaError

        return UnsupportedSchemaError(
            f"unsupported JSON Schema (llguidance cannot enforce it): {message[:400]}"
        )

    def forced_continuation(self, max_len: int = 128) -> str:
        """Text the schema forces next (llguidance's fast-forward bytes), for
        jump-forward decoding; ``""`` when the next token is a real choice."""
        if self._matcher is None or self._dead or self._done:
            return ""
        try:
            forced = bytes(self._matcher.compute_ff_bytes())
        except Exception:  # noqa: BLE001 - a failed probe just means no jump
            return ""
        return forced.decode("utf-8", errors="ignore")[:max_len]


def validate_llg_json_schema(schema: dict) -> None:
    """Raise ``UnsupportedSchemaError`` when llguidance cannot compile ``schema``."""
    from .grammar_compile import llg_grammar, schema_source
    from .json_schema import UnsupportedSchemaError

    try:
        llg_grammar("json_schema", schema_source(schema))
        err = None
    except Exception as exc:  # noqa: BLE001 - llguidance raises plain exceptions
        err = str(exc)
    if err:
        raise UnsupportedSchemaError(
            f"unsupported JSON Schema (llguidance cannot enforce it): {err[:400]}"
        )


def validate_llg_cfg(grammar: str) -> None:
    """Raise ``UnsupportedGrammarError`` for a Lark grammar llguidance cannot compile.

    Needs no tokenizer, so the gateway can reject a bad grammar with a 400 at
    request validation instead of when the engine builds the constraint.
    """
    try:
        import llguidance  # noqa: F401
    except ImportError:  # no llguidance: the engine reports it when it builds
        return
    try:
        from .grammar_compile import llg_grammar

        llg_grammar("cfg", grammar)
        err = None
    except Exception as exc:  # noqa: BLE001 - llguidance raises plain exceptions
        err = str(exc)
    if err:
        raise UnsupportedGrammarError(f"invalid CFG grammar: {err[:400]}")


def build_json_constraint(
    schema: Any, tokenizer: Any = None, *, compact: bool = False
) -> Any:
    """The constraint for a JSON schema (``None`` = any JSON object).

    llguidance enforces every schema it can compile (``YUNSHU_JSON_SCHEMA_ENGINE``
    selects the in-house state machine instead).  A schema llguidance cannot
    compile falls back to the in-house engine when that supports it; only what
    neither can enforce is rejected.  ``compact`` (jump-forward decoding) asks for
    llguidance without structural whitespace.
    """
    from yunshu_engine import settings

    from .json_schema import (
        JsonSchemaConstraint,
        UnsupportedSchemaError,
        validate_supported_schema,
    )

    # Without a tokenizer there is nothing to bind llguidance to: keep the in-house
    # engine (a schema only llguidance can enforce still binds lazily).
    inhouse = (
        tokenizer is None or settings.get("YUNSHU_JSON_SCHEMA_ENGINE") == "inhouse"
    )
    if inhouse:
        if schema is None:
            return JsonSchemaConstraint(None)
        try:
            validate_supported_schema(schema)
        except UnsupportedSchemaError:
            validate_llg_json_schema(schema)
            return LlgJsonSchemaConstraint(schema, tokenizer)
        return JsonSchemaConstraint(schema)
    llg_schema = {"type": "object"} if schema is None else schema
    try:
        validate_llg_json_schema(llg_schema)
        return LlgJsonSchemaConstraint(llg_schema, tokenizer, compact=compact)
    except (UnsupportedSchemaError, UnsupportedGrammarError):
        # llguidance cannot compile the schema, or cannot bind this tokenizer
        # (it needs a Hugging Face fast tokenizer): the in-house engine, if it can.
        if schema is None:
            return JsonSchemaConstraint(None)
        validate_supported_schema(schema)  # raises when neither engine can
        return JsonSchemaConstraint(schema)


# ── Shared utilities ────────────────────────────────────────────────────────


def _build_token_text_map(tokenizer: Any) -> dict[int, str]:
    """Build a mapping from token_id → its decoded text (cached once).

    Used by ChoiceConstraint to filter candidate tokens by their FULL text
    against the remaining choice continuations (not just the first char).
    """
    text_map: dict[int, str] = {}
    if hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
    elif hasattr(tokenizer, "vocab") and isinstance(tokenizer.vocab, dict):
        vocab = tokenizer.vocab
    else:
        vocab_size = getattr(tokenizer, "vocab_size", 32000)
        vocab = {str(i): i for i in range(vocab_size)}
    for _token_text, token_id in vocab.items():
        try:
            decoded = tokenizer.decode([token_id])
        except Exception:
            decoded = ""
        text_map[token_id] = decoded or ""
    return text_map


def _build_token_char_map(tokenizer: Any) -> dict[str, list[int]]:
    """Build a mapping from first character to list of token IDs.

    Multi-character special tokens of the form <...> (e.g. <tool_call>,
    <|im_start|>) ARE included so grammar constraints can match against
    their literal first character when relevant — the downstream full-text
    DFA filter then removes ones that do not actually extend the buffer.
    """
    char_map: dict[str, list[int]] = {}

    if hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
    elif hasattr(tokenizer, "vocab") and isinstance(tokenizer.vocab, dict):
        vocab = tokenizer.vocab
    else:
        vocab_size = getattr(tokenizer, "vocab_size", 32000)
        vocab = {str(i): i for i in range(vocab_size)}

    for token_text, token_id in vocab.items():
        if not token_text:
            continue
        first_char = token_text[0] if token_text else ""
        try:
            decoded = tokenizer.decode([token_id])
            if decoded:
                char_map.setdefault(decoded[0], []).append(token_id)
            else:
                char_map.setdefault(first_char, []).append(token_id)
        except Exception:
            logger.debug(
                "tokenizer decode failed for token %d in char map build",
                token_id,
                exc_info=True,
            )
            char_map.setdefault(first_char, []).append(token_id)

    return char_map


def _build_token_text_map(tokenizer: Any) -> dict[int, str]:
    """Build a mapping from token ID to decoded text for fast lookup
    during DFA-based full-token prefix validation."""
    text_map: dict[int, str] = {}
    if hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
    elif hasattr(tokenizer, "vocab") and isinstance(tokenizer.vocab, dict):
        vocab = tokenizer.vocab
    else:
        return text_map

    eos_ids: set[int] = set(normalize_eos_ids(tokenizer))

    for _token_text, token_id in vocab.items():
        if token_id in eos_ids:
            # EOS tokens are handled separately; their decoded text is empty
            # or meta — keep out so they bypass DFA filtering.
            text_map[token_id] = ""
            continue
        try:
            decoded = tokenizer.decode([token_id])
            text_map[token_id] = decoded or ""
        except Exception:
            text_map[token_id] = ""
    return text_map


def _get_all_token_ids(tokenizer: Any) -> list[int]:
    from .json_schema import without_special_ids

    if hasattr(tokenizer, "get_vocab"):
        return without_special_ids(tokenizer, tokenizer.get_vocab().values())
    if hasattr(tokenizer, "vocab") and isinstance(tokenizer.vocab, dict):
        return without_special_ids(tokenizer, tokenizer.vocab.values())
    return list(range(getattr(tokenizer, "vocab_size", 32000)))


# ── Constraint Factory ──────────────────────────────────────────────────────


class ConstraintFactory:
    """Create the appropriate constraint from a grammar specification.

    Supports:
    - json_schema → JsonSchemaConstraint (from json_schema.py)
    - json_object → JsonSchemaConstraint (no schema)
    - regex → RegexConstraint
    - choice → ChoiceConstraint
    - cfg → CfgGrammarConstraint (Lark syntax, llguidance engine)
    """

    @staticmethod
    def create(
        grammar_type: str,
        grammar: Any = None,
        tokenizer: Any = None,
    ) -> Any:
        """Create a constraint from grammar type and specification.

        Args:
            grammar_type: One of "json_schema", "json_object", "regex", "choice", "cfg"
            grammar: The grammar specification (schema dict, regex string, choice list, etc.)
            tokenizer: Tokenizer for token-level masking

        Returns:
            Constraint object with advance/get_allowed_tokens/is_done/reset interface
        """
        if grammar_type in ("json_schema", "json_object"):
            schema = grammar if grammar_type == "json_schema" else None
            return build_json_constraint(schema, tokenizer)

        if grammar_type == "regex":
            if not isinstance(grammar, str):
                raise ValueError("regex constraint requires a string pattern")
            return RegexConstraint(grammar)

        if grammar_type == "choice":
            if not isinstance(grammar, list):
                raise ValueError("choice constraint requires a list of strings")
            return ChoiceConstraint(grammar)

        if grammar_type == "cfg":
            if not isinstance(grammar, str):
                raise ValueError("cfg constraint requires a grammar string")
            return CfgGrammarConstraint(grammar, tokenizer=tokenizer)

        raise ValueError(f"Unknown grammar_type: {grammar_type}")


def validate_constraint_spec(spec: Any) -> None:
    """Validate a parsed ``response_format`` / ``grammar`` spec before generation.

    Raises a ``ValueError`` subclass (``UnsupportedRegexError``,
    ``UnsupportedSchemaError``, ``UnsupportedGrammarError``) naming the
    unsupported construct, so a gateway can answer 400 instead of generating
    against an approximation.  CFG grammars are validated against the engine
    once a tokenizer is bound (``CfgGrammarConstraint``).
    """
    if spec is None or spec == "json_object":
        return
    if not isinstance(spec, dict):
        return
    gtype = spec.get("type")
    if gtype == "regex" and "pattern" in spec:
        pattern = spec["pattern"]
        if not isinstance(pattern, str):
            raise UnsupportedRegexError("regex pattern must be a string")
        from .grammar_compile import regex_dfa

        regex_dfa(pattern)
        return
    if gtype == "choice" and "choices" in spec:
        choices = spec["choices"]
        if not isinstance(choices, list) or not all(
            isinstance(c, str) for c in choices
        ):
            raise ValueError("choice constraint requires a list of strings")
        return
    if gtype == "cfg" and "grammar" in spec:
        if not isinstance(spec["grammar"], str):
            raise UnsupportedGrammarError("cfg constraint requires a grammar string")
        validate_llg_cfg(spec["grammar"])
        return
    from .json_schema import UnsupportedSchemaError, validate_supported_schema

    try:
        validate_supported_schema(spec)
    except UnsupportedSchemaError:
        validate_llg_json_schema(spec)  # enforced by llguidance, or rejected here
