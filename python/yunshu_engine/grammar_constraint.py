from __future__ import annotations

"""Structured output constraints beyond JSON Schema — regex and CFG support.

Provides token-level constraint masking for:
1. RegexConstraint — regex pattern enforcement (e.g., date, email, phone)
2. LarkGrammarConstraint — context-free grammar via Lark parser
3. ChoiceConstraint — enumeration from a fixed list of strings
4. ConstraintFactory — unified interface matching grammar_type to constraint

All constraints implement the same interface as JsonSchemaConstraint:
- advance(token_text) — update state after each token
- get_allowed_tokens(tokenizer, generated_ids) — return valid next tokens
- is_done — whether generation is complete
- reset() — reset to initial state

Integration: plug into ConstrainedSampler alongside JsonSchemaConstraint.
"""

import contextlib
import logging
import re
import re._parser as _sre_parse
import weakref
from typing import Any

logger = logging.getLogger(__name__)


# ── Regex DFA for prefix matching ────────────────────────────────────────────


class _ExceptChars:
    """Compact representation for 'all chars except these'.

    Avoids materialising sets of ~1M codepoints for NOT_LITERAL, negated []
    character classes, and CATEGORY_NOT_* patterns.  The subset construction
    resolves these against the query range at DFA build time.
    """

    __slots__ = ("excluded",)

    def __init__(self, excluded: set[int]) -> None:
        self.excluded = excluded

    def __contains__(self, item: int) -> bool:
        return item not in self.excluded

    def __len__(self) -> int:
        # Not meaningful but prevents empty-set checks from short-circuiting.
        return 1

    def intersect(self, query_range: set[int]) -> set[int]:
        """Return the members of *query_range* that are not excluded."""
        return query_range - self.excluded


class _RegexDFA:
    """Builds a DFA from a regex pattern to support prefix-valid checks.

    The key capability: given a partial string, determine which characters
    can be appended so that the result is still a prefix of some string that
    fully matches the pattern.

    This solves the fundamental problem with using re.match() for prefix
    checking: re.match(r'\\d{3}', '1') returns None because the pattern
    requires 3 digits, but '1' is a perfectly valid prefix of '123'.

    Approach:
    1. Parse regex via _sre_parse to get the parse tree
    2. Build an NFA with epsilon transitions
    3. Convert to DFA via subset construction
    4. For each DFA state, compute the set of characters that transition
       to another (non-dead) state
    """

    def __init__(self, pattern: str) -> None:
        self._pattern = pattern
        self._compiled = re.compile(pattern)
        # Build DFA
        self._nfa_start: int = 0
        self._nfa_accept: set[int] = set()
        self._nfa_transitions: dict[int, list[tuple[set[int] | _ExceptChars | None, int]]] = {}
        self._nfa_epsilon: dict[int, list[int]] = {}
        self._dfa_transitions: dict[frozenset[int], dict[int, frozenset[int]]] = {}
        self._dfa_accept_states: set[frozenset[int]] = set()
        self._dfa_start: frozenset[int] = frozenset()
        self._nfa_counter = 0
        self._build_dfa(pattern)

    # SECURITY (ReDoS/DoS): a user-supplied regex like "a{100000000}" or
    # nested "(a{5000}){5000}" materializes one NFA copy per repeat → unbounded
    # state allocation + subset-construction blowup that hangs the serial executor
    # before a single token is generated. Cap total NFA states; _new_nfa_state
    # raises the moment the bound is crossed, short-circuiting the build loops.
    _MAX_NFA_STATES = 200_000

    def _new_nfa_state(self) -> int:
        if self._nfa_counter >= self._MAX_NFA_STATES:
            raise ValueError(
                f"regex too complex: NFA exceeds {self._MAX_NFA_STATES} states "
                "(repeat counts too large)")
        s = self._nfa_counter
        self._nfa_counter += 1
        return s

    def _build_dfa(self, pattern: str) -> None:
        """Parse pattern and build DFA via NFA + subset construction."""
        try:
            parsed = _sre_parse.parse(pattern)
        except re.error:
            # If parsing fails, fall back — DFA will be empty, constraint
            # will return None (unrestricted) for safety.
            return

        # Build NFA from parse tree
        start = self._new_nfa_state()
        accept = self._new_nfa_state()
        self._nfa_accept = {accept}
        self._nfa_start = start
        self._build_nfa(parsed, start, accept)

        # Convert NFA to DFA via subset construction
        self._subset_construction()

    def _build_nfa(
        self,
        parsed: _sre_parse.SubPattern,
        start: int,
        accept: int,
    ) -> None:
        """Recursively build NFA fragments from _sre_parse output."""
        items = list(parsed)
        if not items:
            # Empty pattern — epsilon transition
            self._nfa_epsilon.setdefault(start, []).append(accept)
            return

        # Build chain: connect each item sequentially
        current = start
        for i, item in enumerate(items):
            op, av = item
            if op == _sre_parse.LITERAL:
                # Single character match
                next_state = accept if i == len(items) - 1 else self._new_nfa_state()
                self._nfa_transitions.setdefault(current, []).append((set([av]), next_state))
                current = next_state

            elif op == _sre_parse.NOT_LITERAL:
                next_state = accept if i == len(items) - 1 else self._new_nfa_state()
                # Use a compact "except" representation instead of expanding
                # 1.1M Unicode codepoints. The subset construction resolves
                # this against the query range at DFA build time.
                self._nfa_transitions.setdefault(current, []).append(
                    (_ExceptChars({av}), next_state)
                )
                current = next_state

            elif op == _sre_parse.ANY:
                # Match any character (except newline by default)
                next_state = accept if i == len(items) - 1 else self._new_nfa_state()
                self._nfa_transitions.setdefault(current, []).append(
                    (_ExceptChars({ord('\n')}), next_state)
                )
                current = next_state

            elif op == _sre_parse.IN:
                next_state = accept if i == len(items) - 1 else self._new_nfa_state()
                char_set = self._parse_charset(av)
                self._nfa_transitions.setdefault(current, []).append(
                    (char_set, next_state)
                )
                current = next_state

            elif op == _sre_parse.BRANCH:
                # av is (None, [branch1, branch2, ...])
                _, branches = av
                next_state = accept if i == len(items) - 1 else self._new_nfa_state()
                for branch in branches:
                    self._build_nfa(branch, current, next_state)
                current = next_state

            elif op == _sre_parse.SUBPATTERN:
                # av is (group, add_flags, del_flags, parsed_subpattern)
                _, _, _, parsed_sub = av
                next_state = accept if i == len(items) - 1 else self._new_nfa_state()
                self._build_nfa(parsed_sub, current, next_state)
                current = next_state

            elif op == _sre_parse.MAX_REPEAT or op == _sre_parse.MIN_REPEAT:
                # av is (min, max, parsed_subpattern)
                min_count, max_count, parsed_sub = av
                next_state = accept if i == len(items) - 1 else self._new_nfa_state()
                self._build_repeat_nfa(parsed_sub, min_count, max_count, current, next_state)
                current = next_state

            elif op == _sre_parse.AT:
                # Anchors (^, $, \b, etc.) — treat as epsilon for
                # prefix matching since we track state per character
                if i == len(items) - 1:
                    self._nfa_epsilon.setdefault(current, []).append(accept)
                # Otherwise just continue (anchor doesn't consume input)

            elif op == _sre_parse.ASSERT or op == _sre_parse.ASSERT_NOT:
                # Lookahead/lookbehind — treat as epsilon (approximate)
                if i == len(items) - 1:
                    self._nfa_epsilon.setdefault(current, []).append(accept)

            else:
                # Unknown op — epsilon as fallback
                if i == len(items) - 1:
                    self._nfa_epsilon.setdefault(current, []).append(accept)

    def _build_repeat_nfa(
        self,
        parsed: _sre_parse.SubPattern,
        min_count: int,
        max_count: int,
        start: int,
        accept: int,
    ) -> None:
        """Build NFA for repetition (quantifier) constructs.

        Handles: *, +, ?, {n}, {n,}, {n,m}
        Strategy:
        - Build `min_count` mandatory copies in sequence
        - For optional copies (between min and max), add epsilon bypass
        - For unbounded (max == _sre_parse.MAXREPEAT), loop back
        """
        if min_count == 0 and max_count == 1:
            # ? — zero or one
            self._nfa_epsilon.setdefault(start, []).append(accept)
            self._build_nfa(parsed, start, accept)
            return

        if max_count == _sre_parse.MAXREPEAT:
            # Unbounded: *, +, {n,}
            # Strategy: chain min mandatory copies, then add a loop
            if min_count == 0:
                # * or {0,} — epsilon to accept
                self._nfa_epsilon.setdefault(start, []).append(accept)

            current = start
            for _ in range(min_count):
                next_s = self._new_nfa_state()
                self._build_nfa(parsed, current, next_s)
                current = next_s

            # Loop: from current, match one more and loop back
            self._nfa_epsilon.setdefault(current, []).append(accept)
            self._build_nfa(parsed, current, current)
        else:
            # Bounded: {n,m}
            # Chain min mandatory copies, then (max - min) optional copies
            current = start
            for _ in range(min_count):
                next_s = self._new_nfa_state()
                self._build_nfa(parsed, current, next_s)
                current = next_s

            self._nfa_epsilon.setdefault(current, []).append(accept)

            for _ in range(max_count - min_count):
                next_s = self._new_nfa_state()
                self._build_nfa(parsed, current, next_s)
                self._nfa_epsilon.setdefault(next_s, []).append(accept)
                current = next_s

    def _parse_charset(self, items: list) -> set[int] | _ExceptChars:
        """Parse _sre_parse IN items into a set of character ordinals.

        Returns an _ExceptChars for negated char classes to avoid creating
        huge sets of ~1M elements.
        """
        char_set: set[int] = set()
        # Track whether any _ExceptChars was encountered — if so the final
        # result is also an _ExceptChars (union of exclusions).
        except_accum: _ExceptChars | None = None
        negate = False

        for op, av in items:
            if op == _sre_parse.NEGATE:
                negate = True
            elif op == _sre_parse.LITERAL:
                char_set.add(av)
            elif op == _sre_parse.RANGE:
                lo, hi = av
                char_set.update(range(lo, hi + 1))
            elif op == _sre_parse.CATEGORY:
                expanded = self._expand_category(av)
                if isinstance(expanded, _ExceptChars):
                    # Merge exclusions
                    if except_accum is None:
                        except_accum = _ExceptChars(set(expanded.excluded))
                    else:
                        except_accum.excluded.update(expanded.excluded)
                else:
                    char_set.update(expanded)
            else:
                pass  # Unknown

        # If we accumulated any _ExceptChars, combine with the explicit set.
        # The final semantics: match chars NOT in (explicit ∪ exclusions).
        if except_accum is not None:
            except_accum.excluded.update(char_set)
            if negate:
                # Double-negate: [^[:^digit:]] = [[:digit:]].
                # Expand against a limited range to avoid huge sets.
                limited = set(range(0, 0x10000))
                return limited - except_accum.excluded
            return except_accum

        if negate:
            # Return compact "except" representation instead of expanding
            # 65k+ Unicode codepoints.
            return _ExceptChars(char_set)

        return char_set

    def _expand_category(self, category: int) -> set[int] | _ExceptChars:
        """Expand _sre_parse category to a set of character ordinals.

        Returns _ExceptChars for negated categories to avoid creating
        ~65k element sets.
        """
        if category == _sre_parse.CATEGORY_DIGIT:
            return set(range(ord('0'), ord('9') + 1))
        elif category == _sre_parse.CATEGORY_NOT_DIGIT:
            return _ExceptChars(set(range(ord('0'), ord('9') + 1)))
        elif category == _sre_parse.CATEGORY_SPACE:
            return {ord(c) for c in ' \t\n\r\f\v'}
        elif category == _sre_parse.CATEGORY_NOT_SPACE:
            return _ExceptChars({ord(c) for c in ' \t\n\r\f\v'})
        elif category == _sre_parse.CATEGORY_WORD:
            chars: set[int] = set()
            chars.update(range(ord('a'), ord('z') + 1))
            chars.update(range(ord('A'), ord('Z') + 1))
            chars.update(range(ord('0'), ord('9') + 1))
            chars.add(ord('_'))
            return chars
        elif category == _sre_parse.CATEGORY_NOT_WORD:
            word = (set(range(ord('a'), ord('z') + 1))
                    | set(range(ord('A'), ord('Z') + 1))
                    | set(range(ord('0'), ord('9') + 1))
                    | {ord('_')})
            return _ExceptChars(word)
        return set()

    def _epsilon_closure(self, states: frozenset[int]) -> frozenset[int]:
        """Compute epsilon closure of a set of NFA states."""
        closure = set(states)
        stack = list(states)
        while stack:
            s = stack.pop()
            for ns in self._nfa_epsilon.get(s, []):
                if ns not in closure:
                    closure.add(ns)
                    stack.append(ns)
        return frozenset(closure)

    def _subset_construction(self) -> None:
        """Convert NFA to DFA using subset construction algorithm.

        Uses a two-phase approach to handle large character sets
        (NOT_LITERAL, negated IN, CATEGORY_NOT_*) efficiently:
        1. Find all "small" char_sets and iterate those directly.
        2. For large char_sets, intersect with a limited query range
           (printable ASCII + common scripts) since valid_next_chars()
           only ever queries characters within that range.
        """
        start_closure = self._epsilon_closure(frozenset({self._nfa_start}))
        self._dfa_start = start_closure

        # Check if start state is accept (empty string matches)
        if start_closure & self._nfa_accept:
            self._dfa_accept_states.add(start_closure)

        # Precompute the query range — characters that valid_next_chars()
        # will ever ask about. Expanding the full 1.1M-char Unicode range
        # for NOT_LITERAL is pointless if we only ever query ~2k chars.
        _QUERY_RANGE = set(range(1, 128))  # ASCII
        _QUERY_RANGE.update(range(0x4E00, 0x4E00 + 500))  # CJK
        _QUERY_RANGE.update(range(0xAC00, 0xAC00 + 100))  # Hangul
        _QUERY_RANGE.update(range(0x3040, 0x30FF))  # Hiragana + Katakana
        _QUERY_RANGE.update(range(0x0600, 0x0660))  # Arabic
        _QUERY_RANGE.update(range(0x0E00, 0x0E50))  # Thai
        _QUERY_RANGE.update(range(0x0900, 0x0970))  # Devanagari
        _QUERY_RANGE.update(range(0x00C0, 0x0250))  # Latin Extended
        _QUERY_RANGE.update(range(0x1F600, 0x1F6C8))  # Emoji
        _QUERY_RANGE.add(ord('\n'))
        _QUERY_RANGE.add(ord('\t'))
        _QUERY_RANGE.add(ord('\r'))

        worklist = [start_closure]
        visited: set[frozenset[int]] = set()

        while worklist:
            current = worklist.pop()
            if current in visited:
                continue
            visited.add(current)

            # Collect all transitions from NFA states in current DFA state
            nfa_transitions: list[tuple[set[int] | _ExceptChars | None, int]] = []
            for nfa_state in current:
                nfa_transitions.extend(self._nfa_transitions.get(nfa_state, []))

            if not nfa_transitions:
                continue

            # Build char_to_next by iterating each char_set.
            # For _ExceptChars (NOT_LITERAL, negated IN), intersect with
            # _QUERY_RANGE to produce a concrete set. For large concrete
            # sets (> 256 elements, from CATEGORY_NOT_* etc.), also intersect.
            char_to_next: dict[int, set[int]] = {}
            for char_set, target in nfa_transitions:
                if char_set is None:
                    continue
                if isinstance(char_set, _ExceptChars):
                    effective = char_set.intersect(_QUERY_RANGE)
                elif len(char_set) > 256:
                    # Intersect with query range — characters outside this
                    # range are never queried, so omitting them is safe.
                    effective = char_set & _QUERY_RANGE
                else:
                    effective = char_set
                for ch in effective:
                    char_to_next.setdefault(ch, set()).add(target)

            for ch, target_states in char_to_next.items():
                next_dfa = self._epsilon_closure(frozenset(target_states))
                self._dfa_transitions.setdefault(current, {})[ch] = next_dfa

                if next_dfa & self._nfa_accept:
                    self._dfa_accept_states.add(next_dfa)

                if next_dfa not in visited:
                    worklist.append(next_dfa)

    def is_prefix_valid(self, text: str) -> bool:
        """Check if text is a valid prefix of some string matching the pattern.

        Returns True if text can be extended to match the pattern.
        """
        if not self._dfa_transitions:
            # DFA construction failed — fall back to regex-based check
            return self._fallback_prefix_check(text)

        state = self._dfa_start
        for ch in text:
            code = ord(ch)
            trans = self._dfa_transitions.get(state, {})
            if code not in trans:
                return False
            state = trans[code]

        # After consuming all characters, we're in a valid state.
        # The text is a valid prefix if the current state is an accept
        # state OR if there's any path from this state to an accept state.
        # A state is a dead-end if no accept state is reachable from it.
        if state in self._dfa_accept_states:
            return True
        # Check reachability: BFS from current state to any accept state
        visited = set()
        queue = [state]
        while queue:
            s = queue.pop(0)
            if s in visited:
                continue
            visited.add(s)
            trans = self._dfa_transitions.get(s, {})
            for target in trans.values():
                if target in self._dfa_accept_states:
                    return True
                queue.append(target)
        return False

    def is_full_match(self, text: str) -> bool:
        """Check if text fully matches the pattern."""
        if not self._dfa_transitions:
            return bool(self._compiled.fullmatch(text))

        state = self._dfa_start
        for ch in text:
            code = ord(ch)
            trans = self._dfa_transitions.get(state, {})
            if code not in trans:
                return False
            state = trans[code]
        return state in self._dfa_accept_states

    def valid_next_chars(self, text: str, char_range: list[int]) -> set[int]:
        """Return the set of character ordinals that are valid after text.

        For each character codepoint, checks if text + chr(cp) is a valid
        prefix or full match.
        """
        if not self._dfa_transitions:
            # DFA construction failed — use fallback
            return self._fallback_valid_chars(text, char_range)

        # Run the DFA to current position
        state = self._dfa_start
        for ch in text:
            code = ord(ch)
            trans = self._dfa_transitions.get(state, {})
            if code not in trans:
                return set()  # Dead state — no valid continuation
            state = trans[code]

        # Now check which characters lead to a valid next state
        trans = self._dfa_transitions.get(state, {})
        valid = set()
        for cp in char_range:
            if cp in trans:
                valid.add(cp)

        return valid

    def _fallback_prefix_check(self, text: str) -> bool:
        """Fallback prefix check using regex when DFA construction fails.

        Uses the approach: a string is a valid prefix if either:
        1. It's a full match, OR
        2. The pattern's match() consumes the entire string (meaning
           the string is on a valid path through the pattern)
        3. As a last resort, check if pattern + '.*' matches the text
        """
        if self._compiled.fullmatch(text):
            return True
        m = self._compiled.match(text)
        if m and m.end() == len(text):
            return True
        # Try wrapping: check if the text could be a prefix by trying
        # the original pattern with a wildcard suffix
        try:
            extended = re.compile(self._pattern + r'.*')
            return bool(extended.fullmatch(text))
        except re.error:
            return False

    def _fallback_valid_chars(self, text: str, char_range: list[int]) -> set[int]:
        """Fallback valid-next-chars when DFA is unavailable."""
        valid = set()
        for cp in char_range:
            ch = chr(cp)
            candidate = text + ch
            if self._compiled.fullmatch(candidate):
                valid.add(cp)
                continue
            m = self._compiled.match(candidate)
            if m and m.end() == len(candidate):
                valid.add(cp)
                continue
            # Try extended pattern
            try:
                extended = re.compile(self._pattern + r'.*')
                if extended.fullmatch(candidate):
                    valid.add(cp)
            except re.error:
                pass
        return valid


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
        self._compiled = re.compile(pattern)
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
        # Build DFA for prefix matching
        self._dfa = _RegexDFA(pattern)
        # Cache: buffer_text -> list of token IDs whose full decoded text
        # keeps the buffer on a valid DFA path. This is the correct prefix
        # check for multi-character tokens (single-char check via
        # _valid_next_chars is insufficient — see Bug-1 fix).
        self._valid_tokens_cache: dict[str, list[int]] = {}

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
        }

    def rollback(self, saved: dict[str, Any]) -> None:
        """Restore state from a checkpoint."""
        self._text_buffer = saved["text_buffer"]
        self._done = saved["done"]
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
        base.extend([ord('\n'), ord('\t'), ord('\r')])
        if self._query_codepoints:
            base.extend(self._query_codepoints)
            return base
        base.extend(range(0x4E00, 0x4E00 + 1000))  # CJK Unified (broader sample)
        base.extend(range(0xAC00, 0xAC00 + 100))    # Hangul
        base.extend(range(0x3040, 0x30FF))          # Hiragana + Katakana
        base.extend(range(0x0400, 0x0500))          # Cyrillic
        base.extend(range(0x0370, 0x0400))          # Greek + Coptic
        base.extend(range(0x0590, 0x0600))          # Hebrew
        base.extend(range(0x0531, 0x0590))          # Armenian
        base.extend(range(0x10A0, 0x1100))          # Georgian
        base.extend(range(0x0600, 0x0660))          # Arabic
        base.extend(range(0x0E00, 0x0E50))          # Thai
        base.extend(range(0x0900, 0x0970))          # Devanagari
        base.extend(range(0x1F600, 0x1F6C8))        # Common Emoji
        base.extend(range(0x00C0, 0x0250))          # Latin Extended
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
        cache_key = self._text_buffer
        if cache_key in self._valid_chars_cache:
            return self._valid_chars_cache[cache_key]
        if len(self._valid_chars_cache) >= _MAX_CACHE_SIZE:
            self._valid_chars_cache.clear()

        # Build the set of character codepoints to test (vocab-derived when available).
        char_range = self._probe_codepoints()
        total_tested = len(char_range)

        # Use DFA to find valid next character codepoints
        valid_codepoints = self._dfa.valid_next_chars(self._text_buffer, char_range)

        # Convert codepoints back to characters
        valid = {chr(cp) for cp in valid_codepoints}

        # If >90% of tested chars are valid, treat as unrestricted.
        # This avoids false negatives for permissive patterns like ".*".
        result = None if len(valid) > total_tested * 0.9 else valid
        self._valid_chars_cache[cache_key] = result
        return result

    def get_allowed_tokens(self, tokenizer: Any, generated_token_ids: list[int]) -> list[int]:
        if self._done:
            eos_ids = []
            if hasattr(tokenizer, 'eos_token_ids'):
                eos_ids = list(tokenizer.eos_token_ids)
            elif hasattr(tokenizer, 'eos_token_id'):
                eos_ids = [tokenizer.eos_token_id]
            return eos_ids

        # seed the DFA probe with the tokenizer's actual first-char
        # universe so non-Latin scripts aren't silently un-probed (→ empty allow-set →
        # premature EOS). Done once per (cached) tokenizer char-map.
        if self._query_codepoints is None:
            if not hasattr(self.__class__, '_token_char_cache'):
                self.__class__._token_char_cache = weakref.WeakKeyDictionary()
            if tokenizer not in self.__class__._token_char_cache:
                self.__class__._token_char_cache[tokenizer] = _build_token_char_map(tokenizer)
            _cmap = self.__class__._token_char_cache[tokenizer]
            self._query_codepoints = {ord(c) for c in _cmap if c}
            self._valid_chars_cache.clear()  # any pre-tokenizer cached probe is incomplete

        valid_chars = self._valid_next_chars()
        if valid_chars is None:
            return self._get_all_token_ids(tokenizer)
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
        if self._dfa.is_full_match(self._text_buffer):
            eos_ids = []
            if hasattr(tokenizer, 'eos_token_ids'):
                eos_ids = list(tokenizer.eos_token_ids)
            elif hasattr(tokenizer, 'eos_token_id'):
                eos_ids = [tokenizer.eos_token_id]
            if eos_ids:
                # _filter_tokens_by_dfa returns a cached list; build a new list so
                # we never mutate the cached entry with EOS ids.
                allowed = list(allowed) + [e for e in eos_ids if e not in allowed]
        return allowed

    def _get_all_token_ids(self, tokenizer: Any) -> list[int]:
        if hasattr(tokenizer, 'get_vocab'):
            return list(tokenizer.get_vocab().values())
        if hasattr(tokenizer, 'vocab') and isinstance(tokenizer.vocab, dict):
            return list(tokenizer.vocab.values())
        vocab_size = getattr(tokenizer, 'vocab_size', 32000)
        return list(range(vocab_size))

    def _find_tokens_for_chars(self, tokenizer: Any, chars: set[str]) -> list[int]:
        if not hasattr(self.__class__, '_token_char_cache'):
            self.__class__._token_char_cache = weakref.WeakKeyDictionary()
        if tokenizer not in self.__class__._token_char_cache:
            self.__class__._token_char_cache[tokenizer] = _build_token_char_map(tokenizer)

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
        cache_key = self._text_buffer
        cached = self._valid_tokens_cache.get(cache_key)
        if cached is not None:
            return cached

        # Cap cache to avoid unbounded growth on long generations.
        if len(self._valid_tokens_cache) >= 64:
            self._valid_tokens_cache.clear()

        # Build/lookup id->decoded-text map for this tokenizer.
        if not hasattr(self.__class__, '_token_text_cache'):
            self.__class__._token_text_cache = weakref.WeakKeyDictionary()
        if tokenizer not in self.__class__._token_text_cache:
            self.__class__._token_text_cache[tokenizer] = _build_token_text_map(tokenizer)
        text_map = self.__class__._token_text_cache[tokenizer]

        filtered: list[int] = []
        buf = self._text_buffer
        for tid in candidates:
            text = text_map.get(tid)
            if not text:
                # Unknown / undecodable — keep (will be re-checked on advance)
                filtered.append(tid)
                continue
            try:
                if self._dfa.is_prefix_valid(buf + text):
                    filtered.append(tid)
            except Exception:
                # On any DFA error, fall back to keeping the candidate
                filtered.append(tid)

        self._valid_tokens_cache[cache_key] = filtered
        return filtered

    def reset(self) -> None:
        self._text_buffer = ""
        self._done = False
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
        self._has_partial_match = False  # True when matched text is also a prefix of a longer choice
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

    def get_allowed_tokens(self, tokenizer: Any, generated_token_ids: list[int]) -> list[int]:
        if self._done:
            eos_ids = []
            if hasattr(tokenizer, 'eos_token_ids'):
                eos_ids = list(tokenizer.eos_token_ids)
            elif hasattr(tokenizer, 'eos_token_id'):
                eos_ids = [tokenizer.eos_token_id]
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
            eos_ids = []
            if hasattr(tokenizer, 'eos_token_ids'):
                eos_ids = list(tokenizer.eos_token_ids)
            elif hasattr(tokenizer, 'eos_token_id'):
                eos_ids = [tokenizer.eos_token_id]
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
        if not hasattr(self.__class__, '_token_char_cache'):
            self.__class__._token_char_cache = weakref.WeakKeyDictionary()
        if tokenizer not in self.__class__._token_char_cache:
            self.__class__._token_char_cache[tokenizer] = _build_token_char_map(tokenizer)
        if not hasattr(self.__class__, '_token_text_cache'):
            self.__class__._token_text_cache = weakref.WeakKeyDictionary()
        if tokenizer not in self.__class__._token_text_cache:
            self.__class__._token_text_cache[tokenizer] = _build_token_text_map(tokenizer)
        char_map = self.__class__._token_char_cache[tokenizer]
        text_map = self.__class__._token_text_cache[tokenizer]

        # Remaining continuations from the current buffer toward each choice.
        choices_cmp = self._choices if self._case_sensitive else [c.lower() for c in self._choices]
        continuations = [c[len(buf):] for c in choices_cmp if c.startswith(buf)]

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
            eos_ids = []
            if hasattr(tokenizer, 'eos_token_ids'):
                eos_ids = list(tokenizer.eos_token_ids)
            elif hasattr(tokenizer, 'eos_token_id'):
                eos_ids = [tokenizer.eos_token_id]
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


class LarkGrammarConstraint:
    """Context-free grammar constraint using Lark parser.

    Falls back gracefully if lark is not installed — in that case,
    only JSON schema and regex constraints are available.
    """

    def __init__(self, grammar: str, start_rule: str = "start") -> None:
        self._grammar_text = grammar
        self._start_rule = start_rule
        self._text_buffer = ""
        self._done = False
        self._parser = None

        try:
            from lark import Lark, Token
            self._lark_token = Token
            self._parser = Lark(
                grammar,
                start=start_rule,
                parser="earley",
                ambiguity="resolve",
            )
        except ImportError:
            logger.warning("lark not installed — CFG constraint unavailable")
        except Exception as e:
            logger.warning(f"Lark grammar parse failed: {e}")

    @property
    def state(self) -> str:
        return "done" if self._done else "active" if self._parser else "unavailable"

    @property
    def is_done(self) -> bool:
        return self._done

    def advance(self, token_text: str) -> None:
        if self._done or self._parser is None:
            return
        self._text_buffer += token_text
        try:
            self._parser.parse(self._text_buffer)
            # Full match succeeded — but check if the match can be extended.
            # Without this, patterns like `start: /[a-z]+/` terminate after
            # the first character "a" because parse("a") succeeds.
            extendable = False
            for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-.,:;!?@#$%":
                try:
                    self._parser.parse(self._text_buffer + ch)
                    extendable = True
                    break
                except Exception:
                    pass
            if not extendable:
                self._done = True
        except Exception:
            logger.debug("CFG parse incomplete, continuing generation", exc_info=True)

    def get_allowed_tokens(self, tokenizer: Any, generated_token_ids: list[int]) -> list[int]:
        if self._done or self._parser is None:
            if self._done:
                eos_ids = []
                if hasattr(tokenizer, 'eos_token_ids'):
                    eos_ids = list(tokenizer.eos_token_ids)
                elif hasattr(tokenizer, 'eos_token_id'):
                    eos_ids = [tokenizer.eos_token_id]
                return eos_ids
            return []

        # For CFG, we use a broader approach: try single-char extensions
        # and check if they produce valid partial parses
        valid_chars = self._valid_next_chars()
        if valid_chars is None:
            return _get_all_token_ids(tokenizer)
        if not valid_chars:
            return []

        if not hasattr(self.__class__, '_token_char_cache'):
            self.__class__._token_char_cache = weakref.WeakKeyDictionary()
        if tokenizer not in self.__class__._token_char_cache:
            self.__class__._token_char_cache[tokenizer] = _build_token_char_map(tokenizer)

        char_map = self.__class__._token_char_cache[tokenizer]
        candidates = set()
        for ch in valid_chars:
            if ch in char_map:
                candidates.update(char_map[ch])
        # char_map only matches the FIRST char, so a multi-char token whose
        # tail violates the grammar (e.g. "fox" for grammar "foo"|"bar") was wrongly
        # admitted. The JSON/regex/choice constraints all reject such tokens; do the
        # same here by validating the FULL token text against the parser. Single-char
        # tokens are already fully validated by valid_chars. Cap the candidate set to
        # bound cost (CFG is opt-in); fall back to first-char-only above the cap.
        _decode = getattr(tokenizer, "decode", None)
        if _decode is None or len(candidates) > 5000:
            return list(candidates)
        allowed = []
        for tid in candidates:
            try:
                ttext = _decode([tid])
            except Exception:
                continue
            if len(ttext) <= 1 or self._token_text_extends(ttext):
                allowed.append(tid)
        return allowed

    def _token_text_extends(self, text: str) -> bool:
        """True if appending the whole token ``text`` keeps the parse valid/extendable."""
        if self._parser is None:
            return True
        cand = self._text_buffer + text
        try:
            self._parser.parse(cand)
            return True
        except Exception:
            try:
                interactive = self._parser.parse_interactive(cand)
                interactive.exhaust_lexer()
                return True
            except Exception:
                return False

    def _valid_next_chars(self) -> set[str] | None:
        """Test which characters can extend the current partial parse."""
        if self._parser is None:
            return None

        has_interactive = hasattr(self._parser, 'parse_interactive')
        if not has_interactive:
            import logging
            logging.getLogger(__name__).warning(
                "LarkGrammarConstraint: parse_interactive unavailable (Lark < 1.2). "
                "CFG constraint will be permissive — all characters allowed. "
                "Upgrade to Lark >= 1.2 for proper grammar-constrained generation."
            )
            return None

        valid = set()
        test_chars = [chr(i) for i in range(32, 127)]
        test_chars.extend(['\n', '\t'])
        # CJK Unified Ideographs sample and Latin Extended sample
        test_chars.extend(chr(i) for i in range(0x4E00, 0x4E00 + 100))
        test_chars.extend(chr(i) for i in range(0x00C0, 0x00C0 + 50))

        for ch in test_chars:
            candidate = self._text_buffer + ch
            try:
                self._parser.parse(candidate)
                valid.add(ch)
            except Exception:
                try:
                    interactive = self._parser.parse_interactive(candidate)
                    interactive.exhaust_lexer()
                    valid.add(ch)
                except Exception:
                    pass

        if len(valid) > len(test_chars) - 2:
            return None
        return valid

    def checkpoint(self) -> dict[str, Any]:
        """Save current state for rollback (speculative decoding support)."""
        return {
            "text_buffer": self._text_buffer,
            "done": self._done,
        }

    def rollback(self, saved: dict[str, Any]) -> None:
        """Restore state from a checkpoint."""
        self._text_buffer = saved["text_buffer"]
        self._done = saved["done"]

    def reset(self) -> None:
        self._text_buffer = ""
        self._done = False

    def get_stats(self) -> dict[str, Any]:
        return {
            "type": "cfg",
            "has_parser": self._parser is not None,
            "buffer_len": len(self._text_buffer),
            "is_done": self._done,
        }


# ── Shared utilities ────────────────────────────────────────────────────────


def _build_token_text_map(tokenizer: Any) -> dict[int, str]:
    """Build a mapping from token_id → its decoded text (cached once).

    Used by ChoiceConstraint to filter candidate tokens by their FULL text
    against the remaining choice continuations (not just the first char).
    """
    text_map: dict[int, str] = {}
    if hasattr(tokenizer, 'get_vocab'):
        vocab = tokenizer.get_vocab()
    elif hasattr(tokenizer, 'vocab') and isinstance(tokenizer.vocab, dict):
        vocab = tokenizer.vocab
    else:
        vocab_size = getattr(tokenizer, 'vocab_size', 32000)
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

    if hasattr(tokenizer, 'get_vocab'):
        vocab = tokenizer.get_vocab()
    elif hasattr(tokenizer, 'vocab') and isinstance(tokenizer.vocab, dict):
        vocab = tokenizer.vocab
    else:
        vocab_size = getattr(tokenizer, 'vocab_size', 32000)
        vocab = {str(i): i for i in range(vocab_size)}

    for token_text, token_id in vocab.items():
        if not token_text:
            continue
        first_char = token_text[0] if token_text else ''
        try:
            decoded = tokenizer.decode([token_id])
            if decoded:
                char_map.setdefault(decoded[0], []).append(token_id)
            else:
                char_map.setdefault(first_char, []).append(token_id)
        except Exception:
            logger.debug("tokenizer decode failed for token %d in char map build", token_id, exc_info=True)
            char_map.setdefault(first_char, []).append(token_id)

    return char_map


def _build_token_text_map(tokenizer: Any) -> dict[int, str]:
    """Build a mapping from token ID to decoded text for fast lookup
    during DFA-based full-token prefix validation."""
    text_map: dict[int, str] = {}
    if hasattr(tokenizer, 'get_vocab'):
        vocab = tokenizer.get_vocab()
    elif hasattr(tokenizer, 'vocab') and isinstance(tokenizer.vocab, dict):
        vocab = tokenizer.vocab
    else:
        return text_map

    eos_ids: set[int] = set()
    if hasattr(tokenizer, 'eos_token_ids'):
        with contextlib.suppress(Exception):
            eos_ids = set(tokenizer.eos_token_ids)
    elif hasattr(tokenizer, 'eos_token_id'):
        with contextlib.suppress(Exception):
            eos_ids = {tokenizer.eos_token_id}

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
    if hasattr(tokenizer, 'get_vocab'):
        return list(tokenizer.get_vocab().values())
    if hasattr(tokenizer, 'vocab') and isinstance(tokenizer.vocab, dict):
        return list(tokenizer.vocab.values())
    return list(range(getattr(tokenizer, 'vocab_size', 32000)))


# ── Constraint Factory ──────────────────────────────────────────────────────


class ConstraintFactory:
    """Create the appropriate constraint from a grammar specification.

    Supports:
    - json_schema → JsonSchemaConstraint (from json_schema.py)
    - json_object → JsonSchemaConstraint (no schema)
    - regex → RegexConstraint
    - choice → ChoiceConstraint
    - cfg → LarkGrammarConstraint (requires lark)
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
            from .json_schema import JsonSchemaConstraint
            schema = grammar if grammar_type == "json_schema" else None
            return JsonSchemaConstraint(schema)

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
            return LarkGrammarConstraint(grammar)

        raise ValueError(f"Unknown grammar_type: {grammar_type}")
