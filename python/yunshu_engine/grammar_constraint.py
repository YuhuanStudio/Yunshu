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

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class RegexConstraint:
    """Constrains output to match a regex pattern.

    Uses incremental regex matching: after each token, checks which
    single-character extensions of the current output still have a
    valid full-match path. Only allows tokens starting with valid chars.

    This is a character-level FSM approach — for each state in the
    partial match, compute which characters can extend it.
    """

    def __init__(self, pattern: str) -> None:
        self._pattern = pattern
        self._compiled = re.compile(pattern)
        self._text_buffer = ""
        self._done = False
        self._valid_chars_cache: dict[str, set[str] | None] = {}

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
        # Check if current buffer is a full match
        m = self._compiled.fullmatch(self._text_buffer)
        if m:
            self._done = True

    def _valid_next_chars(self) -> set[str] | None:
        """Compute characters that can follow the current partial match.

        Returns None if any character is valid, or a set of valid chars.
        Results are cached per accumulated text state so repeated calls
        for the same state are O(1).

        Uses a three-pronged check:
          1. fullmatch(candidate) — char completes the pattern
          2. match(candidate) consuming ALL of candidate — candidate is a
             valid prefix (the regex matched to the end of the candidate)
          3. No match — candidate is not a valid prefix, skip
        """
        if self._done:
            return set()

        # Cache hit: same accumulated text always produces the same valid set
        cache_key = self._text_buffer
        if cache_key in self._valid_chars_cache:
            return self._valid_chars_cache[cache_key]

        # Try extending with each printable ASCII char + common whitespace
        # + common Unicode ranges (CJK, Hangul, Arabic, Thai, Emoji)
        valid = set()
        test_chars = [chr(i) for i in range(32, 127)]
        test_chars.extend(['\n', '\t', '\r'])
        # CJK Unified Ideographs (broader sample)
        for cp in range(0x4E00, 0x4E00 + 500):
            test_chars.append(chr(cp))
        # Hangul Syllables (Korean)
        for cp in range(0xAC00, 0xAC00 + 100):
            test_chars.append(chr(cp))
        # Hiragana + Katakana (Japanese)
        for cp in range(0x3040, 0x30FF):
            test_chars.append(chr(cp))
        # Arabic
        for cp in range(0x0600, 0x0660):
            test_chars.append(chr(cp))
        # Thai
        for cp in range(0x0E00, 0x0E50):
            test_chars.append(chr(cp))
        # Devanagari (Hindi)
        for cp in range(0x0900, 0x0970):
            test_chars.append(chr(cp))
        # Common Emoji (first 200)
        for cp in range(0x1F600, 0x1F6C8):
            test_chars.append(chr(cp))
        # Latin Extended
        for cp in range(0x00C0, 0x0250):
            test_chars.append(chr(cp))
        total_tested = len(test_chars)

        for ch in test_chars:
            candidate = self._text_buffer + ch
            # A char is valid if the candidate is a full match
            if self._compiled.fullmatch(candidate) is not None:
                valid.add(ch)
                continue
            # Or if candidate is a valid prefix that can be extended.
            # Key: match must consume the ENTIRE candidate string to be
            # considered a valid prefix (not just a prefix of candidate).
            m = self._compiled.match(candidate)
            if m is not None and m.end() == len(candidate):
                valid.add(ch)

        # If >90% of tested chars are valid, treat as unrestricted.
        # This avoids false negatives for permissive patterns like ".*".
        if len(valid) > total_tested * 0.9:
            result = None
        else:
            result = valid
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

        valid_chars = self._valid_next_chars()
        if valid_chars is None:
            return self._get_all_token_ids(tokenizer)
        if not valid_chars:
            return []

        return self._find_tokens_for_chars(tokenizer, valid_chars)

    def _get_all_token_ids(self, tokenizer: Any) -> list[int]:
        if hasattr(tokenizer, 'get_vocab'):
            return list(tokenizer.get_vocab().values())
        if hasattr(tokenizer, 'vocab') and isinstance(tokenizer.vocab, dict):
            return list(tokenizer.vocab.values())
        vocab_size = getattr(tokenizer, 'vocab_size', 32000)
        return list(range(vocab_size))

    def _find_tokens_for_chars(self, tokenizer: Any, chars: set[str]) -> list[int]:
        cache_key = id(tokenizer)
        if not hasattr(self.__class__, '_token_char_cache'):
            self.__class__._token_char_cache = {}
        if cache_key not in self.__class__._token_char_cache:
            self.__class__._token_char_cache[cache_key] = _build_token_char_map(tokenizer)

        char_map = self.__class__._token_char_cache[cache_key]
        allowed = set()
        for ch in chars:
            if ch in char_map:
                allowed.update(char_map[ch])
        return list(allowed)

    def reset(self) -> None:
        self._text_buffer = ""
        self._done = False
        self._valid_chars_cache.clear()

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

        # If only __eos__ is valid, return EOS tokens
        if valid_chars == {"__eos__"}:
            eos_ids = []
            if hasattr(tokenizer, 'eos_token_ids'):
                eos_ids = list(tokenizer.eos_token_ids)
            elif hasattr(tokenizer, 'eos_token_id'):
                eos_ids = [tokenizer.eos_token_id]
            return eos_ids

        # When we have a partial match (text is also a prefix of a longer
        # choice), include EOS tokens alongside continuation chars.
        has_eos = "__eos__" in valid_chars

        # Build allowed tokens from valid chars
        cache_key = id(tokenizer)
        if not hasattr(self.__class__, '_token_char_cache'):
            self.__class__._token_char_cache = {}
        if cache_key not in self.__class__._token_char_cache:
            self.__class__._token_char_cache[cache_key] = _build_token_char_map(tokenizer)

        char_map = self.__class__._token_char_cache[cache_key]
        allowed = set()
        for ch in valid_chars:
            if ch == "__eos__":
                continue
            if ch in char_map:
                allowed.update(char_map[ch])
        # When we have a partial match (completed choice is also a prefix
        # of a longer choice), include EOS tokens so the sampler can pick
        # the shorter match.
        if has_eos or self._has_partial_match:
            eos_ids = []
            if hasattr(tokenizer, 'eos_token_ids'):
                eos_ids = list(tokenizer.eos_token_ids)
            elif hasattr(tokenizer, 'eos_token_id'):
                eos_ids = [tokenizer.eos_token_id]
            allowed.update(eos_ids)
        return list(allowed)

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

        cache_key = id(tokenizer)
        if not hasattr(self.__class__, '_token_char_cache'):
            self.__class__._token_char_cache = {}
        if cache_key not in self.__class__._token_char_cache:
            self.__class__._token_char_cache[cache_key] = _build_token_char_map(tokenizer)

        char_map = self.__class__._token_char_cache[cache_key]
        allowed = set()
        for ch in valid_chars:
            if ch in char_map:
                allowed.update(char_map[ch])
        return list(allowed)

    def _valid_next_chars(self) -> set[str] | None:
        """Test which characters can extend the current partial parse."""
        if self._parser is None:
            return None

        valid = set()
        test_chars = [chr(i) for i in range(32, 127)]
        test_chars.extend(['\n', '\t'])

        for ch in test_chars:
            candidate = self._text_buffer + ch
            try:
                self._parser.parse(candidate)
                # Full parse succeeded — char completes the grammar
                valid.add(ch)
            except Exception:
                # Incomplete parse — the char might still be valid as a
                # prefix. Use parse_interactive if available (Lark >= 1.2).
                if hasattr(self._parser, 'parse_interactive'):
                    try:
                        interactive = self._parser.parse_interactive(candidate)
                        interactive.exhaust_lexer()
                        # If exhaust_lexer succeeds, the candidate is a valid prefix
                        valid.add(ch)
                    except Exception:
                        pass
                # If parse_interactive is unavailable, we cannot confirm
                # the char is a valid prefix — do NOT add it.

        if len(valid) > 90:
            return None
        return valid

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


def _build_token_char_map(tokenizer: Any) -> dict[str, list[int]]:
    """Build a mapping from first character to list of token IDs."""
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
        if first_char == '<' and len(token_text) > 1 and token_text.endswith('>'):
            continue
        try:
            decoded = tokenizer.decode([token_id])
            if decoded:
                char_map.setdefault(decoded[0], []).append(token_id)
        except Exception:
            logger.debug("tokenizer decode failed for token %d in char map build", token_id, exc_info=True)
            char_map.setdefault(first_char, []).append(token_id)

    return char_map


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
