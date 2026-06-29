"""Tests for grammar_constraint.py — regex, choice, CFG constraints."""

import pytest
from python.yunshu_engine.grammar_constraint import (
    ChoiceConstraint,
    ConstraintFactory,
    LarkGrammarConstraint,
    RegexConstraint,
)


class FakeTokenizer:
    """Minimal tokenizer with vocab for constraint testing."""

    def __init__(self):
        self._vocab = {}
        for i in range(256):
            ch = chr(i)
            self._vocab[ch] = i
        # Add some multi-char tokens
        self._vocab["hello"] = 300
        self._vocab["world"] = 301
        self._vocab["true"] = 302
        self._vocab["false"] = 303
        self.eos_token_id = 0
        self.eos_token_ids = [0]

    def get_vocab(self):
        return dict(self._vocab)

    def decode(self, ids):
        result = ""
        for i in ids:
            for text, token_id in self._vocab.items():
                if token_id == i:
                    result += text
                    break
        return result


class TestRegexConstraint:
    def test_basic_pattern(self):
        c = RegexConstraint(r"\d{3}-\d{4}")
        assert not c.is_done
        assert c.state == "active"

    def test_advance_completes(self):
        c = RegexConstraint(r"\d{3}")
        c.advance("123")
        assert c.is_done

    def test_advance_partial(self):
        c = RegexConstraint(r"\d{3}")
        c.advance("1")
        assert not c.is_done
        c.advance("23")
        assert c.is_done

    def test_advance_invalid_then_valid(self):
        c = RegexConstraint(r"[abc]+")
        c.advance("abc")
        # Unbounded pattern [abc]+ — "abc" is a full match but can be
        # extended (e.g., "abca"), so is_done must be False.  Termination
        # should come from max_tokens/stop_tokens/EOS, not the constraint.
        assert not c.is_done
        # Bounded pattern — "abc" fully matches and cannot be extended
        c2 = RegexConstraint(r"[abc]{3}")
        c2.advance("abc")
        assert c2.is_done

    def test_get_allowed_tokens_initial(self):
        tok = FakeTokenizer()
        c = RegexConstraint(r"\d+")
        allowed = c.get_allowed_tokens(tok, [])
        # Should allow digit chars (ASCII 48-57)
        for d in "0123456789":
            assert ord(d) in allowed

    def test_get_allowed_tokens_done(self):
        tok = FakeTokenizer()
        c = RegexConstraint(r"\d{2}")
        c.advance("12")
        allowed = c.get_allowed_tokens(tok, [])
        assert allowed == [0]  # EOS only

    def test_reset(self):
        c = RegexConstraint(r"\d{3}")
        c.advance("12")
        c.reset()
        assert not c.is_done
        assert c.state == "active"

    def test_get_stats(self):
        c = RegexConstraint(r"[a-z]+")
        stats = c.get_stats()
        assert stats["type"] == "regex"
        assert stats["pattern"] == r"[a-z]+"
        assert not stats["is_done"]

    def test_email_pattern(self):
        c = RegexConstraint(r"[a-z]+@[a-z]+\.[a-z]{2,4}")
        c.advance("test@example.com")
        # {2,4} allows up to 4 chars — "com" (3) can extend to "comm" (4),
        # so the constraint must NOT mark done yet.
        assert not c.is_done
        c.advance("m")  # Now "comm" = 4 chars, max of {2,4} reached
        assert c.is_done

    def test_date_pattern(self):
        c = RegexConstraint(r"\d{4}-\d{2}-\d{2}")
        c.advance("2025-01-15")
        assert c.is_done

    def test_unbounded_full_match_allows_eos(self):
        # Regression: an UNBOUNDED pattern (\d+, [a-z]+, email …) reaches a
        # COMPLETE valid match that is still extendable, so is_done stays False.
        # EOS MUST be in the allow-set at such a state, otherwise the model is
        # forced to keep emitting matching chars until max_tokens (runaway, the
        # same class as the JSON-number bug).
        tok = FakeTokenizer()  # eos id == 0
        c = RegexConstraint(r"\d+")
        c.advance("5")  # "5" fully matches \d+ but can be extended
        assert not c.is_done
        allowed = c.get_allowed_tokens(tok, [])
        assert 0 in allowed, "EOS must be allowed once buffer is a full match"
        # …and the model may still continue: digit continuations remain allowed.
        assert any(ord(d) in allowed for d in "0123456789")

    def test_unbounded_partial_match_forbids_eos(self):
        # The complement of the above: at a PARTIAL (not-yet-full) match EOS must
        # NOT be allowed, or the model could stop mid-pattern → invalid output.
        tok = FakeTokenizer()  # eos id == 0
        c = RegexConstraint(r"\d+\.\d+")
        c.advance("1")  # "1" does NOT fully match \d+\.\d+
        allowed = c.get_allowed_tokens(tok, [])
        assert 0 not in allowed, "EOS must be forbidden at a partial match"
        c.advance(".5")  # now "1.5" is a full match
        allowed2 = c.get_allowed_tokens(tok, [])
        assert 0 in allowed2


class TestChoiceConstraint:
    def test_basic_choice(self):
        c = ChoiceConstraint(["hello", "world", "hi"])
        assert not c.is_done

    def test_advance_completes(self):
        c = ChoiceConstraint(["hello", "world"])
        c.advance("hello")
        assert c.is_done

    def test_advance_partial(self):
        c = ChoiceConstraint(["hello", "world"])
        c.advance("hel")
        assert not c.is_done
        c.advance("lo")
        assert c.is_done

    def test_advance_invalid(self):
        c = ChoiceConstraint(["hello", "world"])
        c.advance("xyz")
        assert c.is_done  # invalid path → done

    def test_get_allowed_tokens_initial(self):
        tok = FakeTokenizer()
        c = ChoiceConstraint(["hello", "hi"])
        allowed = c.get_allowed_tokens(tok, [])
        # Should allow 'h' (first char of both choices)
        assert ord('h') in allowed

    def test_get_allowed_tokens_after_prefix(self):
        tok = FakeTokenizer()
        c = ChoiceConstraint(["hello", "hi"])
        c.advance("h")
        allowed = c.get_allowed_tokens(tok, [])
        # Should allow 'e' and 'i' (next chars)
        assert ord('e') in allowed
        assert ord('i') in allowed

    def test_case_insensitive(self):
        c = ChoiceConstraint(["Hello", "World"], case_sensitive=False)
        c.advance("hello")
        assert c.is_done

    def test_done_returns_eos(self):
        tok = FakeTokenizer()
        c = ChoiceConstraint(["ok"])
        c.advance("ok")
        allowed = c.get_allowed_tokens(tok, [])
        assert allowed == [0]  # EOS

    def test_reset(self):
        c = ChoiceConstraint(["hello"])
        c.advance("hel")
        c.reset()
        assert not c.is_done

    def test_get_stats(self):
        c = ChoiceConstraint(["a", "b", "c"])
        stats = c.get_stats()
        assert stats["type"] == "choice"
        assert stats["num_choices"] == 3


class TestLarkGrammarConstraint:
    def test_init_without_lark(self):
        """Should initialize gracefully without lark."""
        c = LarkGrammarConstraint("start: NUMBER\nNUMBER: /[0-9]+/")
        # May or may not have parser depending on env
        assert c.state in ("active", "unavailable")

    def test_is_done_initially_false(self):
        c = LarkGrammarConstraint("start: \"hello\"")
        assert not c.is_done

    def test_reset(self):
        c = LarkGrammarConstraint("start: \"hello\"")
        c.reset()
        assert not c.is_done

    def test_get_stats(self):
        c = LarkGrammarConstraint("start: \"test\"")
        stats = c.get_stats()
        assert stats["type"] == "cfg"
        assert "buffer_len" in stats


class TestConstraintFactory:
    def test_create_regex(self):
        c = ConstraintFactory.create("regex", r"\d+")
        assert isinstance(c, RegexConstraint)

    def test_create_choice(self):
        c = ConstraintFactory.create("choice", ["a", "b", "c"])
        assert isinstance(c, ChoiceConstraint)

    def test_create_json_schema(self):
        from python.yunshu_engine.json_schema import JsonSchemaConstraint
        c = ConstraintFactory.create("json_schema", {"type": "object"})
        assert isinstance(c, JsonSchemaConstraint)

    def test_create_json_object(self):
        from python.yunshu_engine.json_schema import JsonSchemaConstraint
        c = ConstraintFactory.create("json_object")
        assert isinstance(c, JsonSchemaConstraint)

    def test_create_cfg(self):
        c = ConstraintFactory.create("cfg", 'start: "hello"')
        assert isinstance(c, LarkGrammarConstraint)

    def test_regex_requires_string(self):
        with pytest.raises(ValueError, match="string pattern"):
            ConstraintFactory.create("regex", 123)

    def test_choice_requires_list(self):
        with pytest.raises(ValueError, match="list of strings"):
            ConstraintFactory.create("choice", "not a list")

    def test_cfg_requires_string(self):
        with pytest.raises(ValueError, match="grammar string"):
            ConstraintFactory.create("cfg", 123)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown grammar_type"):
            ConstraintFactory.create("unknown")


class TestChoiceConstraintCaseInsensitive:
    """Tests for case-insensitive ChoiceConstraint bug fixes."""

    def test_case_insensitive_allows_uppercase_start(self):
        """Bug fix: case-insensitive matching must allow uppercase tokens.

        Previously, the trie was built with lowercase keys and get_allowed_tokens
        only looked up trie keys (lowercase) in the token char map. This meant
        tokens starting with uppercase letters were never allowed even though
        case_sensitive=False.
        """
        cc = ChoiceConstraint(["Hello", "World"], case_sensitive=False)
        tokenizer = FakeTokenizer()
        # At the start, valid next chars should include both 'h'/'w' (trie keys)
        # AND 'H'/'W' (uppercase versions) for case-insensitive matching.
        allowed = cc.get_allowed_tokens(tokenizer, [])
        # 'H' = ord(72), 'h' = ord(104), 'W' = ord(87), 'w' = ord(119)
        assert 72 in allowed, "uppercase H should be allowed"
        assert 87 in allowed, "uppercase W should be allowed"

    def test_case_insensitive_matches_uppercase(self):
        """Case-insensitive matching should accept uppercase input."""
        cc = ChoiceConstraint(["Hello", "World"], case_sensitive=False)
        cc.advance("H")
        assert not cc.is_done
        cc.advance("e")
        cc.advance("l")
        cc.advance("l")
        cc.advance("o")
        assert cc.is_done

    def test_case_insensitive_allows_mixed_case(self):
        """Case-insensitive matching should accept mixed case."""
        cc = ChoiceConstraint(["Hello"], case_sensitive=False)
        cc.advance("h")
        cc.advance("E")
        cc.advance("l")
        cc.advance("L")
        cc.advance("o")
        assert cc.is_done


class TestRegexConstraintDFAMemory:
    """Tests for regex DFA memory optimization (_ExceptChars)."""

    def test_not_literal_matches_correctly(self):
        """NOT_LITERAL should work correctly with compact representation."""
        rc = RegexConstraint(r"a.c")  # a, any char, c
        rc.advance("a")
        assert not rc.is_done
        rc.advance("x")
        assert not rc.is_done
        rc.advance("c")
        assert rc.is_done

    def test_negated_charset_matches_correctly(self):
        """Negated char class [^x] should work with compact representation."""
        rc = RegexConstraint(r"[^a-z]")  # not lowercase
        rc.advance("A")
        assert rc.is_done

    def test_negated_charset_rejects_member(self):
        """Negated char class should reject member characters."""
        rc = RegexConstraint(r"[^a-z]")
        # After advancing a lowercase letter, the constraint should fail
        # (no valid continuation)
        rc.advance("a")
        # The DFA should be in a dead state — no valid next chars
        tokenizer = FakeTokenizer()
        allowed = rc.get_allowed_tokens(tokenizer, [])
        # Should only return EOS (or nothing) since we're in a dead state
        # The buffer "a" is not a valid prefix for [^a-z]
        assert len(allowed) <= 1  # at most EOS

    def test_digit_pattern_with_compact_dfa(self):
        """\\d pattern should work correctly (uses CATEGORY_DIGIT, not negated)."""
        rc = RegexConstraint(r"\d{3}")
        rc.advance("1")
        assert not rc.is_done
        rc.advance("2")
        assert not rc.is_done
        rc.advance("3")
        assert rc.is_done
