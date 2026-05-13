"""Tests for grammar_constraint.py — regex, choice, CFG constraints."""

import pytest

from python.yunshu_engine.grammar_constraint import (
    ChoiceConstraint,
    ConstraintFactory,
    RegexConstraint,
    LarkGrammarConstraint,
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
        assert c.is_done

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
        assert c.is_done

    def test_date_pattern(self):
        c = RegexConstraint(r"\d{4}-\d{2}-\d{2}")
        c.advance("2025-01-15")
        assert c.is_done


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
