"""Tests for grammar_bitmask.py — bitmask-based grammar engine."""

import os
import pytest

from python.yunshu_engine.grammar_bitmask import (
    BitmaskApplicator,
    BitmaskConstrainedSampler,
    GrammarBitmaskEngine,
    TokenStringTable,
    build_bitmask_engine,
    is_bitmask_enabled,
)


class FakeTokenizer:
    """Minimal tokenizer with vocab for bitmask testing.

    Uses non-overlapping IDs: ASCII chars get their char code as ID,
    special JSON tokens get IDs in the 400+ range.  No ID collisions.
    """

    def __init__(self):
        self._vocab = {}
        # ASCII chars 0-126 mapped to their code points
        for i in range(127):
            self._vocab[chr(i)] = i
        # Special multi-char tokens with unique IDs above 127
        self._vocab["hello"] = 300
        self._vocab["world"] = 301
        self._vocab["true"] = 302
        self._vocab["false"] = 303
        self._vocab["name"] = 500
        self._vocab["age"] = 501
        # Special tokens
        self._vocab["<eos>"] = 2  # EOS uses id 2, same as chr(2) below
        self.eos_token_id = 2
        self.eos_token_ids = [2]
        self._vocab_size = max(self._vocab.values()) + 1

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


# ── TokenStringTable ────────────────────────────────────────────────────────


class TestTokenStringTable:
    def setup_method(self):
        TokenStringTable._cache.clear()

    def test_build_from_tokenizer(self):
        tok = FakeTokenizer()
        table = TokenStringTable.get(tok)
        assert table.vocab_size > 256
        assert len(table.full_vocab_ids) == table.vocab_size
        assert len(table.id_to_string) == table.vocab_size

    def test_cached_by_id(self):
        tok = FakeTokenizer()
        table1 = TokenStringTable.get(tok)
        table2 = TokenStringTable.get(tok)
        assert table1 is table2

    def test_char_to_ids_mapping(self):
        tok = FakeTokenizer()
        table = TokenStringTable.get(tok)
        # "{" is chr(123), mapped to id 123
        assert 123 in table.char_to_ids.get("{", [])
        # '"' is chr(34), mapped to id 34
        assert 34 in table.char_to_ids.get('"', [])

    def test_ids_for_chars(self):
        tok = FakeTokenizer()
        table = TokenStringTable.get(tok)
        ids = table.ids_for_chars({"}"})
        # "}" is chr(125), mapped to id 125
        assert 125 in ids

    def test_eos_ids(self):
        tok = FakeTokenizer()
        table = TokenStringTable.get(tok)
        assert table.eos_ids == [2]


# ── BitmaskApplicator ──────────────────────────────────────────────────────


class TestBitmaskApplicator:
    def test_apply_all_allowed(self):
        import mlx.core as mx

        app = BitmaskApplicator(10)
        logits = mx.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        mask = mx.ones((10,), dtype=mx.bool_)
        result = app.apply(logits, mask)
        # All logits unchanged
        for i in range(10):
            assert float(result[i]) == pytest.approx(float(logits[i]), rel=1e-4)

    def test_apply_none_allowed(self):
        import mlx.core as mx

        app = BitmaskApplicator(10)
        logits = mx.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        mask = mx.zeros((10,), dtype=mx.bool_)
        result = app.apply(logits, mask)
        # All-False bitmask now falls back to argmax (token 9 = 10.0) to avoid NaN.
        # Token 9 should keep its value, all others should be -inf.
        assert float(result[9]) == pytest.approx(10.0, rel=1e-4)
        for i in range(9):
            assert float(result[i]) < -1e10, f"token {i} should be -inf but got {float(result[i])}"

    def test_apply_partial_mask(self):
        import mlx.core as mx

        app = BitmaskApplicator(5)
        logits = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = mx.array([True, False, True, False, True])
        result = app.apply(logits, mask)
        assert float(result[0]) == pytest.approx(1.0, rel=1e-4)
        assert float(result[1]) == float("-inf")
        assert float(result[2]) == pytest.approx(3.0, rel=1e-4)
        assert float(result[3]) == float("-inf")
        assert float(result[4]) == pytest.approx(5.0, rel=1e-4)

    def test_apply_allowlist(self):
        import mlx.core as mx

        app = BitmaskApplicator(5)
        logits = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = app.apply_allowlist(logits, [0, 2, 4])
        assert float(result[0]) == pytest.approx(1.0, rel=1e-4)
        assert float(result[1]) == float("-inf")
        assert float(result[2]) == pytest.approx(3.0, rel=1e-4)
        assert float(result[3]) == float("-inf")
        assert float(result[4]) == pytest.approx(5.0, rel=1e-4)


# ── GrammarBitmaskEngine ───────────────────────────────────────────────────


class TestGrammarBitmaskEngine:
    def setup_method(self):
        TokenStringTable._cache.clear()

    def test_json_schema_basic(self):
        from python.yunshu_engine.json_schema import JsonSchemaConstraint, JsonState

        constraint = JsonSchemaConstraint({"type": "object"})
        engine = GrammarBitmaskEngine(constraint)
        assert not engine.is_done
        assert engine.state == JsonState.START

    def test_json_schema_bitmask(self):
        import mlx.core as mx
        from python.yunshu_engine.json_schema import JsonSchemaConstraint

        tok = FakeTokenizer()
        constraint = JsonSchemaConstraint({"type": "object"})
        engine = GrammarBitmaskEngine(constraint)

        bitmask = engine.compute_bitmask(tok)
        assert bitmask.dtype == mx.bool_
        assert bitmask.shape[0] == tok._vocab_size
        # Token 123 ("{") should be allowed at start
        assert bool(bitmask[123]) is True
        # Token 125 ("}") should NOT be allowed at start
        assert bool(bitmask[125]) is False

    def test_json_schema_advance(self):
        from python.yunshu_engine.json_schema import JsonSchemaConstraint, JsonState

        constraint = JsonSchemaConstraint({"type": "object"})
        engine = GrammarBitmaskEngine(constraint)

        engine.advance("{")
        assert engine.state == JsonState.OBJECT_OPEN

    def test_json_schema_done(self):
        import mlx.core as mx
        from python.yunshu_engine.json_schema import JsonSchemaConstraint

        tok = FakeTokenizer()
        constraint = JsonSchemaConstraint({"type": "object"})
        engine = GrammarBitmaskEngine(constraint)

        engine.advance("{}")
        assert engine.is_done

        bitmask = engine.compute_bitmask(tok)
        # Only EOS should be allowed
        assert bool(bitmask[2]) is True
        # Regular tokens should not be allowed
        assert bool(bitmask[123]) is False

    def test_regex_constraint(self):
        from python.yunshu_engine.grammar_constraint import RegexConstraint

        constraint = RegexConstraint(r"\d{3}")
        engine = GrammarBitmaskEngine(constraint)
        assert not engine.is_done

        engine.advance("123")
        assert engine.is_done

    def test_choice_constraint(self):
        from python.yunshu_engine.grammar_constraint import ChoiceConstraint

        constraint = ChoiceConstraint(["hello", "world"])
        engine = GrammarBitmaskEngine(constraint)
        assert not engine.is_done

        engine.advance("hello")
        assert engine.is_done

    def test_reset(self):
        from python.yunshu_engine.json_schema import JsonSchemaConstraint, JsonState

        constraint = JsonSchemaConstraint({"type": "object"})
        engine = GrammarBitmaskEngine(constraint)
        engine.advance("{")
        assert engine.state == JsonState.OBJECT_OPEN

        engine.reset()
        assert engine.state == JsonState.START
        assert not engine.is_done

    def test_checkpoint_rollback(self):
        from python.yunshu_engine.json_schema import JsonSchemaConstraint, JsonState

        constraint = JsonSchemaConstraint({"type": "object"})
        engine = GrammarBitmaskEngine(constraint)

        engine.checkpoint()
        engine.advance("{")
        assert engine.state == JsonState.OBJECT_OPEN

        engine.rollback()
        assert engine.state == JsonState.START

    def test_get_stats(self):
        from python.yunshu_engine.json_schema import JsonSchemaConstraint

        constraint = JsonSchemaConstraint({"type": "object"})
        engine = GrammarBitmaskEngine(constraint)
        stats = engine.get_stats()
        assert stats["bitmask_engine"] is True

    def test_get_allowed_tokens_compat(self):
        tok = FakeTokenizer()
        from python.yunshu_engine.json_schema import JsonSchemaConstraint

        constraint = JsonSchemaConstraint({"type": "object"})
        engine = GrammarBitmaskEngine(constraint)

        allowed = engine.get_allowed_tokens(tok, [])
        assert 123 in allowed  # "{" token (chr(123))


# ── BitmaskConstrainedSampler ──────────────────────────────────────────────


class TestBitmaskConstrainedSampler:
    def setup_method(self):
        TokenStringTable._cache.clear()

    def test_constrains_to_json_start(self):
        import mlx.core as mx

        tok = FakeTokenizer()
        from python.yunshu_engine.json_schema import JsonSchemaConstraint

        constraint = JsonSchemaConstraint({"type": "object"})
        engine = GrammarBitmaskEngine(constraint)

        # Sampler: always pick argmax
        def base_sampler(logits):
            return mx.argmax(logits)

        sampler = BitmaskConstrainedSampler(base_sampler, engine, tok)

        # Logits: token 125 ("}") has highest logit, but should be masked
        logits = mx.zeros((tok._vocab_size,))
        logits[125] = 10.0  # "}" — should be blocked
        logits[123] = 5.0  # "{" — should be selected

        token = sampler(logits)
        assert int(token) == 123  # Should pick "{"

    def test_done_forces_eos(self):
        import mlx.core as mx

        tok = FakeTokenizer()
        from python.yunshu_engine.json_schema import JsonSchemaConstraint

        constraint = JsonSchemaConstraint({"type": "object"})
        engine = GrammarBitmaskEngine(constraint)

        def base_sampler(logits):
            return mx.argmax(logits)

        sampler = BitmaskConstrainedSampler(base_sampler, engine, tok)

        # Advance to done
        engine.advance("{}")
        assert engine.is_done

        # Even if another token has higher logits, should pick EOS
        logits = mx.zeros((tok._vocab_size,))
        logits[123] = 10.0  # "{" — should be blocked
        logits[2] = 1.0  # EOS

        token = sampler(logits)
        assert int(token) == 2  # Should pick EOS


# ── build_bitmask_engine factory ───────────────────────────────────────────


class TestBuildBitmaskEngine:
    def test_json_schema(self):
        engine = build_bitmask_engine("json_schema", {"type": "object"})
        assert isinstance(engine, GrammarBitmaskEngine)
        assert not engine.is_done

    def test_json_object(self):
        engine = build_bitmask_engine("json_object")
        assert isinstance(engine, GrammarBitmaskEngine)

    def test_json_schema_string(self):
        engine = build_bitmask_engine("json_schema", '{"type": "object"}')
        assert isinstance(engine, GrammarBitmaskEngine)

    def test_regex(self):
        engine = build_bitmask_engine("regex", r"\d{3}")
        assert isinstance(engine, GrammarBitmaskEngine)

    def test_choice(self):
        engine = build_bitmask_engine("choice", ["a", "b"])
        assert isinstance(engine, GrammarBitmaskEngine)

    def test_regex_requires_string(self):
        with pytest.raises(ValueError, match="string pattern"):
            build_bitmask_engine("regex", 123)

    def test_choice_requires_list(self):
        with pytest.raises(ValueError, match="list of strings"):
            build_bitmask_engine("choice", "not a list")

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown"):
            build_bitmask_engine("unknown")


# ── is_bitmask_enabled ─────────────────────────────────────────────────────


class TestIsBitmaskEnabled:
    def test_default_off(self):
        os.environ.pop("YUNSHU_GRAMMAR_BITMASK", None)
        assert is_bitmask_enabled() is False

    def test_enabled(self):
        os.environ["YUNSHU_GRAMMAR_BITMASK"] = "1"
        try:
            assert is_bitmask_enabled() is True
        finally:
            del os.environ["YUNSHU_GRAMMAR_BITMASK"]

    def test_not_one(self):
        os.environ["YUNSHU_GRAMMAR_BITMASK"] = "0"
        try:
            assert is_bitmask_enabled() is False
        finally:
            del os.environ["YUNSHU_GRAMMAR_BITMASK"]


# ── Integration: _build_constrained_sampler with bitmask ───────────────────


class TestBuildConstrainedBitmask:
    def setup_method(self):
        TokenStringTable._cache.clear()

    def test_bitmask_path_selected(self):
        """When YUNSHU_GRAMMAR_BITMASK=1, _build_constrained_sampler uses bitmask."""
        from python.yunshu_engine.batched_engine import _build_constrained_sampler

        os.environ["YUNSHU_GRAMMAR_BITMASK"] = "1"
        try:
            import mlx.core as mx

            tok = FakeTokenizer()

            def base_sampler(logits):
                return mx.argmax(logits)

            sampler = _build_constrained_sampler(
                base_sampler,
                {"type": "object"},
                tok,
            )
            assert isinstance(sampler, BitmaskConstrainedSampler)
        finally:
            del os.environ["YUNSHU_GRAMMAR_BITMASK"]

    def test_standard_path_when_disabled(self):
        """Without YUNSHU_GRAMMAR_BITMASK, standard ConstrainedSampler is used."""
        from python.yunshu_engine.batched_engine import _build_constrained_sampler
        from python.yunshu_engine.json_schema import ConstrainedSampler

        os.environ.pop("YUNSHU_GRAMMAR_BITMASK", None)
        import mlx.core as mx

        tok = FakeTokenizer()

        def base_sampler(logits):
            return mx.argmax(logits)

        sampler = _build_constrained_sampler(
            base_sampler,
            {"type": "object"},
            tok,
        )
        assert isinstance(sampler, ConstrainedSampler)
        assert not isinstance(sampler, BitmaskConstrainedSampler)

    def test_bitmask_grammar_types(self):
        """Bitmask path works with regex/choice grammar types too."""
        from python.yunshu_engine.batched_engine import _build_constrained_sampler

        os.environ["YUNSHU_GRAMMAR_BITMASK"] = "1"
        try:
            import mlx.core as mx

            tok = FakeTokenizer()

            def base_sampler(logits):
                return mx.argmax(logits)

            sampler = _build_constrained_sampler(
                base_sampler,
                {"type": "regex", "pattern": r"\d+"},
                tok,
            )
            assert isinstance(sampler, BitmaskConstrainedSampler)
        finally:
            del os.environ["YUNSHU_GRAMMAR_BITMASK"]
