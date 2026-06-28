"""Tests for JSON Schema constrained generation.

Tests cover:
- JsonSchemaConstraint state machine transitions
- Simple objects, nested objects, arrays
- Typed values (string, number, boolean, null)
- apply_json_constraint() logit masking
- ConstrainedSampler integration
- Response format parameter parsing
- Edge cases: empty schema, complex nested schema, enum values
"""


from yunshu_engine.json_schema import (
    ConstrainedSampler,
    JsonSchemaConstraint,
    JsonState,
    apply_json_constraint,
    make_constrained_sampler,
)
from yunshu_engine.request import SamplingParams

# ── Helpers ─────────────────────────────────────────────────────────────────


class FakeTokenizer:
    """Minimal tokenizer for testing token mapping.

    Uses ASCII chars as single-char tokens (IDs 0-126), plus
    multi-char tokens starting at ID 128 to avoid collisions.
    """

    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.eos_token_ids = [0]
        # Build a simple vocab: ASCII chars as token text
        self._vocab = {}
        for i in range(127):
            self._vocab[chr(i)] = i
        # Add multi-char tokens starting at 128 to avoid collision with ASCII
        self._vocab["true"] = 128
        self._vocab["false"] = 129
        self._vocab["null"] = 130
        self._vocab["name"] = 131
        self._vocab["age"] = 132

    def get_vocab(self):
        return dict(self._vocab)

    def decode(self, token_ids):
        reverse = {v: k for k, v in self._vocab.items()}
        return "".join(reverse.get(tid, "") for tid in token_ids)


# ── JsonSchemaConstraint State Machine Tests ────────────────────────────────


class TestJsonSchemaConstraintBasicObject:
    """Test basic JSON object state transitions."""

    def test_starts_in_start_state(self):
        c = JsonSchemaConstraint()
        assert c.state == JsonState.START

    def test_start_expects_open_brace(self):
        c = JsonSchemaConstraint()
        chars = c._get_expected_chars()
        assert '{' in chars

    def test_advance_with_open_brace(self):
        c = JsonSchemaConstraint()
        c.advance("{")
        assert c.state == JsonState.OBJECT_OPEN

    def test_object_open_expects_quote_or_close(self):
        c = JsonSchemaConstraint()
        c.advance("{")
        chars = c._get_expected_chars()
        assert '"' in chars
        assert '}' in chars

    def test_object_open_with_close(self):
        c = JsonSchemaConstraint()
        c.advance("{")
        c.advance("}")
        assert c.state == JsonState.DONE

    def test_empty_object(self):
        c = JsonSchemaConstraint()
        c.advance("{")
        c.advance("}")
        assert c.is_done

    def test_simple_key_value_string(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        })
        c.advance('{')
        assert c.state == JsonState.OBJECT_OPEN
        c.advance('"name"')
        assert c.state == JsonState.OBJECT_COLON
        c.advance(':')
        assert c.state == JsonState.OBJECT_VALUE
        c.advance('"John"')
        # Should be in OBJECT_COMMA state now
        assert c.state == JsonState.OBJECT_COMMA
        c.advance('}')
        assert c.is_done

    def test_whitespace_between_tokens(self):
        c = JsonSchemaConstraint()
        c.advance("{")
        c.advance("  ")
        c.advance('"key"')
        c.advance(" : ")
        c.advance('"value"')
        c.advance(" }")
        assert c.is_done


class TestJsonSchemaConstraintNestedObjects:
    """Test nested JSON object state transitions."""

    def test_nested_object(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        })
        c.advance('{')
        c.advance('"user"')
        c.advance(':')
        c.advance('{')
        assert c.state == JsonState.OBJECT_OPEN  # nested object
        c.advance('"name"')
        c.advance(':')
        c.advance('"Alice"')
        assert c.state == JsonState.OBJECT_COMMA
        c.advance('}')
        assert c.state == JsonState.OBJECT_COMMA  # back to outer object
        c.advance('}')
        assert c.is_done

    def test_deeply_nested(self):
        c = JsonSchemaConstraint()
        c.advance('{')
        c.advance('"a"')
        c.advance(':')
        c.advance('{')
        c.advance('"b"')
        c.advance(':')
        c.advance('{')
        c.advance('"c"')
        c.advance(':')
        c.advance('"d"')
        c.advance('}')
        c.advance('}')
        c.advance('}')
        assert c.is_done


class TestJsonSchemaConstraintArrays:
    """Test JSON array state transitions."""

    def test_empty_array(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}}
            }
        })
        c.advance('{')
        c.advance('"items"')
        c.advance(':')
        c.advance('[')
        c.advance(']')
        assert c.state == JsonState.OBJECT_COMMA
        c.advance('}')
        assert c.is_done

    def test_string_array(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}}
            }
        })
        c.advance('{')
        c.advance('"tags"')
        c.advance(':')
        c.advance('[')
        c.advance('"a"')
        c.advance(',')
        c.advance('"b"')
        c.advance(',')
        c.advance('"c"')
        c.advance(']')
        assert c.state == JsonState.OBJECT_COMMA
        c.advance('}')
        assert c.is_done

    def test_number_array(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {
                "nums": {"type": "array", "items": {"type": "integer"}}
            }
        })
        c.advance('{')
        c.advance('"nums"')
        c.advance(':')
        c.advance('[')
        c.advance('1')
        # Number ends when non-number char is seen
        # But we need to handle the continuation properly

    def test_top_level_array_not_yet_done(self):
        """Top-level arrays should also be handled via schema."""
        c = JsonSchemaConstraint({"type": "array", "items": {"type": "string"}})
        # This should default to object since START expects {
        # For array support, schema type "array" needs special handling
        assert c.state == JsonState.START

    @staticmethod
    def _feed_all_accepted(c, text):
        """Feed text char-by-char; a None expected-set means 'unconstrained
        continuation' (e.g. mid-number) — only a non-None set that excludes the
        char is a real rejection."""
        for i, ch in enumerate(text):
            exp = c._get_expected_chars()
            if exp is not None and ch not in exp:
                return f"REJECT {ch!r} at {i} (expected {sorted(exp)[:8]})"
            c.advance(ch)
        return "ACCEPT"

    def test_array_no_items_accepts_any_type(self):
        """an array with NO `items` means 'any type' (JSON Schema);
        defaulting to string forbade [1,2,3] etc. — masking valid output."""
        assert self._feed_all_accepted(
            JsonSchemaConstraint({"type": "array"}), "[1, 2, 3]") == "ACCEPT"
        assert self._feed_all_accepted(
            JsonSchemaConstraint({"type": "array"}), '[1, "a", true, null, 2.5]') == "ACCEPT"

    def test_prefixitems_tuple_accepted(self):
        """prefixItems tuple validation — the leading non-string
        element was rejected because only `items` was understood."""
        c = JsonSchemaConstraint({
            "type": "array",
            "prefixItems": [{"type": "integer"}, {"type": "string"}],
        })
        assert self._feed_all_accepted(c, '[1, "a"]') == "ACCEPT"

    def test_typed_items_still_strict(self):
        """Regression guard: the no-items='any' fix must NOT loosen a typed
        array — items:integer must still reject a string element."""
        c = JsonSchemaConstraint({"type": "array", "items": {"type": "integer"}})
        c.advance("[")
        assert '"' not in c._get_expected_chars()  # string start forbidden
        assert "1" in c._get_expected_chars()


class TestJsonSchemaConstraintTypedValues:
    """Test typed value states (string, number, boolean, null)."""

    def test_string_value(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {"name": {"type": "string"}}
        })
        c.advance('{')
        c.advance('"name"')
        c.advance(':')
        assert c.state == JsonState.OBJECT_VALUE
        chars = c._get_expected_chars()
        assert '"' in chars  # string should start with quote

    def test_number_value(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {"age": {"type": "integer"}}
        })
        c.advance('{')
        c.advance('"age"')
        c.advance(':')
        chars = c._get_expected_chars()
        assert any(c in chars for c in '0123456789-')

    def test_boolean_value(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {"active": {"type": "boolean"}}
        })
        c.advance('{')
        c.advance('"active"')
        c.advance(':')
        chars = c._get_expected_chars()
        assert 't' in chars
        assert 'f' in chars

    def test_null_value(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {"data": {"type": "null"}}
        })
        c.advance('{')
        c.advance('"data"')
        c.advance(':')
        chars = c._get_expected_chars()
        assert 'n' in chars

    def test_boolean_true_advance(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {"flag": {"type": "boolean"}}
        })
        c.advance('{')
        c.advance('"flag"')
        c.advance(':')
        c.advance('true')
        # Should have transitioned to OBJECT_COMMA
        assert c.state == JsonState.OBJECT_COMMA

    def test_boolean_false_advance(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {"flag": {"type": "boolean"}}
        })
        c.advance('{')
        c.advance('"flag"')
        c.advance(':')
        c.advance('false')
        assert c.state == JsonState.OBJECT_COMMA

    def test_null_advance(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {"data": {"type": "null"}}
        })
        c.advance('{')
        c.advance('"data"')
        c.advance(':')
        c.advance('null')
        assert c.state == JsonState.OBJECT_COMMA


class TestJsonNumberStateMachine:
    """Number sub-state transitions (NUMBER_FRACTION / NUMBER_EXPONENT edges).

    These exercise the float number state machine that is harder to reach
    via the integer-typed property tests above.
    """

    def _number_constraint(self):
        # 'value' typed as number permits floats and exponents
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {"value": {"type": "number"}},
        })
        c.advance('{')
        c.advance('"value"')
        c.advance(':')
        return c

    def test_fraction_requires_digit_after_dot(self):
        c = self._number_constraint()
        c.advance('3')
        c.advance('.')
        assert c.state == JsonState.NUMBER_FRACTION
        # In NUMBER_FRACTION only digits are valid next
        chars = c._get_expected_chars()
        assert all(ch in '0123456789' for ch in chars)
        assert '.' not in chars

    def test_fraction_digit_returns_to_number(self):
        c = self._number_constraint()
        c.advance('3')
        c.advance('.')
        c.advance('5')
        # after a fraction digit we transition back to NUMBER (exponent allowed)
        assert c.state == JsonState.NUMBER
        chars = c._get_expected_chars()
        assert 'e' in chars or 'E' in chars
        # second dot must not be allowed once a dot was seen
        assert '.' not in chars

    def test_exponent_after_digit(self):
        c = self._number_constraint()
        c.advance('1')
        c.advance('e')
        assert c.state == JsonState.NUMBER_EXPONENT
        chars = c._get_expected_chars()
        # sign or digits valid before exponent digit
        assert '+' in chars and '-' in chars
        assert '5' in chars

    def test_exponent_sign_then_digit(self):
        c = self._number_constraint()
        c.advance('1')
        c.advance('e')
        c.advance('+')
        assert c.state == JsonState.NUMBER_EXPONENT_SIGN
        # only digits valid after the sign
        chars = c._get_expected_chars()
        assert all(ch in '0123456789' for ch in chars)
        c.advance('5')
        assert c.state == JsonState.NUMBER_EXPONENT

    def test_exponent_no_second_sign(self):
        c = self._number_constraint()
        c.advance('1')
        c.advance('e')
        c.advance('5')  # exponent digit seen
        # after an exponent digit, '+'/'-' are no longer valid
        chars = c._get_expected_chars()
        assert '+' not in chars
        assert '-' not in chars

    def test_full_float_with_exponent_completes(self):
        c = self._number_constraint()
        c.advance('3')
        c.advance('.')
        c.advance('1')
        c.advance('4')
        c.advance('e')
        c.advance('-')
        c.advance('2')
        c.advance('}')
        assert c.is_done

    def test_leading_zero_then_fraction(self):
        c = self._number_constraint()
        c.advance('0')
        assert c.state == JsonState.NUMBER_ZERO
        chars = c._get_expected_chars()
        # leading zero allows '.' / 'eE' / terminators but no more digits
        assert '.' in chars
        c.advance('.')
        assert c.state == JsonState.NUMBER_FRACTION
        c.advance('5')
        c.advance('}')
        assert c.is_done

    def test_integer_type_rejects_dot_and_exponent(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {"n": {"type": "integer"}},
        })
        c.advance('{')
        c.advance('"n"')
        c.advance(':')
        c.advance('1')
        chars = c._get_expected_chars()
        assert '.' not in chars
        assert 'e' not in chars and 'E' not in chars


class TestJsonSchemaConstraintMultipleKeys:
    """Test objects with multiple keys."""

    def test_two_string_keys(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {
                "first": {"type": "string"},
                "last": {"type": "string"},
            }
        })
        c.advance('{')
        c.advance('"first"')
        c.advance(':')
        c.advance('"John"')
        assert c.state == JsonState.OBJECT_COMMA
        c.advance(',')
        assert c.state == JsonState.OBJECT_KEY
        c.advance('"last"')
        c.advance(':')
        c.advance('"Doe"')
        c.advance('}')
        assert c.is_done

    def test_mixed_types(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "active": {"type": "boolean"},
            }
        })
        c.advance('{')
        c.advance('"name"')
        c.advance(':')
        c.advance('"Alice"')
        c.advance(',')
        c.advance('"age"')
        c.advance(':')
        c.advance('30')
        # Number state transitions... need special handling
        # Let's test that the key parts work


class TestJsonSchemaConstraintReset:
    """Test constraint reset."""

    def test_reset_returns_to_start(self):
        c = JsonSchemaConstraint()
        c.advance('{')
        c.advance('"key"')
        c.reset()
        assert c.state == JsonState.START
        assert not c.is_done

    def test_reset_allows_reuse(self):
        c = JsonSchemaConstraint()
        c.advance('{')
        c.advance('}')
        assert c.is_done
        c.reset()
        assert c.state == JsonState.START
        c.advance('{')
        c.advance('}')
        assert c.is_done


class TestJsonSchemaConstraintNoSchema:
    """Test with no schema (generic JSON object)."""

    def test_no_schema_defaults_to_object(self):
        c = JsonSchemaConstraint()
        chars = c._get_expected_chars()
        assert '{' in chars

    def test_no_schema_accepts_any_value_types(self):
        c = JsonSchemaConstraint()
        c.advance('{')
        c.advance('"key"')
        c.advance(':')
        # Should accept any value type
        chars = c._get_expected_chars()
        assert '"' in chars
        assert '{' in chars
        assert '[' in chars
        assert 't' in chars
        assert 'n' in chars


# ── apply_json_constraint Tests ─────────────────────────────────────────────


class TestApplyJsonConstraint:
    """Test logit masking."""

    def test_masks_disallowed_tokens(self):
        import mlx.core as mx
        logits = mx.zeros((1, 10))
        allowed = [2, 5, 7]
        masked = apply_json_constraint(logits, allowed)
        # Allowed tokens should be 0, disallowed should be -inf
        for i in range(10):
            val = float(masked[0, i])
            if i in allowed:
                assert val == 0.0, f"Token {i} should be 0, got {val}"
            else:
                assert val < -1e10, f"Token {i} should be -inf, got {val}"

    def test_all_allowed_returns_same(self):
        import mlx.core as mx
        logits = mx.array([[1.0, 2.0, 3.0]])
        allowed = [0, 1, 2]
        masked = apply_json_constraint(logits, allowed)
        for i in range(3):
            assert abs(float(masked[0, i]) - float(logits[0, i])) < 1e-6

    def test_empty_allowed_falls_back_to_argmax(self):
        import mlx.core as mx
        logits = mx.array([[1.0, 2.0, 3.0]])
        masked = apply_json_constraint(logits, [])
        # Empty allowed now falls back to argmax (token 2 = 3.0) to avoid NaN.
        assert float(masked[0, 2]) == 3.0, "argmax token should keep its original logit"
        assert float(masked[0, 0]) < -1e10, "non-argmax tokens should be -inf"
        assert float(masked[0, 1]) < -1e10, "non-argmax tokens should be -inf"
        # Verify softmax does not produce NaN
        probs = mx.softmax(masked)
        assert not mx.any(mx.isnan(probs)).item(), "softmax should not produce NaN"

    def test_single_token_allowed(self):
        import mlx.core as mx
        logits = mx.zeros((1, 100))
        allowed = [42]
        masked = apply_json_constraint(logits, allowed)
        assert float(masked[0, 42]) == 0.0
        assert float(masked[0, 0]) < -1e10


# ── ConstrainedSampler Tests ────────────────────────────────────────────────


class TestConstrainedSampler:
    """Test the ConstrainedSampler wrapper."""

    def test_make_constrained_sampler(self):
        def base_sampler(logits):
            import mlx.core as mx
            return mx.argmax(logits, axis=-1)

        tokenizer = FakeTokenizer()
        sampler = make_constrained_sampler(base_sampler, None, tokenizer)
        assert isinstance(sampler, ConstrainedSampler)
        assert sampler.constraint is not None

    def test_constrained_sampler_tracks_generated_tokens(self):
        import mlx.core as mx

        def base_sampler(logits):
            return mx.argmax(logits, axis=-1)

        tokenizer = FakeTokenizer()
        sampler = make_constrained_sampler(base_sampler, None, tokenizer)

        # Simulate first token (should be `{`)
        logits = mx.zeros((1, tokenizer.vocab_size))
        # Set high logit for `{` (ASCII 123)
        logits[0, ord('{')] = 10.0

        token = sampler(logits)
        assert int(token) == ord('{')

    def test_constrained_sampler_with_schema(self):
        import mlx.core as mx

        def base_sampler(logits):
            return mx.argmax(logits, axis=-1)

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}}
        }
        tokenizer = FakeTokenizer()
        sampler = make_constrained_sampler(base_sampler, schema, tokenizer)

        # First token should pick `{` if it has highest logit
        logits = mx.full((1, tokenizer.vocab_size), -100.0)
        logits[0, ord('{')] = 10.0
        token = sampler(logits)
        assert int(token) == ord('{')


# ── SamplingParams Integration Tests ────────────────────────────────────────


class TestSamplingParamsJsonSchema:
    """Test SamplingParams with json_schema field."""

    def test_default_json_schema_is_none(self):
        sp = SamplingParams()
        assert sp.json_schema is None

    def test_json_schema_dict(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        sp = SamplingParams(json_schema=schema)
        assert sp.json_schema == schema

    def test_json_schema_string(self):
        sp = SamplingParams(json_schema="json_object")
        assert sp.json_schema == "json_object"


# ── Response Format Parsing Tests ───────────────────────────────────────────


class TestResponseFormatParsing:
    """Test the _parse_response_format helper in the chat router."""

    def test_none_input(self):
        from yunshu_gateway.routers.chat import _parse_response_format
        assert _parse_response_format(None) is None

    def test_json_object_type(self):
        from yunshu_gateway.routers.chat import _parse_response_format
        result = _parse_response_format({"type": "json_object"})
        assert result == "json_object"

    def test_json_schema_type_with_schema(self):
        from yunshu_gateway.routers.chat import _parse_response_format
        schema = {"type": "object", "properties": {"x": {"type": "number"}}}
        result = _parse_response_format({
            "type": "json_schema",
            "json_schema": {"name": "test", "schema": schema}
        })
        assert result == schema

    def test_json_schema_type_without_schema(self):
        from yunshu_gateway.routers.chat import _parse_response_format
        result = _parse_response_format({
            "type": "json_schema",
            "json_schema": {"name": "test"}
        })
        assert result == "json_object"

    def test_unknown_type_returns_none(self):
        from yunshu_gateway.routers.chat import _parse_response_format
        result = _parse_response_format({"type": "text"})
        assert result is None

    def test_empty_dict_returns_none(self):
        from yunshu_gateway.routers.chat import _parse_response_format
        result = _parse_response_format({})
        assert result is None


# ── Edge Cases ──────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases for JSON schema constraint."""

    def test_empty_schema(self):
        """Empty schema should default to object."""
        c = JsonSchemaConstraint({})
        assert c.state == JsonState.START
        chars = c._get_expected_chars()
        assert '{' in chars

    def test_schema_with_additional_properties(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "additionalProperties": {"type": "string"}
        })
        c.advance('{')
        c.advance('"any_key"')
        c.advance(':')
        chars = c._get_expected_chars()
        assert '"' in chars

    def test_string_with_escape(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {"msg": {"type": "string"}}
        })
        c.advance('{')
        c.advance('"msg"')
        c.advance(':')
        c.advance('"hello\\"')
        c.advance('world"')
        assert c.state == JsonState.OBJECT_COMMA

    def test_multiple_comma_separated_values(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
                "c": {"type": "string"},
            }
        })
        c.advance('{')
        c.advance('"a"')
        c.advance(':')
        c.advance('"1"')
        assert c.state == JsonState.OBJECT_COMMA
        c.advance(',')
        c.advance('"b"')
        c.advance(':')
        c.advance('"2"')
        assert c.state == JsonState.OBJECT_COMMA
        c.advance(',')
        c.advance('"c"')
        c.advance(':')
        c.advance('"3"')
        assert c.state == JsonState.OBJECT_COMMA
        c.advance('}')
        assert c.is_done

    def test_nested_array_in_object(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        })
        c.advance('{')
        c.advance('"items"')
        c.advance(':')
        c.advance('[')
        c.advance('"x"')
        c.advance(',')
        c.advance('"y"')
        c.advance(']')
        assert c.state == JsonState.OBJECT_COMMA
        c.advance('}')
        assert c.is_done

    def test_object_with_null_value(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {"data": {"type": "null"}}
        })
        c.advance('{')
        c.advance('"data"')
        c.advance(':')
        c.advance('null')
        c.advance('}')
        assert c.is_done

    def test_object_with_boolean_values(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {
                "a": {"type": "boolean"},
                "b": {"type": "boolean"},
            }
        })
        c.advance('{')
        c.advance('"a"')
        c.advance(':')
        c.advance('true')
        c.advance(',')
        c.advance('"b"')
        c.advance(':')
        c.advance('false')
        c.advance('}')
        assert c.is_done

    def test_whitespace_handling(self):
        c = JsonSchemaConstraint()
        c.advance('{')
        c.advance('  ')
        c.advance('"key"')
        c.advance(' ')
        c.advance(':')
        c.advance(' ')
        c.advance('"val"')
        c.advance(' ')
        c.advance('}')
        assert c.is_done

    def test_done_state_get_allowed_returns_eos(self):
        c = JsonSchemaConstraint()
        c.advance('{')
        c.advance('}')
        assert c.is_done
        tok = FakeTokenizer()
        allowed = c.get_allowed_tokens(tok, [])
        assert 0 in allowed  # EOS token ID

    def test_get_allowed_tokens_start_state(self):
        c = JsonSchemaConstraint()
        tok = FakeTokenizer()
        allowed = c.get_allowed_tokens(tok, [])
        # Should include token for `{` character
        assert ord('{') in allowed

    def test_get_allowed_tokens_object_open(self):
        c = JsonSchemaConstraint()
        c.advance('{')
        tok = FakeTokenizer()
        allowed = c.get_allowed_tokens(tok, [])
        # Should include tokens for `"` and `}`
        assert ord('"') in allowed
        assert ord('}') in allowed


class TestComplexNestedSchema:
    """Test complex nested schemas."""

    def test_schema_with_nested_object_and_array(self):
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                },
                "active": {"type": "boolean"}
            }
        }
        c = JsonSchemaConstraint(schema)
        c.advance('{')
        c.advance('"user"')
        c.advance(':')
        c.advance('{')
        c.advance('"name"')
        c.advance(':')
        c.advance('"Alice"')
        c.advance(',')
        c.advance('"tags"')
        c.advance(':')
        c.advance('[')
        c.advance('"admin"')
        c.advance(',')
        c.advance('"dev"')
        c.advance(']')
        c.advance('}')
        c.advance(',')
        c.advance('"active"')
        c.advance(':')
        c.advance('true')
        c.advance('}')
        assert c.is_done


class TestMultiTypeSchemas:
    """Test schemas with multiple types (oneOf-style)."""

    def test_type_as_list(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {
                "value": {"type": ["string", "null"]}
            }
        })
        c.advance('{')
        c.advance('"value"')
        c.advance(':')
        chars = c._get_expected_chars()
        # Should allow both string and null
        assert '"' in chars
        assert 'n' in chars


class TestEnumValues:
    """Test schema with enum values."""

    def test_enum_string_type(self):
        schema = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "inactive"]
                }
            }
        }
        c = JsonSchemaConstraint(schema)
        c.advance('{')
        c.advance('"status"')
        c.advance(':')
        # Enum values are still strings
        chars = c._get_expected_chars()
        assert '"' in chars


# ── Bug Fix Regression Tests ─────────────────────────────────────────────────


class TestTopLevelArray:
    """Test that top-level arrays are supported (START state supports '[')."""

    def test_top_level_array_starts_with_bracket(self):
        """Schema type 'array' should expect '[' at START."""
        c = JsonSchemaConstraint({"type": "array", "items": {"type": "string"}})
        chars = c._get_expected_chars()
        assert '[' in chars
        assert '{' not in chars

    def test_top_level_array_empty(self):
        c = JsonSchemaConstraint({"type": "array", "items": {"type": "string"}})
        c.advance('[')
        c.advance(']')
        assert c.is_done

    def test_top_level_array_with_items(self):
        c = JsonSchemaConstraint({"type": "array", "items": {"type": "string"}})
        c.advance('[')
        c.advance('"a"')
        c.advance(',')
        c.advance('"b"')
        c.advance(']')
        assert c.is_done


class TestBooleanNullLiteralCounter:
    """Test that boolean/null detection uses a counter, not fragile string matching."""

    def test_true_literal_completes(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {"flag": {"type": "boolean"}}
        })
        c.advance('{')
        c.advance('"flag"')
        c.advance(':')
        c.advance('true')
        assert c.state == JsonState.OBJECT_COMMA

    def test_false_literal_completes(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {"flag": {"type": "boolean"}}
        })
        c.advance('{')
        c.advance('"flag"')
        c.advance(':')
        c.advance('false')
        assert c.state == JsonState.OBJECT_COMMA

    def test_null_literal_completes(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {"data": {"type": "null"}}
        })
        c.advance('{')
        c.advance('"data"')
        c.advance(':')
        c.advance('null')
        assert c.state == JsonState.OBJECT_COMMA

    def test_boolean_in_array(self):
        """Boolean values inside arrays should also transition correctly."""
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {"flags": {"type": "array", "items": {"type": "boolean"}}}
        })
        c.advance('{')
        c.advance('"flags"')
        c.advance(':')
        c.advance('[')
        c.advance('true')
        assert c.state == JsonState.ARRAY_COMMA
        c.advance(',')
        c.advance('false')
        assert c.state == JsonState.ARRAY_COMMA
        c.advance(']')
        assert c.state == JsonState.OBJECT_COMMA
        c.advance('}')
        assert c.is_done


class TestNumberStateNoCorruption:
    """Test that number state transitions don't corrupt the text buffer."""

    def test_number_followed_by_comma(self):
        """Number followed by comma should correctly transition to OBJECT_COMMA."""
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "string"}
            }
        })
        c.advance('{')
        c.advance('"x"')
        c.advance(':')
        c.advance('42,')
        # After '42,' the number 42 completes and comma transitions to OBJECT_KEY
        assert c.state == JsonState.OBJECT_KEY
        c.advance('"y"')
        c.advance(':')
        c.advance('"test"')
        c.advance('}')
        assert c.is_done

    def test_number_followed_by_close_brace(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {"count": {"type": "integer"}}
        })
        c.advance('{')
        c.advance('"count"')
        c.advance(':')
        c.advance('99}')
        assert c.is_done


class TestCheckpointRollbackWithLiteralCounter:
    """Test that checkpoint/rollback preserves literal_remaining state."""

    def test_rollback_restores_boolean_state(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {"flag": {"type": "boolean"}}
        })
        c.advance('{')
        c.advance('"flag"')
        c.advance(':')
        # In BOOLEAN_TRUE state, 't' consumed, 3 chars remaining
        c.checkpoint()
        c.advance('ru')  # advance by 2 chars, 1 remaining
        # Rollback should restore to before 'ru'
        c.rollback()
        # The state should allow completing 'true' still
        assert c.state in (JsonState.OBJECT_VALUE, JsonState.BOOLEAN_TRUE)


class TestBuildConstrainedSamplerJsonString:
    """Test that _build_constrained_sampler handles 'json_object' string correctly."""

    def test_json_object_string(self):
        import mlx.core as mx

        from yunshu_engine.batched_engine import _build_constrained_sampler

        def base_sampler(logits):
            return mx.argmax(logits)

        tokenizer = FakeTokenizer()
        # This used to crash with json.JSONDecodeError
        sampler = _build_constrained_sampler(base_sampler, "json_object", tokenizer)
        assert isinstance(sampler, ConstrainedSampler)

    def test_json_schema_string(self):
        import json

        import mlx.core as mx

        from yunshu_engine.batched_engine import _build_constrained_sampler

        def base_sampler(logits):
            return mx.argmax(logits)

        tokenizer = FakeTokenizer()
        schema = json.dumps({"type": "object", "properties": {"x": {"type": "string"}}})
        sampler = _build_constrained_sampler(base_sampler, schema, tokenizer)
        assert isinstance(sampler, ConstrainedSampler)


class TestGatewayCompletionsResponseFormat:
    """Test completions.py response_format handling consistency."""

    def test_json_object_mode(self):
        """json_object type should produce a valid constraint."""
        import mlx.core as mx

        from yunshu_engine.batched_engine import _build_constrained_sampler

        def base_sampler(logits):
            return mx.argmax(logits)

        tokenizer = FakeTokenizer()
        # Empty dict should create a valid constraint (generic object)
        sampler = _build_constrained_sampler(base_sampler, {}, tokenizer)
        assert isinstance(sampler, ConstrainedSampler)
        # Should constrain first token to '{'
        logits = mx.full((1, tokenizer.vocab_size), -100.0)
        logits[0, ord('{')] = 10.0
        token = sampler(logits)
        assert int(token) == ord('{')


class TestConstrainedSamplerCheckpointRollback:
    """Test that ConstrainedSampler forwards checkpoint/rollback."""

    def test_checkpoint_rollback(self):
        import mlx.core as mx

        def base_sampler(logits):
            return mx.argmax(logits)

        tokenizer = FakeTokenizer()
        sampler = make_constrained_sampler(base_sampler, None, tokenizer)

        # Should not raise
        sampler.checkpoint()
        sampler.rollback()

    def test_rollback_restores_state(self):
        import mlx.core as mx

        def base_sampler(logits):
            return mx.argmax(logits)

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        tokenizer = FakeTokenizer()
        sampler = make_constrained_sampler(base_sampler, schema, tokenizer)

        # Generate '{'
        logits = mx.full((1, tokenizer.vocab_size), -100.0)
        logits[0, ord('{')] = 10.0
        sampler(logits)

        # Checkpoint after '{'
        sampler.checkpoint()
        assert sampler.constraint.state == JsonState.OBJECT_OPEN

        # Generate '"'
        logits = mx.full((1, tokenizer.vocab_size), -100.0)
        logits[0, ord('"')] = 10.0
        sampler(logits)

        # Rollback should restore to OBJECT_OPEN
        sampler.rollback()
        assert sampler.constraint.state == JsonState.OBJECT_OPEN


class TestSchemaAnyType:
    """Tests for schema {} allowing any JSON value type at top level."""

    def test_empty_schema_allows_object(self):
        """Schema {} should allow generating objects."""
        c = JsonSchemaConstraint({})
        chars = c._get_expected_chars()
        assert '{' in chars

    def test_empty_schema_allows_string(self):
        """Schema {} should allow generating strings."""
        c = JsonSchemaConstraint({})
        chars = c._get_expected_chars()
        assert '"' in chars

    def test_empty_schema_allows_array(self):
        """Schema {} should allow generating arrays."""
        c = JsonSchemaConstraint({})
        chars = c._get_expected_chars()
        assert '[' in chars

    def test_empty_schema_allows_number(self):
        """Schema {} should allow generating numbers."""
        c = JsonSchemaConstraint({})
        chars = c._get_expected_chars()
        assert '-' in chars
        assert '0' in chars

    def test_empty_schema_allows_boolean(self):
        """Schema {} should allow generating booleans."""
        c = JsonSchemaConstraint({})
        chars = c._get_expected_chars()
        assert 't' in chars
        assert 'f' in chars

    def test_empty_schema_allows_null(self):
        """Schema {} should allow generating null."""
        c = JsonSchemaConstraint({})
        chars = c._get_expected_chars()
        assert 'n' in chars

    def test_empty_schema_generates_string(self):
        """Schema {} should correctly track state for a top-level string."""
        c = JsonSchemaConstraint({})
        c.advance('"hello"')
        assert c.state == JsonState.DONE

    def test_empty_schema_generates_number(self):
        """Schema {} should correctly track state for a top-level number.

        Note: top-level numbers don't have a terminator, so the state stays
        NUMBER until generation stops (EOS). This is expected behavior.
        """
        c = JsonSchemaConstraint({})
        c.advance('42')
        assert c.state == JsonState.NUMBER  # no terminator -> stays NUMBER

    def test_empty_schema_generates_boolean(self):
        """Schema {} should correctly track state for a top-level boolean."""
        c = JsonSchemaConstraint({})
        c.advance('true')
        assert c.state == JsonState.DONE

    def test_empty_schema_generates_null(self):
        """Schema {} should correctly track state for a top-level null."""
        c = JsonSchemaConstraint({})
        c.advance('null')
        assert c.state == JsonState.DONE

    def test_empty_schema_generates_array(self):
        """Schema {} should correctly track state for a top-level array."""
        c = JsonSchemaConstraint({})
        c.advance('[1, 2, 3]')
        assert c.state == JsonState.DONE


# ── composite-schema tests (allOf / oneOf / anyOf / $ref / NUMBER_ZERO) ──


class TestCompositeSchemas:
    """cover composite schemas + number-state edge cases that
    were previously unexercised ."""

    def test_anyOf_string_or_null_collapses_to_oneOf(self):
        """anyOf:[string, null] is normalized to oneOf — should accept both."""
        c = JsonSchemaConstraint({"anyOf": [{"type": "string"}, {"type": "null"}]})
        # null path
        c.advance("null")
        assert c.state == JsonState.DONE

    def test_anyOf_string_path(self):
        c = JsonSchemaConstraint({"anyOf": [{"type": "string"}, {"type": "null"}]})
        c.advance('"hello"')
        assert c.state == JsonState.DONE

    def test_allOf_merges_object_fields(self):
        """allOf intersects object fields — both required fields must be present."""
        schema = {
            "allOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
                {"type": "object", "properties": {"b": {"type": "number"}}, "required": ["b"]},
            ]
        }
        c = JsonSchemaConstraint(schema)
        c.advance('{"a": "x", "b": 42}')
        assert c.state == JsonState.DONE

    def test_oneOf_picks_one_branch(self):
        """oneOf with two distinct value types — pick string branch."""
        schema = {"oneOf": [{"type": "string"}, {"type": "number"}]}
        c = JsonSchemaConstraint(schema)
        c.advance('"value"')
        assert c.state == JsonState.DONE

    def test_oneOf_number_branch(self):
        schema = {"oneOf": [{"type": "string"}, {"type": "number"}]}
        c = JsonSchemaConstraint(schema)
        c.advance('123')
        assert c.state in (JsonState.NUMBER, JsonState.NUMBER_ZERO, JsonState.NUMBER_EXPONENT, JsonState.NUMBER_FRACTION, JsonState.DONE)

    def test_ref_resolution(self):
        """$ref into #/$defs/foo — engine should resolve it before walking."""
        schema = {
            "$defs": {
                "foo": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
            },
            "$ref": "#/$defs/foo",
        }
        c = JsonSchemaConstraint(schema)
        c.advance('{"x": "value"}')
        assert c.state == JsonState.DONE

    def test_nested_ref_resolution_is_inlined(self):
        """a $ref inside a property (the dominant Pydantic/OpenAI
        structured-output pattern) must be inlined, not left as {"$ref": ...}
        resolving to type 'any' (which would drop all of the referenced model's
        structure → zero enforcement)."""
        from python.yunshu_engine.json_schema import _repair_json_schema
        schema = {
            "type": "object",
            "properties": {"pet": {"$ref": "#/$defs/Pet"}},
            "required": ["pet"],
            "$defs": {
                "Pet": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                }
            },
        }
        repaired = _repair_json_schema(schema)
        pet = repaired["properties"]["pet"]
        assert "$ref" not in pet, "nested $ref was not inlined"
        assert pet.get("type") == "object"
        assert "name" in pet.get("properties", {})

    def test_nested_ref_enforced_by_constraint(self):
        """The constraint built from a nested-$ref schema enforces the nested
        object's type (a non-object value for the ref'd property is rejected)."""
        schema = {
            "type": "object",
            "properties": {"pet": {"$ref": "#/$defs/Pet"}},
            "required": ["pet"],
            "$defs": {"Pet": {"type": "object", "properties": {"n": {"type": "string"}}, "required": ["n"]}},
        }
        c = JsonSchemaConstraint(schema)
        c.advance('{"pet": {"n": "rex"}}')
        assert c.state == JsonState.DONE

    def test_number_zero_terminal(self):
        """'0' is a valid number — must not be classified as malformed."""
        c = JsonSchemaConstraint({"type": "number"})
        c.advance('0')
        assert c.state in (JsonState.NUMBER, JsonState.NUMBER_ZERO, JsonState.NUMBER_EXPONENT, JsonState.NUMBER_FRACTION, JsonState.DONE)

    def test_number_zero_in_array(self):
        """[0] is a valid array containing a single zero."""
        c = JsonSchemaConstraint({"type": "array", "items": {"type": "number"}})
        c.advance('[0]')
        assert c.state == JsonState.DONE

    def test_number_zero_in_object(self):
        c = JsonSchemaConstraint({
            "type": "object",
            "properties": {"count": {"type": "number"}},
            "required": ["count"],
        })
        c.advance('{"count": 0}')
        assert c.state == JsonState.DONE

    def test_number_negative_zero(self):
        """JSON allows -0; engine must accept."""
        c = JsonSchemaConstraint({"type": "number"})
        c.advance('-0')
        assert c.state in (JsonState.NUMBER, JsonState.NUMBER_ZERO, JsonState.NUMBER_EXPONENT, JsonState.NUMBER_FRACTION, JsonState.DONE)

    def test_number_float_with_zero(self):
        c = JsonSchemaConstraint({"type": "number"})
        c.advance('0.5')
        assert c.state in (JsonState.NUMBER, JsonState.NUMBER_ZERO, JsonState.NUMBER_EXPONENT, JsonState.NUMBER_FRACTION, JsonState.DONE)

    def test_number_exponent_with_zero(self):
        c = JsonSchemaConstraint({"type": "number"})
        c.advance('1e0')
        assert c.state in (JsonState.NUMBER, JsonState.NUMBER_ZERO, JsonState.NUMBER_EXPONENT, JsonState.NUMBER_FRACTION, JsonState.DONE)


class TestTextBufferCap:
    """verify the _text_buffer trimming (O(N²)→amortized-O(1) perf
    fix) does not corrupt parsing on long structured output."""

    def test_long_string_value_still_parses(self):
        """A very long string value (> buffer cap) must still validate as
        a complete JSON string without the trim corrupting state."""
        from yunshu_engine.json_schema import JsonSchemaConstraint, JsonState
        c = JsonSchemaConstraint({"type": "string"})
        # Feed a 100KB string in chunks (exceeds the 64KB cap → trim fires)
        c.advance('"')
        for _ in range(2000):
            c.advance("x" * 50)  # 100K chars total
        c.advance('"')
        assert c.state == JsonState.DONE

    def test_long_object_with_many_keys_parses(self):
        """Long object generation with trim active still completes."""
        from yunshu_engine.json_schema import JsonSchemaConstraint, JsonState
        c = JsonSchemaConstraint({})
        c.advance("{")
        for i in range(500):
            c.advance(f'"key_{i}_with_long_padding_xxxxxxxxxxxxxxxxxxxxxxxx": {i}')
            if i < 499:
                c.advance(", ")
        c.advance("}")
        assert c.state == JsonState.DONE

    def test_buffer_trimmed_between_values(self):
        """For array/object generation (many distinct values), the buffer
        trims between values and stays bounded. A single contiguous giant
        string legitimately can't trim (content must be validated whole)."""
        from yunshu_engine.json_schema import JsonSchemaConstraint, JsonState
        c = JsonSchemaConstraint({})
        c.advance("[")
        for i in range(3000):
            c.advance(f'"item_{i}_padding_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"')
            if i < 2999:
                c.advance(", ")
        # Fed ~150K chars; buffer should be trimmed well below that.
        assert len(c._text_buffer) < 70000, f"buffer not trimmed: {len(c._text_buffer)}"
        c.advance("]")
        assert c.state == JsonState.DONE
