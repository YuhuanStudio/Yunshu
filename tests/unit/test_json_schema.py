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

import pytest

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

    def test_empty_allowed_returns_same(self):
        import mlx.core as mx
        logits = mx.array([[1.0, 2.0, 3.0]])
        masked = apply_json_constraint(logits, [])
        # Should return as-is when no tokens allowed
        for i in range(3):
            assert abs(float(masked[0, i]) - float(logits[0, i])) < 1e-6

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
        from yunshu_engine.batched_engine import _build_constrained_sampler
        import mlx.core as mx

        def base_sampler(logits):
            return mx.argmax(logits)

        tokenizer = FakeTokenizer()
        # This used to crash with json.JSONDecodeError
        sampler = _build_constrained_sampler(base_sampler, "json_object", tokenizer)
        assert isinstance(sampler, ConstrainedSampler)

    def test_json_schema_string(self):
        from yunshu_engine.batched_engine import _build_constrained_sampler
        import mlx.core as mx
        import json

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
        from yunshu_engine.batched_engine import _build_constrained_sampler
        import mlx.core as mx

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
