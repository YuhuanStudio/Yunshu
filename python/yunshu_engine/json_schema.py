"""Yunshu JSON Schema Constrained Generation — token-level constraint masking.

Since xgrammar/outlines are not available on Apple Silicon, this implements
a self-contained token-level constrained generation approach:

1. Build a state machine that tracks what characters are expected next in JSON
2. For each state, compute the set of allowed token IDs (tokens whose text
   representation is compatible with the expected characters)
3. Mask disallowed tokens to -inf before sampling

Two modes:
- json_object: Any valid JSON object (top-level `{...}`)
- json_schema: JSON object conforming to a specific JSON Schema

Architecture:
  JsonSchemaConstraint  — state machine tracking expected JSON structure
  apply_json_constraint() — logit masking utility
  ConstrainedSampler    — wraps mlx-lm sampler with per-step constraint masking

The constraint is designed to be called from the Scheduler's step loop,
where the sampler is invoked per-token.
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum, auto
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ── JSON State Machine ──────────────────────────────────────────────────────


class JsonState(Enum):
    """States in the JSON generation state machine."""
    START = auto()
    # Object
    OBJECT_OPEN = auto()        # just emitted `{`
    OBJECT_KEY = auto()         # expecting a key string (or `}`)
    OBJECT_KEY_STRING = auto()  # inside a key string
    OBJECT_COLON = auto()       # expecting `:` after key
    OBJECT_VALUE = auto()       # expecting a value (depends on schema)
    OBJECT_COMMA = auto()       # expecting `,` or `}`
    # Array
    ARRAY_OPEN = auto()         # just emitted `[`
    ARRAY_VALUE = auto()        # expecting a value (depends on items schema)
    ARRAY_COMMA = auto()        # expecting `,` or `]`
    # Primitives
    STRING = auto()             # inside a string value
    STRING_ESCAPE = auto()      # after `\` inside a string
    NUMBER = auto()             # inside a number
    BOOLEAN_TRUE = auto()       # expecting `true`
    BOOLEAN_FALSE = auto()      # expecting `false`
    NULL = auto()               # expecting `null`
    # Terminal
    DONE = auto()               # generation complete
    WHITESPACE = auto()         # consuming whitespace between tokens


# Characters allowed in various JSON contexts
_WHITESPACE_CHARS = {' ', '\t', '\n', '\r'}
_DIGIT_CHARS = set('0123456789')
_HEX_CHARS = set('0123456789abcdefABCDEF')


class JsonSchemaConstraint:
    """State machine that tracks JSON structure during generation.

    Tracks the current position in the JSON structure and computes which
    tokens are valid next tokens given the current state.

    Usage:
        constraint = JsonSchemaConstraint(schema)
        # After each generated token:
        allowed = constraint.get_allowed_tokens(tokenizer, generated_token_ids)
    """

    def __init__(self, schema: dict | None = None) -> None:
        """Initialize constraint with optional JSON Schema.

        Args:
            schema: JSON Schema dict. If None, accepts any valid JSON object.
        """
        self._schema = schema
        self._state = JsonState.START
        self._text_buffer = ""  # decoded text so far
        self._schema_stack: list[tuple[JsonState, dict]] = []
        # Stack of (return_state, sub_schema) for nested structures
        # When we enter an object/array, we push where to return when done
        self._object_keys_remaining: list[list[str]] = []
        # For objects with defined properties, tracks which keys still need values
        self._current_key: str | None = None
        self._in_string: bool = False
        self._string_start: int = 0  # position in text_buffer where string started
        self._number_start: int = 0
        self._is_first_value: bool = True  # track first value in object/array
        # Snapshot stack for rollback (speculative draft validation)
        self._snapshots: list[tuple] = []

        # If no schema, default to generic object
        if schema is None:
            self._schema = {"type": "object"}

    @property
    def state(self) -> JsonState:
        return self._state

    @property
    def is_done(self) -> bool:
        return self._state == JsonState.DONE

    def _get_type_from_schema(self, schema: dict) -> str | list[str]:
        """Extract the type from a schema, with default."""
        if "type" in schema:
            return schema["type"]
        # Infer type from other keywords
        if "properties" in schema:
            return "object"
        if "items" in schema:
            return "array"
        if "enum" in schema:
            return "enum"
        if "const" in schema:
            return "const"
        if "anyOf" in schema or "oneOf" in schema:
            # Use the first option's type
            options = schema.get("anyOf") or schema.get("oneOf") or []
            non_null = [o for o in options if o.get("type") != "null"]
            if non_null:
                return self._get_type_from_schema(non_null[0])
            return "any"
        return "any"

    def _resolve_schema_for_value(self, schema: dict, key: str | None = None) -> dict:
        """Resolve which schema applies for a value position.

        For objects, look up the key in properties.
        For arrays, use items schema.
        Handles anyOf/oneOf by resolving to first non-null option.
        """
        schema_type = self._get_type_from_schema(schema)

        # Resolve anyOf/oneOf to concrete schema
        if "anyOf" in schema:
            non_null = [o for o in schema["anyOf"] if o.get("type") != "null"]
            if non_null:
                return self._resolve_schema_for_value(non_null[0], key)
        if "oneOf" in schema:
            non_null = [o for o in schema["oneOf"] if o.get("type") != "null"]
            if non_null:
                return self._resolve_schema_for_value(non_null[0], key)

        if schema_type == "object" or "properties" in schema:
            if key and "properties" in schema:
                return schema["properties"].get(key, {"type": "string"})
            # Check additionalProperties for unknown keys
            if key and "additionalProperties" in schema:
                add_props = schema["additionalProperties"]
                if isinstance(add_props, dict):
                    return add_props
            return {"type": "string"}  # default for unknown keys

        if schema_type == "array" or "items" in schema:
            return schema.get("items", {"type": "string"})

        if schema_type == "enum":
            return {"type": "string"}  # enum values are strings

        return schema

    def _get_expected_chars(self) -> set[str] | None:
        """Get the set of characters that are valid at the current state.

        Returns None if any character is allowed (inside strings/numbers).
        Returns a set of specific characters otherwise.
        """
        state = self._state

        if state == JsonState.START:
            return {'{'}

        if state == JsonState.OBJECT_OPEN:
            # After `{`, expect `"` (key) or `}`
            return {'"', '}', ' ', '\t', '\n', '\r'}

        if state == JsonState.OBJECT_KEY:
            # Expecting key string or close brace
            return {'"', '}', ' ', '\t', '\n', '\r'}

        if state == JsonState.OBJECT_KEY_STRING:
            # Inside a string key — any char except unescaped `"`
            return None  # any char allowed

        if state == JsonState.OBJECT_COLON:
            return {':', ' ', '\t', '\n', '\r'}

        if state == JsonState.OBJECT_VALUE:
            # Depends on schema type for this value
            return self._get_value_start_chars()

        if state == JsonState.OBJECT_COMMA:
            return {',', '}', ' ', '\t', '\n', '\r'}

        if state == JsonState.ARRAY_OPEN:
            return self._get_array_value_start_chars()

        if state == JsonState.ARRAY_VALUE:
            return self._get_array_value_start_chars()

        if state == JsonState.ARRAY_COMMA:
            return {',', ']', ' ', '\t', '\n', '\r'}

        if state == JsonState.STRING:
            return None  # any char inside string

        if state == JsonState.STRING_ESCAPE:
            return {'"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'}

        if state == JsonState.NUMBER:
            return None  # digits and number chars handled separately

        if state == JsonState.BOOLEAN_TRUE:
            return {'t'}

        if state == JsonState.BOOLEAN_FALSE:
            return {'f'}

        if state == JsonState.NULL:
            return {'n'}

        if state == JsonState.DONE:
            return set()  # nothing allowed

        return set()

    def _get_value_start_chars(self) -> set[str]:
        """Get characters that can start a value in the current object context."""
        # Look at what type the current value should be
        value_schema = self._get_current_value_schema()

        if value_schema is None:
            # Any value type allowed
            return {'"', '{', '[', 't', 'f', 'n', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '\t', '\n', '\r'}

        schema_type = self._get_type_from_schema(value_schema)

        if isinstance(schema_type, list):
            chars = set()
            for t in schema_type:
                chars.update(self._type_to_start_chars(t))
            chars.update(_WHITESPACE_CHARS)
            return chars

        chars = self._type_to_start_chars(schema_type)
        chars.update(_WHITESPACE_CHARS)
        return chars

    def _get_array_value_start_chars(self) -> set[str]:
        """Get characters that can start a value in array context."""
        # Get the items schema from the parent array schema
        if self._schema_stack:
            _, parent_schema = self._schema_stack[-1]
            items_schema = parent_schema.get("items", {"type": "string"})
        else:
            items_schema = {"type": "string"}

        schema_type = self._get_type_from_schema(items_schema)

        if isinstance(schema_type, list):
            chars = set()
            for t in schema_type:
                chars.update(self._type_to_start_chars(t))
            chars.update({']', ' ', '\t', '\n', '\r'})
            return chars

        chars = self._type_to_start_chars(schema_type)
        chars.update({']', ' ', '\t', '\n', '\r'})
        return chars

    def _type_to_start_chars(self, schema_type: str) -> set[str]:
        """Map a JSON Schema type to the characters that can start it."""
        mapping = {
            "string": {'"'},
            "object": {'{'},
            "array": {'['},
            "boolean": {'t', 'f'},
            "null": {'n'},
            "number": {'-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'},
            "integer": {'-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'},
            "any": {'"', '{', '[', 't', 'f', 'n', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'},
        }
        return mapping.get(schema_type, {'"', '{', '[', 't', 'f', 'n', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'})

    def _get_current_value_schema(self) -> dict | None:
        """Get the schema for the current value position."""
        if not self._schema_stack:
            return self._schema

        _, parent_schema = self._schema_stack[-1]
        parent_type = self._get_type_from_schema(parent_schema)

        if parent_type == "object" or "properties" in parent_schema:
            if self._current_key and "properties" in parent_schema:
                return parent_schema["properties"].get(self._current_key)
            return None  # any type allowed for additional properties

        if parent_type == "array" or "items" in parent_schema:
            return parent_schema.get("items")

        return None

    def advance(self, token_text: str) -> None:
        """Advance the state machine by one token's worth of text.

        This is called after each token is generated to update the state.

        Args:
            token_text: The decoded text of the generated token.
        """
        if self._state == JsonState.DONE:
            return

        self._text_buffer += token_text

        # Process characters to update state
        self._process_text(token_text)

    def checkpoint(self) -> None:
        """Save current state for later rollback (speculative draft validation)."""
        self._snapshots.append((
            self._state,
            self._text_buffer,
            list(self._schema_stack),
            [list(k) for k in self._object_keys_remaining],
            self._current_key,
            self._in_string,
            self._string_start,
            self._number_start,
            self._is_first_value,
        ))

    def rollback(self) -> None:
        """Restore state to last checkpoint."""
        if not self._snapshots:
            return
        (
            self._state,
            self._text_buffer,
            self._schema_stack,
            self._object_keys_remaining,
            self._current_key,
            self._in_string,
            self._string_start,
            self._number_start,
            self._is_first_value,
        ) = self._snapshots.pop()

    def _process_text(self, text: str) -> None:
        """Process the generated text to update state machine."""
        i = 0
        while i < len(text):
            ch = text[i]

            if self._state == JsonState.START:
                if ch == '{':
                    self._state = JsonState.OBJECT_OPEN
                    self._is_first_value = True
                    self._init_object_keys(self._schema)
                    self._schema_stack.append((JsonState.DONE, self._schema))
                i += 1
                continue

            if self._state == JsonState.OBJECT_OPEN:
                if ch in _WHITESPACE_CHARS:
                    i += 1
                    continue
                if ch == '"':
                    self._state = JsonState.OBJECT_KEY_STRING
                    self._string_start = len(self._text_buffer) - len(text) + i + 1
                    self._in_string = True
                    i += 1
                    continue
                if ch == '}':
                    self._pop_schema()
                    i += 1
                    continue
                i += 1
                continue

            if self._state == JsonState.OBJECT_KEY:
                if ch in _WHITESPACE_CHARS:
                    i += 1
                    continue
                if ch == '"':
                    self._state = JsonState.OBJECT_KEY_STRING
                    self._string_start = len(self._text_buffer) - len(text) + i + 1
                    self._in_string = True
                    i += 1
                    continue
                if ch == '}':
                    self._pop_schema()
                    i += 1
                    continue
                i += 1
                continue

            if self._state == JsonState.OBJECT_KEY_STRING:
                if ch == '\\':
                    # Next char is escaped
                    i += 2  # skip escape and escaped char
                    continue
                if ch == '"':
                    # End of key
                    key_start = self._string_start
                    key_end = len(self._text_buffer) - len(text) + i
                    self._current_key = self._text_buffer[key_start:key_end]
                    self._in_string = False
                    self._state = JsonState.OBJECT_COLON
                    i += 1
                    continue
                i += 1
                continue

            if self._state == JsonState.OBJECT_COLON:
                if ch in _WHITESPACE_CHARS:
                    i += 1
                    continue
                if ch == ':':
                    self._state = JsonState.OBJECT_VALUE
                    i += 1
                    continue
                i += 1
                continue

            if self._state == JsonState.OBJECT_VALUE:
                if ch in _WHITESPACE_CHARS:
                    i += 1
                    continue
                # Determine value type from schema
                self._enter_value(ch)
                i += 1
                continue

            if self._state == JsonState.OBJECT_COMMA:
                if ch in _WHITESPACE_CHARS:
                    i += 1
                    continue
                if ch == ',':
                    self._state = JsonState.OBJECT_KEY
                    self._is_first_value = False
                    i += 1
                    continue
                if ch == '}':
                    self._pop_schema()
                    i += 1
                    continue
                i += 1
                continue

            if self._state == JsonState.ARRAY_OPEN:
                if ch in _WHITESPACE_CHARS:
                    i += 1
                    continue
                if ch == ']':
                    self._pop_schema()
                    i += 1
                    continue
                # First value
                self._is_first_value = True
                self._state = JsonState.ARRAY_VALUE
                self._enter_array_value(ch)
                i += 1
                continue

            if self._state == JsonState.ARRAY_VALUE:
                if ch in _WHITESPACE_CHARS:
                    i += 1
                    continue
                if ch == ']':
                    self._pop_schema()
                    i += 1
                    continue
                self._enter_array_value(ch)
                i += 1
                continue

            if self._state == JsonState.ARRAY_COMMA:
                if ch in _WHITESPACE_CHARS:
                    i += 1
                    continue
                if ch == ',':
                    self._state = JsonState.ARRAY_VALUE
                    self._is_first_value = False
                    i += 1
                    continue
                if ch == ']':
                    self._pop_schema()
                    i += 1
                    continue
                i += 1
                continue

            if self._state == JsonState.STRING:
                if ch == '\\':
                    self._state = JsonState.STRING_ESCAPE
                    i += 1
                    continue
                if ch == '"':
                    # End of string value
                    self._value_completed()
                    i += 1
                    continue
                i += 1
                continue

            if self._state == JsonState.STRING_ESCAPE:
                # After \, next char is the escape type
                if ch == 'u':
                    # Unicode escape: need 4 hex digits
                    # For simplicity, just consume and stay in escape
                    self._state = JsonState.STRING
                else:
                    self._state = JsonState.STRING
                i += 1
                continue

            if self._state == JsonState.NUMBER:
                # Numbers end when we see a non-number character
                if ch in _DIGIT_CHARS or ch in '.eE+-':
                    i += 1
                    continue
                # Number ended — the terminating char will be re-processed
                # in the parent context
                self._text_buffer = self._text_buffer[:-len(text) + i]
                self._value_completed()
                # Re-process remaining text in new state
                remaining = text[i:]
                if remaining:
                    self._process_text(remaining)
                return

            if self._state == JsonState.BOOLEAN_TRUE:
                # Check if "true" is completed
                buf_end = self._text_buffer[-(5 - (len(self._text_buffer) - self._text_buffer.rfind('t'))):]
                # Simpler: check if we have "true" somewhere
                if self._text_buffer.rstrip().endswith('true'):
                    self._value_completed()
                elif ch == 'e' and self._text_buffer.rstrip().endswith('tru'):
                    self._value_completed()
                i += 1
                continue

            if self._state == JsonState.BOOLEAN_FALSE:
                if self._text_buffer.rstrip().endswith('false'):
                    self._value_completed()
                elif ch == 'e' and self._text_buffer.rstrip().endswith('fals'):
                    self._value_completed()
                i += 1
                continue

            if self._state == JsonState.NULL:
                if self._text_buffer.rstrip().endswith('null'):
                    self._value_completed()
                elif ch == 'l' and self._text_buffer.rstrip().endswith('nul'):
                    self._value_completed()
                i += 1
                continue

            i += 1

    def _enter_value(self, ch: str) -> None:
        """Enter a value state based on the first character."""
        if ch == '"':
            self._state = JsonState.STRING
            self._string_start = len(self._text_buffer)
        elif ch == '{':
            value_schema = self._get_current_value_schema()
            obj_schema = value_schema if value_schema and self._get_type_from_schema(value_schema) == "object" else {"type": "object"}
            self._init_object_keys(obj_schema)
            self._schema_stack.append((JsonState.OBJECT_COMMA, obj_schema))
            self._state = JsonState.OBJECT_OPEN
            self._is_first_value = True
        elif ch == '[':
            value_schema = self._get_current_value_schema()
            arr_schema = value_schema if value_schema and self._get_type_from_schema(value_schema) == "array" else {"type": "array"}
            self._schema_stack.append((JsonState.ARRAY_COMMA, arr_schema))
            self._state = JsonState.ARRAY_OPEN
            self._is_first_value = True
        elif ch == 't':
            self._state = JsonState.BOOLEAN_TRUE
        elif ch == 'f':
            self._state = JsonState.BOOLEAN_FALSE
        elif ch == 'n':
            self._state = JsonState.NULL
        elif ch == '-' or ch in _DIGIT_CHARS:
            self._state = JsonState.NUMBER
            self._number_start = len(self._text_buffer) - 1

    def _enter_array_value(self, ch: str) -> None:
        """Enter a value state in array context."""
        if ch == '"':
            self._state = JsonState.STRING
            self._string_start = len(self._text_buffer)
        elif ch == '{':
            # Get items schema
            items_schema = {"type": "object"}
            if self._schema_stack:
                _, parent_schema = self._schema_stack[-1]
                items_schema = parent_schema.get("items", {"type": "object"})
                if isinstance(items_schema, dict) and self._get_type_from_schema(items_schema) == "object":
                    pass
                else:
                    items_schema = {"type": "object"}
            self._init_object_keys(items_schema)
            self._schema_stack.append((JsonState.ARRAY_COMMA, items_schema))
            self._state = JsonState.OBJECT_OPEN
            self._is_first_value = True
        elif ch == '[':
            items_schema = {"type": "array"}
            if self._schema_stack:
                _, parent_schema = self._schema_stack[-1]
                items_schema = parent_schema.get("items", {"type": "array"})
            self._schema_stack.append((JsonState.ARRAY_COMMA, items_schema))
            self._state = JsonState.ARRAY_OPEN
            self._is_first_value = True
        elif ch == 't':
            self._state = JsonState.BOOLEAN_TRUE
        elif ch == 'f':
            self._state = JsonState.BOOLEAN_FALSE
        elif ch == 'n':
            self._state = JsonState.NULL
        elif ch == '-' or ch in _DIGIT_CHARS:
            self._state = JsonState.NUMBER
            self._number_start = len(self._text_buffer) - 1

    def _value_completed(self) -> None:
        """Called when a primitive value has been fully generated."""
        if self._schema_stack:
            return_state, _ = self._schema_stack[-1]
            parent_type = self._get_type_from_schema(self._schema_stack[-1][1])
            if parent_type == "array":
                self._state = JsonState.ARRAY_COMMA
            else:
                self._state = JsonState.OBJECT_COMMA
        else:
            self._state = JsonState.DONE

    def _pop_schema(self) -> None:
        """Pop a completed object/array from the schema stack."""
        if self._schema_stack:
            self._schema_stack.pop()
            if self._object_keys_remaining:
                self._object_keys_remaining.pop()
            self._current_key = None

        if not self._schema_stack:
            self._state = JsonState.DONE
        else:
            return_state, parent_schema = self._schema_stack[-1]
            parent_type = self._get_type_from_schema(parent_schema)
            if parent_type == "array":
                self._state = JsonState.ARRAY_COMMA
            else:
                self._state = JsonState.OBJECT_COMMA

    def _init_object_keys(self, schema: dict) -> None:
        """Initialize the list of expected object keys from schema properties."""
        if "properties" in schema:
            self._object_keys_remaining.append(list(schema["properties"].keys()))
        else:
            self._object_keys_remaining.append([])  # any keys allowed

    def get_allowed_tokens(self, tokenizer: Any, generated_token_ids: list[int]) -> list[int]:
        """Return the list of allowed token IDs for the next token.

        Args:
            tokenizer: The model's tokenizer (must have decode/vocab).
            generated_token_ids: IDs of tokens generated so far.

        Returns:
            List of token IDs that are valid at the current state.
        """
        if self._state == JsonState.DONE:
            # Return EOS token(s) if available
            eos_ids = []
            if hasattr(tokenizer, 'eos_token_ids'):
                eos_ids = list(tokenizer.eos_token_ids)
            elif hasattr(tokenizer, 'eos_token_id'):
                eos_ids = [tokenizer.eos_token_id]
            return eos_ids

        expected_chars = self._get_expected_chars()

        if expected_chars is None:
            # Any token allowed (inside string, number, etc.)
            return self._get_all_token_ids(tokenizer)

        if not expected_chars:
            # Nothing allowed
            return []

        # Find tokens whose text starts with an expected character
        allowed = self._find_tokens_for_chars(tokenizer, expected_chars)
        return allowed

    def _get_all_token_ids(self, tokenizer: Any) -> list[int]:
        """Get all token IDs from the tokenizer vocabulary."""
        if hasattr(tokenizer, 'get_vocab'):
            vocab = tokenizer.get_vocab()
            return list(vocab.values())
        if hasattr(tokenizer, 'vocab'):
            vocab = tokenizer.vocab
            if isinstance(vocab, dict):
                return list(vocab.values())
            return list(range(len(vocab)))
        # Fallback: try to determine vocab size from tokenizer config
        vocab_size = getattr(tokenizer, 'vocab_size', None)
        if not vocab_size:
            # Try reading from the model config attached to tokenizer
            config = getattr(tokenizer, 'config', None)
            if config:
                vocab_size = config.get('vocab_size') or config.get('model_type') and 32000
        if vocab_size:
            return list(range(vocab_size))
        # Last resort: use actual tokenizer length
        try:
            return list(range(len(tokenizer.get_vocab())))
        except Exception:
            logger.debug("tokenizer vocab size detection failed, using fallback", exc_info=True)
            return list(range(32000))

    def _find_tokens_for_chars(self, tokenizer: Any, chars: set[str]) -> list[int]:
        """Find all tokens whose decoded text starts with one of the expected chars.

        Uses precomputed token cache for efficiency on repeated calls.
        """
        # Build token-to-first-char mapping if not cached
        cache_key = id(tokenizer)
        if not hasattr(self.__class__, '_token_char_cache'):
            self.__class__._token_char_cache = {}
        if cache_key not in self.__class__._token_char_cache:
            self.__class__._token_char_cache[cache_key] = self._build_token_char_map(tokenizer)

        char_map = self.__class__._token_char_cache[cache_key]
        allowed = set()
        for ch in chars:
            if ch in char_map:
                allowed.update(char_map[ch])
        return list(allowed)

    @staticmethod
    def _build_token_char_map(tokenizer: Any) -> dict[str, list[int]]:
        """Build a mapping from first character to list of token IDs.

        This is expensive but only done once per tokenizer.
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
            # Get the decoded character(s) that this token starts with
            first_char = token_text[0] if token_text else ''
            # For special tokens (starting with <), skip
            if first_char == '<' and len(token_text) > 1 and token_text.endswith('>'):
                continue
            # For byte-level tokens (starting with Ġ or similar), decode properly
            try:
                decoded = tokenizer.decode([token_id])
                if decoded:
                    first_decoded = decoded[0]
                    char_map.setdefault(first_decoded, []).append(token_id)
            except Exception:
                logger.debug("tokenizer decode failed for token %d, using raw char", token_id, exc_info=True)
                # Fallback: use raw first char
                char_map.setdefault(first_char, []).append(token_id)

        return char_map

    def reset(self) -> None:
        """Reset the constraint to start state."""
        self._state = JsonState.START
        self._text_buffer = ""
        self._schema_stack.clear()
        self._object_keys_remaining.clear()
        self._current_key = None
        self._in_string = False
        self._string_start = 0
        self._number_start = 0
        self._is_first_value = True

    def get_stats(self) -> dict[str, Any]:
        """Return constraint statistics for monitoring."""
        return {
            "state": self._state.name,
            "schema_stack_depth": len(self._schema_stack),
            "has_schema": self._schema is not None,
            "is_done": self.is_done,
            "text_buffer_len": len(self._text_buffer),
        }
        self._in_string = False
        self._is_first_value = True


# ── Logit Masking ───────────────────────────────────────────────────────────


def apply_json_constraint(
    logits: Any,
    allowed_token_ids: list[int],
) -> Any:
    """Mask logits for disallowed tokens to -inf.

    Args:
        logits: mx.array of shape (1, vocab_size) or (vocab_size,)
        allowed_token_ids: List of token IDs that are allowed

    Returns:
        mx.array with disallowed tokens set to -inf
    """
    import mlx.core as mx

    if not allowed_token_ids:
        # No tokens allowed — return logits as-is (shouldn't happen normally)
        return logits

    # Create mask: True where token is NOT allowed
    vocab_size = logits.shape[-1]
    mask = mx.ones((vocab_size,), dtype=mx.bool_)
    allowed = mx.array(allowed_token_ids)
    mask[allowed] = False

    # Apply mask
    neg_inf = mx.array(float('-inf'), dtype=logits.dtype)
    result = mx.where(mask, neg_inf, logits)
    return result


# ── Constrained Sampler ────────────────────────────────────────────────────


class ConstrainedSampler:
    """Wraps an mlx-lm sampler with JSON constraint masking.

    This is a callable that takes logits and returns a sampled token,
    but first masks logits to only allow tokens valid for the JSON state.

    Usage in scheduler:
        sampler = ConstrainedSampler(base_sampler, constraint, tokenizer)
        # Each call to sampler(logprobs) returns a valid token
    """

    def __init__(
        self,
        base_sampler: Callable,
        constraint: JsonSchemaConstraint,
        tokenizer: Any,
    ) -> None:
        self._base_sampler = base_sampler
        self._constraint = constraint
        self._tokenizer = tokenizer
        self._generated_ids: list[int] = []

    def __call__(self, logits: Any) -> Any:
        """Sample a token with JSON constraint masking.

        Args:
            logits: mx.array log-probabilities from the model

        Returns:
            mx.array with sampled token ID
        """
        # Get allowed tokens for current state
        allowed = self._constraint.get_allowed_tokens(self._tokenizer, self._generated_ids)

        if allowed:
            # Mask disallowed tokens
            masked_logits = apply_json_constraint(logits, allowed)
        else:
            masked_logits = logits

        # Sample using base sampler
        token = self._base_sampler(masked_logits)

        # Update constraint state
        import mlx.core as mx
        token_id = int(token)
        self._generated_ids.append(token_id)

        # Decode token text to advance state machine
        try:
            token_text = self._tokenizer.decode([token_id])
        except Exception:
            logger.debug("tokenizer decode failed for constrained sampler token %d", token_id, exc_info=True)
            token_text = ""
        self._constraint.advance(token_text)

        return token

    @property
    def constraint(self) -> JsonSchemaConstraint:
        return self._constraint


def make_constrained_sampler(
    base_sampler: Callable,
    schema: dict | None,
    tokenizer: Any,
    mode: str = "json_object",
) -> ConstrainedSampler:
    """Create a ConstrainedSampler for JSON generation.

    Args:
        base_sampler: The underlying mlx-lm sampler function
        schema: JSON Schema dict, or None for generic object
        tokenizer: The model's tokenizer
        mode: "json_object" for any JSON object, "json_schema" for schema-constrained

    Returns:
        ConstrainedSampler that wraps base_sampler with constraint masking
    """
    constraint = JsonSchemaConstraint(schema)
    return ConstrainedSampler(base_sampler, constraint, tokenizer)
