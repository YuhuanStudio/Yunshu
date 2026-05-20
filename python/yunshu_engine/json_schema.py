from __future__ import annotations
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


import copy
import json
import logging
from enum import Enum, auto
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── JSON Schema Auto-Repair (vLLM pattern) ──────────────────────────────────


def _repair_json_schema(
    schema: dict,
    _depth: int = 0,
    _root_defs: dict | None = None,
    _seen_refs: set[str] | None = None,
) -> dict:
    """Repair common JSON Schema issues before constrained decoding (vLLM pattern).

    Handles patterns that users commonly send from the OpenAI API but that
    our constrained decoder does not natively support:

    1. Resolves ``$ref`` by inlining the referenced definition from
       ``definitions`` / ``$defs``.
    2. Adds ``"type": "object"`` when ``properties`` is present but ``type``
       is missing.
    3. Converts ``"anyOf": [{"type": "string"}, {"type": "null"}]`` (and
       permutations) to ``{"oneOf": [...]}`` for cleaner resolution.
    4. Removes ``additionalProperties: false`` if explicitly set (we don't
       enforce it; it only causes errors in the state machine).
    5. Sets ``additionalProperties: false`` on objects that never mentioned
       ``additionalProperties`` at all (common OpenAI pattern — prevents
       generating extra keys).

    Args:
        schema: A JSON Schema dict.
        _depth: Recursion guard (max 10).
        _root_defs: Definitions block inherited from the root schema for
                    resolving nested ``$ref``.

    Returns:
        A repaired copy of the schema.
    """
    if not isinstance(schema, dict):
        return schema
    if _depth > 10:
        return schema

    schema = copy.deepcopy(schema)

    # Capture root-level definitions for nested $ref resolution
    if _root_defs is None:
        _root_defs = {}
        for key in ("definitions", "$defs"):
            if key in schema and isinstance(schema[key], dict):
                _root_defs.update(schema[key])
    if _seen_refs is None:
        _seen_refs = set()

    # 1. Resolve $ref by inlining the referenced definition
    if "$ref" in schema:
        ref_path = schema["$ref"]
        # Cycle detection: skip already-seen refs to prevent infinite recursion
        if ref_path in _seen_refs:
            del schema["$ref"]
            return schema
        _seen_refs.add(ref_path)
        if isinstance(ref_path, str) and ref_path.startswith("#/"):
            parts = ref_path[2:].split("/")
            target = None
            # Try local definitions first
            for key in ("definitions", "$defs"):
                if key in schema and isinstance(schema[key], dict):
                    if len(parts) == 2 and parts[0] in ("definitions", "$defs"):
                        target = schema[key].get(parts[1])
                    elif len(parts) == 1:
                        target = schema[key].get(parts[0])
                    if target is not None:
                        break
            # Fall back to root definitions
            if target is None and _root_defs:
                if len(parts) == 2 and parts[0] in ("definitions", "$defs"):
                    target = _root_defs.get(parts[1])
                elif len(parts) == 1:
                    target = _root_defs.get(parts[0])
            if target is not None and isinstance(target, dict):
                # Merge the referenced schema, removing $ref
                del schema["$ref"]
                # Remove the definitions block to avoid re-processing
                schema.pop("definitions", None)
                schema.pop("$defs", None)
                # Merge target into schema (target keys override)
                for k, v in target.items():
                    schema[k] = v
                # Recurse to repair the merged schema
                return _repair_json_schema(schema, _depth + 1, _root_defs, _seen_refs)

    # 2. Add "type": "object" if properties is present but type is missing
    if "properties" in schema and "type" not in schema:
        schema["type"] = "object"

    # 3. Convert anyOf with exactly one non-null + null to oneOf
    if "anyOf" in schema and isinstance(schema["anyOf"], list):
        options = schema["anyOf"]
        has_null = any(
            isinstance(o, dict) and o.get("type") == "null"
            for o in options
        )
        non_null = [o for o in options if isinstance(o, dict) and o.get("type") != "null"]
        if has_null and len(non_null) == 1:
            # Pattern: anyOf: [SomeType, null] → oneOf: [SomeType, null]
            schema["oneOf"] = schema.pop("anyOf")

    # 4. Remove additionalProperties: false (we don't enforce it; it causes
    #    errors in the constraint state machine).  Track whether it was
    #    explicitly present so step 5 doesn't re-add it.
    _had_explicit_additional_props = "additionalProperties" in schema
    if schema.get("additionalProperties") is False:
        del schema["additionalProperties"]

    # 5. Set additionalProperties: false for objects when it was never
    #    specified at all (common OpenAI pattern — prevents generating
    #    unexpected keys).  Do NOT re-add if the user explicitly set it
    #    (even if to false, which we removed above — the intent was to
    #    disallow extras, which we can't enforce, so we just omit it).
    schema_type = schema.get("type")
    if (
        schema_type == "object"
        and "additionalProperties" not in schema
        and "properties" in schema
        and not _had_explicit_additional_props
    ):
        schema["additionalProperties"] = False

    # 6. Strip if/then/else — these conditional schema keywords cannot be
    #    enforced during token-level constrained generation and would confuse
    #    _get_type_from_schema if left in place.
    for kw in ("if", "then", "else"):
        schema.pop(kw, None)

    # Recurse into sub-schemas
    if "properties" in schema and isinstance(schema["properties"], dict):
        for key, value in schema["properties"].items():
            if isinstance(value, dict):
                schema["properties"][key] = _repair_json_schema(value, _depth + 1, _root_defs, set(_seen_refs))

    if "items" in schema and isinstance(schema["items"], dict):
        schema["items"] = _repair_json_schema(schema["items"], _depth + 1, _root_defs, set(_seen_refs))

    if "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
        schema["additionalProperties"] = _repair_json_schema(
            schema["additionalProperties"], _depth + 1, _root_defs, set(_seen_refs)
        )

    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema and isinstance(schema[key], list):
            schema[key] = [
                _repair_json_schema(o, _depth + 1, _root_defs, set(_seen_refs)) if isinstance(o, dict) else o
                for o in schema[key]
            ]

    # Recurse into definitions/$defs so that $ref chains and nested schemas
    # inside definition entries are also repaired.  Without this, a definition
    # containing {"$ref": "#/definitions/Other"} would never be resolved.
    for defs_key in ("definitions", "$defs"):
        if defs_key in schema and isinstance(schema[defs_key], dict):
            for def_name, def_schema in schema[defs_key].items():
                if isinstance(def_schema, dict):
                    schema[defs_key][def_name] = _repair_json_schema(
                        def_schema, _depth + 1, _root_defs, set(_seen_refs)
                    )

    return schema


# ── JSON State Machine ──────────────────────────────────────────────────────


class JsonState(Enum):
    """States in the JSON generation state machine."""
    START = auto()
    # Object
    OBJECT_OPEN = auto()        # just emitted `{`
    OBJECT_KEY = auto()         # expecting a key string (or `}`)
    OBJECT_KEY_STRING = auto()  # inside a key string
    OBJECT_KEY_STRING_ESCAPE = auto()  # after \ in a key string
    OBJECT_KEY_STRING_UNICODE = auto()  # after \u in a key string
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
    STRING_UNICODE = auto()     # after `\u` — consuming 4 hex digits
    NUMBER = auto()             # inside a number (after [1-9] or 0)
    NUMBER_ZERO = auto()        # after leading '0' — only '.', 'eE', or terminators allowed
    NUMBER_FRACTION = auto()    # after `.` in a number
    NUMBER_EXPONENT = auto()    # after `e`/`E` in a number
    NUMBER_EXPONENT_SIGN = auto()  # after `e`/`E`+`+/-` — digit required
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


def json_encode_value(val: Any) -> str:
    """Encode a Python value to its JSON text representation for char extraction.

    Used by enum/const handling to determine which start characters are valid.
    """
    try:
        return json.dumps(val, ensure_ascii=False)
    except (TypeError, ValueError):
        return ""


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
                    The schema is auto-repaired via ``_repair_json_schema``
                    before use (vLLM pattern).
        """
        self._schema = _repair_json_schema(schema) if schema is not None else None
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
        self._number_seen_digit: bool = False  # True once at least one digit consumed
        self._number_has_dot: bool = False     # True once '.' consumed
        self._number_exponent_digit: bool = False  # True once at least one exponent digit consumed
        self._is_first_value: bool = True  # track first value in object/array
        self._is_integer: bool = False  # True when current number context requires integer (no .eE)
        # Snapshot stack for rollback (speculative draft validation)
        self._snapshots: list[tuple] = []
        # Track length of value literals for robust detection
        self._literal_remaining: int = 0  # chars remaining in true/false/null
        self._unicode_remaining: int = 0  # hex digits remaining in \uXXXX

        # If no schema, default to generic object
        if schema is None:
            self._schema = {"type": "object"}

        # Detect top-level type for START state initialization
        self._top_level_type = self._get_type_from_schema(self._schema)

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
        if "patternProperties" in schema:
            logger.warning(
                "patternProperties is not supported by the constrained decoder; "
                "these constraints will be ignored: %s",
                list(schema["patternProperties"].keys()),
            )
            return "object"
        if "items" in schema:
            return "array"
        if "enum" in schema:
            return "enum"
        if "const" in schema:
            return "const"
        if "allOf" in schema:
            # Work on a copy to avoid mutating the caller's schema dict.
            schema = dict(schema)
            # Merge fields from all sub-schemas: properties, required,
            # items, etc.  Previously only properties were merged, causing
            # required constraints from sub-schemas to be silently dropped.
            merged_props: dict = {}
            merged_required: list[str] = []
            merged_items: dict | None = None
            for sub in schema["allOf"]:
                if isinstance(sub, dict):
                    if "properties" in sub:
                        merged_props.update(sub["properties"])
                    if "required" in sub and isinstance(sub["required"], list):
                        merged_required.extend(sub["required"])
                    if "items" in sub and isinstance(sub["items"], dict):
                        if merged_items is None:
                            merged_items = sub["items"].copy()
                        else:
                            merged_items.update(sub["items"])
            if merged_props:
                # Persist merged results back into the schema so downstream
                # constraint resolution sees the full merged definition.
                schema["properties"] = merged_props
                if merged_required:
                    schema["required"] = list(dict.fromkeys(merged_required))
                return "object"
            if merged_items is not None:
                schema["items"] = merged_items
                return "array"
            # Otherwise collect unique types from sub-schemas
            types: list[str] = []
            for sub in schema["allOf"]:
                if isinstance(sub, dict):
                    t = self._get_type_from_schema(sub)
                    if t != "any":
                        types.append(t) if isinstance(t, str) else types.extend(t)
            unique_types = list(dict.fromkeys(types))
            if not unique_types:
                return "any"
            if len(unique_types) == 1:
                return unique_types[0]
            # allOf means intersection — conflicting types is a schema error.
            # Warn and fall back to "any" rather than silently widening to union.
            logger.warning(
                "allOf sub-schemas have conflicting types %s — treating as 'any'",
                unique_types,
            )
            return "any"
        if "anyOf" in schema or "oneOf" in schema:
            # Collect ALL option types (not just first) for union semantics
            options = schema.get("anyOf") or schema.get("oneOf") or []
            non_null = [o for o in options if isinstance(o, dict) and o.get("type") != "null"]
            has_null = any(
                isinstance(o, dict) and o.get("type") == "null"
                for o in options
            )
            if not non_null and not has_null:
                return "any"
            # Collect unique types across all options
            all_types: list[str] = []
            for opt in non_null:
                t = self._get_type_from_schema(opt)
                if isinstance(t, list):
                    all_types.extend(t)
                elif t != "any":
                    all_types.append(t)
            if has_null:
                all_types.append("null")
            unique = list(dict.fromkeys(all_types))  # preserve order, dedupe
            if not unique:
                return "any"
            if len(unique) == 1:
                return unique[0]
            return unique
        return "any"

    def _get_expected_chars(self) -> set[str] | None:
        """Get the set of characters that are valid at the current state.

        Returns None if any character is allowed (inside strings/numbers).
        Returns a set of specific characters otherwise.
        """
        state = self._state

        if state == JsonState.START:
            ws = {' ', '\t', '\n', '\r'}
            if self._top_level_type == "array":
                return {'['} | ws
            if self._top_level_type == "string":
                return {'"'} | ws
            if self._top_level_type == "boolean":
                return {'t', 'f'} | ws
            if self._top_level_type == "null":
                return {'n'} | ws
            if self._top_level_type in ("number", "integer"):
                return {'-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'} | ws
            if isinstance(self._top_level_type, list):
                # Multiple types possible
                chars = set()
                for t in self._top_level_type:
                    chars.update(self._type_to_start_chars(t))
                return chars | ws
            return {'{', ' ', '\t', '\n', '\r'}  # default: object

        if state == JsonState.OBJECT_OPEN:
            # After `{`, expect `"` (key) or `}`
            return {'"', '}', ' ', '\t', '\n', '\r'}

        if state == JsonState.OBJECT_KEY:
            # Expecting key string or close brace
            return {'"', '}', ' ', '\t', '\n', '\r'}

        if state in (JsonState.OBJECT_KEY_STRING, JsonState.OBJECT_KEY_STRING_ESCAPE,
                     JsonState.OBJECT_KEY_STRING_UNICODE):
            if state == JsonState.OBJECT_KEY_STRING:
                return None  # any char inside key string
            if state == JsonState.OBJECT_KEY_STRING_ESCAPE:
                return {'"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'}
            if state == JsonState.OBJECT_KEY_STRING_UNICODE:
                return _HEX_CHARS

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

        if state == JsonState.STRING_UNICODE:
            # Must provide hex digits
            return _HEX_CHARS

        if state == JsonState.NUMBER:
            chars = set(_DIGIT_CHARS)
            if not self._is_integer:
                if self._number_seen_digit and not self._number_has_dot:
                    chars.add('.')
                if self._number_seen_digit:
                    chars.update('eE')
            if self._number_seen_digit:
                chars.update({',', '}', ']', ' ', '\t', '\n', '\r'})
            return chars

        if state == JsonState.NUMBER_ZERO:
            # After leading '0', only '.', 'eE', or terminators (no more digits).
            # For integer type, exclude '.' and 'eE'.
            chars: set[str] = set()
            if not self._is_integer:
                chars.add('.')
                chars.update('eE')
            chars.update({',', '}', ']', ' ', '\t', '\n', '\r'})
            return chars

        if state == JsonState.NUMBER_FRACTION:
            # After '.', at least one digit is REQUIRED. 'eE' only allowed
            # after at least one digit (handled by NUMBER state transition).
            chars = set(_DIGIT_CHARS)
            return chars

        if state == JsonState.NUMBER_EXPONENT:
            # After 'e'/'E', may have sign or digits.
            chars = set(_DIGIT_CHARS)
            chars.update('+-')
            if self._number_exponent_digit:
                chars.update({',', '}', ']', ' ', '\t', '\n', '\r'})
            return chars

        if state == JsonState.NUMBER_EXPONENT_SIGN:
            # After e+/e-, only digits are valid
            return _DIGIT_CHARS

        if state == JsonState.BOOLEAN_TRUE:
            literal = "true"
            # After entering BOOLEAN_TRUE, first char 't' was consumed.
            # _literal_remaining tracks how many chars of the suffix remain.
            # idx = position in literal we need next (1='r', 2='u', 3='e')
            remaining = getattr(self, '_literal_remaining', 3)
            idx = len(literal) - remaining
            if 0 <= idx < len(literal):
                return {literal[idx]}
            return set()

        if state == JsonState.BOOLEAN_FALSE:
            literal = "false"
            remaining = getattr(self, '_literal_remaining', 4)
            idx = len(literal) - remaining
            if 0 <= idx < len(literal):
                return {literal[idx]}
            return set()

        if state == JsonState.NULL:
            literal = "null"
            remaining = getattr(self, '_literal_remaining', 3)
            idx = len(literal) - remaining
            if 0 <= idx < len(literal):
                return {literal[idx]}
            return set()

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

        # Handle enum: restrict to first chars of each enum value
        if "enum" in value_schema and isinstance(value_schema["enum"], list):
            chars: set[str] = set()
            for val in value_schema["enum"]:
                s = json_encode_value(val)
                if s:
                    chars.add(s[0])
            chars.update(_WHITESPACE_CHARS)
            return chars if chars else {' ', '\t', '\n', '\r'}

        # Handle const: restrict to first char of the constant value
        if "const" in value_schema:
            s = json_encode_value(value_schema["const"])
            if s:
                return {s[0]} | _WHITESPACE_CHARS
            return _WHITESPACE_CHARS

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

        # Handle enum in items schema
        if isinstance(items_schema, dict) and "enum" in items_schema and isinstance(items_schema["enum"], list):
            chars: set[str] = {']', ' ', '\t', '\n', '\r'}
            for val in items_schema["enum"]:
                s = json_encode_value(val)
                if s:
                    chars.add(s[0])
            return chars if len(chars) > 5 else chars | {']', ' ', '\t', '\n', '\r'}

        # Handle const in items schema
        if isinstance(items_schema, dict) and "const" in items_schema:
            s = json_encode_value(items_schema["const"])
            if s:
                return {s[0]} | {']', ' ', '\t', '\n', '\r'}
            return {']', ' ', '\t', '\n', '\r'}

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

    def _resolve_to_concrete_schema(self, schema: dict, target_type: str) -> dict | None:
        """Resolve a schema (possibly anyOf/oneOf) to find a concrete sub-schema
        matching *target_type* (e.g. "object" or "array").

        Returns the first matching sub-schema, or None if no match exists.
        This is needed because _get_type_from_schema returns a *list* for
        anyOf/oneOf schemas, so callers like _enter_value that check
        ``type == "object"`` would otherwise miss the concrete sub-schema
        and fall back to a generic ``{"type": "object"}`` — losing all
        property constraints.
        """
        if "anyOf" in schema:
            for opt in schema["anyOf"]:
                if not isinstance(opt, dict):
                    continue
                t = self._get_type_from_schema(opt)
                if t == target_type or (isinstance(t, list) and target_type in t):
                    return opt
        if "oneOf" in schema:
            for opt in schema["oneOf"]:
                if not isinstance(opt, dict):
                    continue
                t = self._get_type_from_schema(opt)
                if t == target_type or (isinstance(t, list) and target_type in t):
                    return opt
        return None

    def _is_integer_schema(self, schema: dict | None) -> bool:
        """Check if a schema requires an integer (no decimal/exponent allowed)."""
        if schema is None:
            return False
        schema_type = self._get_type_from_schema(schema)
        if isinstance(schema_type, list):
            # Multiple types: integer only if all types are integer
            return all(t == "integer" for t in schema_type)
        return schema_type == "integer"

    def _get_current_value_schema(self) -> dict | None:
        """Get the schema for the current value position."""
        if not self._schema_stack:
            return self._schema

        _, parent_schema = self._schema_stack[-1]
        parent_type = self._get_type_from_schema(parent_schema)

        if parent_type == "object" or "properties" in parent_schema:
            if self._current_key and "properties" in parent_schema:
                result = parent_schema["properties"].get(self._current_key)
                if result is not None:
                    return result
            # Key not found in properties — check additionalProperties schema.
            # If additionalProperties is a dict, it defines the value schema
            # for unknown keys.  If True (or absent), return None to allow any type.
            add_props = parent_schema.get("additionalProperties")
            if isinstance(add_props, dict):
                return add_props
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
            self._literal_remaining,
            self._unicode_remaining,
            self._number_seen_digit,
            self._number_has_dot,
            self._number_exponent_digit,
            self._is_integer,
        ))

    def discard_checkpoint(self) -> None:
        """Discard the most recent checkpoint without restoring state.

        Used in speculative decoding when all draft tokens are accepted
        so the checkpoint is no longer needed.
        """
        if self._snapshots:
            self._snapshots.pop()

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
            self._literal_remaining,
            self._unicode_remaining,
            self._number_seen_digit,
            self._number_has_dot,
            self._number_exponent_digit,
            self._is_integer,
        ) = self._snapshots.pop()

    def _process_text(self, text: str) -> None:
        """Process the generated text to update state machine."""
        i = 0
        # Precompute buffer offset: position in _text_buffer of text[0]
        buf_offset = len(self._text_buffer) - len(text)
        while i < len(text):
            ch = text[i]

            if self._state == JsonState.START:
                if ch == '{':
                    self._state = JsonState.OBJECT_OPEN
                    self._is_first_value = True
                    self._init_object_keys(self._schema)
                    self._schema_stack.append((JsonState.DONE, self._schema))
                elif ch == '[':
                    self._state = JsonState.ARRAY_OPEN
                    self._is_first_value = True
                    self._schema_stack.append((JsonState.DONE, self._schema))
                elif ch == '"':
                    # Top-level string value
                    self._state = JsonState.STRING
                    self._string_start = buf_offset + i + 1
                    self._schema_stack.append((JsonState.DONE, self._schema))
                elif ch in 'tf':
                    # Top-level boolean
                    self._state = JsonState.BOOLEAN_TRUE if ch == 't' else JsonState.BOOLEAN_FALSE
                    self._literal_remaining = 3 if ch == 't' else 4
                    self._schema_stack.append((JsonState.DONE, self._schema))
                elif ch == 'n':
                    # Top-level null
                    self._state = JsonState.NULL
                    self._literal_remaining = 3
                    self._schema_stack.append((JsonState.DONE, self._schema))
                elif ch == '-' or ch in _DIGIT_CHARS:
                    # Top-level number
                    self._state = JsonState.NUMBER_ZERO if ch == '0' else JsonState.NUMBER
                    self._number_start = buf_offset + i
                    self._number_seen_digit = ch != '-'
                    self._number_has_dot = False
                    self._number_exponent_digit = False
                    self._is_integer = self._is_integer_schema(self._schema)
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
                    # Transition to escape handling state to properly track
                    # escapes that span token boundaries.
                    self._state = JsonState.OBJECT_KEY_STRING_ESCAPE
                    i += 1
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

            if self._state == JsonState.OBJECT_KEY_STRING_ESCAPE:
                # Consuming the escaped character after \ in an object key
                if ch == 'u':
                    self._state = JsonState.OBJECT_KEY_STRING_UNICODE
                    self._unicode_remaining = 4
                else:
                    self._state = JsonState.OBJECT_KEY_STRING
                i += 1
                continue

            if self._state == JsonState.OBJECT_KEY_STRING_UNICODE:
                # Consuming hex digits after \u in an object key
                self._unicode_remaining -= 1
                if self._unicode_remaining <= 0:
                    self._state = JsonState.OBJECT_KEY_STRING
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
                self._enter_value(ch, buf_offset + i)
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
                self._enter_array_value(ch, buf_offset + i)
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
                self._enter_array_value(ch, buf_offset + i)
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
                    self._state = JsonState.STRING_UNICODE
                    self._unicode_remaining = 4
                else:
                    self._state = JsonState.STRING
                i += 1
                continue

            if self._state == JsonState.STRING_UNICODE:
                # Consuming hex digits after \u
                self._unicode_remaining -= 1
                if self._unicode_remaining <= 0:
                    self._state = JsonState.STRING
                i += 1
                continue

            if self._state in (JsonState.NUMBER, JsonState.NUMBER_ZERO,
                               JsonState.NUMBER_FRACTION,
                               JsonState.NUMBER_EXPONENT, JsonState.NUMBER_EXPONENT_SIGN):
                # JSON number: -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?
                # We use five sub-states to track what's valid next:
                #   NUMBER: after [1-9], more digits / '.' / 'eE' / terminators
                #   NUMBER_ZERO: after '0', only '.' / 'eE' / terminators (no more digits)
                #   NUMBER_FRACTION: after '.', digits required (no 'eE' until digit seen)
                #   NUMBER_EXPONENT: after 'e'/'E', sign or digit required
                #   NUMBER_EXPONENT_SIGN: after 'e+/e-', digit required
                if ch in _DIGIT_CHARS:
                    prior_seen_digit = self._number_seen_digit
                    self._number_seen_digit = True
                    if self._state == JsonState.NUMBER_ZERO:
                        # Leading zero followed by digit is invalid JSON (e.g., "07").
                        # Complete the current number as "0". The stray digit must
                        # be re-processed by the new state (e.g. OBJECT_COMMA),
                        # NOT silently consumed — otherwise the state machine
                        # loses sync with the generated text.
                        self._value_completed()
                        self._number_seen_digit = False
                        # Do NOT advance i — re-process this digit in the new state
                        continue
                    elif self._state == JsonState.NUMBER_FRACTION:
                        # After digit in fraction, exponent is now allowed.
                        # Transition to NUMBER so _get_expected_chars includes 'eE'.
                        # _number_has_dot stays True, so '.' won't be re-allowed.
                        self._state = JsonState.NUMBER
                    elif self._state == JsonState.NUMBER_EXPONENT_SIGN:
                        self._state = JsonState.NUMBER_EXPONENT
                        self._number_exponent_digit = True
                    elif self._state == JsonState.NUMBER_EXPONENT:
                        self._number_exponent_digit = True
                    elif self._state == JsonState.NUMBER and ch == '0' and not prior_seen_digit:
                        # e.g. after '-' then '0': treat as leading zero
                        self._state = JsonState.NUMBER_ZERO
                    i += 1
                    continue
                if ch == '.' and self._state in (JsonState.NUMBER, JsonState.NUMBER_ZERO) and not self._number_has_dot and not self._is_integer:
                    self._state = JsonState.NUMBER_FRACTION
                    self._number_has_dot = True
                    i += 1
                    continue
                if ch in 'eE' and self._state in (JsonState.NUMBER, JsonState.NUMBER_ZERO) and self._number_seen_digit and not self._is_integer:
                    self._state = JsonState.NUMBER_EXPONENT
                    self._number_exponent_digit = False
                    i += 1
                    continue
                if ch in '+-' and self._state == JsonState.NUMBER_EXPONENT:
                    self._state = JsonState.NUMBER_EXPONENT_SIGN
                    i += 1
                    continue
                # Number ended — validate required digits were produced.
                self._value_completed()
                continue

            if self._state == JsonState.BOOLEAN_TRUE:
                # Use literal_remaining counter for robust detection
                self._literal_remaining -= 1
                if self._literal_remaining <= 0:
                    self._value_completed()
                i += 1
                continue

            if self._state == JsonState.BOOLEAN_FALSE:
                self._literal_remaining -= 1
                if self._literal_remaining <= 0:
                    self._value_completed()
                i += 1
                continue

            if self._state == JsonState.NULL:
                self._literal_remaining -= 1
                if self._literal_remaining <= 0:
                    self._value_completed()
                i += 1
                continue

            i += 1

    def _enter_value(self, ch: str, buf_pos: int | None = None) -> None:
        """Enter a value state based on the first character."""
        if buf_pos is None:
            buf_pos = len(self._text_buffer) - 1
        if ch == '"':
            self._state = JsonState.STRING
            self._string_start = buf_pos + 1  # position after the opening quote
        elif ch == '{':
            value_schema = self._get_current_value_schema()
            obj_schema = self._resolve_object_schema(value_schema)
            self._init_object_keys(obj_schema)
            self._schema_stack.append((JsonState.OBJECT_COMMA, obj_schema))
            self._state = JsonState.OBJECT_OPEN
            self._is_first_value = True
        elif ch == '[':
            value_schema = self._get_current_value_schema()
            arr_schema = self._resolve_array_schema(value_schema)
            self._schema_stack.append((JsonState.ARRAY_COMMA, arr_schema))
            self._state = JsonState.ARRAY_OPEN
            self._is_first_value = True
        elif ch == 't':
            self._state = JsonState.BOOLEAN_TRUE
            self._literal_remaining = 3  # "rue" remaining after 't'
        elif ch == 'f':
            self._state = JsonState.BOOLEAN_FALSE
            self._literal_remaining = 4  # "alse" remaining after 'f'
        elif ch == 'n':
            self._state = JsonState.NULL
            self._literal_remaining = 3  # "ull" remaining after 'n'
        elif ch == '-' or ch in _DIGIT_CHARS:
            self._state = JsonState.NUMBER_ZERO if ch == '0' else JsonState.NUMBER
            self._number_start = buf_pos
            self._number_seen_digit = ch != '-'
            self._number_has_dot = False
            self._number_exponent_digit = False
            # Determine if schema expects integer (no .eE allowed)
            value_schema = self._get_current_value_schema()
            self._is_integer = self._is_integer_schema(value_schema)

    def _resolve_object_schema(self, value_schema: dict | None) -> dict:
        """Resolve a value schema to a concrete object schema, unwrapping anyOf/oneOf."""
        if value_schema is None:
            return {"type": "object"}
        # Direct match
        schema_type = self._get_type_from_schema(value_schema)
        if schema_type == "object":
            return value_schema
        # anyOf/oneOf: find the first object-typed option
        concrete = self._resolve_to_concrete_schema(value_schema, "object")
        if concrete is not None:
            return concrete
        return {"type": "object"}

    def _resolve_array_schema(self, value_schema: dict | None) -> dict:
        """Resolve a value schema to a concrete array schema, unwrapping anyOf/oneOf."""
        if value_schema is None:
            return {"type": "array"}
        schema_type = self._get_type_from_schema(value_schema)
        if schema_type == "array":
            return value_schema
        concrete = self._resolve_to_concrete_schema(value_schema, "array")
        if concrete is not None:
            return concrete
        return {"type": "array"}

    def _enter_array_value(self, ch: str, buf_pos: int | None = None) -> None:
        """Enter a value state in array context."""
        if buf_pos is None:
            buf_pos = len(self._text_buffer) - 1
        if ch == '"':
            self._state = JsonState.STRING
            self._string_start = buf_pos + 1  # position after the opening quote
        elif ch == '{':
            # Get items schema and resolve anyOf/oneOf to find object option
            items_schema = {"type": "object"}
            if self._schema_stack:
                _, parent_schema = self._schema_stack[-1]
                raw_items = parent_schema.get("items", {"type": "object"})
                if isinstance(raw_items, dict):
                    items_schema = self._resolve_object_schema(raw_items)
            self._init_object_keys(items_schema)
            self._schema_stack.append((JsonState.ARRAY_COMMA, items_schema))
            self._state = JsonState.OBJECT_OPEN
            self._is_first_value = True
        elif ch == '[':
            items_schema = {"type": "array"}
            if self._schema_stack:
                _, parent_schema = self._schema_stack[-1]
                raw_items = parent_schema.get("items", {"type": "array"})
                if isinstance(raw_items, dict):
                    items_schema = self._resolve_array_schema(raw_items)
            self._schema_stack.append((JsonState.ARRAY_COMMA, items_schema))
            self._state = JsonState.ARRAY_OPEN
            self._is_first_value = True
        elif ch == 't':
            self._state = JsonState.BOOLEAN_TRUE
            self._literal_remaining = 3  # "rue"
        elif ch == 'f':
            self._state = JsonState.BOOLEAN_FALSE
            self._literal_remaining = 4  # "alse"
        elif ch == 'n':
            self._state = JsonState.NULL
            self._literal_remaining = 3  # "ull"
        elif ch == '-' or ch in _DIGIT_CHARS:
            self._state = JsonState.NUMBER_ZERO if ch == '0' else JsonState.NUMBER
            self._number_start = buf_pos
            self._number_seen_digit = ch != '-'
            self._number_has_dot = False
            self._number_exponent_digit = False
            # Determine if items schema expects integer
            items_schema = {"type": "string"}
            if self._schema_stack:
                _, parent_schema = self._schema_stack[-1]
                items_schema = parent_schema.get("items", {"type": "string"})
            self._is_integer = self._is_integer_schema(items_schema)

    def _value_completed(self) -> None:
        """Called when a primitive value has been fully generated."""
        if self._schema_stack:
            return_state, _ = self._schema_stack[-1]
            parent_type = self._get_type_from_schema(self._schema_stack[-1][1])
            # For primitives at the true top level (stack has only one entry
            # whose return_state is DONE), completion means the entire JSON
            # value is done.
            if len(self._schema_stack) == 1 and return_state == JsonState.DONE:
                # Top-level primitive — check if schema allows only primitives
                is_container = parent_type in ("object", "array") or (
                    isinstance(parent_type, list) and
                    any(t in ("object", "array") for t in parent_type)
                )
                if not is_container:
                    self._schema_stack.pop()
                    self._state = JsonState.DONE
                    return
            if parent_type == "array" or (isinstance(parent_type, list) and "array" in parent_type):
                self._state = JsonState.ARRAY_COMMA
            else:
                self._state = JsonState.OBJECT_COMMA
        else:
            self._state = JsonState.DONE

    def _pop_schema(self) -> None:
        """Pop a completed object/array from the schema stack."""
        if self._schema_stack:
            popped_state, popped_schema = self._schema_stack[-1]
            popped_type = self._get_type_from_schema(popped_schema)
            self._schema_stack.pop()
            # Only pop object_keys_remaining for objects (not arrays),
            # since _init_object_keys only pushes for objects.
            if (popped_type == "object"
                or (isinstance(popped_type, list) and "object" in popped_type)
                or (popped_schema and "properties" in popped_schema)):
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
        if not hasattr(self.__class__, '_token_char_cache'):
            import weakref
            self.__class__._token_char_cache = weakref.WeakKeyDictionary()
        if tokenizer not in self.__class__._token_char_cache:
            self.__class__._token_char_cache[tokenizer] = self._build_token_char_map(tokenizer)

        char_map = self.__class__._token_char_cache[tokenizer]
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
        self._number_seen_digit = False
        self._number_has_dot = False
        self._number_exponent_digit = False
        self._is_first_value = True
        self._is_integer = False
        self._literal_remaining = 0
        self._unicode_remaining = 0
        self._snapshots.clear()

    def get_stats(self) -> dict[str, Any]:
        """Return constraint statistics for monitoring."""
        return {
            "state": self._state.name,
            "schema_stack_depth": len(self._schema_stack),
            "has_schema": self._schema is not None,
            "is_done": self.is_done,
            "text_buffer_len": len(self._text_buffer),
        }


# ── Logit Masking ───────────────────────────────────────────────────────────


def apply_json_constraint(
    logits: Any,
    allowed_token_ids: list[int],
) -> Any:
    """Mask logits for disallowed tokens to -inf.

    When no tokens are allowed (empty ``allowed_token_ids``), falls back to
    the argmax of the original logits instead of masking all to -inf, which
    would cause softmax NaN.

    Args:
        logits: mx.array of shape (1, vocab_size) or (vocab_size,)
        allowed_token_ids: List of token IDs that are allowed

    Returns:
        mx.array with disallowed tokens set to -inf
    """
    import mlx.core as mx

    neg_inf = mx.array(float('-inf'), dtype=logits.dtype)

    if not allowed_token_ids:
        # No valid tokens in current state — fall back to argmax of original
        # logits to avoid all-inf -> softmax NaN
        logger.warning(
            "JSON constraint: no allowed tokens in current state, "
            "falling back to argmax of original logits"
        )
        best = int(mx.argmax(logits.reshape(-1)))
        mask = mx.ones(logits.shape[-1:], dtype=mx.bool_)
        mask[best] = False
        return mx.where(mask, neg_inf, logits)

    # Create mask: True where token is NOT allowed
    vocab_size = logits.shape[-1]
    mask = mx.ones((vocab_size,), dtype=mx.bool_)
    allowed = mx.array(allowed_token_ids)
    mask[allowed] = False

    # Apply mask
    result = mx.where(mask, neg_inf, logits)

    # Safety: if all allowed tokens already had -inf logits, fall back to argmax
    is_finite = mx.isfinite(result.reshape(-1))
    if not mx.any(is_finite).item():
        logger.warning(
            "JSON constraint: all allowed tokens have -inf logits, "
            "falling back to argmax of original logits"
        )
        best = int(mx.argmax(logits.reshape(-1)))
        fallback_mask = mx.ones(logits.shape[-1:], dtype=mx.bool_)
        fallback_mask[best] = False
        return mx.where(fallback_mask, neg_inf, logits)

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
            # No valid tokens in current state — force EOS to avoid
            # producing invalid output.  Setting all logits to -inf
            # causes softmax NaN, so we allowlist only EOS tokens.
            eos_ids = []
            if hasattr(self._tokenizer, 'eos_token_ids'):
                eos_ids = list(self._tokenizer.eos_token_ids)
            elif hasattr(self._tokenizer, 'eos_token_id'):
                eos_ids = [self._tokenizer.eos_token_id]
            if eos_ids:
                masked_logits = apply_json_constraint(logits, eos_ids)
            else:
                masked_logits = apply_json_constraint(logits, [])

        # Sample using base sampler
        token = self._base_sampler(masked_logits)

        # Update constraint state
        import mlx.core as mx
        token_id = int(token)
        self._generated_ids.append(token_id)

        # Decode token text to advance state machine.
        # Skip advance() for EOS tokens — their decoded text (e.g. "</s>")
        # would corrupt the constraint's text buffer and break
        # checkpoint/rollback correctness.
        eos_ids = set()
        if hasattr(self._tokenizer, 'eos_token_ids'):
            eos_ids = set(self._tokenizer.eos_token_ids)
        elif hasattr(self._tokenizer, 'eos_token_id'):
            eos_ids = {self._tokenizer.eos_token_id}

        if token_id not in eos_ids:
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

    def checkpoint(self) -> None:
        """Forward checkpoint to underlying constraint (for spec decode)."""
        if hasattr(self._constraint, 'checkpoint'):
            self._constraint.checkpoint()

    def rollback(self) -> None:
        """Forward rollback to underlying constraint (for spec decode)."""
        if hasattr(self._constraint, 'rollback'):
            self._constraint.rollback()


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
