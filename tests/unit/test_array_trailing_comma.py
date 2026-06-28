"""(HIGH): json_schema FSM accepted array trailing commas (`[1,]`).

ARRAY_VALUE as a resting state is reached ONLY after a comma (the after-`[`
empty-array case stays in ARRAY_OPEN; its first value char transitions to
ARRAY_VALUE while simultaneously consuming the char). So once we are *resting* in
ARRAY_VALUE, a comma has already committed to another element and `]` is invalid
— but `_get_expected_chars()` for ARRAY_VALUE returned
`_get_array_value_start_chars()` which unconditionally includes `]`, letting the
constrained-decode mask emit `[1,]` / `[{"x":1},]` (invalid JSON). Fixed by
subtracting `{']'}` for ARRAY_VALUE (ARRAY_OPEN keeps `]` for the empty-array
case). Mirrors the OBJECT_KEY fix.
"""
from __future__ import annotations

from yunshu_engine.json_schema import JsonSchemaConstraint, JsonState


def _at(schema, text):
    c = JsonSchemaConstraint(schema)
    c.advance(text)
    return c


def test_empty_array_close_still_allowed_after_open():
    """`[]` must still be valid — ARRAY_OPEN keeps `]`."""
    c = _at({"type": "array", "items": {"type": "integer"}}, "[")
    assert c._state == JsonState.ARRAY_OPEN
    assert "]" in c._get_expected_chars()


def test_close_forbidden_immediately_after_comma():
    """`[1,` must NOT permit `]` next — that would be a trailing comma."""
    c = _at({"type": "array", "items": {"type": "integer"}}, "[1,")
    assert c._state == JsonState.ARRAY_VALUE
    chars = c._get_expected_chars()
    assert "]" not in chars
    assert any(d in chars for d in "0123456789-")  # a real value starter remains


def test_close_allowed_after_value_via_array_comma():
    """A completed value (no comma) → ARRAY_COMMA still offers `]` (valid `["a"]`)."""
    c = _at({"type": "array", "items": {"type": "string"}}, '["a"')
    assert c._state == JsonState.ARRAY_COMMA
    assert "]" in c._get_expected_chars()


def test_object_array_trailing_comma_forbidden():
    """`[{"x":1},` must not allow `]` next."""
    c = _at({"type": "array", "items": {"type": "object",
                                        "properties": {"x": {"type": "integer"}}}}, '[{"x":1},')
    assert c._state == JsonState.ARRAY_VALUE
    assert "]" not in c._get_expected_chars()
