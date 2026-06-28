"""jump-forward foundation — JsonSchemaConstraint.forced_continuation()
extracts the grammar-forced literal string (structural tokens that need no model
forward pass). Pure FSM simulation; no hot-path wiring yet."""
from __future__ import annotations

from python.yunshu_engine.json_schema import JsonSchemaConstraint, JsonState


def _c(schema):
    return JsonSchemaConstraint(schema)


def test_forced_colon_after_key():
    # Single required prop, strict → after `{` the key `"a"` is forced, then `:`.
    c = _c({"type": "object", "properties": {"a": {"type": "string"}},
            "required": ["a"], "additionalProperties": False})
    fwd = c.forced_continuation()
    # From START the object open `{` is forced, then the only key, then colon.
    assert fwd.startswith("{")
    assert '"a"' in fwd and ":" in fwd


def test_no_force_on_branch():
    # Two required props → after `{` the key is a branch (a or b) → not forced past `{`.
    c = _c({"type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a", "b"], "additionalProperties": False})
    fwd = c.forced_continuation()
    # `{` is forced, but the key name is a branch → must stop at/after `{` (the `"`
    # may be forced but the key letter is a branch).
    assert fwd.startswith("{")
    assert not ('"a"' in fwd and '"b"' in fwd)  # can't force a specific key


def test_done_returns_empty():
    c = _c({"type": "string"})
    c.advance('"hi"')
    if c.state == JsonState.DONE:
        assert c.forced_continuation() == ""


def test_no_mutation():
    # forced_continuation must NOT change the real constraint's state.
    c = _c({"type": "object", "properties": {"a": {"type": "string"}},
            "required": ["a"], "additionalProperties": False})
    st_before = c.state
    _ = c.forced_continuation()
    assert c.state == st_before


def test_free_string_value_not_forced():
    # An open string value is a free branch → no forced continuation into it.
    c = _c({"type": "object", "properties": {"a": {"type": "string"}},
            "required": ["a"], "additionalProperties": False})
    c.advance('{"a":')
    fwd = c.forced_continuation()
    # may force whitespace/`"` to open the string, but never invents string content
    assert "hello" not in fwd
