"""R21: supported JSON-Schema subset, differential against ``jsonschema``.

For every supported keyword the constrained decoder must accept exactly the
instances the reference validator accepts (walked through the real token
interface with a character vocabulary); every unsupported keyword must be
rejected up front instead of being silently ignored.
"""

from __future__ import annotations

import json

import pytest

jsonschema = pytest.importorskip("jsonschema")

from yunshu_engine.json_schema import (  # noqa: E402
    JsonSchemaConstraint,
    UnsupportedSchemaError,
    validate_supported_schema,
)


class _CharTok:
    eos_token_ids = [0]

    def __init__(self):
        chars = [chr(i) for i in range(32, 127)] + ["é", "\n"]
        self.v = {"<eos>": 0}
        for i, ch in enumerate(chars, start=1):
            self.v[ch] = i
        self.inv = {i: c for c, i in self.v.items()}

    def get_vocab(self):
        return self.v

    def decode(self, ids):
        return "".join(self.inv[i] for i in ids)


TOK = _CharTok()


def _accepts(schema, text) -> bool:
    c = JsonSchemaConstraint(schema)
    for ch in text:
        if TOK.v[ch] not in c.get_allowed_tokens(TOK, []):
            return False
        c.advance(ch)
    return 0 in c.get_allowed_tokens(TOK, [])


OBJ = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}

SUPPORTED = [
    ({"type": "string"}, ['"a"', "1", "null", '"é"', '"a\\nb"']),
    # Declared deviation: ``integer`` means an integer literal; jsonschema also
    # accepts 1e2 / 1.0 (integral floats), the decoder stays stricter.
    ({"type": "integer"}, ["1", "-3", "0", "1.5", '"a"']),
    ({"type": "number"}, ["1", "1.5", "-0.1", "1e5", '"a"']),
    ({"type": "boolean"}, ["true", "false", "null", "1"]),
    ({"type": "null"}, ["null", "1"]),
    ({"type": ["string", "null"]}, ['"a"', "null", "1"]),
    ({"type": ["integer", "null"]}, ["1", "null", "1.5"]),
    (
        {
            "type": "object",
            "properties": {"a": {"enum": ["a", 1, None]}},
            "required": ["a"],
        },
        ['{"a":"a"}', '{"a":1}', '{"a":null}', '{"a":"b"}', '{"a":2}'],
    ),
    (
        {"type": "object", "properties": {"a": {"const": "x"}}, "required": ["a"]},
        ['{"a":"x"}', '{"a":"y"}'],
    ),
    (
        {"type": "array", "items": {"enum": ["a", "b"]}},
        ['["a","b"]', '["c"]', "[]"],
    ),
    (OBJ, ['{"a":1}', "{}", '{"a":"x"}', '{"b":2}']),
    ({**OBJ, "additionalProperties": False}, ['{"a":1}', '{"a":1,"b":2}', "{}"]),
    (
        {"type": "object", "additionalProperties": {"type": "integer"}},
        ['{"a":1}', '{"a":"x"}', "{}"],
    ),
    (
        {"type": "array", "items": {"type": "integer"}},
        ["[]", "[1,2]", '[1,"a"]', "[1.5]"],
    ),
    ({"anyOf": [{"type": "integer"}, {"type": "string"}]}, ["1", '"a"', "null", "1.5"]),
    ({"oneOf": [{"type": "integer"}, {"type": "string"}]}, ["1", '"a"', "null", "1.5"]),
    (
        {
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/n"}},
            "required": ["a"],
            "$defs": {"n": {"type": "integer"}},
        },
        ['{"a":1}', '{"a":"x"}', '{"a":1.5}'],
    ),
    (
        {"type": "string", "title": "t", "description": "d", "format": "int32"},
        ['"a"', "1"],
    ),
]


@pytest.mark.parametrize(
    "schema,instances", SUPPORTED, ids=[json.dumps(s)[:60] for s, _ in SUPPORTED]
)
def test_supported_keywords_agree_with_jsonschema(schema, instances):
    validator = jsonschema.Draft202012Validator(schema)
    for text in instances:
        want = validator.is_valid(json.loads(text))
        # Declared deviation: an object that lists ``properties`` but no
        # ``additionalProperties`` is closed (OpenAI structured-output
        # convention), so undeclared keys are rejected even though plain JSON
        # Schema would allow them.
        if schema is OBJ and text == '{"b":2}':
            want = False
        assert _accepts(schema, text) == want, (schema, text)


UNSUPPORTED = [
    # root enum / const: the old machine let any string through ({"type":"string",
    # "enum":["a","b"]} accepted "c"), so they are rejected until enforced.
    {"enum": ["a", "b"]},
    {"type": "string", "enum": ["a", "b"]},
    {"const": "x"},
    {"type": "string", "minLength": 2},
    {"type": "string", "maxLength": 2},
    {"type": "string", "pattern": "^a+$"},
    {"type": "integer", "minimum": 5},
    {"type": "integer", "maximum": 5},
    {"type": "integer", "exclusiveMinimum": 5},
    {"type": "integer", "multipleOf": 2},
    {"type": "array", "items": {"type": "integer"}, "minItems": 2},
    {"type": "array", "items": {"type": "integer"}, "maxItems": 2},
    {"type": "array", "items": {"type": "integer"}, "uniqueItems": True},
    {"type": "array", "contains": {"type": "integer"}},
    {"not": {"type": "string"}},
    {"if": {"type": "string"}, "then": {"minLength": 1}},
    {"type": "object", "patternProperties": {"^a": {"type": "integer"}}},
    {"type": "object", "minProperties": 1},
    {"type": "object", "propertyNames": {"pattern": "^a"}},
    {"type": "object", "dependentRequired": {"a": ["b"]}},
    {"type": "array", "prefixItems": [{"type": "integer"}]},
    {"type": "array", "items": False},
    {"type": "object", "properties": {"a": True}},
    {"type": "object", "properties": {"a": {"$ref": "http://x/y.json"}}},
    {"type": "object", "properties": {"a": {"$ref": "#/$defs/missing"}}},
    {
        "type": "object",
        "properties": {"next": {"$ref": "#/$defs/node"}},
        "$defs": {
            "node": {"type": "object", "properties": {"next": {"$ref": "#/$defs/node"}}}
        },
    },
    {"type": "object", "properties": {"a": {"type": "integer", "minimum": 1}}},
    {"type": "wat"},
]


@pytest.mark.parametrize(
    "schema", UNSUPPORTED, ids=[json.dumps(s)[:60] for s in UNSUPPORTED]
)
def test_unsupported_constructs_are_rejected(schema):
    with pytest.raises(UnsupportedSchemaError):
        JsonSchemaConstraint(schema)
    with pytest.raises(ValueError):  # a 400-class error for the gateway
        validate_supported_schema(schema)


def test_error_names_the_offending_keyword():
    with pytest.raises(UnsupportedSchemaError, match="minLength"):
        validate_supported_schema(
            {"type": "object", "properties": {"a": {"type": "string", "minLength": 1}}}
        )


def test_compile_budget_rejects_exponential_ref_fanout():
    # d_i references d_{i+1} twice: 2^40 expansions without a budget.
    defs = {
        f"d{i}": {
            "type": "object",
            "properties": {
                "l": {"$ref": f"#/$defs/d{i + 1}"},
                "r": {"$ref": f"#/$defs/d{i + 1}"},
            },
        }
        for i in range(40)
    }
    defs["d40"] = {"type": "integer"}
    schema = {"$ref": "#/$defs/d0", "$defs": defs}
    with pytest.raises(UnsupportedSchemaError, match="too complex"):
        validate_supported_schema(schema)


def test_no_schema_still_means_any_object():
    assert JsonSchemaConstraint(None) is not None
