"""Schemas outside the in-house subset are enforced by llguidance, differential vs jsonschema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

from yunshu_engine.grammar_constraint import (  # noqa: E402
    LlgJsonSchemaConstraint,
    build_json_constraint,
    validate_constraint_spec,
)
from yunshu_engine.json_schema import (  # noqa: E402
    JsonSchemaConstraint,
    UnsupportedSchemaError,
)

TOKENIZER = Path("/Volumes/P5Plus/models/Qwen2.5-3B-Instruct-4bit")
EOS = (151645, 151643)


@pytest.fixture(scope="module")
def tok():
    if not TOKENIZER.exists():
        pytest.skip("Qwen2.5 tokenizer not available")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(TOKENIZER))


def _accepts(tok, schema, text) -> bool:
    c = build_json_constraint(schema, tok)
    for tid in tok.encode(text):
        if tid not in c.get_allowed_tokens(tok, []):
            return False
        c.advance(tok.decode([tid]))
    return any(e in c.get_allowed_tokens(tok, []) for e in EOS)


def _valid(schema, text) -> bool:
    return jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    ).is_valid(json.loads(text))


CASES = [
    ({"type": "string", "pattern": "^a+$"}, ['"aa"', '"b"', '""', '"aab"']),
    ({"type": "string", "pattern": "^[0-9]{3}-[0-9]{4}$"}, ['"555-1234"', '"55-1234"']),
    ({"type": "string", "minLength": 2}, ['"a"', '"ab"', '"abc"']),
    ({"type": "string", "maxLength": 2}, ['"abc"', '"ab"', '""']),
    ({"type": "string", "minLength": 2, "maxLength": 3}, ['"é"', '"éé"', '"abcd"']),
    ({"type": "integer", "minimum": 5}, ["1", "5", "6", "10"]),
    ({"type": "integer", "maximum": 5}, ["6", "5", "-3"]),
    ({"type": "integer", "exclusiveMinimum": 5}, ["5", "6"]),
    ({"type": "integer", "exclusiveMaximum": 5}, ["5", "4"]),
    ({"type": "integer", "minimum": 5, "maximum": 10}, ["4", "7", "11"]),
    ({"type": "number", "minimum": 0.5}, ["0.4", "0.5", "2"]),
    ({"type": "integer", "multipleOf": 2}, ["3", "4", "10", "7"]),
    ({"type": "number", "multipleOf": 0.5}, ["1.5", "1.2", "2"]),
    (
        {"type": "array", "items": {"type": "integer"}, "minItems": 2},
        ["[1]", "[1,2]", "[]"],
    ),
    (
        {"type": "array", "items": {"type": "integer"}, "maxItems": 2},
        ["[1,2,3]", "[1,2]", "[]"],
    ),
    (
        {
            "type": "array",
            "prefixItems": [{"type": "integer"}, {"type": "string"}],
            "items": False,
        },
        ['[1,"a"]', '["a",1]', "[1]", '[1,"a","b"]'],
    ),
    ({"type": "string", "format": "date"}, ['"2024-02-29"', '"2024-13-01"', '"x"']),
    (
        {"type": "string", "format": "uuid"},
        ['"123e4567-e89b-12d3-a456-426614174000"', '"123"'],
    ),
    ({"type": "string", "format": "ipv4"}, ['"192.168.0.1"', '"999.1.1.1"', '"a"']),
    (
        {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "minimum": 1, "maximum": 3},
                "s": {"type": "string", "minLength": 2},
            },
            "required": ["n", "s"],
            "additionalProperties": False,
        },
        ['{"n":2,"s":"ab"}', '{"n":4,"s":"ab"}', '{"n":2,"s":"a"}'],
    ),
    ({"enum": ["a", "b"]}, ['"a"', '"c"']),
    ({"type": "string", "const": "x"}, ['"x"', '"y"']),
]


@pytest.mark.parametrize(
    "schema,instances", CASES, ids=[json.dumps(s)[:70] for s, _ in CASES]
)
def test_enforced_keywords_agree_with_jsonschema(tok, schema, instances):
    assert isinstance(build_json_constraint(schema, tok), LlgJsonSchemaConstraint), (
        "schema should be routed to llguidance"
    )
    for text in instances:
        assert _accepts(tok, schema, text) == _valid(schema, text), (schema, text)


def test_email_format_accepts_addresses_and_rejects_non_addresses(tok):
    schema = {"type": "string", "format": "email"}
    assert _accepts(tok, schema, '"a@b.co"')
    assert not _accepts(tok, schema, '"not an email"')


def test_pydantic_field_constraints(tok):
    pydantic = pytest.importorskip("pydantic")

    class Person(pydantic.BaseModel):
        name: str = pydantic.Field(min_length=2, max_length=8)
        age: int = pydantic.Field(ge=0, le=120)
        score: float = pydantic.Field(gt=0, le=1)
        tags: list[str] = pydantic.Field(min_length=1, max_length=3)
        code: str = pydantic.Field(pattern=r"^[A-Z]{2}[0-9]{2}$")

    schema = Person.model_json_schema()
    assert isinstance(build_json_constraint(schema, tok), LlgJsonSchemaConstraint)
    good = '{"name":"Ada","age":36,"score":0.5,"tags":["x"],"code":"AB12"}'
    assert _valid(schema, good) and _accepts(tok, schema, good)
    for bad in (
        '{"name":"A","age":36,"score":0.5,"tags":["x"],"code":"AB12"}',
        '{"name":"Ada","age":130,"score":0.5,"tags":["x"],"code":"AB12"}',
        '{"name":"Ada","age":36,"score":0,"tags":["x"],"code":"AB12"}',
        '{"name":"Ada","age":36,"score":0.5,"tags":[],"code":"AB12"}',
        '{"name":"Ada","age":36,"score":0.5,"tags":["x"],"code":"ab12"}',
    ):
        assert not _valid(schema, bad)
        assert not _accepts(tok, schema, bad), bad


def test_recursive_ref_is_enforced_not_dropped(tok):
    schema = {
        "type": "object",
        "properties": {"v": {"type": "integer"}, "next": {"$ref": "#/$defs/node"}},
        "required": ["v"],
        "$defs": {
            "node": {
                "type": "object",
                "properties": {
                    "v": {"type": "integer"},
                    "next": {"$ref": "#/$defs/node"},
                },
                "required": ["v"],
            }
        },
    }
    assert isinstance(build_json_constraint(schema, tok), LlgJsonSchemaConstraint)
    assert _accepts(tok, schema, '{"v":1,"next":{"v":2,"next":{"v":3}}}')
    assert not _accepts(tok, schema, '{"v":1,"next":{"v":"x"}}')


def test_in_subset_schema_keeps_the_in_house_constraint():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"],
    }
    assert isinstance(build_json_constraint(schema), JsonSchemaConstraint)


@pytest.mark.parametrize(
    "schema,needle",
    [
        (
            {"type": "array", "uniqueItems": True, "items": {"type": "integer"}},
            "uniqueItems",
        ),
        ({"not": {"type": "string"}}, "not"),
        ({"if": {"type": "string"}, "then": {"minLength": 1}}, "if"),
        ({"type": "array", "contains": {"type": "integer"}}, "contains"),
    ],
)
def test_only_what_llguidance_cannot_compile_is_rejected(schema, needle):
    with pytest.raises(UnsupportedSchemaError, match=needle):
        validate_constraint_spec(schema)
    with pytest.raises(UnsupportedSchemaError):
        build_json_constraint(schema)


def test_value_constraints_are_accepted_at_validation_time():
    for schema in (
        {"type": "string", "minLength": 2},
        {"type": "integer", "minimum": 1},
        {"type": "array", "prefixItems": [{"type": "integer"}]},
    ):
        validate_constraint_spec(schema)


def test_llg_constraint_checkpoint_rollback_no_arg(tok):
    c = build_json_constraint({"type": "string", "minLength": 2}, tok)
    c.get_allowed_tokens(tok, [])
    c.checkpoint()
    c.advance('"')
    c.advance("ab")
    c.rollback()
    assert not c.is_done
    first = tok.encode('"')[0]
    assert first in c.get_allowed_tokens(tok, [])
