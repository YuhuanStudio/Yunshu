"""llguidance is the JSON-schema engine; our conventions are applied to the schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from yunshu_engine import settings
from yunshu_engine.grammar_constraint import (
    LlgJsonSchemaConstraint,
    _normalize_for_llg,
    build_json_constraint,
)
from yunshu_engine.json_schema import JsonSchemaConstraint

TOKENIZER = Path("/Volumes/P5Plus/models/Qwen2.5-3B-Instruct-4bit")


@pytest.fixture(scope="module")
def tok():
    if not TOKENIZER.exists():
        pytest.skip("Qwen2.5 tokenizer not available")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(TOKENIZER))


def _accepts(c, tok, text: str) -> bool:
    for tid in tok.encode(text):
        if tid not in c.get_allowed_tokens(tok, []):
            return False
        c.advance(tok.decode([tid]))
    return any(e in c.get_allowed_tokens(tok, []) for e in (151645, 151643))


# ── schema normalisation (no tokenizer needed) ─────────────────────────────


def test_objects_with_properties_are_closed_and_typed():
    out = _normalize_for_llg(
        {
            "properties": {
                "a": {"type": "object", "properties": {"b": {"type": "string"}}},
                "open": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {"c": {}},
                },
                "free": {"type": "object"},
            }
        }
    )
    assert out["type"] == "object" and out["additionalProperties"] is False
    assert out["properties"]["a"]["additionalProperties"] is False
    assert out["properties"]["open"]["additionalProperties"] is True
    assert "additionalProperties" not in out["properties"]["free"]


def test_property_names_and_enum_values_are_not_schemas():
    out = _normalize_for_llg(
        {
            "type": "object",
            "properties": {
                "properties": {"type": "string"},
                "type": {"enum": [{"properties": {"x": 1}}]},
            },
        }
    )
    assert out["properties"]["properties"] == {"type": "string"}
    assert out["properties"]["type"] == {"enum": [{"properties": {"x": 1}}]}


def test_all_of_objects_merge_before_closing():
    out = _normalize_for_llg(
        {
            "allOf": [
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "required": ["a"],
                },
                {
                    "type": "object",
                    "properties": {"b": {"type": "integer"}},
                    "required": ["b"],
                },
            ]
        }
    )
    assert "allOf" not in out
    assert set(out["properties"]) == {"a", "b"} and out["required"] == ["a", "b"]
    assert out["additionalProperties"] is False


def test_unmergeable_all_of_branches_stay_open():
    out = _normalize_for_llg(
        {"allOf": [{"$ref": "#/$defs/A"}, {"properties": {"b": {"type": "integer"}}}]}
    )
    assert "additionalProperties" not in out["allOf"][1]


# ── routing ────────────────────────────────────────────────────────────────


class _SlowTokenizer:
    """Not a Hugging Face fast tokenizer: llguidance cannot bind it."""

    def get_vocab(self):
        return {"{": 0, "}": 1}

    def decode(self, ids):
        return "".join("{}"[i] for i in ids)


SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
}


def test_unbindable_tokenizer_falls_back_to_inhouse():
    assert isinstance(
        build_json_constraint(SCHEMA, _SlowTokenizer()), JsonSchemaConstraint
    )


def test_engine_setting_selects_inhouse(tok, monkeypatch):
    assert isinstance(build_json_constraint(SCHEMA, tok), LlgJsonSchemaConstraint)
    assert isinstance(build_json_constraint(None, tok), LlgJsonSchemaConstraint)
    monkeypatch.setenv("YUNSHU_JSON_SCHEMA_ENGINE", "inhouse")
    assert settings.get("YUNSHU_JSON_SCHEMA_ENGINE") == "inhouse"
    assert isinstance(build_json_constraint(SCHEMA, tok), JsonSchemaConstraint)


def test_no_tokenizer_keeps_inhouse():
    assert isinstance(build_json_constraint(SCHEMA), JsonSchemaConstraint)
    assert isinstance(build_json_constraint(None), JsonSchemaConstraint)


# ── semantics on a real tokenizer ──────────────────────────────────────────

PREFIX = [
    (
        {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
        [
            '{"a": "x"}',
            '{"a":"x"}',
            '{\n  "a": "x"\n}',
            '{"a": 1}',
            '{"a": "x", "z": 1}',
            "{}",
            '{"a": "x"} ',
        ],
    ),
    (
        {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
        [
            '{"n": 1}',
            '{"n": -0}',
            '{"n": 1.0}',
            '{"n": 01}',
            '{"n": 1e2}',
            '{"n": "1"}',
        ],
    ),
    (
        {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]},
        ['{"x": 1.5}', '{"x": 1e5}', '{"x": .5}', '{"x": 1.}', '{"x": -0.0}'],
    ),
    (
        {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a"],
        },
        ['{"a": "x"}', '{"a": "x", "b": 2}', '{"a": "x", "c": 2}'],
    ),
    (
        {"type": "array", "items": {"type": "boolean"}},
        ["[]", "[true, false]", "[true,]", "[1]"],
    ),
]


def _inhouse(schema):
    return JsonSchemaConstraint(schema)


@pytest.mark.parametrize("schema,texts", PREFIX)
def test_accept_sets_match_inhouse_on_documented_conventions(tok, schema, texts):
    for text in texts:
        a = _accepts(_inhouse(schema), tok, text)
        b = _accepts(build_json_constraint(schema, tok), tok, text)
        assert a == b, text


def test_whitespace_runs_match_inhouse(tok):
    for text in [
        '{"a":\t"x"}',
        '{"a":   "x"}',
        '{"a": \n "x"}',
        '{\r\n"a": "x"}',
        '{\n                 "a": "x"}',
    ]:
        assert _accepts(_inhouse(SCHEMA2), tok, text) == _accepts(
            build_json_constraint(SCHEMA2, tok), tok, text
        ), text


SCHEMA2 = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}


def test_discriminated_one_of_objects_are_enforced(tok):
    schema = {
        "oneOf": [
            {
                "type": "object",
                "properties": {"kind": {"const": "c"}, "r": {"type": "number"}},
                "required": ["kind", "r"],
            },
            {
                "type": "object",
                "properties": {"kind": {"const": "s"}, "w": {"type": "number"}},
                "required": ["kind", "w"],
            },
        ]
    }
    assert _accepts(build_json_constraint(schema, tok), tok, '{"kind": "s", "w": 2}')
    assert not _accepts(
        build_json_constraint(schema, tok), tok, '{"kind": "s", "r": 2}'
    )


def test_forced_continuation_is_the_schema_literal(tok):
    c = build_json_constraint(SCHEMA2, tok, compact=True)
    assert isinstance(c, LlgJsonSchemaConstraint)
    assert c.forced_continuation() == '{"a":"'
    c.advance(c.forced_continuation())
    assert c.forced_continuation() == ""  # the value is a real choice
    # with structural whitespace allowed only what follows the opening quote is forced
    c = build_json_constraint(SCHEMA2, tok)
    c.advance('{"')
    assert c.forced_continuation() == 'a"'
