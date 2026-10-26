"""response_format.strict: Chat Completions and Responses follow one OpenAI semantics --
strict true closes every object (at any depth); false / omitted leaves the schema alone."""

import copy

import pytest

from yunshu_gateway.routers.chat import _parse_response_format_unchecked as chat_parse
from yunshu_gateway.routers.responses import (
    ResponsesRequest,
)
from yunshu_gateway.routers.responses import (
    _parse_response_format_unchecked as responses_parse,
)
from yunshu_gateway.schemas.strict import strict_schema

SCHEMA = {
    "type": "object",
    "properties": {
        "a": {"type": "string"},
        "inner": {"type": "object", "properties": {"b": {"type": "integer"}}},
        "list": {
            "type": "array",
            "items": {"type": "object", "properties": {"c": {"type": "string"}}},
        },
        "either": {
            "anyOf": [{"type": "object", "properties": {"d": {}}}, {"type": "null"}]
        },
    },
    "$defs": {"d": {"type": "object", "properties": {"e": {"type": "string"}}}},
}


def _closed(schema) -> bool:
    nodes = [
        schema,
        schema["properties"]["inner"],
        schema["properties"]["list"]["items"],
        schema["properties"]["either"]["anyOf"][0],
        schema["$defs"]["d"],
    ]
    return all(n.get("additionalProperties") is False for n in nodes)


def _rf(strict):
    js = {"name": "s", "schema": copy.deepcopy(SCHEMA)}
    if strict is not None:
        js["strict"] = strict
    return {"type": "json_schema", "json_schema": js}


@pytest.mark.parametrize("parse", [chat_parse, responses_parse])
def test_strict_true_closes_every_object_in_both_dialects(parse):
    assert _closed(parse(_rf(True)))


@pytest.mark.parametrize("parse", [chat_parse, responses_parse])
@pytest.mark.parametrize("strict", [False, None])
def test_strict_false_or_omitted_leaves_the_schema_alone(parse, strict):
    assert parse(_rf(strict)) == SCHEMA


def test_responses_text_format_strict_reaches_the_decoder():
    def build(fmt):
        req = ResponsesRequest(
            model="m",
            input="hi",
            text={"format": {"type": "json_schema", "name": "s", **fmt}},
        )
        return responses_parse(req.response_format)

    assert _closed(build({"schema": copy.deepcopy(SCHEMA), "strict": True}))
    # OpenAI's default is strict false: an omitted flag no longer counts as true
    assert build({"schema": copy.deepcopy(SCHEMA)}) == SCHEMA


def test_helper_does_not_mutate_input_and_keeps_map_types():
    src = {"type": "object", "properties": {"properties": {"type": "string"}}}
    out = strict_schema(src, True)
    assert "additionalProperties" not in src
    assert out["additionalProperties"] is False
    assert out["properties"]["properties"] == {
        "type": "string"
    }  # a property named "properties"
    mapped = {"type": "object", "additionalProperties": {"type": "integer"}}
    assert strict_schema(mapped, True)["additionalProperties"] == {"type": "integer"}
