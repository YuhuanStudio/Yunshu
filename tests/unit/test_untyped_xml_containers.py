"""Untyped XML parameters may carry containers; scalar text stays exact."""

import json

import pytest

from yunshu_engine.tool_arguments import coerce_tool_arguments


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"kind":"new"}', {"kind": "new"}),
        ('["a","b"]', ["a", "b"]),
        ("123", "123"),
        ("false", "false"),
        ('"quoted"', '"quoted"'),
        ('["unfinished"', '["unfinished"'),
    ],
)
def test_untyped_xml_parameter(text, expected):
    schemas = {"f": {"properties": {"value": {}}}}
    arguments = json.dumps({"value": text})
    assert json.loads(
        coerce_tool_arguments("f", arguments, schemas, raw_text_values=True)
    ) == {"value": expected}
    assert coerce_tool_arguments("f", arguments, schemas) == arguments


def test_explicit_string_and_string_union_keep_container_text():
    for prop in (
        {"type": "string"},
        {"anyOf": [{"type": "object"}, {"type": "string"}]},
    ):
        arguments = json.dumps({"value": '{"id":1}'})
        assert (
            coerce_tool_arguments(
                "f",
                arguments,
                {"f": {"properties": {"value": prop}}},
                raw_text_values=True,
            )
            == arguments
        )
