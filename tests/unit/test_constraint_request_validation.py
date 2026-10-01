"""Unsupported constraint constructs become a 400 before any generation."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from yunshu_engine.grammar_constraint import validate_constraint_spec
from yunshu_gateway.routers import chat, responses


@pytest.mark.parametrize(
    "parse", [chat._parse_response_format, responses._parse_response_format]
)
@pytest.mark.parametrize(
    "response_format,grammar,needle",
    [
        (None, {"type": "regex", "pattern": "(a)\\1"}, "back-reference"),
        (None, {"type": "regex", "pattern": "(?=a)b"}, "lookahead"),
        (
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "x",
                    "schema": {"type": "array", "uniqueItems": True},
                },
            },
            None,
            "uniqueItems",
        ),
        (
            None,
            {"type": "json", "schema": {"not": {"type": "integer"}}},
            "not",
        ),
    ],
)
def test_unsupported_constraint_is_400(parse, response_format, grammar, needle):
    with pytest.raises(HTTPException) as err:
        parse(response_format, grammar)
    assert err.value.status_code == 400
    assert needle in err.value.detail


def test_supported_constraints_pass_through():
    assert chat._parse_response_format({"type": "json_object"}) == "json_object"
    assert chat._parse_response_format(
        None, {"type": "regex", "pattern": r"\d{3}-\d{4}"}
    ) == {"type": "regex", "pattern": r"\d{3}-\d{4}"}
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"],
    }
    got = chat._parse_response_format(
        {"type": "json_schema", "json_schema": {"name": "n", "schema": schema}}
    )
    assert got["properties"] == schema["properties"]


def test_final_validation_violation_is_logged_and_reported(caplog):
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    with caplog.at_level("WARNING"):
        error = chat._vlm_json_output_error('{"a":"x"}', schema)
    assert error and "does not match" in error
    assert "final validation" in caplog.text
    assert chat._validation_extension(error)["validation"]["valid"] is False
    assert chat._vlm_json_output_error('{"a":1}', schema) is None


def test_validate_none_and_choice():
    validate_constraint_spec(None)
    validate_constraint_spec({"type": "choice", "choices": ["a", "b"]})
    with pytest.raises(ValueError):
        validate_constraint_spec({"type": "choice", "choices": [1]})
