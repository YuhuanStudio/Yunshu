"""Responses usage carries every field of the current openai `ResponseUsage` type."""

from openai.types.responses import ResponseUsage

from yunshu_gateway.usage_shapes import responses_usage


def test_usage_validates_against_sdk_type():
    ResponseUsage.model_validate(responses_usage(10, 5, 2, 4))
