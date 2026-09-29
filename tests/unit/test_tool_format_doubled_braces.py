"""Small models sometimes emit ``{{"name": ...}}`` (an escaped-brace template echo)."""

import pytest

from yunshu_engine import tool_format as tf


def test_doubled_braces_parse():
    body = '{{"name": "get_weather", "arguments": {"city": "Paris"}}}\n'
    calls = tf.INJECTED_JSON.parse(body, None)
    assert calls == [{"name": "get_weather", "arguments": '{"city": "Paris"}'}]


def test_valid_nested_json_is_untouched():
    body = '{"name": "f", "arguments": {"a": {"b": 1}}}'
    assert tf.INJECTED_JSON.parse(body, None)[0]["arguments"] == '{"a": {"b": 1}}'


def test_garbage_still_raises():
    with pytest.raises(ValueError):
        tf.INJECTED_JSON.parse("{{not json}}", None)
