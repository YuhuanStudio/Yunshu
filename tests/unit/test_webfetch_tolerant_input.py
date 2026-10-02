"""C02: web_fetch tool input parsing is tolerant where unambiguous and actionable otherwise."""

from __future__ import annotations

import pytest

from yunshu_gateway.server_tools.runtime import (
    ServerToolRuntime,
    extract_fetch_url,
    parse_tool_args,
)


@pytest.mark.parametrize(
    "args",
    [
        {"url": "https://example.com/a"},
        {"uri": "https://example.com/a"},
        {"link": "https://example.com/a"},
        {"href": "https://example.com/a"},
        {"URL_": 1, "url": " https://example.com/a "},
        {"input": {"url": "https://example.com/a"}},
        {"arguments": '{"uri": "https://example.com/a"}'},
        {"url": ["https://example.com/a"]},
        {"url": "[page](https://example.com/a)"},
        {"url": "<https://example.com/a>"},
        {"url": '"https://example.com/a"'},
        {"url": "example.com/a"},
        {"__raw__": "https://example.com/a"},
        "https://example.com/a",
    ],
)
def test_url_recovered(args):
    url, problem = extract_fetch_url(args)
    assert url == "https://example.com/a", problem


@pytest.mark.parametrize(
    "raw",
    [
        '{"url": "https://example.com/a"}',
        '"https://example.com/a"',
        "https://example.com/a",
        '{"url": "https://example.com/a',  # truncated JSON
        {"url": "https://example.com/a"},
    ],
)
def test_parse_tool_args_then_extract(raw):
    assert extract_fetch_url(parse_tool_args(raw))[0] == "https://example.com/a"


@pytest.mark.parametrize(
    "args,needle",
    [
        ({}, '{"url"'),
        ({"query": "x"}, "got keys: query"),
        ({"url": "/relative/path"}, "absolute http(s) URL"),
        ({"url": "ftp://x/y"}, "absolute http(s) URL"),
        ({"url": "not a url"}, "absolute http(s) URL"),
    ],
)
def test_unrecoverable_input_error_tells_the_model_what_to_send(args, needle):
    url, problem = extract_fetch_url(args)
    assert url is None
    assert needle in problem and "web_fetch takes one JSON object" in problem


async def test_execute_reports_actionable_error_not_none_url():
    rt = ServerToolRuntime.__new__(ServerToolRuntime)
    rt.defs = {}
    rt.uses = {}
    rt.counts = {"web_search": 0, "web_fetch": 0, "mcp": 0}
    rt._http = None
    rt._mcp = {}
    out = await rt._fetch(_Def(), {"query": "x"})
    assert out.is_error and out.error_code == "invalid_tool_input"
    assert "None" not in out.text
    assert '{"url": "https://example.com/page"}' in out.text


class _Def:
    spec: dict = {}
