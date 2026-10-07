"""Unpaired UTF-16 surrogates in request text answer 400 on every API, not 500."""

import pytest

from yunshu_gateway.middleware.surrogate_guard import (
    find_lone_surrogate,
    lone_surrogate_field,
)

LONE = '{"model": "m", "messages": [{"role": "user", "content": "ok"}, {"role": "user", "content": "cut \\ud83d"}]}'


def test_find_lone_surrogate_paths():
    assert find_lone_surrogate({"messages": [{"content": "hi 😀"}]}) is None
    assert (
        find_lone_surrogate({"messages": [{"content": "a"}, {"content": "b\udc00"}]})
        == "messages[1].content"
    )
    assert find_lone_surrogate({"k\ud800": 1}) == "k\ud800"


def test_raw_prefilter():
    # A paired escape (an emoji written as two \u escapes) is valid text.
    assert lone_surrogate_field(b'{"content": "\\ud83d\\ude00"}') is None
    assert lone_surrogate_field(b'{"content": "plain"}') is None
    assert lone_surrogate_field(b"not json \\ud800") is None
    assert lone_surrogate_field(LONE.encode()) == "messages[1].content"
    assert lone_surrogate_field(b'{"x": "\\uDE00 low first"}') == "x"


@pytest.mark.parametrize(
    "path,body,dialect",
    [
        ("/v1/chat/completions", LONE, "openai"),
        (
            "/v1/messages",
            '{"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": "x \\ud83d"}]}',
            "anthropic",
        ),
        ("/v1/responses", '{"model": "m", "input": "x \\udfff"}', "openai"),
        ("/v1/completions", '{"model": "m", "prompt": "x \\ud800"}', "openai"),
    ],
)
def test_endpoints_answer_400(client, path, body, dialect):
    r = client.post(path, content=body, headers={"content-type": "application/json"})
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert "unpaired UTF-16 surrogate" in err["message"]
    if dialect == "anthropic":
        assert r.json()["type"] == "error"
        assert err["type"] == "invalid_request_error"
