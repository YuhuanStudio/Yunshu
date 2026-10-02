"""C05: a bad Lark grammar is a 400 at request validation, in each dialect's shape."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from yunshu_gateway.routers import chat, responses

_BAD_LARK = 'start: ("a"'


@pytest.mark.parametrize(
    "parse", [chat._parse_response_format, responses._parse_response_format]
)
@pytest.mark.parametrize("bad", [_BAD_LARK, "start: undefined_rule\n", "x ::= y"])
def test_bad_lark_grammar_is_400_at_validation(parse, bad):
    with pytest.raises(HTTPException) as err:
        parse(None, {"type": "cfg", "grammar": bad})
    assert err.value.status_code == 400
    assert "invalid CFG grammar" in err.value.detail


def test_good_lark_grammar_passes():
    spec = {"type": "cfg", "grammar": 'start: "a" | "b"\n'}
    assert chat._parse_response_format(None, spec) == spec


def test_guided_grammar_alias_rejected_when_request_is_built():
    with pytest.raises(ValueError, match="invalid CFG grammar"):
        chat.ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "x"}],
            guided_grammar=_BAD_LARK,
        )


def test_anthropic_bad_grammar_is_400_in_anthropic_shape():
    from fastapi.testclient import TestClient

    from yunshu_gateway.main import create_app

    client = TestClient(create_app(), raise_server_exceptions=False)
    r = client.post(
        "/v1/messages",
        json={
            "model": "m",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
            "grammar": {"type": "cfg", "grammar": _BAD_LARK},
        },
    )
    assert r.status_code == 400
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert "invalid CFG grammar" in body["error"]["message"]


def test_openai_chat_http_400_shape():
    from fastapi.testclient import TestClient

    from yunshu_gateway.main import create_app

    client = TestClient(create_app(), raise_server_exceptions=False)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "guided_grammar": _BAD_LARK,
        },
    )
    assert r.status_code == 400
    assert "invalid CFG grammar" in r.text
