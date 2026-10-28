"""A prompt that renders to zero tokens is refused with a clean error on the text paths
(it used to be silently replaced with a lone BOS/EOS token and generate from that)."""

from __future__ import annotations

import pytest

from yunshu_engine.context_window import EmptyPromptError

from .tiny_text_engine import collect_stream, make_engine


class _NoTokens:
    """Encodes everything to zero tokens (a template that renders to nothing)."""

    def __init__(self, base):
        self._base = base

    def encode(self, text, **kw):
        return []

    def __getattr__(self, name):
        return getattr(self._base, name)


def _engine():
    eng = make_engine()
    eng._tokenizer = _NoTokens(eng._tokenizer)
    return eng


async def test_generate_fast_refuses_an_empty_prompt():
    with pytest.raises(EmptyPromptError, match="empty prompt"):
        await _engine()._generate_fast(prompt="", max_tokens=4, temperature=0.0)


async def test_stream_generate_fast_refuses_an_empty_prompt():
    with pytest.raises(EmptyPromptError, match="empty prompt"):
        await collect_stream(_engine(), prompt="", max_tokens=4, temperature=0.0)


async def test_ngram_and_speculative_refuse_an_empty_prompt():
    eng = _engine()
    eng._new_request_proposer = lambda: None
    for fn in (eng._generate_ngram_spec, eng._generate_speculative):
        with pytest.raises(EmptyPromptError):
            await fn(prompt="", max_tokens=4, temperature=0.0)
    with pytest.raises(EmptyPromptError):
        async for _ in eng._stream_generate_ngram_spec(
            prompt="", max_tokens=4, temperature=0.0
        ):
            pass


def test_gateway_maps_the_error_to_400():
    from fastapi.testclient import TestClient

    from yunshu_gateway.main import create_app

    app = create_app()

    @app.get("/_boom")
    async def boom():
        raise EmptyPromptError("empty prompt: the request renders to zero tokens")

    r = TestClient(app, raise_server_exceptions=False).get("/_boom")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "empty_prompt"
