"""the vLLM/SGLang-parity sampling controls (min_tokens / ignore_eos /
suppress_tokens) were honored on /v1/chat/completions + /v1/completions but were never
declared or plumbed on /v1/responses or /v1/messages (Anthropic) — so a request setting
them got them SILENTLY ignored. They must now be accepted and forwarded."""
from __future__ import annotations

import pytest


def test_responses_request_accepts_w735_fields():
    from yunshu_gateway.routers.responses import ResponsesRequest
    req = ResponsesRequest(
        model="m", input="hi",
        min_tokens=8, ignore_eos=True, suppress_tokens=[1, 2, 3],
    )
    assert req.min_tokens == 8
    assert req.ignore_eos is True
    assert req.suppress_tokens == [1, 2, 3]
    # defaults keep old behavior
    d = ResponsesRequest(model="m", input="hi")
    assert d.min_tokens == 0 and d.ignore_eos is False and d.suppress_tokens is None


def test_messages_request_accepts_w735_fields():
    from yunshu_gateway.routers.anthropic import AnthropicMessagesRequest
    req = AnthropicMessagesRequest(
        model="m", max_tokens=16, messages=[{"role": "user", "content": "hi"}],
        min_tokens=4, ignore_eos=True, suppress_tokens=[9],
    )
    assert req.min_tokens == 4 and req.ignore_eos is True and req.suppress_tokens == [9]


def test_min_tokens_negative_rejected():
    from pydantic import ValidationError

    from yunshu_gateway.routers.responses import ResponsesRequest
    with pytest.raises(ValidationError):
        ResponsesRequest(model="m", input="hi", min_tokens=-1)


def test_responses_forwards_w735_at_every_engine_site():
    """The 3 params must be passed at all 5 responses + 4 anthropic param sites."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    r = (root / "python/yunshu_gateway/routers/responses.py").read_text()
    a = (root / "python/yunshu_gateway/routers/anthropic.py").read_text()
    assert r.count("min_tokens=req.min_tokens") == 5
    assert r.count("suppress_tokens=req.suppress_tokens") == 5
    assert a.count("min_tokens=req.min_tokens") == 4
    assert a.count("suppress_tokens=req.suppress_tokens") == 4
