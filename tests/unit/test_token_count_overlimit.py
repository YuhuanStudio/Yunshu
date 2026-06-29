"""token_count over_context_limit for a list `prompt` must test whether
ANY single prompt overflows the context window, not whether their SUM does (each list
element is an independent prompt sent in its own request). The old `sum > limit`
falsely flagged a batch of individually-fitting prompts."""

from __future__ import annotations

import asyncio
import types

from yunshu_gateway.routers import (
    tokenize as T,  # noqa: N812  # intentional short module alias
)


class _FakeTok:
    """encode returns `len(text)` tokens — 1 token per character — for determinism."""

    def encode(self, text, add_special_tokens=True):
        return [0] * len(text)


def _call(prompt, ctx_limit, monkeypatch):
    monkeypatch.setattr(
        "yunshu_gateway.routers.models._check_permission", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "yunshu_gateway.routers.models._check_model_access", lambda *a, **k: None
    )
    monkeypatch.setattr(T, "_resolve_tokenizer", lambda m: _FakeTok())
    monkeypatch.setattr(T, "_resolve_context_limit", lambda m: ctx_limit)
    req = T.TokenCountRequest(model="m", prompt=prompt)
    fake_req = types.SimpleNamespace(state=types.SimpleNamespace())
    return asyncio.run(T.token_count(req, fake_req))


def test_list_no_single_overflow_is_under_limit(monkeypatch):
    # 3 prompts of 4 chars (=4 tokens) each; sum=12 > 10 but none individually > 10.
    out = _call(["aaaa", "bbbb", "cccc"], ctx_limit=10, monkeypatch=monkeypatch)
    assert out["token_count"] == [4, 4, 4]
    assert out["over_context_limit"] is False  # was True under the old sum logic


def test_list_with_single_overflow_flags(monkeypatch):
    out = _call(["aaaa", "x" * 20, "cccc"], ctx_limit=10, monkeypatch=monkeypatch)
    assert out["over_context_limit"] is True  # the 20-token item overflows


def test_string_prompt_unchanged(monkeypatch):
    out = _call("x" * 15, ctx_limit=10, monkeypatch=monkeypatch)
    assert out["token_count"] == 15
    assert out["over_context_limit"] is True


def test_no_limit_never_flags(monkeypatch):
    out = _call(["a" * 100], ctx_limit=0, monkeypatch=monkeypatch)
    assert out["over_context_limit"] is False


# ── A1: /v1/audio/transcriptions rejects an unknown response_format ──


def test_transcription_invalid_response_format_rejected(monkeypatch):
    import asyncio as _aio
    import types as _types

    from fastapi import HTTPException

    from yunshu_gateway.routers import (
        audio as A,  # noqa: N812  # intentional short module alias
    )

    monkeypatch.setattr(
        "yunshu_gateway.routers.models._check_permission", lambda *a, **k: None
    )
    fake_req = _types.SimpleNamespace(state=_types.SimpleNamespace())
    try:
        _aio.run(
            A.create_transcription(
                request=fake_req,
                file=None,
                model="m",
                response_format="jsonl",  # typo / unsupported → must 400
            )
        )
        raise AssertionError("expected HTTPException for invalid response_format")
    except HTTPException as e:
        assert e.status_code == 400
        assert "response_format" in str(e.detail)
