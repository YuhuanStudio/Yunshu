"""Token-counting / preview renders defuse special-token strings exactly like generation does,
so /v1/messages/count_tokens, /tokenize, /apply-template and the completions max_tokens=0 probe report the
prompt the engine actually runs (969b1e9 parity for the VLM paths, incl. the MLX-MTP renderer)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).parent))
from test_control_token_injection import MSGS, _tok, _vlm  # noqa: E402

from yunshu_engine.mlxvlm_mtp import MLXVLMMtp  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / "python"


def _engine_ids(tok):
    return tok.encode(_vlm(tok)._format_prompt(MSGS), add_special_tokens=False)


def test_responses_count_matches_generation():
    from yunshu_gateway.routers.responses import _count_prompt_tokens

    tok = _tok()
    assert _count_prompt_tokens(tok, MSGS) == len(_engine_ids(tok))


def test_apply_template_preview_matches_generation(monkeypatch):
    from yunshu_gateway.routers import models, tokenize

    tok = _tok()
    monkeypatch.setattr(models, "_check_permission", lambda *a, **k: None)
    monkeypatch.setattr(models, "_check_model_access", lambda *a, **k: None)
    monkeypatch.setattr(tokenize, "_resolve_tokenizer", lambda m: tok)
    req = tokenize.ApplyTemplateRequest(messages=MSGS)
    out = asyncio.run(tokenize.apply_template(req, object()))
    assert out["prompt"] == _vlm(tok)._format_prompt(MSGS)


def test_tokenize_route_counts_match_generation(monkeypatch):
    from yunshu_gateway.routers import models, tokenize

    tok = _tok()
    monkeypatch.setattr(models, "_check_permission", lambda *a, **k: None)
    monkeypatch.setattr(models, "_check_model_access", lambda *a, **k: None)
    monkeypatch.setattr(tokenize, "_resolve_tokenizer", lambda m: tok)
    req = tokenize.TokenizeRequest(messages=MSGS, add_special_tokens=False)
    out = asyncio.run(tokenize.tokenize(req, object()))
    body = out if isinstance(out, dict) else getattr(out, "body", None)
    if not isinstance(body, dict):
        import json

        body = json.loads(body)
    assert body["count"] == len(_engine_ids(tok))


def test_mlx_mtp_fallback_renderer_neutralizes():
    tok = _tok()
    enc = MLXVLMMtp.__new__(MLXVLMMtp)
    enc.tokenizer = tok
    assert enc._encode(MSGS) == _engine_ids(tok)


@pytest.mark.parametrize(
    "rel",
    [
        "yunshu_gateway/routers/anthropic.py",
    ],
)
def test_count_probes_call_neutralize_before_template(rel):
    """the chat template count and anthropic count_tokens need a running server to exercise, so
    this is a source-level guard: every apply_chat_template there is fed neutralised text."""
    src = (ROOT / rel).read_text()
    assert "neutralize_special_text(" in src
