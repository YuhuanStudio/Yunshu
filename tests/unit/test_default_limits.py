"""Defaults for output length, thinking and request size, each tied to its source
(yunshu_engine.output_limits): omitted caps are the model limit, not a magic 32768."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from yunshu_engine import output_limits as ol
from yunshu_engine import settings
from yunshu_engine.memory_guard import MemoryGuard
from yunshu_engine.memory_monitor import MemoryInfo, MemoryMonitor
from yunshu_gateway import token_budget
from yunshu_gateway.routers.anthropic import (
    AnthropicMessagesRequest,
    _thinking_switches,
)
from yunshu_gateway.routers.chat import ChatCompletionRequest
from yunshu_gateway.routers.completions import CompletionRequest
from yunshu_gateway.routers.responses import ResponsesRequest

U = [{"role": "user", "content": "hi"}]


def test_omitted_output_cap_is_the_model_limit():
    assert ol.MODEL_MAX_OUTPUT_TOKENS == 262144  # Qwen3.8 card, thinking
    assert settings.get("YUNSHU_DEFAULT_MAX_TOKENS") == 0
    c = ChatCompletionRequest(model="m", messages=U)
    assert c.effective_max_tokens() == 262144 and c.max_tokens_source() == "model_limit"
    r = ResponsesRequest(model="m", input="x")
    assert r.max_output_tokens == 262144


def test_operator_cap_lowers_only_the_default(monkeypatch):
    monkeypatch.setenv("YUNSHU_DEFAULT_MAX_TOKENS", "4096")
    c = ChatCompletionRequest(model="m", messages=U)
    assert c.effective_max_tokens() == 4096 and c.max_tokens_source() == "operator_cap"
    explicit = ChatCompletionRequest(model="m", messages=U, max_tokens=100000)
    assert explicit.effective_max_tokens() == 100000
    assert explicit.max_tokens_source() == "request"
    assert (
        ChatCompletionRequest(
            model="m", messages=U, max_completion_tokens=5
        ).max_tokens_source()
        == "request"
    )


def test_completions_legacy_default_is_16():
    assert CompletionRequest(model="m", prompt="x").max_tokens == 16


def test_budget_reports_the_default_and_its_source():
    b = token_budget.plan(1048576, 1000, 262144, source="model_limit")
    assert b is not None and not b.clamped and b.max_tokens_granted == 262144
    assert b.report()["max_tokens_source"] == "model_limit"
    assert "clamped_by" not in b.report()
    small = token_budget.plan(32768, 30000, 262144, source="model_limit")
    assert small.max_tokens_granted == 2768 and small.report()["clamped_by"] == (
        "context_window"
    )


def test_thinking_budget_accepts_the_model_thinking_limit():
    for cls, kw in [
        (ChatCompletionRequest, dict(messages=U)),
        (ResponsesRequest, dict(input="x")),
        (CompletionRequest, dict(prompt="x")),
    ]:
        assert cls(model="m", thinking_budget=262144, **kw).thinking_budget == 262144
        with pytest.raises(ValidationError):
            cls(model="m", thinking_budget=262145, **kw)


def test_effort_budget_is_monotone_and_xhigh_max_are_largest():
    ladder = ["low", "medium", "high", "xhigh", "max"]
    vals = [ol.effort_budget(e) for e in ladder]
    assert vals == sorted(vals) and len(set(vals)) == 5
    assert vals[-1] == ol.MAX_THINKING_BUDGET


def test_effort_none_and_minimal_turn_thinking_off():
    for eff in ("none", "minimal", "NONE"):
        assert ol.effort_budget(eff) is None
        c = ChatCompletionRequest(model="m", messages=U, reasoning_effort=eff)
        assert c.enable_thinking is False and c.reasoning_effort is None
        r = ResponsesRequest(model="m", input="x", reasoning={"effort": eff})
        assert r.enable_thinking is False and r.reasoning_effort is None
    c = ChatCompletionRequest(model="m", messages=U, reasoning_effort="high")
    assert c.enable_thinking is None and c.reasoning_effort == "high"
    # an explicit enable_thinking wins
    assert ol.fold_effort(True, "none") == (True, None)


def test_normalize_effort_has_no_none_rung():
    from yunshu_engine.model_card import normalize_effort

    assert normalize_effort("none", ["low", "medium", "xhigh"]) != "low"


def test_anthropic_absent_thinking_is_disabled():
    def req(**kw):
        return AnthropicMessagesRequest(model="m", messages=U, max_tokens=8192, **kw)

    assert _thinking_switches(req()) == (False, None)
    assert _thinking_switches(req(thinking={"type": "adaptive"})) == (None, None)
    assert _thinking_switches(
        req(thinking={"type": "enabled", "budget_tokens": 5000})
    ) == (True, 5000)
    assert _thinking_switches(req(thinking={"type": "disabled"})) == (False, None)


def test_anthropic_max_tokens_is_required():
    with pytest.raises(ValidationError):
        AnthropicMessagesRequest(model="m", messages=U)


def test_request_size_default_is_32_mib():
    assert settings.get("YUNSHU_MAX_REQUEST_SIZE") == 32 * 1024 * 1024


# ── memory admission: long outputs are granted what fits, not a fixed 32K ──


def _guard(available: int, **arch) -> MemoryGuard:
    with patch("python.yunshu_engine.memory_monitor.HAS_MLX_METAL", False):
        mon = MemoryMonitor()
    mon.set_model_info(
        num_layers=arch.get("layers", 16),
        num_kv_heads=4,
        head_dim=256,
        num_attention_heads=24,
    )
    info = MemoryInfo(
        total_bytes=available * 2,
        active_bytes=available,
        peak_bytes=available,
        cache_bytes=0,
        available_bytes=available,
        utilization_pct=50.0,
    )
    mon.get_memory_info = lambda: info  # type: ignore[method-assign]
    return MemoryGuard(memory_monitor=mon)


def test_memory_model_has_no_fixed_32k_cap():
    # 16 full-attention layers x 4 kv heads x 256 x 2 (k,v) x 2 bytes = 64 KiB / token
    g = _guard(available=64 * 1024**3)
    fit = g.get_recommended_max_tokens(1000)
    assert fit > 32768
    granted, _ = g.admit(1000, 262144)
    assert granted == min(fit, 262144)


def test_admit_grants_what_fits_and_refuses_an_unfit_prompt():
    g = _guard(available=2 * 1024**3)
    fit = g.get_recommended_max_tokens(1000)
    assert 0 < fit < 262144
    granted, reason = g.admit(1000, 262144)
    assert granted == fit and "granted" in reason
    ok, _ = g.admit(1000, 10)
    assert ok == 10  # fits as asked
    refused, why = _guard(available=4096).admit(100_000, 10)
    assert refused == 0 and "does not fit" in why


def test_longer_prompt_leaves_fewer_output_tokens():
    g = _guard(available=16 * 1024**3)
    assert g.admit(1000, 262144)[0] > g.admit(20_000, 262144)[0] > 0
