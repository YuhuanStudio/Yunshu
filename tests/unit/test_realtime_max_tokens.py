"""the Realtime response.create per-response override path read the WRONG
key (`max_output_tokens` instead of `max_response_output_tokens`) so every client cap
was silently ignored, AND it skipped the str→int coercion SessionConfig.update applies —
so a documented value like "inf" reached stream_chat(max_tokens=) and crashed the turn
(mlx-lm compares the token counter against a str). The coercion is now a shared helper
used by both paths; the key is fixed; per-response temperature is validated too."""
from __future__ import annotations

import inspect

from yunshu_gateway.routers import (
    realtime as RT,  # noqa: N812  # intentional short module alias
)
from yunshu_gateway.routers.realtime import SessionConfig, _coerce_max_response_tokens


def test_coerce_inf_and_none_to_large_cap():
    assert _coerce_max_response_tokens("inf", 4096) == (1 << 20)
    assert _coerce_max_response_tokens(None, 4096) == (1 << 20)


def test_coerce_numeric_string():
    assert _coerce_max_response_tokens("500", 4096) == 500


def test_coerce_garbage_falls_back():
    assert _coerce_max_response_tokens("high", 4096) == 4096
    assert _coerce_max_response_tokens([], 4096) == 4096


def test_coerce_passthrough_int():
    assert _coerce_max_response_tokens(300, 4096) == 300


def test_session_update_coerces_and_rejects():
    cfg = SessionConfig()
    base = cfg.max_response_output_tokens
    # "inf" coerces (does not crash, does not skip)
    cfg.update({"max_response_output_tokens": "inf"})
    assert cfg.max_response_output_tokens == (1 << 20)
    # garbage is skipped → value unchanged from the prior coerced state
    cfg.update({"max_response_output_tokens": "garbage"})
    assert cfg.max_response_output_tokens == (1 << 20)
    # a real int sticks
    cfg.update({"max_response_output_tokens": 256})
    assert cfg.max_response_output_tokens == 256
    assert base == 4096  # default sanity


def test_response_create_reads_correct_key_and_validates_temp():
    src = inspect.getsource(RT.RealtimeSession._generate_response)
    # correct key (not the old max_output_tokens)
    assert 'config.get("max_response_output_tokens"' in src
    assert 'config.get("max_output_tokens"' not in src
    # routed through the shared coercion helper
    assert "_coerce_max_response_tokens(" in src
    # temperature validated with a float() guard + session fallback
    assert "float(temperature)" in src
