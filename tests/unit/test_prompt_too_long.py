"""A prompt that cannot fit in memory answers the spec's context overflow (API row 258).

The gate finds the longest prompt that fits (``fits_tokens``), the refusal carries it through the
engine's text channel, and the OpenAI / Anthropic bodies expose it under ``yunshu``.
"""

from yunshu_engine.memory_admission import (
    MemoryAdmissionError,
    MemoryGate,
    PromptTooLongError,
    memory_limit_error,
    refusal_text,
)
from yunshu_gateway.memory_refusal import (
    anthropic_prompt_too_long,
    openai_prompt_too_long,
)


def _gate() -> MemoryGate:
    # estimate(p, m) = (p + m) * 1000 bytes: no activation or attention term
    return MemoryGate(kv_per_token=1000, act_per_token=0, attn_heads=0)


def test_longest_fitting_prompt_is_found():
    # limit 500 000 bytes, max_tokens 100: (prompt + 100) * 1000 fits up to 400 prompt tokens
    assert _gate().max_prompt_tokens(1000, 100, 500_000) == 400


def test_nothing_fits_when_max_tokens_alone_exceeds_the_limit():
    assert _gate().max_prompt_tokens(1000, 100, 500_000 - 1) == 399
    assert _gate().max_prompt_tokens(1000, 600, 500_000) == 0


def test_refusal_records_the_limit_it_was_measured_against():
    gate = MemoryGate(
        kv_per_token=1000,
        act_per_token=0,
        attn_heads=0,
        ceiling_free=lambda: 500_000,
        read=lambda: None,
    )
    d = gate.decide(2_000_000, 0, any_active=False)
    assert d.refuse and d.reason == "memory_limit" and d.limit_bytes == 500_000


def test_text_channel_round_trips_both_refusals():
    too_long = PromptTooLongError(1000, 400, "needs 1.0 GiB")
    back = memory_limit_error(refusal_text(too_long))
    assert isinstance(back, PromptTooLongError)
    assert (back.prompt_tokens, back.fits_tokens) == (1000, 400)
    plain = memory_limit_error(refusal_text(MemoryAdmissionError("too big")))
    assert type(plain) is MemoryAdmissionError and str(plain) == "too big"


def test_openai_body_is_context_length_exceeded_with_extension():
    body = openai_prompt_too_long(PromptTooLongError(1000, 400, "x"))
    err = body["error"]
    assert err["code"] == "context_length_exceeded"
    assert err["type"] == "invalid_request_error"
    assert err["yunshu"] == {"reason": "memory_limit", "fits_tokens": 400}
    assert "400 tokens" in err["message"] and "1000 tokens" in err["message"]


def test_anthropic_body_is_prompt_too_long_with_extension():
    body = anthropic_prompt_too_long(PromptTooLongError(1000, 400, "x"))
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["message"] == "prompt is too long: 1000 tokens > 400 maximum"
    assert body["error"]["yunshu"]["fits_tokens"] == 400


def test_other_memory_refusals_keep_their_shape():
    assert openai_prompt_too_long(MemoryAdmissionError("x")) is None
    assert anthropic_prompt_too_long(MemoryError("x")) is None
