"""One definition of the usage objects every route reports.

The engine's ``completion_tokens`` already counts every generated token (reasoning included);
``reasoning_tokens`` is a detail inside it and ``cached_tokens`` a detail inside the prompt, so
neither is ever an addend and neither can exceed its total. Chat, completions, Responses and
Anthropic (stream and not) all build their usage here so the same generation reports the same
numbers on every dialect.
"""

from __future__ import annotations

import contextvars
from typing import Any

# Tokens of a server-added assistant prefill (forced tool_choice on an engine that cannot take
# native tools) in this request's prompt. Usage reports what the client sent, so every builder
# below subtracts it: the one place the prompt count is decided.
_PREFILL_TOKENS: contextvars.ContextVar[int] = contextvars.ContextVar(
    "yunshu_prefill_tokens", default=0
)


def note_prefill(tokenizer: Any, text: str) -> None:
    """Record the token count of the server's own prefill for this request's usage."""
    n = 0
    try:
        n = len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        try:
            n = len(tokenizer.encode(text))
        except Exception:
            n = 0
    _PREFILL_TOKENS.set(n)


def client_prompt_tokens(prompt_tokens: int) -> int:
    """``prompt_tokens`` without the server's prefill."""
    return max(0, int(prompt_tokens or 0) - _PREFILL_TOKENS.get())


def clamp_detail(value: int | None, total: int | None) -> int:
    """``value`` as a detail of ``total``: never negative, never larger than the total."""
    return max(0, min(int(value or 0), int(total or 0)))


def openai_usage(
    prompt_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
) -> dict[str, Any]:
    """Chat Completions / Completions ``usage``."""
    prompt_tokens = client_prompt_tokens(prompt_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "completion_tokens_details": {
            "reasoning_tokens": clamp_detail(reasoning_tokens, completion_tokens)
        },
        "prompt_tokens_details": {
            "cached_tokens": clamp_detail(cached_tokens, prompt_tokens)
        },
    }


def responses_usage(
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
) -> dict[str, Any]:
    """Responses API ``usage``."""
    input_tokens = client_prompt_tokens(input_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "output_tokens_details": {
            "reasoning_tokens": clamp_detail(reasoning_tokens, output_tokens)
        },
        "input_tokens_details": {
            "cached_tokens": clamp_detail(cached_tokens, input_tokens),
            "cache_write_tokens": 0,  # required by the current openai Response type; writes are in x_yunshu
        },
    }
