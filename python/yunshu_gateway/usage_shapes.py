"""One definition of the usage objects every route reports.

The engine's ``completion_tokens`` already counts every generated token (reasoning included);
``reasoning_tokens`` is a detail inside it and ``cached_tokens`` a detail inside the prompt, so
neither is ever an addend and neither can exceed its total. Chat, completions, Responses and
Anthropic (stream and not) all build their usage here so the same generation reports the same
numbers on every dialect.
"""

from __future__ import annotations

from typing import Any


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
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "output_tokens_details": {
            "reasoning_tokens": clamp_detail(reasoning_tokens, output_tokens)
        },
        "input_tokens_details": {
            "cached_tokens": clamp_detail(cached_tokens, input_tokens)
        },
    }
