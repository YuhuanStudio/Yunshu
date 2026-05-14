"""Token counting service — accurate token estimation for billing and rate limiting.

Uses the tokenizer from the active engine to count tokens precisely.
Falls back to heuristic estimation when no tokenizer is available.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Approximate chars-per-token ratio (conservative)
_CHARS_PER_TOKEN = 3.5


def count_tokens(text: str, tokenizer=None) -> int:
    """Count tokens in text using the best available method."""
    if not text:
        return 0

    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            logger.debug("tokenizer.encode() failed", exc_info=True)

    # Heuristic: whitespace-split estimation
    words = len(text.split())
    chars = len(text)
    # Use whichever estimate is higher (conservative)
    return max(words, int(chars / _CHARS_PER_TOKEN))


def count_message_tokens(messages: list[dict], tokenizer=None) -> int:
    """Count total tokens across a list of chat messages."""
    total = 0
    for msg in messages:
        # Role overhead (~4 tokens per message)
        total += 4
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content, tokenizer)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        total += count_tokens(part.get("text", ""), tokenizer)
                    elif part.get("type") == "image_url":
                        total += 85  # Fixed overhead for image tokens
                elif isinstance(part, str):
                    total += count_tokens(part, tokenizer)
    total += 2  # Priming tokens
    return total


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model_id: str = "",
) -> dict:
    """Estimate cost for a request (Phase 2: actual billing)."""
    # Placeholder pricing — will be configurable per-model in Phase 2
    pricing = {
        "default": {"prompt": 0.0, "completion": 0.0},
    }

    model_pricing = pricing.get(model_id, pricing["default"])
    prompt_cost = prompt_tokens * model_pricing["prompt"] / 1_000_000
    completion_cost = completion_tokens * model_pricing["completion"] / 1_000_000

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_cost_usd": prompt_cost,
        "completion_cost_usd": completion_cost,
        "total_cost_usd": prompt_cost + completion_cost,
    }
