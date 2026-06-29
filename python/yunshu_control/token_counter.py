"""Token counting service — accurate token estimation for billing and rate limiting.

Uses the tokenizer from the active engine to count tokens precisely.
Falls back to heuristic estimation when no tokenizer is available.
"""

import logging

logger = logging.getLogger(__name__)

# Approximate chars-per-token ratio (conservative)
_CHARS_PER_TOKEN = 3.5
IMAGE_TOKEN_ESTIMATE = 576


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
    """Count total tokens across a list of chat messages.

    Handles tool call fields in assistant messages and tool_call_id/name
    in tool role messages for accurate multi-turn token estimation.
    """
    import json as _json

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
                    elif part.get("type") in (
                        "image_url",
                        "image",
                        "image_data",
                        "video",
                        "video_url",
                    ):
                        total += IMAGE_TOKEN_ESTIMATE
                    elif part.get("type") in ("input_audio", "audio", "audio_url"):
                        # Audio cost varies with duration; a fixed non-zero estimate
                        # stops an audio prompt silently counting 0 and bypassing
                        # context/prefill validation.
                        total += IMAGE_TOKEN_ESTIMATE
                elif isinstance(part, str):
                    total += count_tokens(part, tokenizer)
        # Account for tool_calls in assistant messages
        tool_calls = msg.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list):
            for tc in tool_calls:
                total += 4  # tool call overhead (id, type)
                func = tc.get("function", {}) if isinstance(tc, dict) else {}
                total += count_tokens(func.get("name", ""), tokenizer)
                args = func.get("arguments", "")
                if isinstance(args, dict):
                    args = _json.dumps(args)
                total += count_tokens(str(args), tokenizer)
        # Account for tool_call_id and name in tool role messages
        if msg.get("tool_call_id"):
            total += 4  # tool_call_id overhead
        if msg.get("name"):
            total += count_tokens(msg["name"], tokenizer)
    total += 2  # Priming tokens
    return total


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model_id: str = "",
) -> dict:
    """Estimate cost for a request based on common model pricing.

    Pricing is per-million tokens (USD). Models matched by substring.
    For local/self-hosted models, cost represents compute-equivalent value.
    """
    pricing = {
        # OpenAI models
        "gpt-4o": {"prompt": 2.50, "completion": 10.00},
        "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
        "gpt-4-turbo": {"prompt": 10.00, "completion": 30.00},
        "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
        # Anthropic models
        "claude-sonnet": {"prompt": 3.00, "completion": 15.00},
        "claude-haiku": {"prompt": 0.25, "completion": 1.25},
        "claude-opus": {"prompt": 15.00, "completion": 75.00},
        # Open source (self-hosted equivalent)
        "llama": {"prompt": 0.05, "completion": 0.10},
        "qwen": {"prompt": 0.05, "completion": 0.10},
        "mistral": {"prompt": 0.10, "completion": 0.20},
        "deepseek": {"prompt": 0.14, "completion": 0.28},
        "gemma": {"prompt": 0.05, "completion": 0.10},
        "phi": {"prompt": 0.03, "completion": 0.06},
        # Default for local/self-hosted
        "default": {"prompt": 0.05, "completion": 0.10},
    }

    model_lower = model_id.lower()
    model_pricing = pricing.get(model_lower)
    if model_pricing is None:
        # Try substring match
        for key, val in pricing.items():
            if key in model_lower:
                model_pricing = val
                break
        if model_pricing is None:
            model_pricing = pricing["default"]

    prompt_cost = prompt_tokens * model_pricing["prompt"] / 1_000_000
    completion_cost = completion_tokens * model_pricing["completion"] / 1_000_000

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_cost_usd": round(prompt_cost, 6),
        "completion_cost_usd": round(completion_cost, 6),
        "total_cost_usd": round(prompt_cost + completion_cost, 6),
    }
