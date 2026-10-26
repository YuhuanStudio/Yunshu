"""Token counting service — accurate token estimation for billing and rate limiting.

Uses the tokenizer from the active engine to count tokens precisely.
Falls back to heuristic estimation when no tokenizer is available.
"""

import logging
from collections import OrderedDict
from collections.abc import Callable
from threading import Lock
from typing import Any

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


class TokenCountCache:
    """Exact text counts for one loaded tokenizer, with bounded LRU text storage."""

    def __init__(self, max_entries: int = 128, max_text_bytes: int = 2 << 20) -> None:
        self._counts: OrderedDict[str, int] = OrderedDict()
        self._tokenizer: Any = None
        self._text_bytes = 0
        self._generation = 0
        self._max_entries = max_entries
        self._max_text_bytes = max_text_bytes
        self._lock = Lock()

    def clear(self) -> None:
        with self._lock:
            self._counts.clear()
            self._tokenizer = None
            self._text_bytes = 0
            self._generation += 1

    def count(self, text: str, tokenizer: Any = None) -> int:
        if not text or tokenizer is None:
            return count_tokens(text, tokenizer)
        with self._lock:
            if self._tokenizer is not tokenizer:
                self._counts.clear()
                self._text_bytes = 0
                self._tokenizer = tokenizer
                self._generation += 1
            cached = self._counts.get(text)
            if cached is not None:
                self._counts.move_to_end(text)
                return cached
            generation = self._generation
        try:
            result = len(tokenizer.encode(text))
        except Exception:
            logger.debug("tokenizer.encode() failed", exc_info=True)
            return count_tokens(text)
        size = len(text) * 4
        if size <= self._max_text_bytes:
            with self._lock:
                if self._tokenizer is tokenizer and self._generation == generation:
                    if text not in self._counts:
                        self._text_bytes += size
                    self._counts[text] = result
                    self._counts.move_to_end(text)
                    while (
                        len(self._counts) > self._max_entries
                        or self._text_bytes > self._max_text_bytes
                    ):
                        key, _ = self._counts.popitem(last=False)
                        self._text_bytes -= len(key) * 4
        return result


def count_message_tokens(
    messages: list[dict],
    tokenizer=None,
    media_counter=None,
    text_counter: Callable[[str, Any], int] | None = None,
) -> int:
    """Count total tokens across a list of chat messages.

    Handles tool call fields in assistant messages and tool_call_id/name
    in tool role messages for accurate multi-turn token estimation.
    """
    import json as _json

    count_text = text_counter or count_tokens
    total = 0
    for msg in messages:
        # Role overhead (~4 tokens per message)
        total += 4
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_text(content, tokenizer)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        total += count_text(part.get("text", ""), tokenizer)
                    elif part.get("type") in (
                        "image_url",
                        "image",
                        "image_data",
                        "video",
                        "video_url",
                    ):
                        total += (
                            media_counter(part)
                            if media_counter
                            else IMAGE_TOKEN_ESTIMATE
                        )
                    elif part.get("type") in ("input_audio", "audio", "audio_url"):
                        # Audio cost varies with duration; a fixed non-zero estimate
                        # stops an audio prompt silently counting 0 and bypassing
                        # context/prefill validation.
                        total += IMAGE_TOKEN_ESTIMATE
                elif isinstance(part, str):
                    total += count_text(part, tokenizer)
        # Account for tool_calls in assistant messages
        tool_calls = msg.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list):
            for tc in tool_calls:
                total += 4  # tool call overhead (id, type)
                func = tc.get("function", {}) if isinstance(tc, dict) else {}
                total += count_text(func.get("name", ""), tokenizer)
                args = func.get("arguments", "")
                if isinstance(args, dict):
                    args = _json.dumps(args)
                total += count_text(str(args), tokenizer)
        # Account for tool_call_id and name in tool role messages
        if msg.get("tool_call_id"):
            total += 4  # tool_call_id overhead
        if msg.get("name"):
            total += count_text(msg["name"], tokenizer)
    total += 2  # Priming tokens
    return total
