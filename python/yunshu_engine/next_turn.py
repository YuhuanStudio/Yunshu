"""Next-turn rehearsal: prefill the assistant reply while the user is reading it.

A chat follow-up re-sends the conversation including the assistant reply the engine just
decoded. The cache holds the *prompt* of the finished turn, so the follow-up must prefill the
reply (about 2 K tokens, ~3 s on a 32 K context for Qwen3.8-27B) plus the new user message.
Decode-time KV cannot be reused (decode and prefill arithmetic differ, so the cache would not
be bit-equal to a miss). Instead, once the response is delivered and the engine is idle, the
engine renders the conversation with the reply in the position it will have next turn and
prefills it with the normal prefill path (``max_tokens=1``), which stores exact checkpoints.
Lossless by construction: it is an ordinary request whose result is discarded. A real request
that arrives cancels the rehearsal at the next prefill atom boundary.

Pure helpers live here so they are unit-testable without a model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

SENTINEL = "⁣next-turn-rehearsal⁣"
MIN_REPLY_TOKENS = 16
MAX_REPLY_TOKENS = 4096


def reply_text(parts: Sequence[tuple[str, str]]) -> str:
    """Content the client will send back: non-reasoning text, stripped like the gateway does."""
    return "".join(t for t, state in parts if state != "reasoning").strip()


def eligible(
    *,
    finish_reason: str | None,
    completion_tokens: int,
    reply: str,
    has_media: bool,
    has_tools: bool,
    constrained: bool,
) -> bool:
    """Only a finished plain-text reply of useful size is worth prefilling."""
    if finish_reason != "stop" or not reply:
        return False
    if has_media or has_tools or constrained:
        return False
    return MIN_REPLY_TOKENS <= completion_tokens <= MAX_REPLY_TOKENS


def with_reply(messages: list[dict], reply: str) -> list[dict]:
    """The conversation the next turn will send, plus a sentinel user message."""
    return [
        *messages,
        {"role": "assistant", "content": reply},
        {"role": "user", "content": SENTINEL},
    ]


def cut_before_user_turn(
    ids: Sequence[int],
    turn_start_id: int | None,
    decode: Callable[[Sequence[int]], str],
) -> list[int] | None:
    """Tokens up to (not including) the sentinel user header; None when not found.

    The rendered next turn ends ``<|im_start|>user\\nSENTINEL<|im_end|>\\n<|im_start|>assistant\\n``,
    so the header to cut at is the second-to-last turn-start token. Fail closed: the text after
    the cut must start with the user role and contain the sentinel.
    """
    if turn_start_id is None:
        return None
    starts = [i for i, t in enumerate(ids) if t == turn_start_id]
    if len(starts) < 2:
        return None
    cut = starts[-2]
    tail = decode(list(ids[cut:]))
    if SENTINEL not in tail:
        return None
    if not tail.lstrip().split("\n", 1)[0].endswith("user"):
        return None
    return list(ids[:cut])
