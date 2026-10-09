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

import logging
import threading
from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

SENTINEL = "⁣next-turn-rehearsal⁣"
MIN_REPLY_TOKENS = 16
MAX_REPLY_TOKENS = 4096


def reply_text(parts: Sequence[tuple[str, str]]) -> str:
    """Content the client will send back: non-reasoning text, stripped like the gateway does."""
    return "".join(t for t, state in parts if state != "reasoning").strip()


def eligible(**kw) -> bool:
    """Only a complete plain-text reply of useful size is worth prefilling (stop or length:
    a length-capped reply is re-sent as is)."""
    return ineligible_reason(**kw) is None


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


OUTCOMES = ("started", "completed", "cancelled", "ineligible", "preempted")
_lock = threading.Lock()
_counts: dict[str, int] = dict.fromkeys(OUTCOMES, 0)
_tokens = 0


def record(outcome: str, tokens: int = 0) -> None:
    """Count one rehearsal outcome (in process, and on /metrics when the exporter exists)."""
    global _tokens
    if outcome not in OUTCOMES:
        raise ValueError(outcome)
    with _lock:
        _counts[outcome] += 1
        _tokens += tokens
    try:
        from yunshu_gateway.middleware.prometheus_exporter import (
            get_prometheus_metrics,
        )

        m = get_prometheus_metrics()
        m.inc_counter("next_turn_rehearsals", {"outcome": outcome})
        if tokens:
            m.inc_counter("next_turn_rehearsed_tokens", None, tokens)
    except Exception:
        logger.debug("next-turn metrics unavailable", exc_info=True)


def snapshot() -> dict:
    with _lock:
        return {**_counts, "tokens": _tokens}


def ineligible_reason(
    *,
    finish_reason: str | None,
    completion_tokens: int,
    reply: str,
    has_media: bool,
    has_tools: bool,
    constrained: bool,
) -> str | None:
    """Why a reply is not rehearsed (None = eligible)."""
    if finish_reason not in ("stop", "length"):
        return f"finish_reason={finish_reason}"
    if not reply:
        return "empty reply"
    if has_media:
        return "media request"
    if has_tools:
        return "tools request"
    if constrained:
        return "constrained decoding"
    if completion_tokens < MIN_REPLY_TOKENS:
        return f"reply too short ({completion_tokens} tokens)"
    if completion_tokens > MAX_REPLY_TOKENS:
        return f"reply too long ({completion_tokens} tokens)"
    return None


def rehearsal_length(prefix_len: int, stride: int, guard: int) -> int | None:
    """Prompt length to rehearse on the canonical prefill grid.

    With a position-keyed grid (``stride`` tokens per atom) the cache restores a follow-up only
    at a stride multiple, so the rehearsal request ends ``guard`` tokens after the largest
    multiple ``S`` that fits in the shared prefix: the cache stores its checkpoint at ``S``
    (request length minus the guard) and the follow-up prefills only ``prefix_len - S`` plus its
    new message. Returns None when no multiple fits.
    """
    if stride <= 0:
        return prefix_len
    s = (prefix_len // stride) * stride
    if s + guard > prefix_len:
        s -= stride
    return s + guard if s >= stride else None


def extends(request_ids: Sequence[int], rehearsed_ids: Sequence[int]) -> bool:
    """True when the request's prompt starts with every rehearsed token (it is the follow-up)."""
    n = len(rehearsed_ids)
    return (
        n > 0 and len(request_ids) >= n and list(request_ids[:n]) == list(rehearsed_ids)
    )


class RehearsalState:
    """One in-flight rehearsal: when a request may adopt it and when it must cancel it.

    A request that arrives after the rehearsal was submitted and whose prompt extends the
    rehearsed prefix needs exactly that prefill: it adopts the work (the runner's single-flight
    wait releases it at the checkpoint). Anything else cancels the rehearsal at the next prefill
    atom; a request that arrives before submission (still tokenizing, or inside the start
    delay) cancels it too, since no useful work is running yet.
    """

    def __init__(self, cancel_event):
        self.cancel_event = cancel_event
        self.ids: tuple | None = None
        self.submitted = False
        self._lock = threading.Lock()

    def set_ids(self, ids: Sequence[int]) -> None:
        with self._lock:
            self.ids = tuple(ids)

    def submit(self) -> bool:
        """Called right before the runner call; False when a request already cancelled it."""
        with self._lock:
            if self.cancel_event.is_set():
                return False
            self.submitted = True
            return True

    def arrive(self, request_ids: Sequence[int]) -> str:
        """A real request's prompt is known: 'adopt' (leave running) or 'cancel'."""
        with self._lock:
            if (
                self.submitted
                and self.ids is not None
                and extends(request_ids, self.ids)
            ):
                return "adopt"
            self.cancel_event.set()
            return "cancel"
