"""Admission control and per-request deadlines for the generation routes.

Two things the engine cannot do for itself, both decided before a request reaches the single MLX
executor and both answered in the dialect of the route (OpenAI chat / completions / Responses,
Anthropic Messages):

- **Refuse instead of hanging.** When the server already has ``YUNSHU_QUEUE_LIMIT`` generation
  requests in flight the next one gets ``429`` with ``Retry-After`` and the queue depth in
  ``error.x_yunshu``; when memory is nearly full while other requests run it gets ``503`` the same
  way. An idle server never refuses on memory: waiting would not help.
- **Deadline.** ``X-Yunshu-Deadline-Ms: N`` is the wall time the client will wait, counted from
  arrival (queue wait included). OpenAI has no such field, so it is a namespaced header, the same
  place ``X-Request-Id`` lives. The middleware enforces it (see ``x_yunshu.py``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from yunshu_engine import settings

logger = logging.getLogger(__name__)

DEADLINE_HEADER = "x-yunshu-deadline-ms"
# Longest deadline accepted (a day); anything above is a client bug, not a plan.
MAX_DEADLINE_MS = 86_400_000
_RETRY_MIN_S = 1
_RETRY_MAX_S = 60
_RETRY_DEFAULT_S = 5


@dataclass(frozen=True)
class Refusal:
    status: int
    code: str
    message: str
    retry_after: int
    detail: dict[str, Any]


def parse_deadline_ms(raw: bytes | str | None) -> int | None:
    """The deadline in ms from the header value; None when absent. ValueError (with the
    message to show the client) when it is not a positive integer within bounds."""
    if raw is None:
        return None
    text = raw.decode("latin-1") if isinstance(raw, bytes) else raw
    text = text.strip()
    if not text:
        return None
    try:
        value = int(text)
    except ValueError:
        raise ValueError(
            f"X-Yunshu-Deadline-Ms must be a whole number of milliseconds, got {text!r}"
        ) from None
    if value <= 0 or value > MAX_DEADLINE_MS:
        raise ValueError(
            f"X-Yunshu-Deadline-Ms must be between 1 and {MAX_DEADLINE_MS}, got {value}"
        )
    return value


def memory_pressure() -> float | None:
    """MLX active memory over the recommended Metal working set (None when unknown).
    Replaced in tests."""
    try:
        import mlx.core as mx

        limit = mx.device_info().get("max_recommended_working_set_size")
        if not limit:
            return None
        return float(mx.get_active_memory()) / float(limit)
    except Exception:
        logger.debug("memory pressure unavailable", exc_info=True)
        return None


def retry_after_s(est_wait_ms: float | None) -> int:
    """Seconds to tell the client to wait: the estimated queue wait rounded up, clamped to
    [1, 60]; 5 when there is no estimate."""
    if not est_wait_ms or est_wait_ms <= 0:
        return _RETRY_DEFAULT_S
    return int(min(_RETRY_MAX_S, max(_RETRY_MIN_S, math.ceil(est_wait_ms / 1000.0))))


def admit(in_flight: int, est_wait_ms: float | None = None) -> Refusal | None:
    """None to admit; a :class:`Refusal` when the server is full or short of memory.
    ``in_flight`` counts the generation requests already admitted (this one excluded)."""
    limit = int(settings.get("YUNSHU_QUEUE_LIMIT") or 0)
    retry = retry_after_s(est_wait_ms)
    if limit and in_flight >= limit:
        return Refusal(
            429,
            "queue_full",
            f"Too many requests in flight ({in_flight} of {limit} allowed); "
            f"retry in {retry} s",
            retry,
            {
                "reason": "queue_full",
                "queue_depth": in_flight,
                "queue_limit": limit,
                "retry_after_s": retry,
            },
        )
    threshold = float(settings.get("YUNSHU_MEMORY_PRESSURE_REJECT") or 0.0)
    if threshold > 0 and in_flight > 0:
        pressure = memory_pressure()
        if pressure is not None and pressure >= threshold:
            return Refusal(
                503,
                "memory_pressure",
                f"The server is under memory pressure ({pressure:.0%} of the Metal "
                f"working set in use) with {in_flight} request(s) running; retry in {retry} s",
                retry,
                {
                    "reason": "memory_pressure",
                    "memory_pressure": round(pressure, 3),
                    "queue_depth": in_flight,
                    "retry_after_s": retry,
                },
            )
    return None


def deadline_message(deadline_ms: int) -> str:
    return f"Deadline exceeded: the request did not finish within {deadline_ms} ms"


def deadline_stream_event(
    kind: str, deadline_ms: int, request_id: str, seq: int
) -> bytes:
    """The terminal SSE event for a stream that is past its deadline, in the dialect of the
    route (``kind`` as in ``x_yunshu.dialect``): an error event, then ``[DONE]`` where the
    dialect has one."""
    import json

    msg = deadline_message(deadline_ms)
    detail = {"reason": "deadline_exceeded", "deadline_ms": deadline_ms}
    body: dict[str, Any]
    if kind == "anthropic":
        body = {"type": "error", "error": {"type": "timeout_error", "message": msg}}
        return f"event: error\ndata: {json.dumps(body)}\n\n".encode()
    if kind == "responses":
        body = {
            "type": "error",
            "code": "deadline_exceeded",
            "message": msg,
            "param": None,
            "sequence_number": seq,
        }
        return f"event: error\ndata: {json.dumps(body)}\n\n".encode()
    body = {
        "error": {
            "message": msg,
            "type": "timeout_error",
            "code": "deadline_exceeded",
            "x_yunshu": {**detail, "request_id": request_id},
        }
    }
    return f"data: {json.dumps(body)}\n\ndata: [DONE]\n\n".encode()
