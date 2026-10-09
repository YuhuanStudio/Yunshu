# Patches upstream mlx-vlm symbols (see vendor.json kind=patches; `just vendor-check` flags source changes)
"""Preemptible prefill: let a background request stop inside a prefill atom.

A 2048-token prefill atom is one lazy graph over every layer that the runner submits with
``async_eval``; once submitted it cannot be recalled, so a request arriving mid-atom waits for
the whole atom (2.4 s at 32K on the 27B). For work that may be thrown away (the next-turn
rehearsal), the decoder layers evaluate their output every ``every`` layers and check the
request's cancel event; on cancel they raise :class:`PreemptedError` and the runner drops the job
and its private cache. Evaluating a lazy array earlier does not change any kernel or its
inputs, so the numbers are unchanged; nothing is stored from the partial atom.

Only the scope set by the runner for a lone, preemptible job is active; every other forward is
untouched (one ContextVar read per layer).
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

EVERY_LAYERS = 4


class PreemptedError(Exception):
    """The preemptible request was cancelled in the middle of a prefill atom."""


@dataclass
class _Scope:
    cancel_event: Any
    every: int = EVERY_LAYERS
    layers: int = 0
    evals: int = 0


_SCOPE: contextvars.ContextVar[_Scope | None] = contextvars.ContextVar(
    "yunshu_preempt_scope", default=None
)


@contextlib.contextmanager
def scope(cancel_event: Any, every: int = EVERY_LAYERS):
    state = _Scope(cancel_event, every)
    token = _SCOPE.set(state)
    try:
        yield state
    finally:
        _SCOPE.reset(token)


def after_layer(out: Any) -> None:
    state = _SCOPE.get()
    if state is None:
        return
    state.layers += 1
    if state.layers % state.every:
        return
    mx.eval(out)
    state.evals += 1
    if state.cancel_event.is_set():
        raise PreemptedError()


def wrap(original):
    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        after_layer(out)
        return out

    forward._yunshu_preempt = True
    return forward


def install() -> bool:
    """Wrap the Qwen3.5 decoder layer (idempotent). False when the class is absent."""
    try:
        from mlx_vlm.models.qwen3_5.language import Qwen3_5DecoderLayer
    except Exception:
        return False
    original = Qwen3_5DecoderLayer.__call__
    if not getattr(original, "_yunshu_preempt", False):
        Qwen3_5DecoderLayer.__call__ = wrap(original)
    return True


def supported() -> bool:
    try:
        from mlx_vlm.models.qwen3_5.language import Qwen3_5DecoderLayer
    except Exception:
        return False
    return bool(getattr(Qwen3_5DecoderLayer.__call__, "_yunshu_preempt", False))
