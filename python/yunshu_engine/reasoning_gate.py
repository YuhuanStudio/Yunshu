"""Constrain structured output (JSON schema / grammar) only AFTER the reasoning ends.

A response-format grammar masks the output from the first generated token. When the prompt
pre-opens ``<think>`` (thinking on) the model is still reasoning there, so the schema used to be
forced into the reasoning block and the answer came back empty (the reason a constrained request
defaults to thinking off). With the gate the reasoning is free text; the grammar starts on the
token after the reasoning close.

While the reasoning is open the gate also masks

* the end-of-sequence ids (a stop there would leave the grammar unbound and the answer empty:
  mlx2 01dcd75 moves the stop probability to the close marker; a bitmask cannot move
  probability, so Yunshu removes the stop and the model keeps reasoning until it closes, the same
  masking the forced tool-call grammar uses), and
* the implicit reasoning ends the grammar rejects as a first content token, e.g. ``<tool_call>``
  (vllm 4ac1f8d2a): sampling one would end the reasoning with an invalid answer.

Everything is a function of the token history, so the speculative lanes' plan / checkpoint /
restore see the same masks as one-token decode. A request whose reasoning never closes (max
tokens inside ``<think>``) reports ``enforced=false, reason=reasoning_not_closed`` instead of an
enforced grammar that never bound.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

NOT_CLOSED = "reasoning_not_closed"


class ReasoningGate:
    """Open until the reasoning close token is generated; blocks stop / implicit-end ids while open."""

    def __init__(
        self,
        close_id: int,
        eos_ids: list[int] | tuple[int, ...],
        implicit_end_ids: list[int] | tuple[int, ...] = (),
        report: dict | None = None,
    ) -> None:
        self.close_id = int(close_id)
        self.blocked = tuple(
            sorted(
                {int(t) for t in (*eos_ids, *implicit_end_ids) if t is not None}
                - {self.close_id}
            )
        )
        self.open = True
        self.report = report
        if report is not None:
            report.update(enforced=False, reason=NOT_CLOSED)

    def feed(self, token: int) -> None:
        if self.open and int(token) == self.close_id:
            self.open = False
            if self.report is not None:
                self.report.update(enforced=True, reason="enforced")

    def restore_open(self, was_open: bool) -> None:
        self.open = was_open
        if self.report is not None:
            self.report.update(
                enforced=not was_open, reason="enforced" if not was_open else NOT_CLOSED
            )


def make_gate(
    *,
    prompt_opens_thinking: bool,
    close_id: int | None,
    eos_ids,
    implicit_end_ids=(),
    report: dict | None = None,
) -> ReasoningGate | None:
    """A gate when generation starts inside an open reasoning block, else None."""
    if not prompt_opens_thinking or close_id is None:
        return None
    return ReasoningGate(close_id, list(eos_ids or ()), list(implicit_end_ids), report)


class ThinkGatedGuide:
    """Bitmask guide: free reasoning (stop and implicit ends masked), then ``inner``."""

    constrained = True

    def __init__(self, inner: Any, gate: ReasoningGate) -> None:
        self.inner = inner
        self.gate = gate
        self.words = inner.words
        self.lane_rounds = 0
        row = np.full(self.words, 0xFFFFFFFF, dtype=np.uint32)
        for t in gate.blocked:
            if 0 <= t < self.words * 32:
                row[t >> 5] &= np.uint32(~(1 << (t & 31)) & 0xFFFFFFFF)
        self._wait_row = row.view(np.int32)

    def mask(self):
        if self.gate.open:
            return self._wait_row
        return self.inner.mask()

    def feed(self, token):
        if self.gate.open:
            self.gate.feed(token)
            return True
        return self.inner.feed(token)

    def advance(self, tokens):
        for token in tokens:
            self.feed(token)
        return len(tokens)

    def checkpoint(self):
        return self.gate.open, self.inner.checkpoint()

    def restore(self, saved):
        was_open, inner = saved
        self.gate.restore_open(was_open)
        self.inner.restore(inner)

    def plan(self, drafts, n_pos):
        from .constrained_spec import ConstraintGuide

        return ConstraintGuide.plan(self, drafts, n_pos)


class ThinkGatedSampler:
    """Text-path sampler: unconstrained (stop and implicit ends masked) until the close token,
    then the wrapped constrained sampler (whose grammar state starts fresh)."""

    def __init__(self, free: Any, constrained: Any, gate: ReasoningGate) -> None:
        self._free = free
        self._constrained = constrained
        self.gate = gate
        self._blocked = None

    def __call__(self, logits):
        if not self.gate.open:
            return self._constrained(logits)
        import mlx.core as mx

        if self.gate.blocked:
            width = logits.shape[-1]
            if self._blocked is None or self._blocked.shape[-1] != width:
                ids = [t for t in self.gate.blocked if 0 <= t < width]
                self._blocked = mx.zeros((width,), dtype=mx.float32)
                if ids:
                    self._blocked = self._blocked.at[mx.array(ids)].add(-float("inf"))
            logits = logits + self._blocked.astype(logits.dtype)
        token = self._free(logits)
        self.gate.feed(int(token))
        return token

    @property
    def constraint(self):
        # The n-gram drafter pre-filters drafts against the live grammar state; during
        # reasoning there is none, so it must not be consulted (acceptance still goes through
        # this sampler).
        if self.gate.open:
            raise AttributeError("constraint")
        return self._constrained.constraint

    def __getattr__(self, name):
        if name.startswith("_") or name in ("gate",):
            raise AttributeError(name)
        return getattr(self._constrained, name)


def prompt_opens_reasoning(input_ids, tokenizer) -> tuple[bool, int | None]:
    """(generation starts inside an open reasoning block, the close token id)."""
    from .thinking_budget import _resolve_think_end_token_ids, detect_needs_think_prefix

    try:
        end = _resolve_think_end_token_ids(tokenizer)
        if not end or len(end) != 1:
            return False, None
        return bool(detect_needs_think_prefix(list(input_ids), tokenizer)), int(end[0])
    except Exception:
        logger.debug("reasoning-open detection failed", exc_info=True)
        return False, None


def gate_constrained_sampler(
    free_sampler, constrained_sampler, input_ids, tokenizer, eos_ids, report=None
):
    """Wrap a text-path constrained sampler so the grammar binds after the reasoning close."""
    if constrained_sampler is free_sampler:
        return constrained_sampler
    opens, close_id = prompt_opens_reasoning(input_ids, tokenizer)
    gate = make_gate(
        prompt_opens_thinking=opens,
        close_id=close_id,
        eos_ids=eos_ids,
        implicit_end_ids=_tool_marker_ids(tokenizer),
        report=report,
    )
    if gate is None:
        return constrained_sampler
    return ThinkGatedSampler(free_sampler, constrained_sampler, gate)


def _tool_marker_ids(tokenizer) -> list[int]:
    try:
        ids = tokenizer.encode("<tool_call>", add_special_tokens=False)
    except Exception:
        return []
    return [int(ids[0])] if len(ids) == 1 else []
