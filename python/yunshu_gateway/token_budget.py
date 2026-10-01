"""One token budget for a request: context window = prompt + thinking + answer.

``validate_context_window`` only rejects a prompt that is longer than the window. A prompt that
fits but leaves less room than ``max_tokens`` asks for (an agent sending ``max_tokens=32000`` with a
240K-token transcript on a 262K window) used to be handed to the engine as is, and the decode ran
past the window. :func:`plan` grants what fits instead: ``max_tokens`` is clamped to the room the
prompt leaves, ``thinking_budget`` (a part of the answer's tokens) to what was granted, and the
clamp is reported in ``x_yunshu.budget`` so the client can tell why the answer is shorter than it
asked. A request that fits is untouched and reports nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenBudget:
    context_window: int
    prompt_tokens: int
    max_tokens_requested: int
    max_tokens_granted: int
    thinking_budget_requested: int | None = None
    thinking_budget_granted: int | None = None

    @property
    def clamped(self) -> bool:
        return (
            self.max_tokens_granted < self.max_tokens_requested
            or self.thinking_budget_granted != self.thinking_budget_requested
        )

    def report(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "context_window": self.context_window,
            "prompt_tokens": self.prompt_tokens,
            "max_tokens_requested": self.max_tokens_requested,
            "max_tokens_granted": self.max_tokens_granted,
            "clamped_by": "context_window",
        }
        if self.thinking_budget_requested is not None:
            out["thinking_budget_requested"] = self.thinking_budget_requested
            out["thinking_budget_granted"] = self.thinking_budget_granted
        return out

    def publish(self) -> None:
        """Attach the report to the request being served (a no-op when nothing was clamped
        or outside a request)."""
        if not self.clamped:
            return
        logger.info("token budget clamped: %s", self.report())
        from yunshu_engine.request_tracker import current_request_info

        info: Any = current_request_info.get()
        if info is not None:
            try:
                info.budget = self.report()
            except Exception:
                logger.debug("budget report not attached", exc_info=True)


def plan(
    context_window: int | None,
    prompt_tokens: int,
    max_tokens: int,
    thinking_budget: int | None = None,
) -> TokenBudget | None:
    """The budget for one request; None when the window is unknown (nothing to enforce).

    Raises ``ValueError`` when the prompt leaves no room for even one token (the caller turns
    that into the same 400 as an over-long prompt)."""
    if not context_window or context_window <= 0:
        return None
    room = context_window - prompt_tokens
    if room < 1:
        raise ValueError(
            f"prompt is too long: {prompt_tokens} tokens leave no room for output; "
            f"the prompt exceeds max context window of {context_window} tokens"
        )
    granted = min(max_tokens, room)
    thinking = thinking_budget
    if thinking is not None:
        thinking = min(thinking, granted)
    return TokenBudget(
        context_window=context_window,
        prompt_tokens=prompt_tokens,
        max_tokens_requested=max_tokens,
        max_tokens_granted=granted,
        thinking_budget_requested=thinking_budget,
        thinking_budget_granted=thinking,
    )


def plan_for_engine(
    prompt_tokens: int,
    max_tokens: int,
    thinking_budget: int | None,
    model_id: str | None,
    engine: Any,
) -> TokenBudget | None:
    """:func:`plan` with the window looked up the way ``validate_context_window`` does; an
    exhausted window is the same 400 as an over-long prompt."""
    from fastapi import HTTPException

    from .streaming import get_max_context_window

    try:
        budget = plan(
            get_max_context_window(model_id, engine, trusted_only=True),
            prompt_tokens,
            max_tokens,
            thinking_budget,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    if budget is not None:
        budget.publish()
    return budget
