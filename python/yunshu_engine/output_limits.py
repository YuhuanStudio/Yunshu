"""Output-length and thinking-budget defaults, each with its source.

Request omits the output cap -> the API specs say "up to the model's limit":

* OpenAI Chat ``max_completion_tokens`` / ``max_tokens`` and Responses ``max_output_tokens``: no
  default; the upper bound is the model's output limit (and the context window).
* Qwen3.8 model card, Best Practices: output length 262,144 tokens for thinking mode,
  131,072 for a final answer.  ``MODEL_MAX_OUTPUT_TOKENS`` is the thinking figure because
  thinking is on by default and its tokens count against the same cap.
* OpenAI legacy ``/v1/completions`` ``max_tokens``: documented default 16.

The request is then clamped to the room the prompt leaves in the context window
(``token_budget.plan``), so the effective cap is min(limit, context - prompt); the operator can
lower the limit with ``YUNSHU_DEFAULT_MAX_TOKENS`` (0 = the model limit).
"""

from __future__ import annotations

# Qwen3.8 card (Best Practices): 262,144 output tokens for thinking, 131,072 for the answer.
MODEL_MAX_OUTPUT_TOKENS = 262144
# OpenAI /v1/completions reference: "max_tokens ... Defaults to 16".
COMPLETIONS_DEFAULT_MAX_TOKENS = 16

# Source labels reported in x_yunshu.budget.max_tokens_source.
SOURCE_REQUEST = "request"
SOURCE_MODEL_LIMIT = "model_limit"
SOURCE_OPERATOR_CAP = "operator_cap"
SOURCE_SPEC_DEFAULT = "api_spec_default"


def default_max_output() -> int:
    """Output cap for a request that sent none: the model limit, or the operator's lower cap."""
    from . import settings

    cap = int(settings.get("YUNSHU_DEFAULT_MAX_TOKENS"))
    return min(cap, MODEL_MAX_OUTPUT_TOKENS) if cap > 0 else MODEL_MAX_OUTPUT_TOKENS


def default_source() -> str:
    from . import settings

    return (
        SOURCE_OPERATOR_CAP
        if int(settings.get("YUNSHU_DEFAULT_MAX_TOKENS")) > 0
        else SOURCE_MODEL_LIMIT
    )


# Thinking budget (tokens) -----------------------------------------------------------------
# Largest thinking budget a request may ask for: the model card's thinking output limit
# (Anthropic budget_tokens and OpenAI-style fields have no smaller API cap).
MAX_THINKING_BUDGET = MODEL_MAX_OUTPUT_TOKENS

# OpenAI SDK ReasoningEffort = none | minimal | low | medium | high | xhigh | max;
# "none" means no reasoning and "minimal" is the lowest-reasoning setting, mapped to off
# because the local template has no rung below "low" that thinks less.
THINKING_OFF_EFFORTS = frozenset({"none", "minimal"})


# OpenAI ReasoningEffort (none, minimal, low, medium, high, xhigh) and Anthropic output_config.effort
# (low, medium, high, xhigh, max): the spec values. "max" is the one level only Anthropic names.
ACCEPTED_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def check_effort(effort: str | None) -> str | None:
    """Lower-cased ``reasoning_effort``; a value outside ``ACCEPTED_EFFORTS`` is a 400."""
    if effort is None:
        return None
    e = str(effort).strip().lower()
    if e not in ACCEPTED_EFFORTS:
        raise ValueError(
            f"{effort!r} is not a supported reasoning_effort; accepted values: "
            + ", ".join(ACCEPTED_EFFORTS)
        )
    return e


def fold_effort(
    enable_thinking: bool | None, effort: str | None
) -> tuple[bool | None, str | None]:
    """``reasoning_effort`` "none"/"minimal" -> thinking disabled (an explicit
    ``enable_thinking`` wins); the effort is dropped so no rung of the ladder runs."""
    if effort is not None and str(effort).strip().lower() in THINKING_OFF_EFFORTS:
        return (False if enable_thinking is None else enable_thinking), None
    return enable_thinking, effort
