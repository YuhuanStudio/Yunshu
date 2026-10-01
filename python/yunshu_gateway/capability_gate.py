"""The capability-contract request gate shared by every dialect.

Each route hands the model name and a chat-shaped probe of what the request uses (``messages``
holding content parts, plus ``tools`` / ``response_format`` / ``logprobs`` ...). A violation is an
``HTTPException(400)``; the app's handler renders it in the dialect's own error shape (OpenAI
``{"error": {...}}``, Anthropic ``{"type": "error", ...}``). Ollama's routes go through the
loopback chat route and relay its message in Ollama's ``{"error": "..."}``.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException

logger = logging.getLogger(__name__)


def enforce(model: str, probe: dict[str, Any]) -> None:
    """400 when ``probe`` uses a field the served ``model``'s contract does not cover."""
    try:
        from yunshu_engine.capability_contract import unsupported

        from .model_cards import find_card

        reasons = unsupported(find_card(model), probe)
    except Exception:  # noqa: BLE001 - a card problem must never break serving
        logger.debug("capability contract check failed", exc_info=True)
        return
    if reasons:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' does not support: " + "; ".join(reasons),
        )


def _dump(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_dump(o) for o in obj]
    return obj.model_dump(mode="json") if hasattr(obj, "model_dump") else obj


def enforce_anthropic(req: Any) -> None:
    enforce(
        req.model,
        {
            "messages": _dump(req.messages),
            "tools": _dump(req.tools),
            "tool_choice": req.tool_choice,
            "response_format": req.response_format,
            "logprobs": req.logprobs,
        },
    )


def enforce_responses(req: Any) -> None:
    items = req.input if isinstance(req.input, list) else []
    enforce(
        req.model,
        {
            "messages": _dump(items),
            "tools": _dump(req.tools),
            "tool_choice": req.tool_choice,
            "response_format": req.response_format,
            "logprobs": req.logprobs,
        },
    )


def enforce_completion(req: Any) -> None:
    enforce(
        req.model,
        {
            "response_format": req.response_format,
            "grammar": req.grammar,
            "logprobs": req.logprobs,
        },
    )
