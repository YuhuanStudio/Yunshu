"""Bodies for a refused prompt that cannot fit in memory: the spec's context overflow.

Agent clients (Codex, Claude Code) auto-compact on a context-overflow error, so a prompt that
does not fit answers with it (OpenAI: 400 ``context_length_exceeded``; Anthropic: 400
``prompt is too long``). The memory reason and the longest prompt that fits stay visible in the
namespaced ``yunshu`` field. Other memory refusals keep their own shape.
"""

from __future__ import annotations

from yunshu_engine.memory_admission import PromptTooLongError


def _ext(exc: PromptTooLongError) -> dict:
    return {"reason": "memory_limit", "fits_tokens": exc.fits_tokens}


def openai_prompt_too_long(exc: BaseException) -> dict | None:
    """The OpenAI error body for ``exc``, or None when it is not a prompt-length refusal."""
    if not isinstance(exc, PromptTooLongError):
        return None
    return {
        "error": {
            "message": (
                f"This model's maximum context length is {exc.fits_tokens} tokens. "
                f"However, your messages resulted in {exc.prompt_tokens} tokens. "
                "Please reduce the length of the messages."
            ),
            "type": "invalid_request_error",
            "param": "messages",
            "code": "context_length_exceeded",
            "yunshu": _ext(exc),
        }
    }


def anthropic_prompt_too_long(exc: BaseException) -> dict | None:
    """The Anthropic error body for ``exc``, or None when it is not a prompt-length refusal."""
    if not isinstance(exc, PromptTooLongError):
        return None
    return {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": (
                f"prompt is too long: {exc.prompt_tokens} tokens > {exc.fits_tokens} maximum"
            ),
            "yunshu": _ext(exc),
        },
    }
