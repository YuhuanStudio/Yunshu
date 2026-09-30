"""Reasoning continuity for stateless Responses clients (Codex sends ``store: false`` and
``include: ["reasoning.encrypted_content"]``).

OpenAI returns the model's reasoning as an opaque ``encrypted_content`` on ``reasoning`` items and
expects the client to send those items back, so the model can continue its chain of thought across tool
calls. Yunshu does the same, honestly: the token is the reasoning text in an opaque envelope (a local
server has nothing to hide from its own client), and when a ``reasoning`` item comes back in ``input`` or
in a stored response chain, its text is handed to the next assistant turn as ``reasoning_content``, the
field chat templates render (Qwen keeps the reasoning of the turns after the last user message).
"""

from __future__ import annotations

import base64

_PREFIX = "yunshu1:"


def seal_reasoning(text: str) -> str:
    return _PREFIX + base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")


def unseal_reasoning(token: str | None) -> str | None:
    if not isinstance(token, str) or not token.startswith(_PREFIX):
        return None
    try:
        return base64.urlsafe_b64decode(token[len(_PREFIX) :].encode("ascii")).decode(
            "utf-8"
        )
    except Exception:
        return None


def reasoning_text_of(item: dict) -> str:
    """The reasoning text a ``reasoning`` item carries: our sealed token, else summary / content text."""
    sealed = unseal_reasoning(item.get("encrypted_content"))
    if sealed:
        return sealed
    parts: list[str] = []
    for key in ("content", "summary"):
        for p in item.get(key) or []:
            if isinstance(p, dict) and p.get("text"):
                parts.append(p["text"])
        if parts:
            break
    return "\n".join(parts)
