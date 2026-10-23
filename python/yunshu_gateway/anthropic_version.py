"""Version-aware request rules of the Anthropic API version 2026-10-21.

A request that declares ``anthropic-version: 2026-10-21`` (or later) gets the stricter Messages
validation of that version (``/v1/messages``, ``/v1/messages/count_tokens`` and every request of a
Message Batch). ``2023-06-01`` and requests without the header keep the lenient behaviour.

Sampling parameters (``temperature`` / ``top_p`` / ``top_k``) are 400 on the hosted API under
2026-10-21; this gateway still accepts them (a local-model decision, see API_SURFACE.md).
"""

from __future__ import annotations

from typing import Any

NEW_VERSION = "2026-10-21"


def is_strict(version: str | None) -> bool:
    """True when the ``anthropic-version`` header names 2026-10-21 or a later date."""
    v = (version or "").strip()
    return len(v) == 10 and v[4] == "-" and v[7] == "-" and v >= NEW_VERSION


def _text_citations(block: Any) -> bool:
    return (
        isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("citations"), list)
        and len(block["citations"]) > 0
    )


def validate_messages(messages: list[Any], prefix: str = "messages") -> str | None:
    """Return the 400 message for the first violation, or None.

    Each item is a dict or an object with ``role`` / ``content``.
    """
    for i, m in enumerate(messages):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        content = (
            m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        )
        if not isinstance(content, list):
            continue
        for j, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            where = f"{prefix}.{i}.content.{j}"
            if role == "assistant" and block.get("type") == "document":
                return (
                    f"{where}: `document` blocks are not supported in `assistant` "
                    "messages; send documents in a `user` message"
                )
            if role == "user":
                if _text_citations(block):
                    return (
                        f"{where}.citations: `citations` on `text` blocks are not supported "
                        "in `user` messages"
                    )
                if block.get("type") == "search_result" and isinstance(
                    block.get("content"), list
                ):
                    for k, inner in enumerate(block["content"]):
                        if _text_citations(inner):
                            return (
                                f"{where}.content.{k}.citations: `citations` on `text` blocks "
                                "are not supported in `user` messages"
                            )
    return None


def validate_request_body(body: dict[str, Any], prefix: str = "") -> str | None:
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return None
    return validate_messages(msgs, f"{prefix}messages")
