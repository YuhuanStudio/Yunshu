"""Constant-time token comparison that tolerates non-ASCII input."""

from __future__ import annotations

import hmac


def tokens_equal(presented: str | bytes | None, expected: str | bytes | None) -> bool:
    """Constant-time equality over UTF-8 bytes; never raises on non-ASCII."""
    if not presented or not expected:
        return False
    a = (
        presented.encode("utf-8", "surrogateescape")
        if isinstance(presented, str)
        else presented
    )
    b = (
        expected.encode("utf-8", "surrogateescape")
        if isinstance(expected, str)
        else expected
    )
    return hmac.compare_digest(a, b)
