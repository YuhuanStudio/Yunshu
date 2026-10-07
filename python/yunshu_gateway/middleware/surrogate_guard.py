"""Answer 400, not 500, for request text holding an unpaired UTF-16 surrogate.

JSON allows escapes such as ``"\\ud83d"`` with no partner (agents produce them when they cut
tool output through an emoji). Python decodes them into ``str`` values that cannot be encoded
to UTF-8, so the prompt hash and the tokenizer fail deep in the request. Following oMLX
c5a86d35, every JSON POST is checked once, before any router. The raw-byte prefilter keeps the
cost of a clean 128K-token body to a single ``bytes.find`` pass; only bodies holding a
surrogate escape are parsed.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..error_envelope import format_error_response
from .body_replay import replay_receive

_ESCAPE = re.compile(rb"\\u[dD][89a-fA-F]")
_LONE = re.compile("[\ud800-\udfff]")


def find_lone_surrogate(value: Any, path: str = "") -> str | None:
    """Path of the first string (or object key) holding an unpaired surrogate, else None."""
    if isinstance(value, str):
        return (path or "body") if _LONE.search(value) else None
    if isinstance(value, dict):
        for key, item in value.items():
            here = f"{path}.{key}" if path else str(key)
            if isinstance(key, str) and _LONE.search(key):
                return here
            found = find_lone_surrogate(item, here)
            if found:
                return found
    elif isinstance(value, list):
        for i, item in enumerate(value):
            found = find_lone_surrogate(item, f"{path}[{i}]")
            if found:
                return found
    return None


def lone_surrogate_field(raw: bytes) -> str | None:
    """The offending field of a JSON body, or None (clean, or not JSON: left to the router)."""
    if not _ESCAPE.search(raw):
        return None
    try:
        body = json.loads(raw)
    except ValueError:
        return None
    return find_lone_surrogate(body)


class SurrogateGuardMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        if b"json" not in headers.get(b"content-type", b""):
            return await self.app(scope, receive, send)
        chunks = []
        while True:
            event = await receive()
            if event["type"] != "http.request":
                return
            chunks.append(event.get("body", b""))
            if not event.get("more_body"):
                break
        raw = b"".join(chunks)
        field = lone_surrogate_field(raw)
        if field:
            response = format_error_response(
                scope.get("path", ""),
                f"Invalid string in '{field}': unpaired UTF-16 surrogate "
                "(text was likely truncated through an emoji)",
                400,
                code="invalid_unicode",
                error_type="invalid_request_error",
            )
            return await response(scope, receive, send)
        await self.app(scope, replay_receive(raw, receive), send)
