"""CORS middleware that follows ``YUNSHU_CORS_ORIGINS`` while the server runs.

Starlette's ``CORSMiddleware`` computes its origin tables once in ``__init__``.
This subclass compares the setting on each request and rebuilds the tables when
it changed, so a console edit (``PATCH /v1/yunshu/cors``) applies at once and
no restart is needed.
"""

from __future__ import annotations

from typing import Any

from starlette.middleware.cors import CORSMiddleware

from yunshu_engine import settings

ALLOW_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
ALLOW_HEADERS = [
    "Authorization",
    "Content-Type",
    "Accept",
    "X-Request-ID",
    # Anthropic SDK headers
    "anthropic-version",
    "anthropic-beta",
    "x-api-key",
    # OpenAI SDK headers
    "OpenAI-Organization",
    "OpenAI-Beta",
]


def current_origins() -> list[str]:
    """The configured origins, trimmed; ``["*"]`` for the wildcard."""
    text = settings.get("YUNSHU_CORS_ORIGINS") or ""
    if text.strip() == "*":
        return ["*"]
    return [o.strip() for o in text.split(",") if o.strip()]


def credentials_allowed(origins: list[str]) -> bool:
    """Credentials are never combined with the wildcard (browsers reject it)."""
    return origins != ["*"]


class LiveCORSMiddleware(CORSMiddleware):
    def __init__(self, app: Any) -> None:
        self._origins = current_origins()
        super().__init__(
            app,
            allow_origins=self._origins,
            allow_credentials=credentials_allowed(self._origins),
            allow_methods=ALLOW_METHODS,
            allow_headers=ALLOW_HEADERS,
            max_age=3600,  # Cache preflight for 1 hour to reduce OPTIONS overhead
        )

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        origins = current_origins()
        if origins != self._origins:
            self.__init__(self.app)  # type: ignore[misc]
        await super().__call__(scope, receive, send)
