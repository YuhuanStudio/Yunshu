"""Serve the built Yunshu Console as a same-origin static application.

The console build is optional in source checkouts. API routes and inference
startup do not depend on it; when the frontend has not been built, /console
returns an actionable 404 instead of breaking app construction.
"""

from pathlib import Path

from fastapi import FastAPI
from starlette.responses import PlainTextResponse
from starlette.staticfiles import StaticFiles
from starlette.types import Scope


class ConsoleStaticFiles(StaticFiles):
    """Hash-routed Vite assets rooted at the packaged console build directory."""

    def __init__(self, directory: Path) -> None:
        self._console_root = directory
        super().__init__(
            directory=str(directory), html=True, check_dir=False, follow_symlink=False
        )

    async def check_config(self) -> None:
        """Leave an absent build as a request-time 404 instead of startup failure."""

    async def get_response(self, path: str, scope: Scope):
        if (
            not self._console_root.is_dir()
            or not (self._console_root / "index.html").is_file()
        ):
            return PlainTextResponse(
                "Yunshu Console is not built. Run `cd frontend && pnpm build` to create python/yunshu_gateway/console_static.",
                status_code=404,
            )
        return await super().get_response(path, scope)


def mount_console_ui(app: FastAPI, *, static_dir: Path | None = None) -> None:
    """Mount the optional console at /console without SPA history fallback."""
    directory = static_dir or Path(__file__).with_name("console_static")
    app.mount("/console", ConsoleStaticFiles(directory), name="console")
