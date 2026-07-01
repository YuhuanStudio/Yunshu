"""Shared CLI output layer for agent-facing use.

Every command renders through here so a single global ``--json`` flag switches the WHOLE
CLI between human (Rich) output and machine-readable JSON on stdout. Agents set ``--json``,
parse stdout, and branch on the process exit code (0 = ok, non-zero = failure).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from rich.console import Console

console = Console()

_json_mode = False


def set_json_mode(enabled: bool) -> None:
    global _json_mode
    _json_mode = enabled


def auth_headers() -> dict[str, str]:
    """Authorization header from ``YUNSHU_AUTH_TOKEN`` so the CLI can reach the gateway's
    token-gated admin/monitoring endpoints. Empty when no token is set (open server)."""
    import os

    tok = os.environ.get("YUNSHU_AUTH_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def is_json() -> bool:
    return _json_mode


def emit(data: Any, human: Callable[[], None] | None = None) -> None:
    """Emit a successful result.

    In ``--json`` mode, print ``data`` as pretty JSON to stdout. Otherwise call ``human``
    (the Rich renderer). ``data`` should be JSON-serializable; non-serializable values fall
    back to ``str``.
    """
    if _json_mode:
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    elif human is not None:
        human()


def fail(message: str, code: int = 1, **extra: Any) -> None:
    """Emit an error and raise typer.Exit(code).

    JSON mode → ``{"error": message, ...extra}`` on stdout; human mode → red text on stderr.
    """
    import typer

    if _json_mode:
        print(json.dumps({"error": message, **extra}, ensure_ascii=False, default=str))
    else:
        console.print(f"[red]{message}[/]")
    raise typer.Exit(code)
