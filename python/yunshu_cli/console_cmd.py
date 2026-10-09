"""Yunshu CLI: ``yunshu console``, the console process (web console, docs, history, engine proxy).

It is light and never loads MLX. ``yunshu serve`` starts one as a sibling by default;
``yunshu console --engine URL`` runs it on its own, e.g. to watch an engine on another machine, and
``yunshu service install`` runs it as its own launchd job so it keeps recording while the engine
restarts. See docs/CONSOLE.md.
"""

from __future__ import annotations

import socket

import typer
from rich.console import Console

from yunshu_engine import settings

console = Console()
DEFAULT_ENGINE = "http://127.0.0.1:8000"


def console_port_busy(host: str, port: int) -> bool:
    """True when something already listens there (e.g. the service's console job)."""
    probe = "127.0.0.1" if host in ("0.0.0.0", "") else host
    try:
        with socket.create_connection((probe, port), timeout=0.4):
            return True
    except OSError:
        return False


def console_command(
    engine: str | None = typer.Option(
        None,
        "--engine",
        "-e",
        help=f"Engine URL to watch and proxy to (default: YUNSHU_CONSOLE_ENGINE, else {DEFAULT_ENGINE}).",
    ),
    host: str | None = typer.Option(
        None, "--host", "-h", help="Bind host (default 127.0.0.1; YUNSHU_CONSOLE_HOST)."
    ),
    port: int | None = typer.Option(
        None, "--port", "-p", help="Bind port (default 8100; YUNSHU_CONSOLE_PORT)."
    ),
    engine_token: str | None = typer.Option(
        None,
        "--engine-token",
        help="Bearer token for reading the engine (default YUNSHU_CONSOLE_ENGINE_TOKEN, "
        "else YUNSHU_AUTH_TOKEN).",
    ),
    no_history: bool = typer.Option(
        False, "--no-history", help="Do not record the metrics history or request log."
    ),
    log_level: str = typer.Option("info", "--log-level", help="uvicorn log level."),
) -> None:
    """Run the console process: the web console and docs on one port, a reverse proxy to the
    engine API, and the recorded history that survives engine restarts."""
    import uvicorn

    if engine:
        settings.set_override("YUNSHU_CONSOLE_ENGINE", engine)
    if engine_token:
        settings.set_override("YUNSHU_CONSOLE_ENGINE_TOKEN", engine_token)
    if no_history:
        settings.set_override("YUNSHU_CONSOLE_HISTORY", False)
    bind_host = host or settings.get("YUNSHU_CONSOLE_HOST") or "127.0.0.1"
    bind_port = int(port or settings.get("YUNSHU_CONSOLE_PORT") or 8100)
    from yunshu_console.app import build_from_settings

    app = build_from_settings(engine)
    shown = "127.0.0.1" if bind_host == "0.0.0.0" else bind_host
    console.print(
        f"[bold]Yunshu console[/] http://{shown}:{bind_port}/console/  ->  engine "
        f"{app.state.engine_url}"
    )
    uvicorn.run(app, host=bind_host, port=bind_port, log_level=log_level)
