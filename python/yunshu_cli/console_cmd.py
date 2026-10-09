"""Yunshu CLI: ``yunshu console``, the console process (web console, docs, history, engine proxy).

It is light and never loads MLX. ``yunshu serve`` starts one as a sibling by default;
``yunshu console --engine URL`` runs it on its own, e.g. to watch an engine on another machine, and
``yunshu service install`` runs it as its own launchd job so it keeps recording while the engine
restarts. See docs/CONSOLE.md.
"""

from __future__ import annotations

import typer
from rich.console import Console

console = Console()
DEFAULT_ENGINE = "http://127.0.0.1:8000"


def console_port_busy(host: str, port: int) -> bool:
    """True when something already listens there (e.g. the service's console job)."""
    from yunshu_console.cli import port_busy

    return port_busy(host, port)


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
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="TOML file of YUNSHU_* settings (same file as the engine's).",
    ),
    static_dir: str | None = typer.Option(
        None, "--static-dir", help="Serve this console build (development, tests)."
    ),
    no_history: bool = typer.Option(
        False, "--no-history", help="Do not record the metrics history or request log."
    ),
    log_level: str = typer.Option("info", "--log-level", help="uvicorn log level."),
) -> None:
    """Run the console process: the web console and docs on one port, a reverse proxy to the
    engine API, and the recorded history that survives engine restarts."""
    import argparse

    from yunshu_console.cli import run

    run(
        argparse.Namespace(
            engine=engine,
            host=host,
            port=port,
            engine_token=engine_token,
            config=config,
            no_history=no_history,
            static_dir=static_dir,
            log_level=log_level,
        )
    )
