"""Command line of the console process, without typer or rich (they alone cost ~25 MiB of RSS).

``python -m yunshu_console`` is what ``yunshu serve`` (sibling) and the launchd job run;
``yunshu console`` is the same thing with the CLI's help formatting.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys

DEFAULT_ENGINE = "http://127.0.0.1:8000"


def port_busy(host: str, port: int) -> bool:
    """True when something already listens there (e.g. the service's console job)."""
    probe = "127.0.0.1" if host in ("0.0.0.0", "") else host
    try:
        with socket.create_connection((probe, port), timeout=0.4):
            return True
    except OSError:
        return False


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="yunshu console",
        description="The console process: the web console and docs on one port, a reverse proxy "
        "to the engine API, and the recorded history that survives engine restarts.",
    )
    p.add_argument(
        "--engine",
        "-e",
        help=f"Engine URL to watch and proxy to (YUNSHU_CONSOLE_ENGINE, else {DEFAULT_ENGINE}).",
    )
    p.add_argument("--host", help="Bind host (default 127.0.0.1; YUNSHU_CONSOLE_HOST).")
    p.add_argument(
        "--port", "-p", type=int, help="Bind port (default 8100; YUNSHU_CONSOLE_PORT)."
    )
    p.add_argument(
        "--engine-token",
        help="Bearer token for reading the engine (YUNSHU_CONSOLE_ENGINE_TOKEN, else YUNSHU_AUTH_TOKEN).",
    )
    p.add_argument(
        "--config", "-c", help="TOML file of YUNSHU_* settings (the engine's)."
    )
    p.add_argument(
        "--no-history",
        action="store_true",
        help="Do not record the metrics history or request log.",
    )
    p.add_argument("--log-level", default="info", help="uvicorn log level.")
    p.add_argument(
        "--static-dir",
        help="Serve this console build instead of the packaged one (development, tests).",
    )
    return p


def run(args: argparse.Namespace) -> None:
    import uvicorn

    from yunshu_engine import settings

    if args.config:
        settings.set_override(
            "YUNSHU_CONFIG", os.path.abspath(os.path.expanduser(args.config))
        )
    if args.engine:
        settings.set_override("YUNSHU_CONSOLE_ENGINE", args.engine)
    if args.engine_token:
        settings.set_override("YUNSHU_CONSOLE_ENGINE_TOKEN", args.engine_token)
    if args.no_history:
        settings.set_override("YUNSHU_CONSOLE_HISTORY", False)
    bind_host = args.host or settings.get("YUNSHU_CONSOLE_HOST") or "127.0.0.1"
    bind_port = int(args.port or settings.get("YUNSHU_CONSOLE_PORT") or 8100)
    from .app import build_from_settings

    app = build_from_settings(args.engine, static_dir=args.static_dir)
    shown = "127.0.0.1" if bind_host == "0.0.0.0" else bind_host
    print(
        f"Yunshu console http://{shown}:{bind_port}/console/  ->  engine {app.state.engine_url}",
        file=sys.stderr,
        flush=True,
    )
    uvicorn.run(app, host=bind_host, port=bind_port, log_level=args.log_level)


def main(argv: list[str] | None = None) -> None:
    run(parser().parse_args(argv))
