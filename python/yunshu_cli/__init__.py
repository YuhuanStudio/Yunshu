"""Yunshu CLI — command-line interface for the Yunshu inference engine.

Getting started: doctor, pull, serve, service. Inference commands (talk to a
running server): complete, embed, tokenize, rerank, transcribe, speak, ocr, image.
Management: chat, model, config, status, launch, eval, bench, diagnose. Every
command honors the global ``--json`` flag for agent use.
"""

from __future__ import annotations

import os

import typer
from rich.console import Console

console = Console()
app = typer.Typer(
    name="yunshu",
    help="Fast local LLM / VLM inference engine for Apple Silicon (MLX).",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


DEFAULT_GATEWAY_URL = "http://localhost:8000"


def _print_version(value: bool) -> None:
    if value:
        from yunshu_engine.version import yunshu_version

        typer.echo(f"yunshu {yunshu_version()}")
        raise typer.Exit()


@app.callback()
def _global_options(
    ctx: typer.Context,
    url: str = typer.Option(
        DEFAULT_GATEWAY_URL,
        "--url",
        "-u",
        envvar="YUNSHU_GATEWAY_URL",
        help="Gateway URL (forwarded to all subcommands that talk to the server).",
    ),
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Machine-readable JSON on stdout (for agents/scripts). Exit code signals "
        "success (0) or failure (non-zero).",
    ),
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        is_eager=True,
        callback=_print_version,
        help="Print the Yunshu version and exit.",
    ),
) -> None:
    """Top-level options shared by every subcommand."""
    from ._output import set_json_mode

    ctx.ensure_object(dict)
    ctx.obj["url"] = url
    ctx.obj["json"] = json_out
    set_json_mode(json_out)
    # Only propagate to the env var when the URL was EXPLICITLY supplied
    # (via --url or YUNSHU_GATEWAY_URL); the option default
    # `DEFAULT_GATEWAY_URL` would otherwise force every leaf subcommand
    # into "talk to live gateway" mode even when the user expected the
    # local fallback (e.g. `yunshu model list --dir /path` should scan
    # disk, not query http://localhost:8000).
    try:
        # Compare by name: typer >= 0.27 vendors its own click, so the enum class
        # differs from click.core.ParameterSource.
        src = ctx.get_parameter_source("url")
        if getattr(src, "name", None) in ("COMMANDLINE", "ENVIRONMENT"):
            os.environ["YUNSHU_GATEWAY_URL"] = url
    except Exception:
        # Click/Typer version without parameter-source introspection —
        # fall back to setting unconditionally (legacy behavior).
        os.environ["YUNSHU_GATEWAY_URL"] = url


from .benchmark import bench_app
from .chat import chat_app
from .config import config_app
from .diagnose import diagnose_app
from .doctor import doctor
from .eval import eval_app
from .integrations import launch_app
from .model import model_app, pull
from .serve import serve_app
from .service import service_app
from .status import status_app

_START = "Get started"
_SERVER = "Server and models"
_TOOLS = "Evaluate and diagnose"
_INFER = "Inference (calls a running server)"

app.command("doctor", rich_help_panel=_START)(doctor)
app.command("pull", rich_help_panel=_START)(pull)
app.add_typer(serve_app, name="serve", rich_help_panel=_START)
app.add_typer(service_app, name="service", rich_help_panel=_START)
app.add_typer(chat_app, name="chat", rich_help_panel=_START)
app.add_typer(model_app, name="model", rich_help_panel=_SERVER)
app.add_typer(config_app, name="config", rich_help_panel=_SERVER)
app.add_typer(status_app, name="status", rich_help_panel=_SERVER)
app.add_typer(launch_app, name="launch", rich_help_panel=_SERVER)
app.add_typer(eval_app, name="eval", rich_help_panel=_TOOLS)
app.add_typer(bench_app, name="bench", rich_help_panel=_TOOLS)
app.add_typer(diagnose_app, name="diagnose", rich_help_panel=_TOOLS)

# Top-level single-shot inference commands (complete/embed/tokenize/rerank/transcribe/
# speak/ocr/image) — the agent-facing surface, all JSON-capable + non-interactive.
from .infer import register as register_infer

_n = len(app.registered_commands)
register_infer(app)
for _cmd in app.registered_commands[_n:]:
    _cmd.rich_help_panel = _INFER


def main() -> None:
    """Console-script entry point. In --json mode, wrap the Typer app so Click usage errors
    (missing/invalid arguments, unknown options) are reported as a JSON error object on the
    real stdout with a non-zero exit — an agent parsing stdout always gets JSON, never an
    empty stream. The normal (non-JSON) path runs the app unchanged."""
    import sys

    if "--json" not in sys.argv:
        app()
        return

    import json as _json

    try:  # typer >= 0.27 vendors its own click; older releases use the real one
        from typer._click import exceptions as click_exc
    except ImportError:
        from click import exceptions as click_exc

    try:
        rv = app(standalone_mode=False)
    except click_exc.ClickException as e:
        # sys.__stdout__ (not sys.stdout, which --json has swapped to stderr) is the real
        # stdout — the group callback that swaps it runs before subcommand arg validation.
        print(_json.dumps({"error": e.format_message()}), file=sys.__stdout__)
        raise SystemExit(e.exit_code or 2) from None
    except click_exc.Abort:
        raise SystemExit(1) from None
    # typer.Exit(code) from a command surfaces as the return value under standalone_mode=False
    if isinstance(rv, int) and rv != 0:
        raise SystemExit(rv)
