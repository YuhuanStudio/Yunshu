"""Yunshu CLI — command-line interface for the Yunshu inference engine.

Inference commands (talk to a running server): complete, embed, tokenize, rerank,
transcribe, speak, ocr, image. Management: serve, chat, model, status, launch, eval,
bench, diagnose. Every command honors the global ``--json`` flag for agent use.
"""

from __future__ import annotations

import os

import typer
from rich.console import Console

console = Console()
app = typer.Typer(
    name="yunshu",
    help="Fast local multimodal (omni) MLX inference engine for Apple Silicon.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


DEFAULT_GATEWAY_URL = "http://localhost:8000"


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
        from click.core import ParameterSource

        src = ctx.get_parameter_source("url")
        if src in (ParameterSource.COMMANDLINE, ParameterSource.ENVIRONMENT):
            os.environ["YUNSHU_GATEWAY_URL"] = url
    except Exception:
        # Click/Typer version without parameter-source introspection —
        # fall back to setting unconditionally (legacy behavior).
        os.environ["YUNSHU_GATEWAY_URL"] = url


from .benchmark import bench_app
from .chat import chat_app
from .diagnose import diagnose_app
from .eval import eval_app
from .integrations import launch_app
from .model import model_app
from .serve import serve_app
from .status import status_app

app.add_typer(serve_app, name="serve")
app.add_typer(chat_app, name="chat")
app.add_typer(model_app, name="model")
app.add_typer(status_app, name="status")
app.add_typer(launch_app, name="launch")
app.add_typer(eval_app, name="eval")
app.add_typer(bench_app, name="bench")
app.add_typer(diagnose_app, name="diagnose")

# Top-level single-shot inference commands (complete/embed/tokenize/rerank/transcribe/
# speak/ocr/image) — the agent-facing surface, all JSON-capable + non-interactive.
from .infer import register as register_infer

register_infer(app)


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

    import click

    try:
        rv = app(standalone_mode=False)
    except click.exceptions.ClickException as e:
        # sys.__stdout__ (not sys.stdout, which --json has swapped to stderr) is the real
        # stdout — the group callback that swaps it runs before subcommand arg validation.
        print(_json.dumps({"error": e.format_message()}), file=sys.__stdout__)
        raise SystemExit(e.exit_code or 2) from None
    except click.exceptions.Abort:
        raise SystemExit(1) from None
    # typer.Exit(code) from a command surfaces as the return value under standalone_mode=False
    if isinstance(rv, int) and rv != 0:
        raise SystemExit(rv)
