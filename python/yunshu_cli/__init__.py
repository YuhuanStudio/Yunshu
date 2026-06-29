"""Yunshu CLI — Production command-line interface for Apple Silicon inference.

Subcommands:
  serve    — Start inference server (single or multi-model)
  chat     — Interactive terminal chat with streaming
  model    — Model management (list, download, info, benchmark)
  status   — Quick server health and stats overview
  config   — View/edit server configuration
  bench    — Run benchmarks (roofline, latency, throughput, inference)
  diagnose — System diagnostics (GPU, memory, MLX, models)
"""

from __future__ import annotations

import os

import typer
from rich.console import Console

console = Console()
app = typer.Typer(
    name="yunshu",
    help="Production-grade MLX inference platform for Apple Silicon.",
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
) -> None:
    """Top-level options shared by every subcommand."""
    ctx.ensure_object(dict)
    ctx.obj["url"] = url
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
from .config import config_app
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
app.add_typer(config_app, name="config")
app.add_typer(launch_app, name="launch")
app.add_typer(eval_app, name="eval")
app.add_typer(bench_app, name="bench")
app.add_typer(diagnose_app, name="diagnose")
