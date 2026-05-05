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

import typer
from rich.console import Console

console = Console()
app = typer.Typer(
    name="yunshu",
    help="Production-grade MLX inference platform for Apple Silicon.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

from .chat import chat_app
from .config import config_app
from .diagnose import diagnose_app
from .eval import eval_app
from .integrations import launch_app
from .model import model_app
from .benchmark import bench_app
from .serve import serve_app
from .status import status_app
from .admin import admin_app

app.add_typer(serve_app, name="serve")
app.add_typer(chat_app, name="chat")
app.add_typer(model_app, name="model")
app.add_typer(status_app, name="status")
app.add_typer(config_app, name="config")
app.add_typer(launch_app, name="launch")
app.add_typer(eval_app, name="eval")
app.add_typer(bench_app, name="bench")
app.add_typer(diagnose_app, name="diagnose")
app.add_typer(admin_app, name="admin")
