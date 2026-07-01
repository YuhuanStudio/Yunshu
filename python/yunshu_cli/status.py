"""Yunshu CLI — status subcommand.

Quick server health check and stats overview.
"""

from __future__ import annotations

import logging

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
# No `no_args_is_help` — status has a single invoke_without_command callback, so bare
# `yunshu status` must RUN it (show status), not print the help screen.
status_app = typer.Typer(help="Server status.")

logger = logging.getLogger(__name__)


@status_app.callback(invoke_without_command=True)
def status(
    url: str = typer.Option("http://localhost:8000", "--url", "-u", help="Server URL."),
):
    """Show Yunshu server status."""
    import httpx

    # Health
    try:
        resp = httpx.get(f"{url}/health", timeout=5)
        healthy = resp.status_code == 200
        resp.json() if healthy else {}
    except httpx.ConnectError:
        console.print(
            Panel(
                f"[red]Cannot connect to {url}[/]\n\n"
                "Start the server with: [bold]yunshu serve[/]",
                title="Server Status",
                border_style="red",
            )
        )
        raise typer.Exit(1) from None

    status_color = "green" if healthy else "red"
    status_text = "Healthy" if healthy else "Unhealthy"

    # System stats
    sys_data = {}
    try:
        resp = httpx.get(f"{url}/api/v1/monitoring/system", timeout=5)
        if resp.status_code == 200:
            sys_data = resp.json()
    except Exception:
        logger.debug("failed to fetch system stats", exc_info=True)

    # Engine stats
    eng_data = {}
    try:
        resp = httpx.get(f"{url}/api/v1/monitoring/engine", timeout=5)
        if resp.status_code == 200:
            eng_data = resp.json()
    except Exception:
        logger.debug("failed to fetch engine stats", exc_info=True)

    # Models
    models_data = []
    try:
        resp = httpx.get(f"{url}/v1/models", timeout=5)
        if resp.status_code == 200:
            models_data = resp.json().get("data", [])
    except Exception:
        logger.debug("failed to fetch models", exc_info=True)

    # Summary panel
    lines = [f"Status: [{status_color}]{status_text}[/]"]
    lines.append(f"URL: {url}")

    if sys_data:
        gpu = sys_data.get("gpu", {})
        if gpu:
            lines.append(
                f"GPU: {_fmt(gpu.get('active_bytes', 0))} / {_fmt(gpu.get('total_bytes', 0))} ({gpu.get('utilization_pct', 0):.0f}%)"
            )
        if sys_data.get("cpu_percent"):
            lines.append(f"CPU: {sys_data['cpu_percent']:.1f}%")
        if sys_data.get("mlx_version"):
            lines.append(f"MLX: {sys_data['mlx_version']}")

    if eng_data:
        lines.append(f"Requests: {eng_data.get('requests_processed', 0)}")
        lines.append(f"Active: {eng_data.get('active_requests', 0)}")

    console.print(
        Panel(
            "\n".join(lines), title="[bold]Yunshu Server[/]", border_style=status_color
        )
    )

    # Models table
    if models_data:
        table = Table(title="Models")
        table.add_column("ID", style="cyan")
        table.add_column("Status", justify="center")

        for m in models_data:
            mid = m.get("id", "unknown")
            loaded = m.get("loaded", False)
            table.add_row(mid, "[green]Loaded[/]" if loaded else "[dim]Registered[/]")

        console.print(table)


def _fmt(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"
