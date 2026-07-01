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

    from ._output import auth_headers, emit, fail, is_json

    _hdr = auth_headers()

    # Health
    try:
        resp = httpx.get(f"{url}/health", timeout=5)
        healthy = resp.status_code == 200
    except httpx.ConnectError:
        fail(f"Cannot connect to {url} — start the server with `yunshu serve`.", code=1)

    status_color = "green" if healthy else "red"
    status_text = "Healthy" if healthy else "Unhealthy"

    # System stats
    sys_data = {}
    try:
        resp = httpx.get(f"{url}/api/v1/gw/monitoring/system", headers=_hdr, timeout=5)
        if resp.status_code == 200:
            sys_data = resp.json()
    except Exception:
        logger.debug("failed to fetch system stats", exc_info=True)

    # Engine stats
    eng_data = {}
    try:
        resp = httpx.get(f"{url}/api/v1/gw/monitoring/engine", headers=_hdr, timeout=5)
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

    if is_json():
        emit(
            {
                "healthy": healthy,
                "url": url,
                "system": sys_data,
                "engine": eng_data,
                "models": [m.get("id") for m in models_data],
            }
        )
        return

    # Summary panel
    lines = [f"Status: [{status_color}]{status_text}[/]"]
    lines.append(f"URL: {url}")

    if sys_data:
        # Keys match /api/v1/gw/monitoring/system: gpu.total_uma_bytes (unified memory),
        # cpu.percent (nested), mlx_version lives under gpu.
        gpu = sys_data.get("gpu", {})
        if gpu:
            lines.append(
                f"GPU: {_fmt(gpu.get('active_bytes', 0))} / {_fmt(gpu.get('total_uma_bytes', 0))} ({gpu.get('utilization_pct', 0):.0f}%)"
            )
            if gpu.get("mlx_version"):
                lines.append(f"MLX: {gpu['mlx_version']}")
        cpu = sys_data.get("cpu", {})
        if cpu.get("percent") is not None:
            lines.append(f"CPU: {cpu['percent']:.1f}%")
        mem = sys_data.get("memory", {})
        if mem.get("used_bytes") is not None:
            lines.append(
                f"RAM: {_fmt(mem.get('used_bytes', 0))} / {_fmt(mem.get('total_bytes', 0))} ({mem.get('percent', 0):.0f}%)"
            )

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
