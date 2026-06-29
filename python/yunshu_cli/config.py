"""Yunshu CLI — config subcommand.

View and edit server configuration.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

console = Console()
config_app = typer.Typer(help="Server configuration.", no_args_is_help=True)


@config_app.callback(invoke_without_command=True)
def show_config(
    url: str = typer.Option("http://localhost:8000", "--url", "-u", help="Server URL."),
):
    """Show current server configuration."""
    import httpx

    try:
        resp = httpx.get(f"{url}/api/v1/admin/config/engine", timeout=5)
    except httpx.ConnectError:
        console.print(
            "[red]Cannot connect to server.[/] Start with: [bold]yunshu serve[/]"
        )
        raise typer.Exit(1) from None

    if resp.status_code != 200:
        console.print(f"[red]Error {resp.status_code}:[/] {resp.text[:200]}")
        raise typer.Exit(1)

    config = resp.json()

    table = Table(title="Engine Configuration", show_lines=True)
    table.add_column("Key", style="bold")
    table.add_column("Value")

    for key, val in sorted(config.items()):
        label = key.replace("_", " ").replace("/", " ").title()
        table.add_row(label, str(val))

    console.print(table)


@config_app.command("set")
def set_config(
    key: str = typer.Argument(help="Configuration key."),
    value: str = typer.Argument(help="New value."),
    url: str = typer.Option("http://localhost:8000", "--url", "-u", help="Server URL."),
):
    """Set a configuration value."""
    import httpx

    num_val: int | float | str
    try:
        num_val = int(value)
    except ValueError:
        try:
            num_val = float(value)
        except ValueError:
            num_val = value

    try:
        resp = httpx.patch(
            f"{url}/api/v1/admin/config/engine",
            json={key: num_val},
            timeout=5,
        )
    except httpx.ConnectError:
        console.print("[red]Cannot connect to server.[/]")
        raise typer.Exit(1) from None

    if resp.status_code == 200:
        console.print(f"[green]✓[/] {key} = {value}")
    else:
        console.print(f"[red]Error {resp.status_code}:[/] {resp.text[:200]}")
        raise typer.Exit(1)
