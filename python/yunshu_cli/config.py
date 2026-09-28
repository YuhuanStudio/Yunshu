"""Yunshu CLI — config subcommand: effective settings and where each came from."""

from __future__ import annotations

import json
import os

import typer
from rich.console import Console
from rich.table import Table

from yunshu_engine import settings

console = Console()

config_app = typer.Typer(help="Show effective settings and their sources.")


@config_app.callback(invoke_without_command=True)
def config(
    all_: bool = typer.Option(
        False, "--all", "-a", help="Include experimental and internal settings."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON."),
    file: str | None = typer.Option(
        None, "--config", "-c", help="Also read this TOML config file."
    ),
):
    """Show every setting's effective value and source (cli/env/file/default)."""
    if file:
        settings.set_override(
            "YUNSHU_CONFIG", os.path.abspath(os.path.expanduser(file))
        )
    include = ("stable", "experimental", "internal") if all_ else ("stable",)
    rows = settings.effective(include)
    warnings = settings.validate(warn=False) if _valid() else []
    if as_json:
        typer.echo(json.dumps({"settings": rows, "warnings": warnings}, default=str))
        return
    table = Table(show_lines=False)
    for col in ("Setting", "Value", "Source", "Category"):
        table.add_column(col)
    for r in rows:
        value = "" if r["value"] is None else str(r["value"])
        name = (
            r["name"]
            if r["stability"] == "stable"
            else f"{r['name']} ({r['stability']})"
        )
        source = r["source"] if r["source"] == "default" else f"[bold]{r['source']}[/]"
        table.add_row(name, value, source, r["category"])
    console.print(table)
    for w in warnings:
        console.print(f"[yellow]Warning:[/] {w}")


def _valid() -> bool:
    try:
        settings.validate(warn=False)
        return True
    except settings.SettingError as exc:
        console.print(f"[red]{exc}[/]")
        return False
