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
    ctx: typer.Context,
    all_: bool = typer.Option(
        False, "--all", "-a", help="Include experimental and internal settings."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON."),
    file: str | None = typer.Option(
        None, "--config", "-c", help="Also read this TOML config file."
    ),
):
    """Show every setting's effective value and source (cli/env/file/default)."""
    if ctx.invoked_subcommand is not None:
        return
    if file:
        settings.set_override(
            "YUNSHU_CONFIG", os.path.abspath(os.path.expanduser(file))
        )
    include = ("stable", "experimental", "internal") if all_ else ("stable",)
    rows = settings.effective(include)
    warnings = settings.validate(warn=False) if _valid() else []
    if settings.get("YUNSHU_WEB_SEARCH_PROVIDER") != "none":
        from yunshu_gateway.server_tools.search import PRIVACY_NOTICE

        warnings.append(PRIVACY_NOTICE)
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


def _name(key: str) -> str:
    name = settings._normalize_key(key)
    if name not in settings.REGISTRY:
        close = settings.close_matches(name)
        hint = f" (did you mean {', '.join(close)}?)" if close else ""
        console.print(f"[red]Error:[/] unknown setting {key!r}{hint}")
        raise typer.Exit(2)
    return name


@config_app.command("set")
def config_set(
    key: str = typer.Argument(help="Setting name, with or without YUNSHU_."),
    value: str = typer.Argument(help="New value."),
    file: str | None = typer.Option(
        None, "--config", "-c", help="Write this TOML file instead of the user file."
    ),
):
    """Save a setting in the user config file (~/.yunshu/config.toml).

    Example: `yunshu config set models_dir /Volumes/Models` moves where
    `yunshu pull` downloads and where the server looks for models.
    """
    name = _name(key)
    if settings.REGISTRY[name].type == "path":
        value = os.path.abspath(os.path.expanduser(value))
    try:
        target = settings.write_config_value(name, value, file)
    except settings.SettingError as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(2) from None
    console.print(f"{name} = {value}  [dim]({target})[/]")
    if name in os.environ:
        console.print(
            f"[yellow]Note:[/] {name} is also set in the environment, which takes "
            "precedence over the config file."
        )


@config_app.command("unset")
def config_unset(
    key: str = typer.Argument(help="Setting name, with or without YUNSHU_."),
    file: str | None = typer.Option(
        None, "--config", "-c", help="Edit this TOML file instead of the user file."
    ),
):
    """Remove a setting from the user config file (back to its default)."""
    name = _name(key)
    target = settings.write_config_value(name, None, file)
    console.print(f"{name} removed  [dim]({target})[/]")


@config_app.command("path")
def config_path():
    """Print the user config file location."""
    typer.echo(str(settings.user_config_path()))
