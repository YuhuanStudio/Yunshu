"""Yunshu CLI — config subcommand: effective settings and where each came from."""

from __future__ import annotations

import json
import os

import typer
from rich.console import Console
from rich.table import Table

from yunshu_engine import settings

from ._output import emit, fail, is_json

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
    try:
        rows = settings.effective(include)
    except settings.SettingError as exc:
        fail(f"{exc}. Fix the value in your environment or config file.", code=2)
    warnings = settings.validate(warn=False) if _valid() else []
    if settings.get("YUNSHU_WEB_SEARCH_PROVIDER") != "none":
        from yunshu_gateway.server_tools.search import PRIVACY_NOTICE

        warnings.append(PRIVACY_NOTICE)
    from yunshu_gateway.server_tools.metasearch import read_health

    health = read_health()
    data = {"settings": rows, "warnings": warnings, "web_search_health": health}
    if is_json():
        emit(data)
        return
    if as_json:
        typer.echo(json.dumps(data, default=str))
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
    if health:
        console.print("Web search provider health (last server snapshot):")
        console.print_json(data=health)
    else:
        console.print("Web search provider health: no server observations yet")
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
        fail(f"Unknown setting {key!r}{hint}. See `yunshu config --all`.", code=2)
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
    except (settings.SettingError, OSError) as exc:
        fail(
            f"{exc}. Check the value, path and permissions; see `yunshu config --all`.",
            code=2,
        )
    emit(
        {
            "name": name,
            "value": "***" if settings.REGISTRY[name].secret else value,
            "path": str(target),
        },
        human=lambda: console.print(f"{name} = {value}  [dim]({target})[/]"),
    )
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
    try:
        target = settings.write_config_value(name, None, file)
    except (settings.SettingError, OSError) as exc:
        fail(f"Cannot edit config: {exc}. Check the path and permissions.", code=2)
    emit(
        {"name": name, "removed": True, "path": str(target)},
        human=lambda: console.print(f"{name} removed  [dim]({target})[/]"),
    )


@config_app.command("path")
def config_path():
    """Print the user config file location."""
    path = str(settings.user_config_path())
    emit({"path": path}, human=lambda: typer.echo(path))
