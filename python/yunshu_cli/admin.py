"""Yunshu CLI — admin subcommand.

Model management, API key management, and mesh status.
"""
from __future__ import annotations

import json
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()
admin_app = typer.Typer(help="Admin operations.", no_args_is_help=True)

DEFAULT_URL = "http://localhost:8000"


# ── Models ──

models_app = typer.Typer(help="Model management.", no_args_is_help=True)
admin_app.add_typer(models_app, name="models")


@models_app.command("list")
def models_list(
    url: str = typer.Option(DEFAULT_URL, "--url", "-u"),
):
    """List available models."""
    import httpx
    try:
        resp = httpx.get(f"{url}/v1/models", timeout=5)
        models = resp.json().get("data", [])
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1)

    if not models:
        console.print("[dim]No models available.[/]")
        return

    table = Table(title="Models")
    table.add_column("ID", style="cyan")
    table.add_column("Owned By", style="dim")
    table.add_column("Object")
    for m in models:
        table.add_row(m.get("id", ""), m.get("owned_by", ""), m.get("object", ""))
    console.print(table)


@models_app.command("discover")
def models_discover(
    url: str = typer.Option(DEFAULT_URL, "--url", "-u"),
):
    """Discover models on disk."""
    import httpx
    try:
        resp = httpx.get(f"{url}/api/v1/admin/models/discover", timeout=5)
        data = resp.json()
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1)

    models = data.get("models", data) if isinstance(data, dict) else data
    if isinstance(models, list):
        for m in models:
            console.print(f"  [cyan]{m}[/]")
    elif isinstance(models, dict):
        table = Table(title="Discovered Models")
        table.add_column("ID", style="cyan")
        table.add_column("Type")
        table.add_column("Engine")
        for mid, info in models.items():
            if isinstance(info, dict):
                table.add_row(mid, info.get("model_type", ""), info.get("engine_type", ""))
            else:
                table.add_row(mid, str(info), "")
        console.print(table)


@models_app.command("load")
def models_load(
    model_id: str = typer.Argument(help="Model ID to load"),
    url: str = typer.Option(DEFAULT_URL, "--url", "-u"),
):
    """Load a model into memory."""
    import httpx
    try:
        resp = httpx.post(f"{url}/api/v1/admin/models/load", json={"model_id": model_id}, timeout=120)
        console.print(f"[green]{resp.json().get('message', resp.text)}[/]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1)


@models_app.command("unload")
def models_unload(
    model_id: str = typer.Argument(help="Model ID to unload"),
    url: str = typer.Option(DEFAULT_URL, "--url", "-u"),
):
    """Unload a model from memory."""
    import httpx
    try:
        resp = httpx.post(f"{url}/api/v1/admin/models/unload", json={"model_id": model_id}, timeout=30)
        console.print(f"[green]{resp.json().get('message', resp.text)}[/]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1)


# ── Keys ──

keys_app = typer.Typer(help="API key management.", no_args_is_help=True)
admin_app.add_typer(keys_app, name="keys")


@keys_app.command("list")
def keys_list(
    url: str = typer.Option(DEFAULT_URL, "--url", "-u"),
):
    """List API keys."""
    import httpx
    try:
        resp = httpx.get(f"{url}/api/v1/admin/keys", timeout=5)
        keys = resp.json()
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1)

    if isinstance(keys, list):
        for k in keys:
            name = k.get("name", k.get("id", ""))
            role = k.get("role", "")
            prefix = k.get("key_prefix", k.get("id", "")[:8])
            console.print(f"  {prefix}... [{role}] {name}")
    elif isinstance(keys, dict):
        items = keys.get("keys", keys.get("data", []))
        for k in items:
            console.print(f"  {k}")


@keys_app.command("create")
def keys_create(
    name: str = typer.Argument(help="Key name"),
    role: str = typer.Option("user", "--role", "-r", help="Role: admin, developer, user"),
    url: str = typer.Option(DEFAULT_URL, "--url", "-u"),
):
    """Create a new API key."""
    import httpx
    try:
        resp = httpx.post(f"{url}/api/v1/admin/keys", json={"name": name, "role": role}, timeout=5)
        data = resp.json()
        key = data.get("key", data.get("token", ""))
        if key:
            console.print(f"[green]Key created:[/] {key}")
        else:
            console.print(f"[yellow]{data}[/]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1)


@keys_app.command("revoke")
def keys_revoke(
    key_id: str = typer.Argument(help="Key ID to revoke"),
    url: str = typer.Option(DEFAULT_URL, "--url", "-u"),
):
    """Revoke an API key."""
    import httpx
    try:
        resp = httpx.delete(f"{url}/api/v1/admin/keys/{key_id}", timeout=5)
        console.print(f"[green]{resp.json().get('message', 'Key revoked')}[/]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1)


# ── Mesh ──

mesh_app = typer.Typer(help="Mesh cluster status.", no_args_is_help=True)
admin_app.add_typer(mesh_app, name="mesh")


@mesh_app.command("status")
def mesh_status(
    url: str = typer.Option(DEFAULT_URL, "--url", "-u"),
):
    """Show mesh cluster status."""
    import httpx
    try:
        resp = httpx.get(f"{url}/api/v1/mesh/status", timeout=5)
        data = resp.json()
    except Exception as e:
        console.print(f"[yellow]Mesh not available: {e}[/]")
        raise typer.Exit(1)

    distributed = data.get("distributed", False)
    topology = data.get("topology", "single")
    nodes = data.get("nodes", [])

    console.print(Panel(
        f"Distributed: {'[green]Yes[/]' if distributed else '[dim]No[/]'}\n"
        f"Topology: {topology}\n"
        f"Nodes: {len(nodes)}",
        title="Mesh Status",
    ))

    if nodes:
        table = Table(title="Nodes")
        table.add_column("ID", style="cyan")
        table.add_column("Host")
        table.add_column("IP")
        table.add_column("State")
        table.add_column("Chip")
        table.add_column("RAM")
        for n in nodes:
            caps = n.get("capabilities", {})
            table.add_row(
                n.get("node_id", "")[:12],
                n.get("hostname", ""),
                n.get("ip", ""),
                n.get("state", ""),
                caps.get("chip", ""),
                f"{caps.get('total_memory_gb', 0):.0f} GB",
            )
        console.print(table)


# ── Config ──

@admin_app.command("config")
def admin_config(
    url: str = typer.Option(DEFAULT_URL, "--url", "-u"),
    key: str = typer.Option(None, "--key", "-k", help="Get specific config key"),
    set_value: str = typer.Option(None, "--set", "-s", help="Set config key=value"),
):
    """View or edit server configuration."""
    import httpx

    if set_value:
        k, _, v = set_value.partition("=")
        if not k or not v:
            console.print("[red]Format: --set key=value[/]")
            raise typer.Exit(1)
        try:
            try:
                num_v = int(v)
            except ValueError:
                try:
                    num_v = float(v)
                except ValueError:
                    num_v = v
            resp = httpx.patch(f"{url}/api/v1/admin/config/engine", json={k: num_v}, timeout=5)
            console.print(f"[green]{resp.json().get('message', 'Updated')}[/]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")
        return

    try:
        resp = httpx.get(f"{url}/api/v1/admin/config/engine", timeout=5)
        data = resp.json()
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1)

    if key:
        console.print(json.dumps(data.get(key, "not found"), indent=2))
    else:
        console.print(json.dumps(data, indent=2))
