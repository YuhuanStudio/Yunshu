"""Yunshu CLI — admin subcommand.

Model management, API key management, and mesh status.
"""

from __future__ import annotations

import json
import os

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
admin_app = typer.Typer(help="Admin operations.", no_args_is_help=True)


def _default_url() -> str:
    """Resolve gateway URL from env (set by top-level --url) or fallback."""
    return os.environ.get("YUNSHU_GATEWAY_URL", "http://localhost:8000")


# Module-level constant for backwards compatibility (re-evaluated at call time
# via os.environ; callers that import DEFAULT_URL at module load get a
# reasonable default). Each typer.Option below uses _default_url() so the
# top-level --url propagates correctly even after this module is imported.
DEFAULT_URL = _default_url()


@admin_app.callback()
def _admin_options(
    ctx: typer.Context,
    url: str = typer.Option(
        None,
        "--url",
        "-u",
        help="Gateway URL (overrides global --url for admin commands).",
    ),
) -> None:
    """Admin-group options. Accepts --url between `admin` and the subcommand."""
    if url is not None:
        os.environ["YUNSHU_GATEWAY_URL"] = url
        if ctx.obj is None:
            ctx.obj = {}
        ctx.obj["url"] = url


# ── Models ──

models_app = typer.Typer(help="Model management.", no_args_is_help=True)
admin_app.add_typer(models_app, name="models")


@models_app.command("list")
def models_list(
    url: str = typer.Option(_default_url(), "--url", "-u", envvar="YUNSHU_GATEWAY_URL"),
):
    """List available models."""
    import httpx

    try:
        resp = httpx.get(f"{url}/v1/models", timeout=5)
        models = resp.json().get("data", [])
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1) from e

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
    url: str = typer.Option(_default_url(), "--url", "-u", envvar="YUNSHU_GATEWAY_URL"),
):
    """Discover models on disk."""
    import httpx

    try:
        resp = httpx.get(f"{url}/api/v1/admin/models/discover", timeout=5)
        data = resp.json()
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1) from e

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
                table.add_row(
                    mid, info.get("model_type", ""), info.get("engine_type", "")
                )
            else:
                table.add_row(mid, str(info), "")
        console.print(table)


@models_app.command("load")
def models_load(
    model_id: str = typer.Argument(help="Model ID to load"),
    url: str = typer.Option(_default_url(), "--url", "-u", envvar="YUNSHU_GATEWAY_URL"),
):
    """Load a model into memory."""
    import httpx

    try:
        resp = httpx.post(
            f"{url}/api/v1/admin/models/load", json={"model_id": model_id}, timeout=120
        )
        console.print(f"[green]{resp.json().get('message', resp.text)}[/]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1) from e


@models_app.command("unload")
def models_unload(
    model_id: str = typer.Argument(help="Model ID to unload"),
    url: str = typer.Option(_default_url(), "--url", "-u", envvar="YUNSHU_GATEWAY_URL"),
):
    """Unload a model from memory."""
    import httpx

    try:
        resp = httpx.post(
            f"{url}/api/v1/admin/models/unload", json={"model_id": model_id}, timeout=30
        )
        console.print(f"[green]{resp.json().get('message', resp.text)}[/]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1) from e


# ── Keys ──

keys_app = typer.Typer(help="API key management.", no_args_is_help=True)
admin_app.add_typer(keys_app, name="keys")


@keys_app.command("list")
def keys_list(
    url: str = typer.Option(_default_url(), "--url", "-u", envvar="YUNSHU_GATEWAY_URL"),
):
    """List API keys."""
    import httpx

    try:
        resp = httpx.get(f"{url}/api/v1/admin/keys", timeout=5)
        keys = resp.json()
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1) from e

    if isinstance(keys, list):
        for k in keys:
            name = k.get("name") or k.get("id") or ""
            role = k.get("role", "")
            prefix = k.get("key_prefix", k.get("id", "")[:8])
            console.print(f"  {prefix}... [{role}] {name}")
    elif isinstance(keys, dict):
        # dict.get returns explicit None even when default is given. Use `or`
        # chain so falsy values (None, "") fall through to the empty list.
        items = keys.get("keys") or keys.get("data") or []
        for k in items:
            console.print(f"  {k}")


@keys_app.command("create")
def keys_create(
    name: str = typer.Argument(help="Key name"),
    role: str = typer.Option(
        "user", "--role", "-r", help="Role: admin, developer, user"
    ),
    url: str = typer.Option(_default_url(), "--url", "-u", envvar="YUNSHU_GATEWAY_URL"),
):
    """Create a new API key."""
    import httpx

    try:
        resp = httpx.post(
            f"{url}/api/v1/admin/keys", json={"name": name, "role": role}, timeout=5
        )
        data = resp.json()
        key = data.get("key") or data.get("token") or ""
        if key:
            console.print(f"[green]Key created:[/] {key}")
        else:
            console.print(f"[yellow]{data}[/]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1) from e


@keys_app.command("revoke")
def keys_revoke(
    key_id: str = typer.Argument(help="Key ID to revoke"),
    url: str = typer.Option(_default_url(), "--url", "-u", envvar="YUNSHU_GATEWAY_URL"),
):
    """Revoke an API key."""
    import httpx

    try:
        resp = httpx.delete(f"{url}/api/v1/admin/keys/{key_id}", timeout=5)
        console.print(f"[green]{resp.json().get('message', 'Key revoked')}[/]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1) from e


# ── Mesh ──

mesh_app = typer.Typer(help="Mesh cluster status.", no_args_is_help=True)
admin_app.add_typer(mesh_app, name="mesh")


@mesh_app.command("status")
def mesh_status(
    url: str = typer.Option(_default_url(), "--url", "-u", envvar="YUNSHU_GATEWAY_URL"),
):
    """Show mesh cluster status."""
    import httpx

    try:
        resp = httpx.get(f"{url}/api/v1/mesh/status", timeout=5)
        data = resp.json()
    except Exception as e:
        console.print(f"[yellow]Mesh not available: {e}[/]")
        raise typer.Exit(1) from e

    distributed = data.get("distributed", False)
    topology = data.get("topology", "single")
    nodes = data.get("nodes", [])

    console.print(
        Panel(
            f"Distributed: {'[green]Yes[/]' if distributed else '[dim]No[/]'}\n"
            f"Topology: {topology}\n"
            f"Nodes: {len(nodes)}",
            title="Mesh Status",
        )
    )

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
    url: str = typer.Option(_default_url(), "--url", "-u", envvar="YUNSHU_GATEWAY_URL"),
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
            resp = httpx.patch(
                f"{url}/api/v1/admin/config/engine", json={k: num_v}, timeout=5
            )
            console.print(f"[green]{resp.json().get('message', 'Updated')}[/]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")
        return

    try:
        resp = httpx.get(f"{url}/api/v1/admin/config/engine", timeout=5)
        data = resp.json()
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise typer.Exit(1) from e

    if key:
        console.print(json.dumps(data.get(key, "not found"), indent=2))
    else:
        console.print(json.dumps(data, indent=2))


# ── Tenants ──

tenants_app = typer.Typer(help="Tenant management.", no_args_is_help=True)
admin_app.add_typer(tenants_app, name="tenants")


def _admin_headers() -> dict[str, str]:
    """Return Bearer auth header from env (YUNSHU_AUTH_TOKEN or YUNSHU_API_KEY)."""
    tok = os.environ.get("YUNSHU_AUTH_TOKEN") or os.environ.get("YUNSHU_API_KEY")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


@tenants_app.command("create")
def tenants_create(
    name: str = typer.Argument(help="Tenant name"),
    tier: str = typer.Option(
        "FREE", "--tier", "-t", help="Tier: FREE / PRO / ENTERPRISE"
    ),
    rpm: int | None = typer.Option(None, "--rpm", help="Requests per minute override"),
    tpm: int | None = typer.Option(None, "--tpm", help="Tokens per minute override"),
    url: str = typer.Option(_default_url(), "--url", "-u", envvar="YUNSHU_GATEWAY_URL"),
):
    """Create a new tenant. Prints the api_key (returned ONCE — save it)."""
    import httpx

    body: dict = {"name": name, "tier": tier.upper()}
    if rpm is not None:
        body["requests_per_minute"] = rpm
    if tpm is not None:
        body["tokens_per_minute"] = tpm
    try:
        resp = httpx.post(
            f"{url}/api/v1/admin/tenants",
            json=body,
            headers=_admin_headers(),
            timeout=10,
        )
        if resp.status_code != 200:
            console.print(f"[red]Error {resp.status_code}: {resp.text}[/]")
            raise typer.Exit(1)
        data = resp.json()
    except httpx.HTTPError as e:
        console.print(f"[red]Network error: {e}[/]")
        raise typer.Exit(1) from e

    console.print(
        Panel.fit(
            f"[green]Tenant created:[/]\n"
            f"  tenant_id: [cyan]{data['tenant_id']}[/]\n"
            f"  name: {data['name']}\n"
            f"  tier: {data['tier']}\n"
            f"  api_key: [yellow]{data['api_key']}[/]  ← save this; cannot recover\n"
            f"  quota.rpm: {data['quota']['requests_per_minute']}\n"
            f"  quota.tpm: {data['quota']['tokens_per_minute']}\n"
            f"  quota.max_concurrent: {data['quota']['max_concurrent']}"
        )
    )


@tenants_app.command("list")
def tenants_list(
    url: str = typer.Option(_default_url(), "--url", "-u", envvar="YUNSHU_GATEWAY_URL"),
):
    """List all tenants (api_keys never shown)."""
    import httpx

    try:
        resp = httpx.get(
            f"{url}/api/v1/admin/tenants", headers=_admin_headers(), timeout=5
        )
        if resp.status_code == 503:
            console.print(
                f"[yellow]{resp.json().get('detail', 'Tenant management not enabled')}[/]"
            )
            return
        data = resp.json()
    except httpx.HTTPError as e:
        console.print(f"[red]Network error: {e}[/]")
        raise typer.Exit(1) from e

    tenants = data.get("tenants", [])
    if not tenants:
        console.print("[dim]No tenants.[/]")
        return
    table = Table(title="Tenants")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Tier")
    table.add_column("RPM")
    table.add_column("Active")
    for t in tenants:
        # list_tenants returns {id, name, tier, active, quota_rpm} — match exactly.
        table.add_row(
            t.get("id", ""),
            t.get("name", ""),
            t.get("tier", ""),
            str(t.get("quota_rpm", "")),
            "✓" if t.get("active", True) else "✗",
        )
    console.print(table)


@tenants_app.command("delete")
def tenants_delete(
    tenant_id: str = typer.Argument(help="Tenant ID (e.g. tn-abc123)"),
    url: str = typer.Option(_default_url(), "--url", "-u", envvar="YUNSHU_GATEWAY_URL"),
):
    """Delete a tenant."""
    import httpx

    try:
        resp = httpx.delete(
            f"{url}/api/v1/admin/tenants/{tenant_id}",
            headers=_admin_headers(),
            timeout=5,
        )
        if resp.status_code == 404:
            console.print(f"[red]Tenant '{tenant_id}' not found.[/]")
            raise typer.Exit(1)
        if resp.status_code != 200:
            console.print(f"[red]Error {resp.status_code}: {resp.text}[/]")
            raise typer.Exit(1)
        console.print(f"[green]Deleted tenant {tenant_id}[/]")
    except httpx.HTTPError as e:
        console.print(f"[red]Network error: {e}[/]")
        raise typer.Exit(1) from e


@tenants_app.command("deactivate")
def tenants_deactivate(
    tenant_id: str = typer.Argument(help="Tenant ID (e.g. tn-abc123)"),
    url: str = typer.Option(_default_url(), "--url", "-u", envvar="YUNSHU_GATEWAY_URL"),
):
    """Soft-disable a tenant (api_key still on disk but auth returns None)."""
    import httpx

    try:
        resp = httpx.post(
            f"{url}/api/v1/admin/tenants/{tenant_id}/deactivate",
            headers=_admin_headers(),
            timeout=5,
        )
        if resp.status_code == 404:
            console.print(f"[red]Tenant '{tenant_id}' not found.[/]")
            raise typer.Exit(1)
        if resp.status_code != 200:
            console.print(f"[red]Error {resp.status_code}: {resp.text}[/]")
            raise typer.Exit(1)
        console.print(f"[yellow]Deactivated tenant {tenant_id}[/]")
    except httpx.HTTPError as e:
        console.print(f"[red]Network error: {e}[/]")
        raise typer.Exit(1) from e


# ── Audit log ──


@admin_app.command("audit-log")
def admin_audit_log(
    limit: int = typer.Option(
        50, "--limit", "-n", help="Max events to display (1-1000)"
    ),
    op: str | None = typer.Option(None, "--op", help="Filter by operation name"),
    url: str = typer.Option(_default_url(), "--url", "-u", envvar="YUNSHU_GATEWAY_URL"),
):
    """Show recent audit events from the gateway's ring buffer."""
    import httpx

    params: dict = {"limit": limit}
    if op:
        params["op"] = op
    try:
        resp = httpx.get(
            f"{url}/api/v1/admin/audit-log",
            params=params,
            headers=_admin_headers(),
            timeout=5,
        )
        data = resp.json()
    except httpx.HTTPError as e:
        console.print(f"[red]Network error: {e}[/]")
        raise typer.Exit(1) from e

    events = data.get("events", [])
    if not events:
        console.print("[dim]No audit events.[/]")
        return
    table = Table(
        title=f"Audit Events (showing {len(events)} of {data.get('total_in_buffer', 0)})"
    )
    table.add_column("Timestamp", style="dim")
    table.add_column("Op", style="cyan")
    table.add_column("Actor")
    table.add_column("Resource")
    table.add_column("Result")
    for e in events:
        result = e.get("result", "")
        result_color = "[green]" if result == "success" else "[red]"
        table.add_row(
            e.get("ts", "")[:19],
            e.get("op", ""),
            e.get("actor", ""),
            e.get("resource", ""),
            f"{result_color}{result}[/]",
        )
    console.print(table)
