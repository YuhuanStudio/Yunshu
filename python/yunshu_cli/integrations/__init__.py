from __future__ import annotations

"""Yunshu CLI — launch subcommand.

Launch external coding tools (Codex, OpenCode, Pi) configured
to use a running Yunshu server.
"""


import contextlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .._output import auth_headers

console = Console()
launch_app = typer.Typer(help="Launch external tools.", no_args_is_help=True)

logger = logging.getLogger(__name__)


# ── Integration Base ──


@dataclass
class Integration:
    name: str
    display_name: str
    install_check: str
    install_hint: str

    def is_installed(self) -> bool:
        return shutil.which(self.install_check) is not None

    def configure(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1"
    ) -> None:
        raise NotImplementedError

    def launch(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1", **kwargs
    ) -> None:
        raise NotImplementedError

    def _write_json_config(self, config_path: Path, updater) -> None:
        existing: dict = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = {}
            backup = config_path.with_suffix(f".{int(time.time())}.bak")
            with contextlib.suppress(OSError):
                shutil.copy2(config_path, backup)

        updater(existing)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


# ── Codex Integration ──


class CodexIntegration(Integration):
    """OpenAI Codex CLI — configures ~/.codex/config.toml."""

    CONFIG_PATH = Path.home() / ".codex" / "config.toml"

    def __init__(self):
        super().__init__(
            name="codex",
            display_name="Codex",
            install_check="codex",
            install_hint="npm install -g @openai/codex",
        )

    def configure(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1"
    ) -> None:
        config_path = self.CONFIG_PATH
        config_path.parent.mkdir(parents=True, exist_ok=True)

        existing = ""
        if config_path.exists():
            with contextlib.suppress(OSError):
                existing = config_path.read_text(encoding="utf-8")

        lines = existing.splitlines()
        new_lines = []
        in_yunshu_section = False

        top_overrides = {
            "model": f'"{model or "default"}"',
            "model_provider": '"yunshu"',
        }

        seen = set()
        in_section = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                in_section = True
                in_yunshu_section = stripped == "[model_providers.yunshu]"

            if not in_section and "=" in stripped:
                key = stripped.split("=")[0].strip()
                if key in top_overrides:
                    new_lines.append(f"{key} = {top_overrides[key]}")
                    seen.add(key)
                    continue

            if in_yunshu_section:
                continue

            new_lines.append(line)

        for key, val in top_overrides.items():
            if key not in seen:
                new_lines.insert(0, f"{key} = {val}")

        new_lines.append("\n[model_providers.yunshu]")
        new_lines.append('name = "Yunshu"')
        new_lines.append(f'base_url = "http://{host}:{port}/v1"')
        new_lines.append('env_key = "YUNSHU_API_KEY"')

        config_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        console.print(f"[green]✓[/] Config updated: {config_path}")

    def launch(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1", **kwargs
    ) -> None:
        self.configure(port, api_key, model, host)
        env = os.environ.copy()
        env["YUNSHU_API_KEY"] = api_key or "yunshu"
        args = ["codex"]
        if model:
            args.extend(["-m", model])
        console.print(f"[bold]Launching[/] Codex with model {model}...")
        os.execvpe("codex", args, env)


# ── OpenCode Integration ──


class OpenCodeIntegration(Integration):
    """OpenCode — configures ~/.config/opencode/opencode.json."""

    CONFIG_PATH = Path.home() / ".config" / "opencode" / "opencode.json"

    def __init__(self):
        super().__init__(
            name="opencode",
            display_name="OpenCode",
            install_check="opencode",
            install_hint="curl -fsSL https://opencode.ai/install | bash",
        )

    def configure(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1"
    ) -> None:
        def updater(config: dict) -> None:
            config.setdefault("provider", {})
            provider_config = {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Yunshu",
                "options": {"baseURL": f"http://{host}:{port}/v1"},
            }
            if api_key:
                provider_config["options"]["apiKey"] = api_key
            if model:
                provider_config["models"] = {
                    model: {
                        "name": model,
                        "modalities": {"input": ["text"], "output": ["text"]},
                    }
                }
            config["provider"]["yunshu"] = provider_config
            if model:
                config["model"] = f"yunshu/{model}"

        self._write_json_config(self.CONFIG_PATH, updater)
        console.print(f"[green]✓[/] Config updated: {self.CONFIG_PATH}")

    def launch(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1", **kwargs
    ) -> None:
        self.configure(port, api_key, model, host)
        console.print(f"[bold]Launching[/] OpenCode with model {model}...")
        os.execvpe("opencode", ["opencode"], os.environ.copy())


# ── Pi Integration ──


class PiIntegration(Integration):
    """Pi coding agent — configures ~/.pi/agent/."""

    MODELS_PATH = Path.home() / ".pi" / "agent" / "models.json"
    SETTINGS_PATH = Path.home() / ".pi" / "agent" / "settings.json"

    def __init__(self):
        super().__init__(
            name="pi",
            display_name="Pi",
            install_check="pi",
            install_hint="npm install -g @mariozechner/pi-coding-agent",
        )

    def configure(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1"
    ) -> None:
        def update_models(config: dict) -> None:
            config.setdefault("providers", {})
            provider_config: dict = {
                "baseUrl": f"http://{host}:{port}/v1",
                "api": "openai-completions",
                "apiKey": api_key or "yunshu",
                "authHeader": True,
            }
            if model:
                provider_config["models"] = [
                    {
                        "id": model,
                        "name": model,
                        "input": ["text"],
                        "cost": {"input": 0, "output": 0},
                    }
                ]
            config["providers"]["yunshu"] = provider_config

        def update_settings(config: dict) -> None:
            config["defaultProvider"] = "yunshu"
            if model:
                config["defaultModel"] = model

        self._write_json_config(self.MODELS_PATH, update_models)
        self._write_json_config(self.SETTINGS_PATH, update_settings)
        console.print(f"[green]✓[/] Config updated: {self.MODELS_PATH}")

    def launch(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1", **kwargs
    ) -> None:
        self.configure(port, api_key, model, host)
        args = ["pi"]
        if model:
            args.extend(["--model", f"yunshu/{model}"])
        console.print(f"[bold]Launching[/] Pi with model {model}...")
        os.execvpe("pi", args, os.environ.copy())


# ── Registry ──

INTEGRATIONS: dict[str, Integration] = {
    "codex": CodexIntegration(),
    "opencode": OpenCodeIntegration(),
    "pi": PiIntegration(),
}


def _resolve_model(url: str) -> str | None:
    import httpx

    try:
        resp = httpx.get(f"{url}/v1/models", headers=auth_headers(), timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            for m in models:
                mid = m.get("id", "")
                if any(
                    k in mid.lower()
                    for k in ("qwen", "llama", "gemma", "mistral", "phi", "deepseek")
                ):
                    return mid
            if models:
                return models[0].get("id")
    except Exception:
        logger.debug("failed to resolve model from server", exc_info=True)
    return None


# ── Commands ──


@launch_app.command("list")
def list_tools():
    """List available tool integrations."""
    from .._output import emit, is_json

    if is_json():
        emit(
            {
                "integrations": [
                    {
                        "name": integ.display_name,
                        "installed": integ.is_installed(),
                        "install_hint": integ.install_hint,
                    }
                    for integ in INTEGRATIONS.values()
                ]
            }
        )
        return

    table = Table(title="Available Integrations")
    table.add_column("Tool", style="bold cyan")
    table.add_column("Status")
    table.add_column("Install")

    for integ in INTEGRATIONS.values():
        installed = integ.is_installed()
        table.add_row(
            integ.display_name,
            "[green]installed[/]" if installed else "[dim]not installed[/]",
            integ.install_hint if not installed else "—",
        )

    console.print(table)


@launch_app.callback(invoke_without_command=True)
def launch_tool(
    tool: str = typer.Argument(
        "list", help="Tool to launch: codex, opencode, pi, or 'list'."
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Model to use."),
    url: str = typer.Option(
        "http://localhost:8000",
        "--url",
        "-u",
        envvar="YUNSHU_GATEWAY_URL",
        help="Server URL.",
    ),
    api_key: str | None = typer.Option(None, "--api-key", "-k", help="API key."),
):
    """Launch an external coding tool configured for Yunshu."""
    if tool == "list":
        return list_tools()

    integration = INTEGRATIONS.get(tool)
    if not integration:
        console.print(f"[red]Unknown tool: {tool}[/]")
        console.print(f"Available: {', '.join(INTEGRATIONS.keys())}")
        raise typer.Exit(1)

    if not integration.is_installed():
        console.print(f"[red]{integration.display_name} is not installed.[/]")
        console.print(f"Install: [bold]{integration.install_hint}[/]")
        raise typer.Exit(1)

    # Check server
    import httpx

    host = url.replace("http://", "").replace("https://", "")
    port = 8000
    if ":" in host:
        parts = host.split(":")
        host = parts[0]
        port = int(parts[1])

    try:
        resp = httpx.get(f"{url}/health", timeout=5)
        resp.raise_for_status()
    except Exception:
        logger.debug("health check failed for %s", url, exc_info=True)
        console.print(f"[red]Server not running at {url}[/]")
        console.print("Start with: [bold]yunshu serve[/]")
        raise typer.Exit(1) from None

    # Resolve model
    resolved_model = model or _resolve_model(url)
    if not resolved_model:
        console.print("[red]No model available.[/]")
        raise typer.Exit(1)

    integration.launch(
        port=port, api_key=api_key or "", model=resolved_model, host=host
    )
