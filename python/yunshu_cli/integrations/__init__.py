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
from .agent_config import (
    ModelInfo,
    claude_code_env,
    codex_catalog,
    codex_provider_toml,
    opencode_provider,
)

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
        self,
        port: int,
        api_key: str,
        model: str,
        host: str = "127.0.0.1",
        info: ModelInfo | None = None,
    ) -> None:
        raise NotImplementedError

    def launch(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1", **kwargs
    ) -> None:
        raise NotImplementedError

    def preview(self, base_url: str, api_key: str, info: ModelInfo, **kwargs) -> str:
        """What ``launch`` would write / export, as text (``--dry-run``)."""
        return ""

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
    """OpenAI Codex CLI — configures ~/.codex/config.toml and a model catalog."""

    CONFIG_PATH = Path.home() / ".codex" / "config.toml"
    CATALOG_PATH = Path.home() / ".codex" / "yunshu-models.json"
    _TOP_KEYS = (
        "model",
        "model_provider",
        "model_catalog_json",
        "model_context_window",
        "model_auto_compact_token_limit",
        "web_search",
        "model_reasoning_summary",
    )

    def __init__(self):
        super().__init__(
            name="codex",
            display_name="Codex",
            install_check="codex",
            install_hint="npm install -g @openai/codex",
        )

    def preview(self, base_url: str, api_key: str, info: ModelInfo, **kwargs) -> str:
        toml = codex_provider_toml(info, base_url, str(self.CATALOG_PATH))
        return f"# {self.CONFIG_PATH}\n{toml}\n# {self.CATALOG_PATH}\n" + json.dumps(
            codex_catalog(info), indent=2
        )

    def configure(
        self,
        port: int,
        api_key: str,
        model: str,
        host: str = "127.0.0.1",
        info: ModelInfo | None = None,
    ) -> None:
        info = info or ModelInfo(id=model or "default")
        config_path = self.CONFIG_PATH
        config_path.parent.mkdir(parents=True, exist_ok=True)
        self.CATALOG_PATH.write_text(
            json.dumps(codex_catalog(info), indent=2) + "\n", encoding="utf-8"
        )
        block = codex_provider_toml(
            info, f"http://{host}:{port}", str(self.CATALOG_PATH)
        )
        top, _, provider = block.partition("\n[model_providers.yunshu]")
        top_lines = [ln for ln in top.splitlines() if ln.strip()]

        existing = ""
        if config_path.exists():
            with contextlib.suppress(OSError):
                existing = config_path.read_text(encoding="utf-8")
            with contextlib.suppress(OSError):
                shutil.copy2(
                    config_path, config_path.with_suffix(f".{int(time.time())}.bak")
                )

        kept: list[str] = []
        in_section = False
        skipping = False
        for line in existing.splitlines():
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                in_section = True
                skipping = stripped == "[model_providers.yunshu]"
            if skipping:
                continue
            if not in_section and "=" in stripped:
                if stripped.split("=")[0].strip() in self._TOP_KEYS:
                    continue
            kept.append(line)
        out = (
            top_lines + kept + ["", "[model_providers.yunshu]"] + provider.splitlines()
        )
        config_path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
        console.print(f"[green]✓[/] Config updated: {config_path}")
        console.print(f"[green]✓[/] Model catalog: {self.CATALOG_PATH}")

    def launch(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1", **kwargs
    ) -> None:
        self.configure(port, api_key, model, host, info=kwargs.get("info"))
        env = os.environ.copy()
        env["YUNSHU_API_KEY"] = api_key or "yunshu"
        args = ["codex"]
        if model:
            args.extend(["-m", model])
        console.print(f"[bold]Launching[/] Codex with model {model}...")
        os.execvpe("codex", args, env)


# ── Claude Code Integration ──


class ClaudeCodeIntegration(Integration):
    """Claude Code — environment only: nothing under ~/.claude is written."""

    def __init__(self):
        super().__init__(
            name="claude",
            display_name="Claude Code",
            install_check="claude",
            install_hint="npm install -g @anthropic-ai/claude-code",
        )

    def env(self, port, api_key, host, info, effort=None) -> dict[str, str]:
        return claude_code_env(info, f"http://{host}:{port}", api_key, effort)

    def preview(self, base_url: str, api_key: str, info: ModelInfo, **kwargs) -> str:
        env = claude_code_env(info, base_url, api_key, kwargs.get("effort"))
        return "\n".join(f"export {k}={v}" for k, v in env.items())

    def configure(self, port, api_key, model, host="127.0.0.1", info=None) -> None:
        console.print(
            "Claude Code needs environment variables only; use `yunshu launch claude`."
        )

    def launch(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1", **kwargs
    ) -> None:
        info = kwargs.get("info") or ModelInfo(id=model)
        env = os.environ.copy()
        # A stray key or Bedrock / Vertex switch in the shell would send requests elsewhere.
        for k in (
            "ANTHROPIC_API_KEY",
            "CLAUDE_CODE_USE_BEDROCK",
            "CLAUDE_CODE_USE_VERTEX",
        ):
            env.pop(k, None)
        env.update(self.env(port, api_key, host, info, kwargs.get("effort")))
        console.print(f"[bold]Launching[/] Claude Code with model {model}...")
        os.execvpe("claude", ["claude"], env)


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

    def preview(self, base_url: str, api_key: str, info: ModelInfo, **kwargs) -> str:
        cfg = {
            "provider": {"yunshu": opencode_provider(info, base_url, api_key)},
            "model": f"yunshu/{info.id}",
        }
        return f"# {self.CONFIG_PATH}\n" + json.dumps(cfg, indent=2)

    def configure(
        self,
        port: int,
        api_key: str,
        model: str,
        host: str = "127.0.0.1",
        info: ModelInfo | None = None,
    ) -> None:
        info = info or ModelInfo(id=model or "default")

        def updater(config: dict) -> None:
            config.setdefault("provider", {})
            config["provider"]["yunshu"] = opencode_provider(
                info, f"http://{host}:{port}", api_key
            )
            config["model"] = f"yunshu/{info.id}"

        self._write_json_config(self.CONFIG_PATH, updater)
        console.print(f"[green]✓[/] Config updated: {self.CONFIG_PATH}")

    def launch(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1", **kwargs
    ) -> None:
        self.configure(port, api_key, model, host, info=kwargs.get("info"))
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
        self,
        port: int,
        api_key: str,
        model: str,
        host: str = "127.0.0.1",
        info: ModelInfo | None = None,
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
        self.configure(port, api_key, model, host, info=kwargs.get("info"))
        args = ["pi"]
        if model:
            args.extend(["--model", f"yunshu/{model}"])
        console.print(f"[bold]Launching[/] Pi with model {model}...")
        os.execvpe("pi", args, os.environ.copy())


# ── Registry ──

INTEGRATIONS: dict[str, Integration] = {
    "claude": ClaudeCodeIntegration(),
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


def _fetch_info(url: str, model: str) -> ModelInfo:
    """The served model's facts (window, output limit, effort levels, vision, web search)."""
    import httpx

    try:
        resp = httpx.get(f"{url}/v1/models", headers=auth_headers(), timeout=5)
        if resp.status_code == 200:
            for item in resp.json().get("data", []):
                if item.get("id") == model:
                    return ModelInfo.from_models_item(item)
    except Exception:
        logger.debug("failed to read the model card from the server", exc_info=True)
    return ModelInfo(id=model)


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
        "list", help="Tool to launch: claude, codex, opencode, pi, or 'list'."
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
    effort: str | None = typer.Option(
        None, "--effort", help="Claude Code reasoning effort (low, medium, high, ...)."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the configuration instead of writing it and launching the tool.",
    ),
):
    """Launch an external coding tool configured for Yunshu.

    The model's real context window, output limit, reasoning-effort levels and vision
    support are read from the server and handed to the tool, which would otherwise guess
    (Claude Code assumes 200K for an unknown model, Codex has no catalog entry for it).
    """
    if tool == "list":
        return list_tools()

    integration = INTEGRATIONS.get(tool)
    if not integration:
        console.print(f"[red]Unknown tool: {tool}[/]")
        console.print(f"Available: {', '.join(INTEGRATIONS.keys())}")
        raise typer.Exit(1)

    if not dry_run and not integration.is_installed():
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

    info = _fetch_info(url, resolved_model)
    if dry_run:
        console.print(
            integration.preview(
                f"http://{host}:{port}", api_key or "", info, effort=effort
            ),
            markup=False,
        )
        return
    integration.launch(
        port=port,
        api_key=api_key or "",
        model=resolved_model,
        host=host,
        info=info,
        effort=effort,
    )
