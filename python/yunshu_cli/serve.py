"""Yunshu CLI — serve subcommand.

Starts the inference server in single-model or multi-model mode.
Matches oMLX's serve command options for seamless migration.
"""

from __future__ import annotations

import os
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

serve_app = typer.Typer(help="Start inference server.", no_args_is_help=True)


@serve_app.callback(invoke_without_command=True)
def serve(
    model: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="Model path or HuggingFace ID (single-model mode).",
    ),
    models_dir: Optional[str] = typer.Option(
        None,
        "--models-dir", "-d",
        help="Directory to scan for models (multi-model mode).",
    ),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind host."),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port."),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of workers."),
    max_memory: Optional[str] = typer.Option(
        None,
        "--max-memory",
        help="Max GPU memory for models (e.g., 32GB, 'disabled'). Default: 80%% of system.",
    ),
    cache_size_mb: int = typer.Option(
        512,
        "--cache-size",
        help="Metal buffer cache size in MB.",
    ),
    prefill_batch_size: int = typer.Option(
        8,
        "--prefill-batch",
        help="Prefill batch size.",
    ),
    completion_batch_size: int = typer.Option(
        32,
        "--completion-batch",
        help="Completion batch size.",
    ),
    max_concurrent: Optional[int] = typer.Option(
        None,
        "--max-concurrent",
        help="Max concurrent requests (default: 8).",
    ),
    auth_token: Optional[str] = typer.Option(
        None,
        "--auth-token",
        help="API bearer token for authentication.",
        envvar="YUNSHU_AUTH_TOKEN",
    ),
    mcp_config: Optional[str] = typer.Option(
        None,
        "--mcp-config",
        help="Path to MCP configuration file (JSON/YAML).",
        envvar="YUNSHU_MCP_CONFIG",
    ),
    hf_endpoint: Optional[str] = typer.Option(
        None,
        "--hf-endpoint",
        help="Custom HuggingFace Hub endpoint URL.",
        envvar="YUNSHU_HF_ENDPOINT",
    ),
    http_proxy: Optional[str] = typer.Option(
        None,
        "--http-proxy",
        help="HTTP proxy URL.",
    ),
    https_proxy: Optional[str] = typer.Option(
        None,
        "--https-proxy",
        help="HTTPS proxy URL.",
    ),
    no_proxy: Optional[str] = typer.Option(
        None,
        "--no-proxy",
        help="Comma-separated hosts to bypass proxy.",
    ),
    base_path: Optional[str] = typer.Option(
        None,
        "--base-path",
        help="Base directory for Yunshu data (default: ~/.yunshu).",
    ),
    log_level: str = typer.Option("info", "--log-level", help="Log level (trace|debug|info|warning|error)."),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Enable auto-reload (development mode).",
    ),
):
    """Start Yunshu inference server."""
    import uvicorn

    # Build environment
    env = os.environ.copy()
    if model:
        env["YUNSHU_MODEL"] = model
    if models_dir:
        env["YUNSHU_MULTI_MODEL"] = "1"
        env["YUNSHU_MODELS_DIR"] = models_dir
    if max_memory:
        env["YUNSHU_MAX_MEMORY"] = max_memory
    if auth_token:
        env["YUNSHU_AUTH_TOKEN"] = auth_token
    if mcp_config:
        env["YUNSHU_MCP_CONFIG"] = mcp_config
    if hf_endpoint:
        env["HF_ENDPOINT"] = hf_endpoint
    if http_proxy:
        env["HTTP_PROXY"] = http_proxy
        env["http_proxy"] = http_proxy
    if https_proxy:
        env["HTTPS_PROXY"] = https_proxy
        env["https_proxy"] = https_proxy
    if no_proxy:
        env["NO_PROXY"] = no_proxy
        env["no_proxy"] = no_proxy
    if base_path:
        env["YUNSHU_BASE_PATH"] = base_path
    if max_concurrent:
        env["YUNSHU_MAX_CONCURRENT"] = str(max_concurrent)

    # Determine effective model source
    effective_model = model or os.environ.get("YUNSHU_MODEL")
    effective_dir = models_dir or os.environ.get("YUNSHU_MODELS_DIR")
    is_multi = effective_dir is not None or os.environ.get("YUNSHU_MULTI_MODEL")

    # Display startup info
    _print_startup_banner(
        model=effective_model,
        models_dir=effective_dir,
        is_multi=is_multi,
        host=host,
        port=port,
        prefill_batch=prefill_batch_size,
        completion_batch=completion_batch_size,
        cache_size_mb=cache_size_mb,
        mcp_config=mcp_config,
        hf_endpoint=hf_endpoint,
        has_proxy=bool(http_proxy or https_proxy),
    )

    # Override os.environ for the child process
    os.environ.update(env)

    uvicorn.run(
        "python.yunshu_gateway.main:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=reload,
        factory=False,
    )


def _print_startup_banner(
    model: str | None,
    models_dir: str | None,
    is_multi: bool,
    host: str,
    port: int,
    prefill_batch: int,
    completion_batch: int,
    cache_size_mb: int,
    mcp_config: str | None = None,
    hf_endpoint: str | None = None,
    has_proxy: bool = False,
):
    """Print a rich startup banner."""
    console.print()

    table = Table(show_header=False, border_style="bright_blue", padding=(0, 2))
    table.add_column(style="bold cyan", width=20)
    table.add_column()

    table.add_row("Yunshu", "[bold green]v0.1.0-dev[/]")
    table.add_row("Mode", "Multi-model" if is_multi else "Single-model")
    if model:
        table.add_row("Model", model)
    if models_dir:
        table.add_row("Models Dir", models_dir)
    table.add_row("Address", f"http://{host}:{port}")
    table.add_row("Prefill Batch", str(prefill_batch))
    table.add_row("Completion Batch", str(completion_batch))
    table.add_row("Cache Size", f"{cache_size_mb} MB")

    if mcp_config:
        table.add_row("MCP Config", mcp_config)
    if hf_endpoint:
        table.add_row("HF Endpoint", hf_endpoint)
    if has_proxy:
        table.add_row("Proxy", "configured")

    table.add_row("Endpoints", "/v1/chat/completions, /v1/models, /health")

    console.print(Panel(table, title="[bold]Yunshu Server[/]", border_style="bright_blue"))
    console.print()
