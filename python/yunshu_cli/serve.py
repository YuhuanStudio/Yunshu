"""Yunshu CLI — serve subcommand.

Starts the inference server in single-model or multi-model mode.
Matches oMLX's serve command options for seamless migration.
"""

from __future__ import annotations

import os

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

serve_app = typer.Typer(help="Start inference server.", no_args_is_help=True)


def _is_omni_model(model: str | None) -> bool:
    """Best-effort: does this model have a Talker (native speech-to-speech)?

    Reads a local ``config.json`` when present (a Qwen3-Omni model carries a
    ``talker_config``); otherwise falls back to the model name. Used only to
    decide whether to auto-enable the voice path — never fatal.
    """
    if not model:
        return False
    cfg = os.path.join(model, "config.json")
    if os.path.isfile(cfg):
        try:
            import json

            with open(cfg) as f:
                c = json.load(f)
            if "talker_config" in c:
                return True
            return "omni" in str(c.get("model_type", "")).lower()
        except Exception:  # noqa: BLE001 - detection is best-effort
            pass
    return "omni" in model.lower()


@serve_app.callback(invoke_without_command=True)
def serve(
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model path or HuggingFace ID (single-model mode).",
    ),
    models_dir: str | None = typer.Option(
        None,
        "--models-dir",
        "-d",
        help="Directory to scan for models (multi-model mode).",
    ),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind host."),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port."),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of workers."),
    max_memory: str | None = typer.Option(
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
    max_concurrent: int | None = typer.Option(
        None,
        "--max-concurrent",
        help="Max concurrent requests (default: 8).",
    ),
    auth_token: str | None = typer.Option(
        None,
        "--auth-token",
        help="API bearer token for authentication.",
        envvar="YUNSHU_AUTH_TOKEN",
    ),
    mcp_config: str | None = typer.Option(
        None,
        "--mcp-config",
        help="Path to MCP configuration file (JSON/YAML).",
        envvar="YUNSHU_MCP_CONFIG",
    ),
    hf_endpoint: str | None = typer.Option(
        None,
        "--hf-endpoint",
        help="Custom HuggingFace Hub endpoint URL.",
        envvar="YUNSHU_HF_ENDPOINT",
    ),
    http_proxy: str | None = typer.Option(
        None,
        "--http-proxy",
        help="HTTP proxy URL.",
    ),
    https_proxy: str | None = typer.Option(
        None,
        "--https-proxy",
        help="HTTPS proxy URL.",
    ),
    no_proxy: str | None = typer.Option(
        None,
        "--no-proxy",
        help="Comma-separated hosts to bypass proxy.",
    ),
    base_path: str | None = typer.Option(
        None,
        "--base-path",
        help="Base directory for Yunshu data (default: ~/.yunshu).",
    ),
    log_level: str = typer.Option(
        "info", "--log-level", help="Log level (trace|debug|info|warning|error)."
    ),
    startup_timeout: float = typer.Option(
        300.0,
        "--startup-timeout",
        help="Max seconds to wait for model loading before giving up.",
        envvar="YUNSHU_STARTUP_TIMEOUT",
    ),
    slow_request_threshold: float = typer.Option(
        30.0,
        "--slow-request-threshold",
        help="Log a warning for requests exceeding this duration (seconds).",
        envvar="YUNSHU_SLOW_REQUEST_THRESHOLD",
    ),
    drain_timeout: float = typer.Option(
        30.0,
        "--drain-timeout",
        help="Max seconds to wait for request draining on shutdown.",
        envvar="YUNSHU_DRAIN_TIMEOUT",
    ),
    keep_alive_timeout: int = typer.Option(
        5,
        "--keep-alive-timeout",
        help="Seconds to keep idle connections alive (0 to disable).",
        envvar="YUNSHU_KEEP_ALIVE_TIMEOUT",
    ),
    max_request_size: int = typer.Option(
        10 * 1024 * 1024,
        "--max-request-size",
        help="Maximum request body size in bytes (default: 10MB).",
        envvar="YUNSHU_MAX_REQUEST_SIZE",
    ),
    server_header: bool = typer.Option(
        False,
        "--server-header",
        help="Include 'Server: Yunshu' header in responses (for reverse proxy compat).",
    ),
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
        env["YUNSHU_MAX_MEMORY_GB"] = max_memory
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
    if max_concurrent is not None:
        env["YUNSHU_MAX_CONCURRENT"] = str(max_concurrent)
    env["YUNSHU_CACHE_SIZE_MB"] = str(cache_size_mb)
    env["YUNSHU_PREFILL_BATCH_SIZE"] = str(prefill_batch_size)
    env["YUNSHU_COMPLETION_BATCH_SIZE"] = str(completion_batch_size)
    env["YUNSHU_STARTUP_TIMEOUT"] = str(startup_timeout)
    env["YUNSHU_SLOW_REQUEST_THRESHOLD"] = str(slow_request_threshold)
    env["YUNSHU_DRAIN_TIMEOUT"] = str(drain_timeout)
    env["YUNSHU_KEEP_ALIVE_TIMEOUT"] = str(keep_alive_timeout)
    env["YUNSHU_MAX_REQUEST_SIZE"] = str(max_request_size)

    # Determine effective model source
    effective_model = model or os.environ.get("YUNSHU_MODEL")
    effective_dir = models_dir or os.environ.get("YUNSHU_MODELS_DIR")
    is_multi = effective_dir is not None or (
        os.environ.get("YUNSHU_MULTI_MODEL", "").strip().lower() in ("1", "true", "yes")
    )

    # Native speech-to-speech is automatic: serving an omni model (one that has a
    # Talker) lights up the voice path by REUSING that same loaded model — no flag,
    # no second copy. Voice is off only for non-omni models or YUNSHU_REALTIME_OMNI=0.
    _omni_off = env.get("YUNSHU_REALTIME_OMNI", "").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    )
    voice_on = not _omni_off and (
        bool(env.get("YUNSHU_OMNI_MODEL")) or _is_omni_model(effective_model)
    )

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
        voice_on=voice_on,
    )

    if voice_on:
        console.print(
            "[dim]Native voice is on (same model serves text + speech). "
            "Run [bold]python examples/talk.py[/] to talk to it.[/]"
        )

    # Override os.environ for the child process
    os.environ.update(env)

    # Warn if reload=True with workers>1 (uvicorn ignores workers in reload mode)
    if reload and workers > 1:
        console.print(
            "[yellow]Warning:[/] --reload with --workers > 1 is not supported by uvicorn. Using workers=1."
        )
        workers = 1

    uvicorn.run(
        "yunshu_gateway.main:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=reload,
        factory=False,
        timeout_keep_alive=keep_alive_timeout,
        # NB: uvicorn.run has no request-size limit kwarg; the limit is enforced
        # by the gateway middleware via YUNSHU_MAX_REQUEST_SIZE (set above).
        server_header="Yunshu" if server_header else None,
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
    voice_on: bool = False,
):
    """Print a rich startup banner."""
    console.print()

    table = Table(show_header=False, border_style="bright_blue", padding=(0, 2))
    table.add_column(style="bold cyan", width=20)
    table.add_column()

    try:
        from importlib.metadata import version as _pkg_version

        _ver = _pkg_version("yunshu")
    except Exception:
        _ver = "0.0.1"
    table.add_row("Yunshu", f"[bold green]v{_ver}[/]")
    table.add_row("Mode", "Multi-model" if is_multi else "Single-model")
    if model:
        table.add_row("Model", model)
    if models_dir:
        table.add_row("Models Dir", models_dir)
    table.add_row("Address", f"http://{host}:{port}")
    table.add_row("Prefill Batch", str(prefill_batch))
    table.add_row("Completion Batch", str(completion_batch))
    table.add_row("Cache Size", f"{cache_size_mb} MB")
    if voice_on:
        table.add_row("Voice", "[bold green]native speech-to-speech (omni) ON[/]")

    if mcp_config:
        table.add_row("MCP Config", mcp_config)
    if hf_endpoint:
        table.add_row("HF Endpoint", hf_endpoint)
    if has_proxy:
        table.add_row("Proxy", "configured")

    table.add_row("Endpoints", "/v1/chat/completions, /v1/models, /health")

    console.print(
        Panel(table, title="[bold]Yunshu Server[/]", border_style="bright_blue")
    )
    console.print()
