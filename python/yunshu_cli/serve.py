"""Yunshu CLI — serve subcommand.

Starts the inference server in single-model or multi-model mode.
"""

from __future__ import annotations

import os

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from yunshu_engine import settings

console = Console()

serve_app = typer.Typer(
    help="Start inference server.",
    no_args_is_help=True,
    context_settings={"allow_interspersed_args": True},
)


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
    model_ref: str | None = typer.Argument(
        None, help="Model path or Hugging Face ID (also accepted through --model/-m)."
    ),
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
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        "-h",
        help="Bind host. 127.0.0.1 serves this Mac only; 0.0.0.0 also serves the "
        "network (set --auth-token then).",
    ),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port."),
    uds: str | None = typer.Option(
        None,
        "--uds",
        help="Serve on this Unix domain socket instead of TCP (same app; "
        "curl --unix-socket PATH http://localhost/v1/models). --host/--port are ignored.",
    ),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of workers."),
    max_memory: str | None = typer.Option(
        None,
        "--max-memory",
        help="Max GPU memory for models (e.g., 32GB, 'disabled'). Default: 80%% of system.",
    ),
    completion_batch_size: int | None = typer.Option(
        None,
        "--completion-batch",
        help="Completion batch size (default 32).",
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
    ),
    mcp_config: str | None = typer.Option(
        None,
        "--mcp-config",
        help="Path to MCP configuration file (JSON/YAML).",
    ),
    hf_endpoint: str | None = typer.Option(
        None,
        "--hf-endpoint",
        help="Custom HuggingFace Hub endpoint URL.",
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
    log_level: str = typer.Option(
        "info", "--log-level", help="Log level (trace|debug|info|warning|error)."
    ),
    startup_timeout: float | None = typer.Option(
        None,
        "--startup-timeout",
        help="Max seconds to wait for model loading before giving up.",
    ),
    slow_request_threshold: float | None = typer.Option(
        None,
        "--slow-request-threshold",
        help="Log a warning for requests exceeding this duration (seconds).",
    ),
    drain_timeout: float | None = typer.Option(
        None,
        "--drain-timeout",
        help="Max seconds to wait for request draining on shutdown.",
    ),
    keep_alive_timeout: int | None = typer.Option(
        None,
        "--keep-alive-timeout",
        help="Seconds to keep idle connections alive (0 to disable).",
    ),
    max_request_size: int | None = typer.Option(
        None,
        "--max-request-size",
        help="Maximum request body size in bytes (default 10MB).",
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
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="TOML file of YUNSHU_* settings (see docs/CONFIGURATION.md).",
    ),
    set_: list[str] = typer.Option(
        [],
        "--set",
        help="Override a setting: --set KEY=VALUE (repeatable; KEY with or "
        "without the YUNSHU_ prefix).",
    ),
):
    """Start Yunshu inference server."""
    import uvicorn

    if model_ref and model and model_ref != model:
        console.print(
            "[red]Error:[/] The positional model and --model disagree. "
            "Use one model reference."
        )
        raise typer.Exit(2)
    model = model_ref or model

    # Flags and --set become highest-precedence settings. They are also
    # exported so worker/reload subprocesses see them.
    overrides: dict[str, object] = {}
    if config:
        overrides["YUNSHU_CONFIG"] = os.path.abspath(os.path.expanduser(config))
    for key, value in (
        ("YUNSHU_MODEL", model),
        ("YUNSHU_MODELS_DIR", models_dir),
        ("YUNSHU_MULTI_MODEL", True if models_dir else None),
        ("YUNSHU_MAX_MEMORY_GB", max_memory),
        ("YUNSHU_AUTH_TOKEN", auth_token),
        ("YUNSHU_MCP_CONFIG", mcp_config),
        ("YUNSHU_HF_ENDPOINT", hf_endpoint),
        ("YUNSHU_MAX_CONCURRENT", max_concurrent),
        ("YUNSHU_COMPLETION_BATCH_SIZE", completion_batch_size),
        ("YUNSHU_STARTUP_TIMEOUT", startup_timeout),
        ("YUNSHU_SLOW_REQUEST_THRESHOLD", slow_request_threshold),
        ("YUNSHU_DRAIN_TIMEOUT", drain_timeout),
        ("YUNSHU_KEEP_ALIVE_TIMEOUT", keep_alive_timeout),
        ("YUNSHU_MAX_REQUEST_SIZE", max_request_size),
        ("YUNSHU_UDS", os.path.abspath(os.path.expanduser(uds)) if uds else None),
    ):
        if value is not None:
            overrides[key] = value
    for item in set_:
        key, sep, value = item.partition("=")
        name = settings._normalize_key(key)
        if not sep or name not in settings.REGISTRY:
            close = settings.close_matches(name)
            hint = f" (did you mean {', '.join(close)}?)" if close else ""
            console.print(f"[red]Error:[/] --set {item!r}: unknown setting{hint}")
            raise typer.Exit(2)
        overrides[name] = value
    for key, value in overrides.items():
        settings.set_override(key, value)
    try:
        for warning in settings.validate(warn=False):
            console.print(f"[yellow]Warning:[/] {warning}")
    except settings.SettingError as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(2) from None

    # Fail before importing/loading a checkpoint when the chosen TCP endpoint
    # cannot bind. Uvicorn remains the authority at startup (another process
    # can acquire the port after this check). Unix sockets use their own path.
    if not uds and not settings.get("YUNSHU_UDS"):
        try:
            _check_bind_address(host, port)
        except OSError as exc:
            console.print(
                f"[red]Error:[/] Cannot listen on {host}:{port}: {exc}. "
                "Choose another --port (for example --port 8001), "
                "or stop the server already using this address."
            )
            raise typer.Exit(2) from None

    env = os.environ.copy()
    env.update({k: settings._to_text(v) for k, v in overrides.items()})
    hf = settings.get("YUNSHU_HF_ENDPOINT")
    if hf:
        env["HF_ENDPOINT"] = hf
    if http_proxy:
        env["HTTP_PROXY"] = http_proxy
        env["http_proxy"] = http_proxy
    if https_proxy:
        env["HTTPS_PROXY"] = https_proxy
        env["https_proxy"] = https_proxy
    if no_proxy:
        env["NO_PROXY"] = no_proxy
        env["no_proxy"] = no_proxy

    effective_model = settings.get("YUNSHU_MODEL")
    effective_dir = settings.get("YUNSHU_MODELS_DIR")
    if effective_model:
        from yunshu_engine.model_discovery import resolve_model_ref

        # A name under the models directory, or a repo id already in the
        # Hugging Face cache, serves that local copy instead of downloading.
        local = resolve_model_ref(effective_model)
        if local != effective_model:
            console.print(f"[dim]Using the local copy of {effective_model}: {local}[/]")
            settings.set_override("YUNSHU_MODEL", local)
            env["YUNSHU_MODEL"] = local
            effective_model = local
        _preflight_model(effective_model)
    if host not in ("127.0.0.1", "localhost", "::1") and not settings.get(
        "YUNSHU_AUTH_TOKEN"
    ):
        console.print(
            f"[yellow]Warning:[/] serving on {host} without --auth-token: anyone "
            "who can reach this port can use the model."
        )
    is_multi = not effective_model and (
        effective_dir is not None or settings.get_bool("YUNSHU_MULTI_MODEL")
    )

    # Native speech-to-speech is automatic: serving an omni model (one that has a
    # Talker) lights up the voice path by REUSING that same loaded model — no flag,
    # no second copy. Voice is off only for non-omni models or YUNSHU_REALTIME_OMNI=off.
    _omni_off = settings.get("YUNSHU_REALTIME_OMNI") in ("0", "false", "no", "off")
    voice_on = not _omni_off and (
        bool(settings.get("YUNSHU_OMNI_MODEL")) or _is_omni_model(effective_model)
    )

    # Display startup info
    _print_startup_banner(
        model=effective_model,
        models_dir=effective_dir,
        is_multi=is_multi,
        host=host,
        port=port,
        completion_batch=settings.get("YUNSHU_COMPLETION_BATCH_SIZE"),
        mcp_config=settings.get("YUNSHU_MCP_CONFIG"),
        hf_endpoint=hf,
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

    uds_path = settings.get("YUNSHU_UDS")
    if uds_path:
        console.print(f"[bold]Unix socket:[/] {uds_path}")
        # A stale socket file from a crashed run would make bind fail.
        import contextlib
        import stat

        with contextlib.suppress(OSError):
            if stat.S_ISSOCK(os.stat(uds_path).st_mode):
                os.unlink(uds_path)
    _rotate_service_log()
    uvicorn.run(
        "yunshu_gateway.main:app",
        **({"uds": uds_path} if uds_path else {"host": host, "port": port}),
        workers=workers,
        log_level=log_level,
        reload=reload,
        factory=False,
        timeout_keep_alive=settings.get("YUNSHU_KEEP_ALIVE_TIMEOUT"),
        # Ctrl-C / SIGTERM: in-flight requests get this long to finish, then
        # their connections are cancelled (which stops their GPU work) and the
        # server exits. Without it uvicorn waits for open streams forever.
        timeout_graceful_shutdown=int(settings.get("YUNSHU_DRAIN_TIMEOUT")) or None,
        # NB: uvicorn.run has no request-size limit kwarg; the limit is enforced
        # by the gateway middleware via YUNSHU_MAX_REQUEST_SIZE (set above).
        server_header="Yunshu" if server_header else None,
    )


def _check_bind_address(host: str, port: int) -> None:
    """Check the actual bind address without contacting or stopping its owner."""
    import socket

    if not 0 <= port <= 65535:
        raise OSError("port must be between 0 and 65535")
    addresses = socket.getaddrinfo(
        host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    )
    for family, socktype, proto, _, address in addresses:
        with socket.socket(family, socktype, proto) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(address)
    if not addresses:
        raise OSError("no usable bind address")


def _rotate_service_log() -> None:
    """Under the launchd agent (stdout is the service log) keep that log rotated."""
    from yunshu_engine import log_rotation, paths

    log = paths.log_dir() / "yunshu.log"
    if log_rotation.stdout_is(log):
        log_rotation.start_background(log)


def _preflight_model(model: str) -> None:
    """Say so before loading when the model cannot work (a path that does not
    exist, a half-finished download, weights larger than memory). The server
    still starts and ``/health/ready`` reports why it is not ready, so a
    supervisor (launchd, Docker) does not restart-loop and a client gets a
    503 with the reason."""
    from .doctor import check_model

    info: dict = {}
    try:
        import mlx.core as mx

        info = dict(mx.device_info())
    except Exception:  # noqa: BLE001 - doctor falls back to sysctl
        pass
    for c in check_model(model, info):
        if c.status == "fail":
            console.print(
                f"[red]Error:[/] {c.detail}\n  {c.fix}\n"
                "  The server starts anyway; /health/ready reports it as not ready."
            )
        if c.status == "warn":
            console.print(f"[yellow]Warning:[/] {c.detail}. {c.fix}")


def _print_startup_banner(
    model: str | None,
    models_dir: str | None,
    is_multi: bool,
    host: str,
    port: int,
    completion_batch: int,
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

    from yunshu_engine.version import yunshu_version

    table.add_row("Yunshu", f"[bold green]{yunshu_version()}[/]")
    table.add_row("Mode", "Multi-model" if is_multi else "Single-model")
    if model:
        table.add_row("Model", model)
    if models_dir:
        table.add_row("Models Dir", models_dir)
    _uds = settings.get("YUNSHU_UDS")
    table.add_row("Address", f"unix:{_uds}" if _uds else f"http://{host}:{port}")
    table.add_row("Completion Batch", str(completion_batch))
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
