"""Yunshu CLI — diagnose subcommand.

System diagnostics: GPU, memory, MLX, model health.
for Apple Silicon validation.
"""

from __future__ import annotations

import contextlib
import logging

import typer
from rich.console import Console
from rich.table import Table

console = Console()
diagnose_app = typer.Typer(help="System diagnostics.", no_args_is_help=True)

logger = logging.getLogger(__name__)


# `yunshu doctor` is the system check; `diagnose system` stays as its alias.
from .doctor import doctor as _doctor  # noqa: E402

diagnose_app.command("system", hidden=True)(_doctor)


@diagnose_app.command("gpu")
def diagnose_gpu():
    """Detailed GPU information and Metal capabilities."""
    import time

    from ._output import emit, fail, is_json

    try:
        import mlx.core as mx
    except ImportError:
        fail("MLX is not installed.", code=1)

    memory = {
        "active_bytes": mx.get_active_memory(),
        "peak_bytes": mx.get_peak_memory(),
        "cache_bytes": mx.get_cache_memory(),
    }
    gemm = []
    for size in [256, 512, 1024, 2048, 4096]:
        a = mx.random.normal((size, size))
        b = mx.random.normal((size, size))
        mx.eval(a, b, a @ b)
        mx.synchronize()
        iters = max(1, 2**24 // (size * size))
        t0 = time.perf_counter()
        outs = [a @ b for _ in range(iters)]
        mx.eval(outs)
        mx.synchronize()
        elapsed = time.perf_counter() - t0
        gemm.append({"size": size, "tflops": 2.0 * size**3 * iters / elapsed / 1e12})

    if is_json():
        emit({"memory": memory, "gemm": gemm})
        return

    console.print("[bold]Apple GPU Diagnostics[/]\n")
    table = Table(title="GPU Memory")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Active", _fmt(memory["active_bytes"]))
    table.add_row("Peak", _fmt(memory["peak_bytes"]))
    table.add_row("Cache", _fmt(memory["cache_bytes"]))
    console.print(table)
    console.print("\n[bold]Quick GEMM Benchmark[/]")
    for g in gemm:
        console.print(f"  {g['size']:5d}×{g['size']:5d}: {g['tflops']:.2f} TFLOPS")
    console.print()


@diagnose_app.command("server")
def diagnose_server(
    url: str = typer.Option(
        "http://localhost:8000",
        "--url",
        "-u",
        envvar="YUNSHU_GATEWAY_URL",
        help="Server URL.",
    ),
):
    """Check a running Yunshu server's health and stats."""
    import httpx

    from ._output import auth_headers, emit, fail, is_json

    info: dict = {"url": url, "healthy": False, "engine": {}, "models": []}
    try:
        resp = httpx.get(f"{url}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            info["healthy"] = True
            info["status"] = data.get("status")
            info["engine"] = data.get("engine", {})
    except httpx.ConnectError:
        fail(
            f"Cannot connect to {url} — is the server running? Try `yunshu serve`.",
            code=2,
        )
    except Exception as e:  # noqa: BLE001
        fail(f"Error: {e}", code=1)

    with contextlib.suppress(Exception):
        resp = httpx.get(f"{url}/v1/models", headers=auth_headers(), timeout=5)
        if resp.status_code == 200:
            info["models"] = [m.get("id") for m in resp.json().get("data", [])]

    if is_json():
        emit(info)
        return

    console.print(f"[bold]Server Diagnostics[/] — {url}\n")
    if info["healthy"]:
        console.print(f"[green]✓ Server healthy[/] — status: {info.get('status')}")
        if info["engine"]:
            table = Table(title="Engine Stats")
            table.add_column("Metric", style="bold")
            table.add_column("Value")
            for k, v in info["engine"].items():
                table.add_row(str(k), str(v))
            console.print(table)
    else:
        console.print("[red]✗ Server unhealthy[/]")
    if info["models"]:
        console.print(f"\n[bold]Loaded Models:[/] {len(info['models'])}")
        for mid in info["models"]:
            console.print(f"  • {mid}")


@diagnose_app.command("bundle")
def diagnose_bundle(
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="File to write (default: ./yunshu-diagnostics-<time>.json).",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Host you serve on."),
    port: int = typer.Option(8000, "--port", "-p", help="Port you serve on."),
):
    """Write a local diagnostics file: version, settings (secrets redacted), doctor, recent errors.

    Never includes prompts or completions, and never uploads anything.
    """
    import time
    from pathlib import Path

    from ._output import emit
    from .bundle import write

    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = Path(output or f"yunshu-diagnostics-{stamp}.json").expanduser()
    path = write(dest, host=host, port=port)
    emit(
        {"path": str(path.resolve())},
        human=lambda: console.print(
            f"[green]✓ Wrote[/] {path.resolve()}\n"
            "Local file only: review it before attaching it to a report."
        ),
    )


def _fmt(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"
