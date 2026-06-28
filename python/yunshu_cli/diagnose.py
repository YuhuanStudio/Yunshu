"""Yunshu CLI — diagnose subcommand.

System diagnostics: GPU, memory, MLX, model health.
for Apple Silicon validation.
"""

from __future__ import annotations

import logging
import platform
import subprocess
import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

console = Console()
diagnose_app = typer.Typer(help="System diagnostics.", no_args_is_help=True)

logger = logging.getLogger(__name__)


@diagnose_app.command("system")
def diagnose_system():
    """Full system diagnostic for Yunshu compatibility."""
    tree = Tree("[bold]Yunshu System Diagnostic[/]")

    # OS & Hardware
    hw = tree.add("[bold cyan]Hardware & OS[/]")
    hw.add(f"macOS: {platform.mac_ver()[0]}")
    hw.add(f"Architecture: {platform.machine()}")
    hw.add(f"Python: {sys.version.split()[0]}")

    # CPU
    try:
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        hw.add(f"CPU: {result.stdout.strip()}")
    except Exception:
        logger.debug("failed to query CPU info", exc_info=True)
        hw.add("CPU: [dim]unknown[/]")

    # Memory
    try:
        result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        total_gb = int(result.stdout.strip()) / (1024 ** 3)
        hw.add(f"Unified Memory: {total_gb:.0f} GB")
    except Exception:
        logger.debug("failed to query memory info", exc_info=True)
        hw.add("Memory: [dim]unknown[/]")

    # GPU
    gpu = tree.add("[bold cyan]GPU[/]")
    try:
        result = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.split("\n"):
            line = line.strip()
            if "Chipset Model" in line:
                gpu.add(f"GPU: {line.split(':')[-1].strip()}")
            elif "Total Number of Cores" in line:
                gpu.add(f"Cores: {line.split(':')[-1].strip()}")
            elif "Metal" in line and "Support" in line:
                gpu.add(line)
    except Exception:
        logger.debug("failed to query GPU info", exc_info=True)
        gpu.add("[dim]Unable to query GPU[/]")

    # MLX
    mlx_node = tree.add("[bold cyan]MLX[/]")
    try:
        import mlx.core as mx
        mlx_node.add(f"Version: {mx.__version__}")
        mlx_node.add(f"Default device: {mx.default_device()}")
        active = mx.get_active_memory()
        peak = mx.get_peak_memory()
        mlx_node.add(f"Active memory: {_fmt(active)}")
        mlx_node.add(f"Peak memory: {_fmt(peak)}")

        # Quick GEMM test
        a = mx.random.normal((1024, 1024))
        b = mx.random.normal((1024, 1024))
        _ = a @ b
        mx.synchronize()
        mlx_node.add("[green]✓ GEMM test passed[/]")
    except ImportError:
        mlx_node.add("[red]✗ MLX not installed[/]")
    except Exception as e:
        mlx_node.add(f"[red]✗ MLX error: {e}[/]")

    # mlx-lm
    mlm = tree.add("[bold cyan]mlx-lm[/]")
    try:
        import mlx_lm
        mlm.add(f"Version: {mlx_lm.__version__}")
        mlm.add("[green]✓ Installed[/]")
    except ImportError:
        mlm.add("[red]✗ Not installed[/]")

    # mlx-vlm
    vlm = tree.add("[bold cyan]mlx-vlm[/]")
    try:
        import mlx_vlm
        vlm.add(f"Version: {mlx_vlm.__version__}")
        vlm.add("[green]✓ Installed[/]")
    except ImportError:
        vlm.add("[yellow]✗ Not installed[/]")

    # mlx-audio
    audio = tree.add("[bold cyan]mlx-audio[/]")
    try:
        import mlx_audio
        try:
            from importlib.metadata import version as _pkg_version
            _audio_ver = _pkg_version("mlx-audio")
        except Exception:
            _audio_ver = getattr(mlx_audio, "__version__", "unknown")
        audio.add(f"Version: {_audio_ver}")
        audio.add("[green]✓ Installed[/]")
    except ImportError:
        audio.add("[yellow]✗ Not installed[/]")

    console.print(tree)
    console.print()

    # Compatibility summary
    checks = []
    try:
        import mlx.core as mx
        checks.append(("MLX", True))
    except ImportError:
        checks.append(("MLX", False))

    try:
        import mlx_lm
        checks.append(("mlx-lm", True))
    except ImportError:
        checks.append(("mlx-lm", False))

    try:
        import fastapi  # noqa: F401  # availability probe only
        checks.append(("FastAPI", True))
    except ImportError:
        checks.append(("FastAPI", False))

    all_ok = all(ok for _, ok in checks)
    status = "[green]✓ Compatible[/]" if all_ok else "[red]✗ Issues found[/]"
    console.print(Panel(status, title="Yunshu Compatibility", border_style="green" if all_ok else "red"))


@diagnose_app.command("gpu")
def diagnose_gpu():
    """Detailed GPU information and Metal capabilities."""
    import mlx.core as mx

    console.print("[bold]Apple GPU Diagnostics[/]\n")

    # Memory
    table = Table(title="GPU Memory")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Active", _fmt(mx.get_active_memory()))
    table.add_row("Peak", _fmt(mx.get_peak_memory()))
    table.add_row("Cache", _fmt(mx.get_cache_memory()))
    console.print(table)

    # Quick benchmark
    console.print("\n[bold]Quick GEMM Benchmark[/]")
    import time
    for size in [256, 512, 1024, 2048, 4096]:
        a = mx.random.normal((size, size))
        b = mx.random.normal((size, size))
        _ = a @ b
        mx.synchronize()

        iters = max(1, 2**24 // (size * size))
        t0 = time.perf_counter()
        for _ in range(iters):
            a @ b
        mx.synchronize()
        elapsed = time.perf_counter() - t0
        tflops = 2.0 * size ** 3 * iters / elapsed / 1e12
        console.print(f"  {size:5d}×{size:5d}: {tflops:.2f} TFLOPS")

    console.print()


@diagnose_app.command("server")
def diagnose_server(
    url: str = typer.Option("http://localhost:8000", "--url", "-u", help="Server URL."),
):
    """Check a running Yunshu server's health and stats."""
    import httpx

    console.print(f"[bold]Server Diagnostics[/] — {url}\n")

    # Health check
    try:
        resp = httpx.get(f"{url}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            console.print(f"[green]✓ Server healthy[/] — status: {data.get('status')}")

            engine = data.get("engine", {})
            if engine:
                table = Table(title="Engine Stats")
                table.add_column("Metric", style="bold")
                table.add_column("Value")
                for k, v in engine.items():
                    table.add_row(str(k), str(v))
                console.print(table)
        else:
            console.print(f"[red]✗ Server returned {resp.status_code}[/]")
    except httpx.ConnectError:
        console.print(f"[red]✗ Cannot connect to {url}[/]")
        console.print("[dim]Is the server running? Try: yunshu serve[/]")
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/]")

    # Models
    try:
        resp = httpx.get(f"{url}/v1/models", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", [])
            console.print(f"\n[bold]Loaded Models:[/] {len(models)}")
            for m in models:
                console.print(f"  • {m.get('id', 'unknown')}")
    except Exception:
        logger.debug("failed to fetch models from server", exc_info=True)


def _fmt(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"
