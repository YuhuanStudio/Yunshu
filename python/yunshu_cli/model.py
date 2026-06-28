"""Yunshu CLI — model subcommand.

Model management: list, download, info, benchmark.
Follows oMLX's model management with HF Hub integration.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

console = Console()
model_app = typer.Typer(help="Model management.", no_args_is_help=True)


def _get_models_dir() -> Path:
    """Resolve models directory from env or default."""
    env_dir = __import__("os").environ.get("YUNSHU_MODELS_DIR")
    if env_dir:
        return Path(env_dir)
    # Default: yunshu/models/
    return Path(__file__).parent.parent.parent.parent / "models"


def _detect_model_type(config_path: Path) -> str:
    """Auto-detect model type from config.json ."""
    if not config_path.exists():
        return "UNKNOWN"
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except Exception:
        logger.debug("Failed to read or parse config.json at %s", config_path, exc_info=True)
        return "UNKNOWN"

    # Check model_index.json first (diffusion models)
    model_index = config_path.parent / "model_index.json"
    if model_index.exists():
        return "IMAGE_GEN"

    model_type = cfg.get("model_type", "")
    if model_type in ("tts",):
        return "TTS"
    if model_type in ("asr",):
        return "ASR"

    architectures = []
    for arch in cfg.get("architectures", []):
        architectures.append(arch.lower())

    arch_str = " ".join(architectures)
    if any(k in arch_str for k in ("omni", "vlm", "vision")):
        return "VLM"

    if cfg.get("model_type") == "encoder_decoder" and "audio" in str(cfg):
        return "ASR"

    return "LLM"


def _format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


@model_app.command("list")
def list_models(
    models_dir: str | None = typer.Option(None, "--dir", "-d", help="Models directory."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed info."),
    url: str | None = typer.Option(
        None,
        "--url",
        "-u",
        envvar="YUNSHU_GATEWAY_URL",
        help="Gateway URL; when provided, list models reported by the running server "
             "instead of scanning the local models directory.",
    ),
):
    """List available models."""
    # If --url is supplied, query the live gateway. Otherwise fall back to a
    # local on-disk scan of the models directory.
    if url:
        import httpx

        try:
            resp = httpx.get(f"{url.rstrip('/')}/v1/models", timeout=5)
            data = resp.json()
        except Exception as e:
            console.print(f"[red]Error querying {url}: {e}[/]")
            raise typer.Exit(1) from e
        models = data.get("data", []) if isinstance(data, dict) else (data or [])
        if not models:
            console.print("[dim]No models reported by the server.[/]")
            return
        table = Table(title=f"Models on {url}")
        table.add_column("ID", style="bold cyan")
        table.add_column("Owned By", style="dim")
        table.add_column("Object")
        for m in models:
            table.add_row(
                str(m.get("id", "")),
                str(m.get("owned_by", "")),
                str(m.get("object", "")),
            )
        console.print(table)
        console.print(f"\n[dim]Total: {len(models)} models on {url}[/]")
        return

    base = Path(models_dir) if models_dir else _get_models_dir()

    if not base.exists():
        console.print(f"[yellow]Models directory not found: {base}[/]")
        console.print("[dim]Download models with: yunshu model download <model-id>[/]")
        return

    models = []
    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir():
            continue
        config_path = subdir / "config.json"
        has_weights = any(subdir.rglob("*.safetensors"))

        if config_path.exists() or has_weights:
            size = sum(f.stat().st_size for f in subdir.rglob("*") if f.is_file())
            model_type = _detect_model_type(config_path)
            models.append({
                "name": subdir.name,
                "type": model_type,
                "size": size,
                "has_config": config_path.exists(),
            })

    if not models:
        console.print("[yellow]No models found.[/]")
        return

    table = Table(title="Available Models", show_lines=True)
    table.add_column("Model", style="bold cyan")
    table.add_column("Type", style="green")
    table.add_column("Size", justify="right")
    table.add_column("Config", justify="center")

    type_colors = {
        "LLM": "bright_blue",
        "VLM": "magenta",
        "TTS": "yellow",
        "ASR": "green",
        "IMAGE_GEN": "red",
        "UNKNOWN": "dim",
    }

    for m in models:
        color = type_colors.get(m["type"], "white")
        table.add_row(
            m["name"],
            f"[{color}]{m['type']}[/]",
            _format_size(m["size"]),
            "✓" if m["has_config"] else "✗",
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(models)} models in {base}[/]")


@model_app.command("download")
def download_model(
    model_id: str = typer.Argument(help="HuggingFace model ID (e.g. mlx-community/Qwen3.5-9B-MLX-4bit)."),
    models_dir: str | None = typer.Option(None, "--dir", "-d", help="Download directory."),
    revision: str | None = typer.Option(None, "--revision", "-r", help="Model revision/branch."),
):
    """Download a model from HuggingFace."""
    from huggingface_hub import snapshot_download
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TransferSpeedColumn,
    )

    base = Path(models_dir) if models_dir else _get_models_dir()
    base.mkdir(parents=True, exist_ok=True)

    # Extract local name from HF ID
    local_name = model_id.rsplit("/", 1)[-1] if "/" in model_id else model_id
    target_dir = base / local_name

    if target_dir.exists():
        console.print(f"[yellow]Model already exists at {target_dir}[/]")
        overwrite = typer.confirm("Overwrite?", default=False)
        if not overwrite:
            return
        import shutil
        shutil.rmtree(target_dir)

    console.print(f"[bold]Downloading[/] {model_id} → {target_dir}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Downloading {model_id}", total=None)
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=str(target_dir),
                revision=revision,
            )
            progress.update(task, completed=1, total=1)
        except Exception as e:
            logger.debug("Model download failed for %s", model_id, exc_info=True)
            console.print(f"[red]Download failed: {e}[/]")
            raise typer.Exit(1) from e

    config_path = target_dir / "config.json"
    model_type = _detect_model_type(config_path)
    console.print(f"[green]✓ Downloaded[/] {model_id} ({model_type}) → {target_dir}")


@model_app.command("info")
def model_info(
    model: str = typer.Argument(help="Model name or path."),
):
    """Show detailed model information."""
    base = _get_models_dir()
    model_path = Path(model)
    if not model_path.exists():
        model_path = base / model
    if not model_path.exists():
        # Try to find by partial match
        for subdir in base.iterdir():
            if subdir.is_dir() and model.lower() in subdir.name.lower():
                model_path = subdir
                break

    if not model_path.exists():
        console.print(f"[red]Model not found: {model}[/]")
        raise typer.Exit(1)

    config_path = model_path / "config.json"
    model_type = _detect_model_type(config_path)
    total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
    num_files = sum(1 for f in model_path.rglob("*") if f.is_file())
    safetensors_files = list(model_path.rglob("*.safetensors"))

    # Build info tree
    tree = Tree(f"[bold cyan]{model_path.name}[/]")

    info = tree.add("[bold]Basic Info[/]")
    info.add(f"Type: [green]{model_type}[/]")
    info.add(f"Path: {model_path}")
    info.add(f"Total Size: {_format_size(total_size)}")
    info.add(f"Files: {num_files}")
    info.add(f"Safetensors: {len(safetensors_files)}")

    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)

        arch = tree.add("[bold]Architecture[/]")
        for key in ("architectures", "model_type", "hidden_size", "num_hidden_layers",
                     "num_attention_heads", "num_key_value_heads", "intermediate_size",
                     "max_position_embeddings", "vocab_size", "quantization_config"):
            if key in cfg:
                val = cfg[key]
                if isinstance(val, dict):
                    for k, v in val.items():
                        arch.add(f"{key}.{k}: {v}")
                else:
                    arch.add(f"{key}: {val}")

        # Show quantization info
        quant = cfg.get("quantization_config", {})
        if quant:
            qtree = tree.add("[bold]Quantization[/]")
            for k, v in quant.items():
                qtree.add(f"{k}: {v}")

    console.print(tree)


@model_app.command("benchmark")
def benchmark_model(
    model: str = typer.Argument(help="Model name or path."),
    prompt_tokens: int = typer.Option(128, "--prompt-tokens", help="Prompt length in tokens."),
    max_tokens: int = typer.Option(256, "--max-tokens", help="Max tokens to generate."),
    num_runs: int = typer.Option(3, "--runs", "-n", help="Number of runs."),
    warmup: int = typer.Option(1, "--warmup", help="Warmup runs."),
):
    """Benchmark a model's inference performance."""
    import time

    import mlx.core as mx

    base = _get_models_dir()
    model_path = Path(model)
    if not model_path.exists():
        model_path = base / model
    if not model_path.exists():
        console.print(f"[red]Model not found: {model}[/]")
        raise typer.Exit(1)

    console.print(f"[bold]Benchmarking[/] {model_path.name}")

    # Load model
    from mlx_lm.utils import load as load_model
    with console.status("[bold]Loading model..."):
        ml_model, tokenizer = load_model(str(model_path))

    console.print("[green]✓ Model loaded[/]")

    # Prepare prompt
    prompt = "The quick brown fox jumps over the lazy dog. " * (prompt_tokens // 10 + 1)
    tokens = tokenizer.encode(prompt)[:prompt_tokens]

    console.print(f"Prompt tokens: {len(tokens)}, Max output: {max_tokens}")

    # Warmup
    for _ in range(warmup):
        from mlx_lm.utils import generate_step
        for _ in generate_step(ml_model, tokens, max_tokens=16, temp=0.0):
            pass
        mx.synchronize()

    # Benchmark runs
    results = []
    for run in range(num_runs):
        t0 = time.perf_counter()
        generated = 0
        for _ in generate_step(ml_model, tokens, max_tokens=max_tokens, temp=0.7):
            generated += 1
        mx.synchronize()
        elapsed = time.perf_counter() - t0
        tok_per_s = generated / elapsed if elapsed > 0 else 0
        results.append(tok_per_s)
        console.print(f"  Run {run + 1}: {tok_per_s:.1f} tok/s ({generated} tokens in {elapsed:.2f}s)")

    avg = sum(results) / len(results)
    table = Table(title=f"Benchmark Results: {model_path.name}")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Avg throughput", f"{avg:.1f} tok/s")
    table.add_row("Min", f"{min(results):.1f} tok/s")
    table.add_row("Max", f"{max(results):.1f} tok/s")
    table.add_row("Runs", str(num_runs))
    table.add_row("Prompt tokens", str(prompt_tokens))
    table.add_row("Output tokens", str(max_tokens))
    console.print(table)
