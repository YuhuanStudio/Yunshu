"""Yunshu CLI — model subcommand.

Local models: list (models directory + Hugging Face cache), pull, info,
benchmark; and load/unload on a running server.
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

from yunshu_engine import paths

from ._output import auth_headers, emit, fail, is_json

console = Console()
model_app = typer.Typer(help="Model management.", no_args_is_help=True)


def _get_models_dir() -> Path:
    """YUNSHU_MODELS_DIR, else ~/.yunshu/models."""
    return paths.models_dir()


def _detect_model_type(model_path: Path) -> str:
    """The server's own detection, so the CLI and the gateway agree."""
    from yunshu_engine.model_manager import _detect_model_type as detect

    try:
        return detect(str(model_path)).name
    except Exception:  # noqa: BLE001 - listing must not fail on one odd folder
        logger.debug("type detection failed for %s", model_path, exc_info=True)
        return "UNKNOWN"


def _format_size(size_bytes: float) -> str:
    """Format bytes to human-readable size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def _is_model_dir(path: Path) -> bool:
    return (
        (path / "config.json").exists()
        or (path / "model_index.json").exists()
        or any(path.glob("*.safetensors"))
    )


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def scan_models_dir(base: Path, *, detect_types: bool = True) -> list[dict]:
    """Model folders directly under ``base`` or one level down (``org/name``),
    the same layout the server's multi-model discovery reads."""
    found: list[Path] = []
    if base.is_dir():
        for sub in sorted(base.iterdir()):
            if not sub.is_dir() or sub.name.startswith("."):
                continue
            if _is_model_dir(sub):
                found.append(sub)
                continue
            found.extend(
                c
                for c in sorted(sub.iterdir())
                if c.is_dir() and not c.name.startswith(".") and _is_model_dir(c)
            )
    return [
        {
            "name": str(p.relative_to(base)),
            "path": str(p),
            "type": _detect_model_type(p) if detect_types else "UNKNOWN",
            "size": _dir_size(p),
            "source": "models-dir",
        }
        for p in found
    ]


def scan_hf_cache() -> list[dict]:
    """Model repos in the Hugging Face cache that have a config and weights
    (anything `yunshu serve -m <repo id>` can load without downloading)."""
    from yunshu_engine.model_discovery import hf_cache_snapshots

    return [
        {
            "name": repo_id,
            "path": str(snapshot),
            "type": _detect_model_type(snapshot),
            "size": size,
            "source": "hf-cache",
        }
        for repo_id, snapshot, size in hf_cache_snapshots()
    ]


def weights_complete(path: Path) -> tuple[bool, str]:
    """Is the model folder a finished download? Returns (complete, reason)."""
    if not path.is_dir():
        return False, "missing"
    if (path / "model_index.json").exists():
        return True, "diffusers pipeline"
    if not (path / "config.json").exists():
        return False, "no config.json"
    partial = path / ".cache" / "huggingface" / "download"
    if partial.is_dir() and any(partial.rglob("*.incomplete")):
        return False, "interrupted download"
    index = path / "model.safetensors.index.json"
    if index.exists():
        try:
            shards = set(json.loads(index.read_text())["weight_map"].values())
        except (OSError, ValueError, KeyError):
            return False, "unreadable model.safetensors.index.json"
        missing = sorted(s for s in shards if not (path / s).exists())
        if missing:
            return False, f"{len(missing)} weight shard(s) missing"
        return True, "complete"
    if any(path.glob("*.safetensors")):
        return True, "complete"
    return False, "no *.safetensors weights"


@model_app.command("list")
def list_models(
    models_dir: str | None = typer.Option(
        None, "--dir", "-d", help="Models directory (default: YUNSHU_MODELS_DIR)."
    ),
    hf_cache: bool = typer.Option(
        True,
        "--hf-cache/--no-hf-cache",
        help="Also list models already in the Hugging Face cache.",
    ),
    url: str | None = typer.Option(
        None,
        "--url",
        "-u",
        envvar="YUNSHU_GATEWAY_URL",
        help="Gateway URL; when provided, list models reported by the running server "
        "instead of scanning the local disk.",
    ),
):
    """List models on disk (models directory + Hugging Face cache), or on a server."""
    # If --url is supplied, query the live gateway. Otherwise scan the disk.
    if url:
        import httpx

        try:
            resp = httpx.get(
                f"{url.rstrip('/')}/v1/models", headers=auth_headers(), timeout=5
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            fail(f"Error querying {url}: {e}", code=1)
        models = data.get("data", []) if isinstance(data, dict) else (data or [])

        if is_json():
            emit({"models": models, "source": url})
            return
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

    base = Path(models_dir).expanduser() if models_dir else _get_models_dir()
    models = scan_models_dir(base) + (scan_hf_cache() if hf_cache else [])

    if is_json():
        emit({"models": models, "models_dir": str(base)})
        return
    if not models:
        where = " or the Hugging Face cache" if hf_cache else ""
        console.print(f"[yellow]No models found[/] in {base}{where}.")
        console.print("[dim]Download one with: yunshu pull <org/name>[/]")
        return

    table = Table(title="Local models")
    table.add_column("Model", style="bold cyan")
    table.add_column("Type")
    table.add_column("Size", justify="right")
    table.add_column("Where", style="dim")
    for m in models:
        where = "models dir" if m["source"] == "models-dir" else "HF cache"
        table.add_row(m["name"], m["type"], _format_size(m["size"]), where)
    console.print(table)
    console.print(
        f"\n[dim]{len(models)} models. Models dir: {base}. "
        "Serve one with: yunshu serve -m <path or repo id>[/]"
    )


def _hf_cached_snapshot(repo_id: str) -> Path | None:
    for m in scan_hf_cache():
        if m["name"] == repo_id:
            return Path(m["path"])
    return None


def pull(
    repo_id: str = typer.Argument(
        help="Hugging Face repo id, e.g. mlx-community/Qwen3.5-9B-MLX-4bit."
    ),
    models_dir: str | None = typer.Option(
        None, "--dir", "-d", help="Download under this directory (default: models dir)."
    ),
    revision: str | None = typer.Option(
        None,
        "--revision",
        "-r",
        help="Branch, tag or commit; resolve it even when the model is already cached.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Download even if the model is already on disk. Only missing or "
        "changed files are fetched; nothing is deleted.",
    ),
):
    """Download a model from Hugging Face into the models directory.

    Refuses to download a model that is already on disk (models directory or
    Hugging Face cache); an interrupted download is resumed.
    """
    from huggingface_hub.utils import HFValidationError, validate_repo_id

    try:
        validate_repo_id(repo_id)
    except HFValidationError:
        fail(f"Invalid Hugging Face repo id: {repo_id!r}.", code=2)
    parts = repo_id.split("/")
    if len(parts) != 2 or not all(parts):
        fail(f"Expected a Hugging Face repo id like org/name, got {repo_id!r}.", code=2)
    base = Path(models_dir).expanduser() if models_dir else _get_models_dir()
    target = base / parts[0] / parts[1]

    # A complete directory/cache proves availability, not the requested revision.
    # Let the Hub resolve explicit revisions; unchanged blobs are still reused.
    if not force and revision is None:
        for candidate in (target, base / parts[1]):
            if weights_complete(candidate)[0]:
                emit(
                    {"status": "present", "path": str(candidate), "repo_id": repo_id},
                    human=lambda c=candidate: console.print(
                        f"[green]Already downloaded[/] at {c} — not downloading "
                        f"again.\nServe it with: [bold]yunshu serve -m {c}[/]\n"
                        "[dim](--force re-checks the files against the Hub.)[/]"
                    ),
                )
                return
        cached = _hf_cached_snapshot(repo_id)
        if cached is not None:
            emit(
                {"status": "present", "path": str(cached), "repo_id": repo_id},
                human=lambda: console.print(
                    f"[green]Already in the Hugging Face cache[/] at {cached} — not "
                    f"downloading again.\nServe it with: [bold]yunshu serve -m "
                    f"{repo_id}[/]\n[dim](--force also copies it into {base}.)[/]"
                ),
            )
            return

    if not is_json():
        verb = "Resuming" if target.exists() else "Downloading"
        console.print(f"[bold]{verb}[/] {repo_id} → {target}")
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(repo_id=repo_id, local_dir=str(target), revision=revision)
    except Exception as e:  # noqa: BLE001
        logger.debug("Model download failed for %s", repo_id, exc_info=True)
        fail(
            f"Download of {repo_id} failed: {e}. Run the same command again to resume.",
            code=1,
        )
    done, reason = weights_complete(target)
    if not done:
        fail(f"Download finished but the model looks incomplete: {reason}.", code=1)
    model_type = _detect_model_type(target)
    emit(
        {
            "status": "downloaded",
            "revision": revision,
            "path": str(target),
            "repo_id": repo_id,
            "type": model_type,
        },
        human=lambda: console.print(
            f"[green]✓ Downloaded[/] {repo_id} ({model_type}) → {target}\n"
            f"Serve it with: [bold]yunshu serve -m {target}[/]"
        ),
    )


# `yunshu pull` is the command; `yunshu model download` stays as its alias.
model_app.command("download", hidden=True)(pull)


def resolve_info_model(model: str) -> Path:
    """Resolve the same local inventory shown by list, refusing ambiguous names."""
    direct = Path(model).expanduser()
    if direct.is_dir() and _is_model_dir(direct):
        return direct
    base = _get_models_dir()
    exact = base / model
    if exact.is_dir() and _is_model_dir(exact):
        return exact
    models = scan_models_dir(base) + scan_hf_cache()
    # Full org/name IDs take precedence over fuzzy matches.
    matches = [m for m in models if m["name"] == model]
    if not matches:
        matches = [m for m in models if model.casefold() in m["name"].casefold()]
    unique = {Path(m["path"]).resolve(): m for m in matches}
    if len(unique) == 1:
        return next(iter(unique))
    if unique:
        names = ", ".join(sorted(m["name"] for m in unique.values()))
        fail(
            f"Model name {model!r} is ambiguous: {names}. Use a full path or org/name.",
            code=2,
        )
    fail(
        f"Model not found: {model}. Run yunshu model list to see local models.", code=1
    )
    raise AssertionError("fail exits")


@model_app.command("info")
def model_info(
    model: str = typer.Argument(help="Model name or path."),
):
    """Show detailed model information."""
    model_path = resolve_info_model(model)

    config_path = model_path / "config.json"
    model_type = _detect_model_type(model_path)
    total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
    num_files = sum(1 for f in model_path.rglob("*") if f.is_file())
    safetensors_files = list(model_path.rglob("*.safetensors"))

    cfg = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            logger.debug("failed to parse config.json", exc_info=True)

    if is_json():
        emit(
            {
                "name": model_path.name,
                "type": model_type,
                "path": str(model_path),
                "total_size": total_size,
                "num_files": num_files,
                "safetensors": len(safetensors_files),
                "config": cfg,
            }
        )
        return

    # Build info tree
    tree = Tree(f"[bold cyan]{model_path.name}[/]")

    info = tree.add("[bold]Basic Info[/]")
    info.add(f"Type: [green]{model_type}[/]")
    info.add(f"Path: {model_path}")
    info.add(f"Total Size: {_format_size(total_size)}")
    info.add(f"Files: {num_files}")
    info.add(f"Safetensors: {len(safetensors_files)}")

    if cfg:
        arch = tree.add("[bold]Architecture[/]")
        for key in (
            "architectures",
            "model_type",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "max_position_embeddings",
            "vocab_size",
            "quantization_config",
        ):
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


_SRV_URL = typer.Option(
    "http://localhost:8000",
    "--url",
    "-u",
    envvar="YUNSHU_GATEWAY_URL",
    help="Server URL.",
)


@model_app.command("load")
def load_model_cmd(
    model: str = typer.Argument(help="Model id/path to load on the running server."),
    url: str = _SRV_URL,
):
    """Load a model on the running server (POST /v1/models/load)."""
    from .infer import _body, _post

    resp = _post(url, "/v1/models/load", json={"model": model}, timeout=600)
    emit(_body(resp), human=lambda: console.print(f"[green]✓ Loaded[/] {model}"))


@model_app.command("unload")
def unload_model_cmd(
    model: str = typer.Argument(help="Model id to unload from the running server."),
    url: str = _SRV_URL,
):
    """Unload a model from the running server (POST /v1/models/unload/{id})."""
    from .infer import _body, _post

    resp = _post(url, f"/v1/models/unload/{model}", json={})
    emit(_body(resp), human=lambda: console.print(f"[green]✓ Unloaded[/] {model}"))


@model_app.command("benchmark")
def benchmark_model(
    model: str = typer.Argument(help="Model name or path."),
    prompt_tokens: int = typer.Option(
        128, "--prompt-tokens", help="Prompt length in tokens."
    ),
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
        fail(f"Model not found: {model}", code=1)

    console.print(f"[bold]Benchmarking[/] {model_path.name}")

    # Load model
    from mlx_lm.utils import load as load_model

    try:
        with console.status("[bold]Loading model..."):
            ml_model, tokenizer = load_model(str(model_path))
    except Exception as e:  # noqa: BLE001
        fail(f"Failed to load model {model_path}: {e}", code=1)

    console.print("[green]✓ Model loaded[/]")

    # Prepare prompt
    prompt = "The quick brown fox jumps over the lazy dog. " * (prompt_tokens // 10 + 1)
    tokens = tokenizer.encode(prompt)[:prompt_tokens]

    console.print(f"Prompt tokens: {len(tokens)}, Max output: {max_tokens}")

    # generate_step signature is (prompt, model, *, max_tokens, sampler); the prompt is an
    # mx.array. (Was imported from the wrong module, only inside the warmup loop so it was
    # unbound when warmup=0, and called with model/tokens reversed + a removed `temp` kwarg.)
    from mlx_lm.generate import generate_step

    _prompt = mx.array(tokens)

    # Warmup
    for _ in range(warmup):
        for _ in generate_step(_prompt, ml_model, max_tokens=16):
            pass
        mx.synchronize()

    # Benchmark runs
    results = []
    for run in range(num_runs):
        t0 = time.perf_counter()
        generated = 0
        for _ in generate_step(_prompt, ml_model, max_tokens=max_tokens):
            generated += 1
        mx.synchronize()
        elapsed = time.perf_counter() - t0
        tok_per_s = generated / elapsed if elapsed > 0 else 0
        results.append(tok_per_s)
        console.print(
            f"  Run {run + 1}: {tok_per_s:.1f} tok/s ({generated} tokens in {elapsed:.2f}s)"
        )

    avg = sum(results) / len(results)

    if is_json():
        emit(
            {
                "model": model_path.name,
                "avg_tok_s": avg,
                "min_tok_s": min(results),
                "max_tok_s": max(results),
                "runs": num_runs,
                "prompt_tokens": prompt_tokens,
                "output_tokens": max_tokens,
            }
        )
        return

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


@model_app.command("rm")
def remove_model(
    model: str = typer.Argument(help="Exact model name under the models directory."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Confirm removal."),
):
    """Remove a downloaded model. HF cache and external/symlinked models are kept."""
    import shutil

    base = _get_models_dir().resolve()
    # Deletion uses exact inventory names only, never fuzzy resolution or arbitrary paths.
    matches = [m for m in scan_models_dir(base) if m["name"] == model]
    if not matches:
        fail(f"No downloaded model named {model!r}. Run `yunshu models list`.", code=2)
    target = Path(matches[0]["path"])
    if (
        not target.resolve().is_relative_to(base)
        or target.is_symlink()
        or target.parent.is_symlink()
    ):
        fail(
            "Refusing to remove a symlinked or external model. Manage its source directly.",
            code=2,
        )
    if not yes:
        if is_json():
            fail("Removal needs --yes: yunshu models rm <org/name> --yes", code=2)
        if not typer.confirm(f"Remove {model} from {target}?"):
            emit(
                {"removed": False, "name": model},
                human=lambda: typer.echo("Kept model."),
            )
            return
    try:
        shutil.rmtree(target)
    except OSError as exc:
        fail(f"Cannot remove {target}: {exc}. Check directory permissions.")
    emit(
        {"removed": True, "name": model, "path": str(target)},
        human=lambda: typer.echo(f"Removed {model}"),
    )


model_app.command("pull")(pull)
model_app.command("show")(model_info)
