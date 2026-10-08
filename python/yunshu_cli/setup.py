"""First-run model selection; downloads require a user-selected repository."""

import sys
from pathlib import Path

import typer

from ._output import emit, fail, is_json


def _interactive() -> bool:
    return sys.stdin.isatty()


def select_model() -> str:
    from .model import (
        _get_models_dir,
        pull,
        scan_hf_cache,
        scan_models_dir,
        weights_complete,
    )

    models = scan_models_dir(_get_models_dir()) + scan_hf_cache()
    models = list(
        {m["path"]: m for m in models if weights_complete(Path(m["path"]))[0]}.values()
    )
    if is_json() or not _interactive():
        fail(
            "Choose a model with `yunshu models list`, then `yunshu serve -m <path or org/name>`. "
            "Download with `yunshu models pull <org/name>`; interactive selection: `yunshu setup`.",
            code=2,
            models=models,
            next_steps=[
                "yunshu models list",
                "yunshu models pull <org/name>",
                "yunshu serve -m <path or org/name>",
            ],
        )
    for i, m in enumerate(models, 1):
        typer.echo(f"{i}. {m['name']} ({m['path']})")
    typer.echo("0. Download a Hugging Face MLX model (org/name)")
    choice = typer.prompt("Select model", type=int, default=0)
    if 1 <= choice <= len(models):
        return str(models[choice - 1]["path"])
    if choice != 0:
        fail("Invalid selection. Run `yunshu setup` again.", code=2)
    repo = typer.prompt("MLX repository (for example Jundot/Qwen3.8-27B-oQ4e-mtp)")
    pull(repo, models_dir=None, revision=None, force=False)
    from .model import resolve_info_model

    return str(resolve_info_model(repo))


def setup() -> None:
    """Select or download a model and print the next command (does not start a server)."""
    import shlex

    model = select_model()
    command = f"yunshu serve -m {shlex.quote(model)}"
    emit({"model": model, "next_command": command}, human=lambda: typer.echo(command))
