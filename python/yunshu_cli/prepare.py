"""Yunshu CLI — prepare: one-time repack of a Flash-Next model's PLE table for faster cold starts.

``yunshu prepare <model>`` checks free disk first, repacks with a progress bar, skips a model that
is already prepared and picks up where a killed run left off (by rebuilding it); the server then uses
the repack automatically. ``--remove`` deletes it again.
"""

from __future__ import annotations

import time

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from ._output import emit, fail, is_json
from .model import resolve_info_model

console = Console()


def _gib(n: float) -> str:
    return f"{n / 2**30:.1f} GiB"


def prepare(
    model: str = typer.Argument(..., help="Model name, org/name or path."),
    remove: bool = typer.Option(
        False, "--remove", help="Delete the prepared copy (the model is untouched)."
    ),
) -> None:
    """Repack a Flash-Next model's n-gram table once so cold first tokens are 5-25% faster."""
    from yunshu_engine import ple_prepare

    path = resolve_info_model(model)
    info = ple_prepare.status(path)
    if not info["applicable"]:
        fail(
            f"{path.name} has no n-gram (PLE) table to prepare; only Qwen3.8-Flash-Next "
            "checkpoints benefit.",
            code=1,
        )
    if remove:
        freed = ple_prepare.remove(path)
        emit(
            {"model": str(path), "removed": freed > 0, "freed_bytes": freed},
            human=lambda: console.print(
                f"Removed the prepared copy of {path.name} ({_gib(freed)} freed)."
                if freed
                else f"{path.name} was not prepared."
            ),
        )
        return
    if info["prepared"]:
        emit(
            {"model": str(path), "prepared": True, "skipped": True, **info},
            human=lambda: console.print(
                f"{path.name} is already prepared ({_gib(info['prepared_bytes'])} in "
                f"{info['rows_dir']}); the server uses it automatically."
            ),
        )
        return
    try:
        job = ple_prepare.start_job(path)
    except ple_prepare.InsufficientDiskError as exc:
        fail(str(exc), code=1, needed_bytes=exc.needed, free_bytes=exc.free)
    started = time.monotonic()
    try:
        with Progress(
            TextColumn(f"Preparing {path.name}"),
            BarColumn(),
            DownloadColumn(binary_units=True),
            TimeRemainingColumn(),
            console=console,
            disable=is_json(),
        ) as bar:
            task = bar.add_task("", total=job.total)
            while job.state() == "running":
                bar.update(task, completed=job.poll()["bytes_done"])
                time.sleep(0.5)
            bar.update(task, completed=job.total if job.state() == "done" else None)
    except KeyboardInterrupt:
        job.cancel()
        fail(
            "Cancelled; nothing was kept. Run the command again to start over.",
            code=130,
        )
    result = job.poll()
    if result["state"] != "done":
        fail(f"Prepare failed: {result['error']}", code=1)
    after = ple_prepare.status(path)
    emit(
        {
            "model": str(path),
            "prepared": True,
            "seconds": time.monotonic() - started,
            **after,
        },
        human=lambda: console.print(
            f"Prepared {path.name} in {time.monotonic() - started:.0f} s "
            f"({_gib(after['prepared_bytes'])} in {after['rows_dir']}).\n"
            "The server uses it automatically (storage ssd-rows). "
            f"Undo with: yunshu prepare {model} --remove"
        ),
    )
