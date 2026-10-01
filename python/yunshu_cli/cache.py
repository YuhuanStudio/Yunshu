"""Yunshu CLI — cache: check and clean the SSD prefix caches.

``yunshu cache status`` reports what is on disk; ``yunshu cache gc`` removes stale namespaces (unused for YUNSHU_CACHE_STALE_DAYS, or whose checkpoint is gone), truncated /
corrupt / old-format entries and orphaned temp files, and trims each cache to its size cap.
``gc`` is a dry run until ``--apply``.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from yunshu_engine import paths, settings

from ._output import emit

console = Console()
cache_app = typer.Typer(
    help="Check and clean the SSD prefix caches.", no_args_is_help=True
)


def cache_targets() -> list[tuple[str, Path, int | None]]:
    """``(label, directory, cap in bytes)`` for each SSD cache that is configured."""
    out: list[tuple[str, Path, int | None]] = []
    apc = paths.apc_dir()
    if apc is not None:
        from yunshu_kv.disk_budget import resolve_cap_gb

        gb = resolve_cap_gb(settings.get("YUNSHU_VLM_APC_DISK_GB"), apc)
        out.append(("apc", apc, int(gb * (1 << 30)) if gb and gb > 0 else None))
    text = settings.get("YUNSHU_SSD_CACHE_DIR")
    if text and settings.get_bool("YUNSHU_SSD_CACHE"):
        gb = settings.get("YUNSHU_SSD_CACHE_MAX_GB")
        out.append(
            ("kv-ssd", Path(text).expanduser(), int(gb * (1 << 30)) if gb else None)
        )
    return out


def _gib(n: int) -> str:
    return f"{n / (1 << 30):.2f} GiB"


def _run(apply: bool) -> list[dict]:
    from yunshu_kv import cache_gc, disk_budget

    rows = []
    stale_days = float(settings.get("YUNSHU_CACHE_STALE_DAYS"))
    for label, directory, cap in cache_targets():
        budget = disk_budget.DiskBudget(
            directory,
            cap_bytes=cap or 0,
            reserve_pct=float(settings.get("YUNSHU_CACHE_RESERVE_PCT")),
            reserve_min_bytes=int(
                float(settings.get("YUNSHU_CACHE_RESERVE_GB")) * 2**30
            ),
        )
        try:
            effective = budget.effective_cap() if directory.is_dir() else cap
        except OSError:
            effective = cap
        rep = cache_gc.scan(
            directory, max_bytes=effective, apply=apply, stale_days=stale_days
        )
        rows.append(
            {
                "cache": label,
                "dir": str(directory),
                "exists": directory.is_dir(),
                "files": rep.files,
                "valid_bytes": rep.bytes,
                "cap_bytes": cap,
                "effective_cap_bytes": effective,
                "namespaces": rep.namespaces,
                "stale_namespaces": rep.stale_namespaces,
                "problems": rep.by_reason(),
                "reclaimable_bytes": sum(f.bytes for f in rep.findings),
                "freed_bytes": rep.freed,
                "applied": apply,
            }
        )
    return rows


@cache_app.command("status")
def status():
    """Size, cap and problems of each SSD cache (changes nothing)."""
    rows = _run(apply=False)

    def human():
        if not rows:
            console.print("No SSD cache is enabled.")
        for r in rows:
            cap = _gib(r["cap_bytes"]) if r["cap_bytes"] else "uncapped"
            console.print(
                f"[bold]{r['cache']}[/] {r['dir']}: {r['files']} files, "
                f"{_gib(r['valid_bytes'])} (cap {cap})"
            )
            for ns, n in sorted(r["namespaces"].items(), key=lambda kv: -kv[1]):
                tag = " [stale]" if ns in r["stale_namespaces"] else ""
                console.print(f"    {ns}: {_gib(n)}{tag}")
            if r["problems"]:
                console.print(
                    f"  [yellow]{r['problems']}[/] — {_gib(r['reclaimable_bytes'])} "
                    "reclaimable: `yunshu cache gc --apply`"
                )

    emit({"caches": rows}, human=human)


@cache_app.command("gc")
def gc(
    apply: bool = typer.Option(
        False, "--apply", help="Delete what is found (default: report only)."
    ),
):
    """Remove truncated, corrupt and old-format entries and orphaned temp files; enforce the cap.

    Safe while the server runs (a missing entry is a cache miss), but the cap trim is
    cleanest with the server stopped.
    """
    rows = _run(apply=apply)

    def human():
        if not rows:
            console.print("No SSD cache is enabled.")
        for r in rows:
            verb = "freed" if apply else "would free"
            n = _gib(r["freed_bytes"] if apply else r["reclaimable_bytes"])
            console.print(
                f"[bold]{r['cache']}[/] {r['dir']}: {r['problems'] or 'clean'} — "
                f"{verb} {n}"
            )
        if rows and not apply and any(r["problems"] for r in rows):
            console.print("Run with --apply to delete.")

    emit({"caches": rows, "applied": apply}, human=human)
