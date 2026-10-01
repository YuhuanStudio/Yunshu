"""Yunshu CLI — doctor: can this machine serve, and if not, what to do.

Every check returns a status (ok / warn / fail), what was found, and for
anything not ok, the fix. ``yunshu doctor`` exits 1 when a check fails, so it
also works as a pre-flight step in scripts. ``yunshu serve`` runs the model
checks (path, memory) itself before it starts.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from yunshu_engine import paths, settings

from ._output import emit, is_json

console = Console()

MIN_MACOS = (14, 0)
MIN_PYTHON = (3, 13)
# Weights above this share of the GPU working set leave too little for the KV
# cache and activations of a long prompt.
WEIGHTS_WARN_FRACTION = 0.8


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail
    detail: str
    fix: str = ""


def _sysctl(key: str) -> str:
    with contextlib.suppress(Exception):
        return subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    return ""


def check_platform() -> list[Check]:
    out = []
    if sys.platform != "darwin" or platform.machine() != "arm64":
        translated = _sysctl("sysctl.proc_translated") == "1"
        if sys.platform == "darwin" and translated:
            out.append(
                Check(
                    "platform",
                    "fail",
                    "Python runs under Rosetta (x86_64) on an Apple Silicon Mac",
                    "Use a native arm64 Python: `uv python install 3.13`, then "
                    "reinstall Yunshu with it.",
                )
            )
        else:
            out.append(
                Check(
                    "platform",
                    "fail",
                    f"{sys.platform} / {platform.machine()}",
                    "Yunshu runs on MLX, which needs a Mac with Apple Silicon.",
                )
            )
        return out
    chip = _sysctl("machdep.cpu.brand_string") or "Apple Silicon"
    out.append(Check("platform", "ok", f"{chip}, arm64"))
    ver = platform.mac_ver()[0]
    try:
        parts = tuple(int(x) for x in ver.split(".")[:2])
        major_minor = (parts + (0,))[:2]
    except ValueError:
        major_minor = (0, 0)
    if major_minor < MIN_MACOS:
        out.append(
            Check(
                "macOS",
                "fail",
                f"macOS {ver}",
                f"MLX needs macOS {MIN_MACOS[0]}.{MIN_MACOS[1]} or newer.",
            )
        )
    else:
        out.append(Check("macOS", "ok", f"macOS {ver}"))
    return out


def check_python() -> Check:
    v = sys.version_info
    if (v.major, v.minor) < MIN_PYTHON:
        return Check(
            "python",
            "fail",
            f"Python {v.major}.{v.minor}",
            "Yunshu needs Python 3.13+: `uv python install 3.13`.",
        )
    return Check("python", "ok", f"Python {v.major}.{v.minor}.{v.micro}")


def _pkg(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_mlx() -> tuple[list[Check], dict]:
    """MLX + Metal, and the GPU memory facts later checks need."""
    info: dict = {}
    try:
        import mlx.core as mx
    except Exception as exc:  # noqa: BLE001 - ImportError or a broken wheel
        return [
            Check(
                "mlx",
                "fail",
                f"cannot import mlx: {exc}",
                "Reinstall Yunshu (MLX is a core dependency) with a native arm64 "
                "Python.",
            )
        ], info
    if not mx.metal.is_available():
        return [
            Check(
                "mlx",
                "fail",
                f"MLX {mx.__version__}, Metal not available",
                "Run on the Mac itself (not in a VM or a sandbox without GPU access).",
            )
        ], info
    with contextlib.suppress(Exception):
        info = dict(mx.device_info())
    device = info.get("device_name", "GPU")
    checks = [Check("mlx", "ok", f"MLX {mx.__version__} on {device} (Metal)")]
    vlm = _pkg("mlx-vlm")
    lm = _pkg("mlx-lm")
    checks.append(
        Check(
            "mlx-lm",
            "ok" if lm else "fail",
            lm or "not installed",
            "" if lm else "Reinstall Yunshu; mlx-lm is a core dependency.",
        )
    )
    checks.append(
        Check(
            "mlx-vlm",
            "ok" if vlm else "warn",
            vlm or "not installed",
            ""
            if vlm
            else "Vision models and the Qwen3.5 / 3.6 / 3.8 family need the vision "
            "extra: install `yunshu[vision]` (source checkout: `uv sync --extra "
            "vision`).",
        )
    )
    return checks, info


def check_memory(info: dict) -> Check:
    total = info.get("memory_size") or int(_sysctl("hw.memsize") or 0)
    working = info.get("max_recommended_working_set_size")
    if not total:
        return Check("memory", "warn", "unknown", "")
    detail = f"{total / 1024**3:.0f} GiB unified"
    if working:
        detail += f", GPU working set {working / 1024**3:.0f} GiB"
    return Check("memory", "ok", detail)


def weights_bytes(model_path: Path) -> int:
    return sum(f.stat().st_size for f in model_path.rglob("*.safetensors"))


def check_model(model: str, info: dict) -> list[Check]:
    """Does ``model`` (a path or a Hugging Face repo id) exist, and does it fit?"""
    from .model import _hf_cached_snapshot, weights_complete

    p = Path(model).expanduser()
    looks_like_path = model.startswith(("/", ".", "~")) or p.exists()
    resolved: Path | None = None
    if looks_like_path:
        if not p.exists():
            return [
                Check(
                    "model",
                    "fail",
                    f"{model}: no such directory",
                    "Check the path, list what is on disk with `yunshu model list`, "
                    "or download a model with `yunshu pull <org/name>`.",
                )
            ]
        done, reason = _model_complete(p, weights_complete)
        if not done:
            # A half-downloaded model is certain to fail; other layouts (npz
            # weights, component folders) may still load, so only warn.
            broken = reason == "interrupted download" or "missing" in reason
            return [
                Check(
                    "model",
                    "fail" if broken else "warn",
                    f"{model}: {reason}",
                    "Run `yunshu pull <org/name>` again to resume the download."
                    if broken
                    else "Expected a folder with config.json and *.safetensors "
                    "weights; a model directory goes to --models-dir.",
                )
            ]
        resolved = p
    else:
        cached = _hf_cached_snapshot(model)
        if cached is None:
            return [
                Check(
                    "model",
                    "warn",
                    f"{model}: not on disk yet",
                    f"It will be downloaded on first start. To download it now: "
                    f"`yunshu pull {model}`.",
                )
            ]
        resolved = cached
    size = weights_bytes(resolved)
    checks = [Check("model", "ok", f"{resolved} ({size / 1024**3:.1f} GiB weights)")]
    total = info.get("memory_size") or int(_sysctl("hw.memsize") or 0)
    working = info.get("max_recommended_working_set_size") or total
    if total and size > total:
        checks.append(
            Check(
                "model fits",
                "fail",
                f"weights {size / 1024**3:.1f} GiB > {total / 1024**3:.0f} GiB memory",
                "Use a smaller or more quantized model (for example a 4-bit MLX "
                "build).",
            )
        )
    elif working and size > WEIGHTS_WARN_FRACTION * working:
        checks.append(
            Check(
                "model fits",
                "warn",
                f"weights {size / 1024**3:.1f} GiB use over "
                f"{WEIGHTS_WARN_FRACTION:.0%} of the {working / 1024**3:.0f} GiB GPU "
                "working set",
                "Long prompts may run out of memory; consider a smaller quantization.",
            )
        )
    elif working:
        checks.append(
            Check(
                "model fits",
                "ok",
                f"{size / working:.0%} of the GPU working set",
            )
        )
    return checks


def _model_complete(p: Path, weights_complete) -> tuple[bool, str]:
    """A model folder, or a models directory handed to --model by mistake."""
    done, reason = weights_complete(p)
    if (
        not done
        and reason == "no config.json"
        and any(c.is_dir() and (c / "config.json").exists() for c in p.iterdir())
    ):
        return False, "no config.json here, but its subfolders are models"
    return done, reason


def check_models_dir(base: Path) -> list[Check]:
    from .model import scan_models_dir

    if not base.exists():
        return [
            Check(
                "models dir",
                "warn",
                f"{base} does not exist",
                "Created on the first `yunshu pull`; or set YUNSHU_MODELS_DIR to "
                "where your models live.",
            )
        ]
    n = len(scan_models_dir(base))
    checks = [Check("models dir", "ok", f"{base} ({n} models)")]
    free = shutil.disk_usage(base).free
    if free < 20 * 1024**3:
        checks.append(
            Check(
                "disk space",
                "warn",
                f"{free / 1024**3:.0f} GiB free on the models volume",
                "Large models need tens of GiB; free space or set YUNSHU_MODELS_DIR "
                "to another volume.",
            )
        )
    return checks


def check_speculative(model: str) -> list[Check]:
    """Which speculative path serving ``model`` will use."""
    import json

    from yunshu_engine import model_discovery, spec_select
    from yunshu_engine.mlxvlm_mtp import is_mtp_capable

    resolved = model_discovery.resolve_model_ref(model)
    cfg_path = Path(resolved or model).expanduser() / "config.json"
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, ValueError):
        return []
    family = cfg.get("model_type") in ("qwen3_5", "qwen3_6", "qwen3_5_moe")
    choice = spec_select.choose(
        cfg,
        spec_family=family,
        mtp_capable=family and is_mtp_capable(str(cfg_path.parent)),
    )
    if choice.kind == "dflash":
        detail = f"DFlash2 drafter: {choice.drafter} ({choice.reason})"
        return [Check("speculative", "ok", detail)]
    if choice.kind == "mtp":
        hint = ""
        if spec_select.is_27b_class(cfg):
            hint = (
                "`yunshu pull incoai/Qwen3.8-27B-DFlash2` enables the faster "
                "DFlash2 drafter."
            )
        return [Check("speculative", "ok", f"MTP head ({choice.reason})", hint)]
    return [Check("speculative", "ok", f"none ({choice.reason})")]


def check_port(host: str, port: int) -> Check:
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "", "localhost") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        in_use = s.connect_ex((probe_host, port)) == 0
    if not in_use:
        return Check("port", "ok", f"{host}:{port} is free")
    if _is_yunshu(probe_host, port):
        return Check("port", "ok", f"Yunshu is already running on {probe_host}:{port}")
    return Check(
        "port",
        "warn",
        f"{probe_host}:{port} is used by another program",
        f"Start Yunshu on another port: `yunshu serve --port {port + 1}`.",
    )


def _is_yunshu(host: str, port: int) -> bool:
    with contextlib.suppress(Exception):
        import httpx

        r = httpx.get(f"http://{host}:{port}/version", timeout=1)
        return r.status_code == 200 and r.json().get("service") == "yunshu"
    return False


def check_settings() -> list[Check]:
    try:
        warnings = settings.validate(warn=False)
    except settings.SettingError as exc:
        return [
            Check(
                "settings",
                "fail",
                str(exc),
                "Fix the value (see `yunshu config` and docs/CONFIGURATION.md).",
            )
        ]
    if warnings:
        return [
            Check("settings", "warn", w, "See `yunshu config --all`.") for w in warnings
        ]
    return [Check("settings", "ok", "all YUNSHU_* values parse")]


def _dir_bytes(d: Path) -> int:
    total = 0
    for f in d.rglob("*"):
        with contextlib.suppress(OSError):
            if f.is_file():
                total += f.stat().st_size
    return total


def check_prefix_disk() -> Check:
    """The APC SSD tier: where it lives, what it holds, whether the disk can carry it."""
    d = paths.apc_dir()
    if d is None:
        return Check(
            "prefix cache disk",
            "ok",
            "off (YUNSHU_VLM_APC_DISK=0): evicted prefixes are re-prefilled",
        )
    cap = float(settings.get("YUNSHU_VLM_APC_DISK_GB") or 0)
    probe = d
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free / 1024**3
    except OSError as exc:
        return Check(
            "prefix cache disk",
            "warn",
            f"{d}: cannot read free space ({exc})",
            "Set YUNSHU_VLM_APC_DISK_DIR to a writable volume, or YUNSHU_VLM_APC_DISK=0.",
        )
    used = _dir_bytes(d) / 1024**3 if d.exists() else 0.0
    msg = f"{d} ({used:.1f} GiB used, cap {cap:.0f} GiB, {free:.0f} GiB free)"
    if free < min(cap, 20.0):
        return Check(
            "prefix cache disk",
            "warn",
            msg,
            "Little free space: lower YUNSHU_VLM_APC_DISK_GB, point "
            "YUNSHU_VLM_APC_DISK_DIR at another volume, or set YUNSHU_VLM_APC_DISK=0.",
        )
    return Check("prefix cache disk", "ok", msg)


def check_service() -> Check:
    plist = paths.launch_agent_plist()
    if plist.exists():
        return Check("service", "ok", f"launchd agent installed ({plist})")
    return Check(
        "service",
        "ok",
        "not installed (optional: `yunshu service install` runs Yunshu at login)",
    )


def run_checks(model: str | None, host: str, port: int) -> list[Check]:
    checks = check_platform()
    checks.append(check_python())
    if any(c.status == "fail" for c in checks):
        return checks
    mlx_checks, info = check_mlx()
    checks += mlx_checks
    checks.append(check_memory(info))
    checks += check_settings()
    checks += check_models_dir(paths.models_dir())
    if model:
        checks += check_model(model, info)
        checks += check_speculative(model)
    checks.append(check_port(host, port))
    checks.append(check_prefix_disk())
    checks.append(check_service())
    return checks


_MARK = {"ok": "[green]✓[/]", "warn": "[yellow]![/]", "fail": "[red]✗[/]"}


def doctor(
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Also check this model (path or repo id). Default: YUNSHU_MODEL.",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Host you will serve on."),
    port: int = typer.Option(8000, "--port", "-p", help="Port you will serve on."),
):
    """Check that this Mac can run Yunshu, and say how to fix what cannot."""
    checks = run_checks(model or settings.get("YUNSHU_MODEL"), host, port)
    failed = any(c.status == "fail" for c in checks)
    from yunshu_engine.version import yunshu_version

    if is_json():
        emit(
            {
                "version": yunshu_version(),
                "ok": not failed,
                "checks": [asdict(c) for c in checks],
            }
        )
    else:
        table = Table(title=f"Yunshu {yunshu_version()} — doctor", show_lines=False)
        table.add_column("", width=1)
        table.add_column("Check", style="bold")
        table.add_column("Found")
        table.add_column("Fix", style="cyan")
        for c in checks:
            table.add_row(_MARK[c.status], c.name, c.detail, c.fix)
        console.print(table)
        if failed:
            console.print("[red]Some checks failed — fix those before serving.[/]")
        else:
            console.print("[green]Ready to serve.[/]")
    if failed:
        raise typer.Exit(1)
