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
    from yunshu_kv.disk_budget import resolve_cap_gb

    cap = resolve_cap_gb(settings.get("YUNSHU_VLM_APC_DISK_GB"), d)
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
    try:
        from yunshu_kv import disk_budget

        per_ns: dict[str, int] = {}
        for e in disk_budget.scan_root(d):
            per_ns[e.ns] = per_ns.get(e.ns, 0) + e.size
        if len(per_ns) > 1:
            msg += "; namespaces: " + ", ".join(
                f"{k} {v / 1024**3:.1f}" for k, v in sorted(per_ns.items())
            )
    except Exception:
        pass
    if free < min(cap, 20.0):
        return Check(
            "prefix cache disk",
            "warn",
            msg,
            "Little free space: lower YUNSHU_VLM_APC_DISK_GB, point "
            "YUNSHU_VLM_APC_DISK_DIR at another volume, or set YUNSHU_VLM_APC_DISK=0.",
        )
    return Check("prefix cache disk", "ok", msg)


# Packages whose minimum version is enforced; the minimums themselves come from this
# release's own dependency metadata (pyproject.toml), so they cannot drift.
CHECKED_PACKAGES = (
    "mlx",
    "mlx-lm",
    "mlx-vlm",
    "mlx-audio",
    "llguidance",
    "transformers",
)


def min_versions() -> dict[str, str]:
    """``name -> minimum version`` for CHECKED_PACKAGES, read from the yunshu distribution's
    requirements (a lower version has known breakage, so it is a failure, not a warning)."""
    from importlib.metadata import PackageNotFoundError, requires

    from packaging.requirements import Requirement

    try:
        reqs = requires("yunshu") or []
    except PackageNotFoundError:
        return {}
    out: dict[str, str] = {}
    for line in reqs:
        r = Requirement(line)
        if r.name not in CHECKED_PACKAGES:
            continue
        for spec in r.specifier:
            if spec.operator == ">=":
                out[r.name] = spec.version
    return out


def check_prefix_tiers() -> Check | None:
    """The lower APC storage tiers (YUNSHU_VLM_APC_DISK_TIERS): mounted or not, free space, and
    the bandwidth measured at the last startup (never probed here)."""
    raw = settings.get("YUNSHU_VLM_APC_DISK_TIERS")
    if not raw or paths.apc_dir() is None:
        return None
    from yunshu_engine.apc_storage import (
        ProfileStore,
        device_name,
        mount_of,
        parse_tiers,
    )

    store = ProfileStore(paths.home() / "cache" / "apc-device-profiles.json")
    parts, warn = [], False
    for spec in parse_tiers(raw):
        probe = spec.path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        mounted = spec.path.exists() or (
            not str(spec.path).startswith(("/Volumes/", "/mnt/", "/media/"))
        )
        if not mounted:
            parts.append(f"{spec.path} NOT MOUNTED (skipped until it is)")
            warn = True
            continue
        try:
            free = f"{shutil.disk_usage(probe).free / 1024**3:.0f} GiB free"
        except OSError:
            free, warn = "free space unreadable", True
        prof = store.get(str(mount_of(spec.path)), ttl_s=float("inf"))
        speed = (
            f", read {prof.read_bps / 1e6:.0f} MB/s, {prof.latency_s * 1e3:.1f} ms"
            if prof
            else ", not profiled yet (measured at the next start)"
        )
        parts.append(f"{spec.path} ({device_name(spec.path)}, {free}{speed})")
    return Check(
        "prefix cache tiers",
        "warn" if warn else "ok",
        "; ".join(parts),
        "Plug in or mount the volume, or remove it from YUNSHU_VLM_APC_DISK_TIERS."
        if warn
        else "",
    )


# Minimums mirror pyproject.toml; a lower version has known breakage (APC, MTP, llguidance
# schemas), so it is a failure with the upgrade command, not a warning.
MIN_VERSIONS = {
    "mlx": "0.32.3",
    "mlx-lm": "0.31.3",
    "mlx-vlm": "0.7.4",
    "mlx-audio": "0.5.7",
    "llguidance": "1.8",
}
# extra name -> (package, what it enables)
EXTRAS = {
    "vision": ("mlx-vlm", "image/video input and the Qwen3.5 / 3.6 / 3.8 family"),
    "audio": ("mlx-audio", "speech-to-text, text-to-speech and the Realtime voice WS"),
}


def _vkey(v: str) -> tuple[int, ...]:
    out = []
    for part in v.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def check_versions(pkg=_pkg) -> list[Check]:
    """Installed packages below the minimum this release needs."""
    out = []
    for name, minimum in min_versions().items():
        have = pkg(name)
        if have is None or _vkey(have) >= _vkey(minimum):
            continue
        out.append(
            Check(
                f"version {name}",
                "fail",
                f"{name} {have} < {minimum}",
                f"Upgrade: `uv pip install -U '{name}>={minimum}'` (or reinstall "
                "Yunshu).",
            )
        )
    if not out:
        out.append(
            Check("versions", "ok", "mlx / mlx-lm / mlx-vlm / llguidance current")
        )
    return out


def check_extras(pkg=_pkg) -> list[Check]:
    """Optional extras (what is installed, what each one unlocks) and llguidance."""
    out = []
    for extra, (name, enables) in EXTRAS.items():
        have = pkg(name)
        if name == "mlx-vlm":
            continue  # reported by check_mlx with its own fix text
        out.append(
            Check(
                f"extra {extra}",
                "ok" if have else "warn",
                f"{name} {have}" if have else f"{name} not installed ({enables})",
                "" if have else f"Install `yunshu[{extra}]`.",
            )
        )
    llg = pkg("llguidance")
    out.append(
        Check(
            "llguidance",
            "ok" if llg else "fail",
            llg or "not installed",
            ""
            if llg
            else "CFG grammars and JSON schemas beyond the in-house subset need it: "
            "reinstall Yunshu (it is a core dependency).",
        )
    )
    return out


def check_api_features(host: str) -> list[Check]:
    """State of the API features that depend on configuration."""
    out = []
    token = settings.get("YUNSHU_AUTH_TOKEN")
    disabled = settings.get_bool("YUNSHU_AUTH_DISABLED")
    exposed = host not in ("127.0.0.1", "localhost", "::1")
    if exposed and not token:
        out.append(
            Check(
                "auth",
                "warn",
                f"serving on {host} with no YUNSHU_AUTH_TOKEN: inference is open to "
                "the network",
                "Set YUNSHU_AUTH_TOKEN, or bind 127.0.0.1.",
            )
        )
    elif disabled:
        out.append(
            Check(
                "auth",
                "warn",
                "YUNSHU_AUTH_DISABLED is on: operational endpoints are open",
                "Unset it, or set YUNSHU_AUTH_TOKEN.",
            )
        )
    else:
        out.append(Check("auth", "ok", "token set" if token else "local, open"))
    provider = settings.get("YUNSHU_WEB_SEARCH_PROVIDER")
    fetch = settings.get_bool("YUNSHU_WEB_FETCH")
    out.append(
        Check(
            "server tools",
            "ok",
            f"web_search provider={provider}, web_fetch={'on' if fetch else 'off'}",
        )
    )
    return out


def _gib(n: float) -> str:
    return f"{n / 1024**3:.1f} GiB"


def check_disk_budget(models_base: Path, free=None) -> list[Check]:
    """Models + APC SSD cache against the free space of their volumes."""
    from .cache import cache_targets

    free = free or (lambda p: shutil.disk_usage(p).free)
    out = []
    for label, directory, cap in cache_targets():
        probe = directory
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            avail = free(probe)
        except OSError:
            continue
        if cap and cap > avail:
            out.append(
                Check(
                    f"disk {label} cache",
                    "warn",
                    f"cap {_gib(cap)} > {_gib(avail)} free on {probe}",
                    "Lower the cap (YUNSHU_VLM_APC_DISK_GB / YUNSHU_SSD_CACHE_MAX_GB)"
                    " or move the cache with YUNSHU_VLM_APC_DISK_DIR.",
                )
            )
        else:
            out.append(
                Check(
                    f"disk {label} cache",
                    "ok",
                    f"{directory}: {_gib(_dir_bytes(directory)) if directory.exists() else '0'}"
                    f" used (all namespaces), cap {_gib(cap) if cap else 'none'}, "
                    f"{_gib(avail)} free",
                )
            )
    return out


def check_cache_integrity() -> list[Check]:
    """Corrupt / truncated / old-format cache entries and orphaned temp files."""
    from yunshu_kv import cache_gc

    from .cache import cache_targets

    out = []
    for label, directory, cap in cache_targets():
        rep = cache_gc.scan(directory, max_bytes=cap)
        if not rep.findings:
            continue
        out.append(
            Check(
                f"cache {label}",
                "warn",
                f"{directory}: {rep.by_reason()} "
                f"({_gib(sum(f.bytes for f in rep.findings))} reclaimable)",
                "Run `yunshu cache gc --apply`.",
            )
        )
    if not out:
        out.append(Check("cache integrity", "ok", "no damaged or orphaned entries"))
    return out


def check_downloads(base: Path) -> list[Check]:
    """Half-downloaded models in the models directory."""
    from .model import scan_models_dir, weights_complete

    if not base.is_dir():
        return []
    out = []
    names = {m["name"] for m in scan_models_dir(base)}
    candidates = [
        c for c in sorted(base.iterdir()) if c.is_dir() and not c.name.startswith(".")
    ]
    candidates += [
        g
        for c in candidates
        if c.name not in names
        for g in sorted(c.iterdir())
        if g.is_dir() and not g.name.startswith(".")
    ]
    for c in candidates:
        partial = c / ".cache" / "huggingface" / "download"
        if partial.is_dir() and any(partial.rglob("*.incomplete")):
            out.append(
                Check(
                    "download",
                    "warn",
                    f"{c.relative_to(base)}: interrupted download",
                    f"Resume with `yunshu pull {c.relative_to(base)}`.",
                )
            )
            continue
        if (c / "config.json").exists():
            done, reason = weights_complete(c)
            if not done:
                out.append(
                    Check(
                        "download",
                        "warn",
                        f"{c.relative_to(base)}: {reason}",
                        f"Resume with `yunshu pull {c.relative_to(base)}`.",
                    )
                )
    if not out:
        out.append(Check("downloads", "ok", "no half-downloaded models"))
    return out


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
    checks += check_versions()
    checks += check_extras()
    checks += check_settings()
    checks += check_models_dir(paths.models_dir())
    checks += check_downloads(paths.models_dir())
    checks += check_disk_budget(paths.models_dir())
    checks += check_cache_integrity()
    checks += check_api_features(host)
    if model:
        checks += check_model(model, info)
        checks += check_speculative(model)
    checks.append(check_port(host, port))
    checks.append(check_prefix_disk())
    tiers = check_prefix_tiers()
    if tiers is not None:
        checks.append(tiers)
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
