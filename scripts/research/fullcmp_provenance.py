#!/usr/bin/env python3
"""Provenance for one full-comparison cell (FULLCMP_SPEC.md, "Hard guards" and "Telemetry").

Used by tfbench.py when TFB_FULLCMP=1. Everything here fails closed: a missing or mismatched
value raises ProvenanceError, and the harness then writes no cell result.

  * Yunshu pin: TFB_YUNSHU_BIN and TFB_YUNSHU_SRC must equal the venv / worktree recorded in
    competitors/compsync/fullcmp-1103/yunshu_pin.json. The server is then started as
    <pinned venv python> -m yunshu_cli and verified after launch (status version + process cmdline).
    The main checkout .venv is never used.
  * Census lookups (census.json): model id, weight sha256 per shard, release date.
  * Weight bits / group size come from the model's own config.json (top-level quantization).
  * File sha256 of the launched binary (resolved symlinks), cached per (path, size, mtime).

This module is CPU-only and never touches MLX or a server.
"""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

PIN_PATH = Path(
    "/Volumes/P5Plus/yunshu-build/competitors/compsync/fullcmp-1103/yunshu_pin.json"
)
CENSUS_PATH = Path(
    "/Volumes/P5Plus/yunshu-build/competitors/compsync/fullcmp-1103/census.json"
)
DEFAULT_TELEMETRY_DIR = "/Volumes/Micron/yunshu-fullcmp/raw"
# Required in every fullcmp cell: identifiers that the audit cross-checks against the manifest.
REQUIRED_ENV = ("TFB_RUN_ID", "TFB_GROUP", "GPUQ_JOB_ID")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class ProvenanceError(RuntimeError):
    """A fullcmp cell cannot be attributed to its pin / census / job; the cell must not run."""


def fullcmp_mode(environ) -> bool:
    return environ.get("TFB_FULLCMP") == "1"


def load_json(path) -> dict:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ProvenanceError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ProvenanceError(f"{path} is not a JSON object")
    return data


def same_path(a: str, b: str) -> bool:
    return os.path.realpath(os.path.expanduser(a)) == os.path.realpath(
        os.path.expanduser(b)
    )


def require_yunshu_pin(environ, pin_path=None) -> dict:
    """Return the pin for this cell's Yunshu server; refuse anything that is not the pinned tree."""
    pin_path = pin_path or PIN_PATH  # resolved per call (tests redirect it)
    bin_path = environ.get("TFB_YUNSHU_BIN")
    src = environ.get("TFB_YUNSHU_SRC")
    if not bin_path or not src:
        raise ProvenanceError(
            "TFB_FULLCMP=1 requires TFB_YUNSHU_BIN and TFB_YUNSHU_SRC (pin: "
            f"{pin_path}); refusing to launch Yunshu"
        )
    pin = load_json(pin_path)
    try:
        venv, worktree = pin["venv"], pin["worktree"]
        pin_python, pin_sha, pin_version = (
            pin["python_executable"],
            pin["sha"],
            pin["version"],
        )
    except KeyError as exc:
        raise ProvenanceError(f"pin {pin_path} lacks {exc}") from exc
    if not same_path(bin_path, os.path.join(venv, "bin", "yunshu")):
        raise ProvenanceError(
            f"TFB_YUNSHU_BIN={bin_path} is not the pinned venv entry "
            f"{os.path.join(venv, 'bin', 'yunshu')}"
        )
    if not same_path(src, os.path.join(worktree, "python")):
        raise ProvenanceError(
            f"TFB_YUNSHU_SRC={src} is not the pinned worktree "
            f"{os.path.join(worktree, 'python')}"
        )
    if not same_path(pin_python, os.path.join(venv, "bin", "python")):
        raise ProvenanceError(f"pin python {pin_python} is not in the pinned venv")
    if not os.path.isfile(pin_python):
        raise ProvenanceError(f"pinned python missing: {pin_python}")
    return {
        "python": pin_python,
        "version": pin_version,
        "git_sha": pin_sha,
        "src": src,
        "bin": bin_path,
        "worktree": worktree,
        "venv": venv,
    }


def pinned_yunshu_cmd(cmd: list, pin: dict, yunshu_args) -> list:
    """Replace whatever entry the harness built with `<pinned python> -m yunshu_cli <args>`."""
    return [pin["python"], "-m", "yunshu_cli", *yunshu_args(cmd)]


def verify_pinned_server(status: dict, cmdline: str, pin: dict) -> None:
    """After launch: the server reports the pinned version and runs from the pinned python."""
    if not isinstance(status, dict) or status.get("version") != pin["version"]:
        raise ProvenanceError(
            f"server reports version {status.get('version') if isinstance(status, dict) else None!r}, "
            f"pinned {pin['version']!r}"
        )
    parts = (cmdline or "").split()
    if not parts or not same_path(parts[0], pin["python"]):
        raise ProvenanceError(f"server process runs {parts[:1]}, not the pinned python")
    if "yunshu_cli" not in parts:
        raise ProvenanceError("server process was not started as -m yunshu_cli")


def file_sha256(path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_SHA_CACHE: dict = {}


def binary_sha256(path) -> str:
    """sha256 of the file a server was launched from (symlinks resolved)."""
    real = os.path.realpath(path)
    st = os.stat(real)
    key = (real, st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        _SHA_CACHE[key] = file_sha256(real)
    return _SHA_CACHE[key]


def census_models(census: dict) -> dict:
    """Model entries of census.json. The lead's census keeps them under `_meta.models`
    (the 2026-11-03 file); a top-level `models` object is accepted too."""
    models = census.get("models")
    if not isinstance(models, dict):
        models = (census.get("_meta") or {}).get("models")
    return models if isinstance(models, dict) else {}


def census_model_id(census: dict, model_path: str) -> str:
    """The census key whose recorded path is this model directory."""
    for key, entry in census_models(census).items():
        if (
            isinstance(entry, dict)
            and entry.get("path")
            and same_path(entry["path"], model_path)
        ):
            return key
    raise ProvenanceError(f"model {model_path} is not in census.json")


def weights_record(model_dir: str, census: dict, model_key: str) -> list:
    """[{file, sha256, bits, group_size}] for every safetensors shard, sha from census.json."""
    entry = census_models(census).get(model_key) or {}
    sums = entry.get("sha256") or {}
    try:
        config = load_json(Path(model_dir) / "config.json")
    except ProvenanceError:
        raise
    quant = config.get("quantization") or {}
    if not (
        isinstance(quant.get("bits"), int) and isinstance(quant.get("group_size"), int)
    ):
        raise ProvenanceError(
            f"{model_dir}/config.json has no top-level quantization bits/group_size"
        )
    shards = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(model_dir, "*.safetensors"))
    )
    if not shards:
        raise ProvenanceError(f"no safetensors shards in {model_dir}")
    out = []
    for name in shards:
        sha = sums.get(name)
        if not isinstance(sha, str) or not _HEX64.fullmatch(sha):
            raise ProvenanceError(f"census.json has no sha256 for {model_key}/{name}")
        out.append(
            {
                "file": name,
                "sha256": sha,
                "bits": quant["bits"],
                "group_size": quant["group_size"],
            }
        )
    return out


def release_date(census: dict, census_key: str, version: str, environ) -> str:
    """YYYY-MM-DD: TFB_RELEASE_DATE when the runner sets it, else census latest_date when the
    census 'latest' names the version this cell actually runs (never a stale or other date)."""
    given = environ.get("TFB_RELEASE_DATE")
    if given:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", given):
            raise ProvenanceError(f"TFB_RELEASE_DATE={given!r} is not YYYY-MM-DD")
        return given
    entry = (census.get(census_key) or {}) if census_key else {}
    latest, when = str(entry.get("latest") or ""), str(entry.get("latest_date") or "")
    if version and latest.startswith(version) and re.match(r"\d{4}-\d{2}-\d{2}", when):
        return when[:10]
    raise ProvenanceError(
        f"no release date for {census_key or '?'} {version!r}: set TFB_RELEASE_DATE"
    )


def telemetry_out(out_path: str, engine: str, telemetry_dir: str) -> str:
    """<telemetry_dir>/<engine>/<cell>-rN.telemetry.jsonl: the audit's telemetry_path() when the
    telemetry dir is the results root (the audit reads it beside each cell file)."""
    stem = os.path.basename(out_path)
    if stem.endswith(".jsonl"):
        stem = stem[: -len(".jsonl")]
    return os.path.join(telemetry_dir, engine, stem + ".telemetry.jsonl")


def swap_snapshot() -> dict:
    """Swap state (`sysctl vm.swapusage`) from the telemetry module's helper; recorded per cell
    at start and end so a cell that pushed the machine into swap is visible in its record."""
    import fullcmp_telemetry as tm

    return tm.swap_record()


def check_required_env(environ) -> None:
    missing = [k for k in REQUIRED_ENV if not environ.get(k)]
    if missing:
        raise ProvenanceError(f"TFB_FULLCMP=1 requires {', '.join(missing)}")
