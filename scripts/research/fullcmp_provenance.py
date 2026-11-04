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
  * gpuq job id: GPUQ_JOB_ID, else TFB_JOB_LABEL resolved to the one running job with that label
    in $GPUQ_DIR/jobs (resolve_job_id). The current daemon may not export GPUQ_JOB_ID yet.

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
REQUIRED_ENV = ("TFB_RUN_ID", "TFB_GROUP")  # the job id: see resolve_job_id
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


PB2_PIN_PATH = Path(
    "/Volumes/P5Plus/yunshu-build/competitors/compsync/fullcmp-1103/yunshu_pb2_pin.json"
)

def pin_path_for(engine=None) -> Path:
    """The pin file of one Yunshu engine id. Unknown Yunshu-like ids fail closed; the default
    (no engine, or the dev0 ids) is the dev0 pin, resolved per call so tests can redirect it."""
    if engine in (None, "yunshu", "yunshu-main"):
        return PIN_PATH
    if engine == "yunshu-pb2":
        return PB2_PIN_PATH
    raise ProvenanceError(f"no Yunshu pin for engine {engine!r}")


def require_yunshu_pin(environ, pin_path=None, engine=None) -> dict:
    """Return the pin for this cell's Yunshu server; refuse anything that is not the pinned tree.
    The pin is the engine's own (pin_path_for): a cell of one tree never validates against another."""
    pin_path = pin_path or pin_path_for(engine)
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
        "engine": engine or "yunshu",
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


def verify_pinned_status(status: dict, pin: dict) -> None:
    """The server reports the pinned version."""
    if not isinstance(status, dict) or status.get("version") != pin["version"]:
        raise ProvenanceError(
            f"server reports version {status.get('version') if isinstance(status, dict) else None!r}, "
            f"pinned {pin['version']!r}"
        )


def verify_pinned_cmdline(cmdline: str, pin: dict) -> None:
    """The live process runs from the pinned python as -m yunshu_cli."""
    parts = (cmdline or "").split()
    if not parts or not same_path(parts[0], pin["python"]):
        raise ProvenanceError(f"server process runs {parts[:1]}, not the pinned python")
    if "yunshu_cli" not in parts:
        raise ProvenanceError("server process was not started as -m yunshu_cli")


def verify_pinned_server(status: dict, cmdline: str, pin: dict) -> None:
    """After launch: the server reports the pinned version and runs from the pinned python."""
    verify_pinned_status(status, pin)
    verify_pinned_cmdline(cmdline, pin)


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
    # A version the census lists with its own release date (release_dates: {version: date}, e.g.
    # an older release that is not the census 'latest') is checked first.
    listed = (entry.get("release_dates") or {}).get(version) if version else None
    if listed and re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(listed)):
        return str(listed)
    latest, when = str(entry.get("latest") or ""), str(entry.get("latest_date") or "")
    if version and latest.startswith(version) and re.match(r"\d{4}-\d{2}-\d{2}", when):
        return when[:10]
    raise ProvenanceError(
        f"no release date for {census_key or '?'} {version!r}: set TFB_RELEASE_DATE"
    )


# Census key (census.json) of each engine's release date; engines not listed need TFB_RELEASE_DATE.
# Lives here (not in tfbench.py) so the runner's dry-run gate and tfbench's header read one map.
FULLCMP_CENSUS_KEY = {
    "yunshu": "yunshu",
    "tf-new": "tensorfold",
    "tf-old": "tensorfold",
    "tf-native": "tensorfold",
    "omlx": "omlx",
    "omlx-latest": "omlx",
    "splash": "splash",
    "llamacpp": "llama_cpp",
    "llamacpp-mtp": "llama_cpp",
    "mlxlm": "mlx_lm_server",
    "mlx-serve": "mlx-serve",
    "kapitsa": "kapitsa",
    # census-named registry ids (2026-11-03 smoke: "no release date for ? '1.2.1'" after launch)
    "tf-1.2.1": "tensorfold",
    "tf-1.1.8": "tensorfold",
    "splash-latest": "splash",
    "splash-1.5.0": "splash",
    "omlx-0.7.3": "omlx",
    "omlx-0.7.3rc1": "omlx",
    "mlx-serve-26.11.1": "mlx-serve",
    "mlx-lm-server": "mlx_lm_server",
    "kapitsa-0.2.26": "kapitsa",
    "mlx-vlm-server": "mlx_vlm_server",
    "llama-cpp-b12061": "llama_cpp",
}


def is_yunshu_engine(engine: str) -> bool:
    """The rule tfbench's Srv._is_yunshu applies: the Yunshu engine and its registry ids."""
    import bench_engines

    return engine == "yunshu" or (
        bench_engines.is_new_engine(engine)
        and bench_engines.ENGINES[engine].kind == "yunshu"
    )


def census_key_for(engine: str):
    """The census entry whose release date a cell of this engine reads (tfbench's rule)."""
    return "yunshu" if is_yunshu_engine(engine) else FULLCMP_CENSUS_KEY.get(engine)


def engine_release_date(census: dict, engine: str, version: str, environ) -> str:
    """release_date(...) for an engine id, through the census key tfbench uses."""
    return release_date(census, census_key_for(engine), version, environ)


def dry_run_problems(engines, census: dict, pin: dict, model_dir: str, environ=None) -> list:
    """CPU-only provenance gate for the runner's dry-run, self-check and submit paths.

    Per engine id: the version must be known, its release date must resolve from census.json
    (census_key_for + release_date, with TFB_RELEASE_DATE removed: a cell's job env never carries
    it), and the launched binary must hash. The model's census entry must list a sha256 for every
    shard. Returns one line per problem, each naming the engine; an empty list means every cell
    of these engines can be attributed."""
    import bench_engines

    environ = dict(os.environ if environ is None else environ)
    environ.pop("TFB_RELEASE_DATE", None)
    problems = []
    # the pinned Yunshu tree and its venv python (the same paths require_yunshu_pin checks)
    yunshu_src = os.path.join(str(pin.get("worktree") or ""), "python")
    try:
        weights_record(model_dir, census, census_model_id(census, model_dir))
    except ProvenanceError as exc:
        problems.append(f"model {model_dir}: {exc}")
    for manifest_id in dict.fromkeys(engines):
        try:
            # the harness id a cell runs under (yunshu -> yunshu-main); stubs never reach here
            engine = bench_engines.tfbench_engine(manifest_id)
            version, _sha = bench_engines.engine_version(engine, yunshu_src)
            if not version or version == "unknown":
                raise ProvenanceError(f"no pinned version for {engine}")
            engine_release_date(census, engine, version, environ)
        except ProvenanceError as exc:
            problems.append(f"{manifest_id}: {exc}")
            continue
        except Exception as exc:  # an unknown id or a broken registry entry refuses too
            problems.append(f"{manifest_id}: version or release date not resolvable: {exc}")
            continue
        try:
            if is_yunshu_engine(engine):
                binary = pin["python_executable"]
            else:
                binary = bench_engines.build_launch(
                    engine, 1, "/nonexistent-home", environ
                ).cmd[0]
            binary_sha256(binary)
        except Exception as exc:
            problems.append(f"{manifest_id}: binary sha256 unavailable: {exc}")
    return problems


PROVENANCE_PACKAGES = ("mlx", "mlx-lm", "mlx-vlm")
_PKG_CACHE: dict = {}


def pinned_package_versions(python: str) -> dict:
    """{package: version} of the MLX stack inside a pinned Yunshu venv (metadata only, no MLX import).
    Recorded in every Yunshu cell header, so the two venvs' versions are part of each result."""
    if python in _PKG_CACHE:
        return dict(_PKG_CACHE[python])
    import subprocess

    code = (
        "import importlib.metadata as m, json\n"
        f"names = {list(PROVENANCE_PACKAGES)!r}\n"
        "out = {}\n"
        "for n in names:\n"
        "    try:\n"
        "        out[n] = m.version(n)\n"
        "    except m.PackageNotFoundError:\n"
        "        out[n] = None\n"
        "print(json.dumps(out))\n"
    )
    res = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=60)
    if res.returncode != 0:
        raise ProvenanceError(f"cannot read package versions of {python}: {res.stderr.strip()[-200:]}")
    versions = json.loads(res.stdout.strip().splitlines()[-1])
    _PKG_CACHE[python] = versions
    return dict(versions)


def telemetry_out(out_path: str, engine: str, telemetry_dir: str) -> str:
    """<telemetry_dir>/<cell dir>/<cell>-rN.telemetry.jsonl: the audit's telemetry_path() when the
    telemetry dir is the results root (the audit reads it beside each cell file). The directory is
    the cell file's own parent name (the manifest engine), not `engine` (the tfbench bench name,
    e.g. yunshu-main, which differs for some engines), so the series always sits beside its cell."""
    stem = os.path.basename(out_path)
    if stem.endswith(".jsonl"):
        stem = stem[: -len(".jsonl")]
    cell_dir = os.path.basename(os.path.dirname(os.path.abspath(out_path))) or engine
    return os.path.join(telemetry_dir, cell_dir, stem + ".telemetry.jsonl")


def swap_snapshot() -> dict:
    """Swap state (`sysctl vm.swapusage`) from the telemetry module's helper; recorded per cell
    at start and end so a cell that pushed the machine into swap is visible in its record."""
    import fullcmp_telemetry as tm

    return tm.swap_record()


def resolve_job_id(environ, jobs_dir=None) -> str:
    """The gpuq job id of this cell.

    GPUQ_JOB_ID when gpuq exports it (the current daemon's env, not the old one on main). Else
    TFB_JOB_LABEL names the job: exactly one job in $GPUQ_DIR/jobs with that label and state
    "running" supplies the id. Zero or several matches fail closed, naming the problem. Job
    files that cannot be read are skipped (they cannot be this job's record)."""
    jid = environ.get("GPUQ_JOB_ID")
    if jid:
        return jid
    label = environ.get("TFB_JOB_LABEL")
    if not label:
        raise ProvenanceError(
            "GPUQ_JOB_ID unset and TFB_JOB_LABEL unset: a fullcmp cell must run inside a gpuq job"
        )
    if jobs_dir is None:
        gdir = environ.get("GPUQ_DIR")
        if not gdir:
            raise ProvenanceError(
                f"GPUQ_JOB_ID unset: cannot resolve label {label!r} without GPUQ_DIR"
            )
        jobs_dir = Path(gdir) / "jobs"
    matches = []
    for path in sorted(Path(jobs_dir).glob("*.json")):
        try:
            job = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if (
            isinstance(job, dict)
            and job.get("label") == label
            and job.get("state") == "running"
            and job.get("id")
        ):
            matches.append(str(job["id"]))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ProvenanceError(
            f"GPUQ_JOB_ID unset: no running gpuq job labelled {label!r} in {jobs_dir}"
        )
    raise ProvenanceError(
        f"GPUQ_JOB_ID unset: {len(matches)} running gpuq jobs labelled {label!r} "
        f"({', '.join(matches)}); cannot choose"
    )


def observe_state(environ, is_yunshu: bool) -> bool | None:
    """Observe mode a fullcmp Yunshu cell must run with (fail closed; None for competitors).
    Timed cells run observe OFF: TFB_OBSERVE must be exactly "0" (unset or "1" is refused, since
    tfbench would otherwise default observe on). Profile cells (TFB_PROFILE=1) run observe ON:
    TFB_OBSERVE must be exactly "1". Observe, stage timers and xctrace never feed table numbers."""
    if not is_yunshu:
        return None
    raw = environ.get("TFB_OBSERVE")
    if environ.get("TFB_PROFILE") == "1":
        if raw != "1":
            raise ProvenanceError(f"profile Yunshu cell requires TFB_OBSERVE=1 (got {raw!r})")
        return True
    if raw != "0":
        raise ProvenanceError(
            f"timed Yunshu cell requires TFB_OBSERVE=0 (got {raw!r}; unset or 1 is refused)"
        )
    return False


def check_required_env(environ, jobs_dir=None) -> str:
    """Fail closed before a cell runs: REQUIRED_ENV present and a job id resolvable. Returns it."""
    missing = [k for k in REQUIRED_ENV if not environ.get(k)]
    if missing:
        raise ProvenanceError(f"TFB_FULLCMP=1 requires {', '.join(missing)}")
    return resolve_job_id(environ, jobs_dir)
