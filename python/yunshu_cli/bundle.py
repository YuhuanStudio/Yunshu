"""Yunshu CLI — diagnose bundle: a local file to attach to a bug report.

Contents: version, platform, package versions, effective settings (secrets redacted),
paths, the doctor checks, SSD cache state, and the recent error lines of the service log with
their trace ids. Never included: prompts, completions, request bodies or model weights.
Log lines are cut to one short line each and scrubbed of the payload fragments that
validation errors echo (``input_value=...``, message ``content``). The bundle is written to
a file on this machine; nothing is uploaded.
"""

from __future__ import annotations

import gzip
import json
import platform
import re
import sys
import time
from pathlib import Path

from yunshu_engine import log_rotation, paths, settings

MAX_LINE = 300
MAX_ERRORS = 100
_ERROR_LINE = re.compile(
    r"\b(ERROR|CRITICAL|WARNING|Traceback|Exception|Error:)\b|\b[45]\d\d\b "
)
_TRACE_ID = re.compile(r"\breq_[0-9a-f]{8,}\b|\b[0-9a-f]{32}\b")
# Fragments of user content that error messages quote.
_PAYLOAD = [
    re.compile(r"input_value=.*?(?=,\s*input_type=|\)\s*$|$)"),
    re.compile(
        r"""(['"]?(?:content|text|prompt|input|arguments)['"]?\s*[:=]\s*)(['"]).*?\2"""
    ),
    re.compile(
        r"""(['"]?(?:content|text|prompt|input|messages)['"]?\s*[:=]\s*)[\[{].*"""
    ),
]
# Settings that hold structured credentials the generic redaction cannot see into.
_OPAQUE = {"YUNSHU_MCP_SERVERS", "YUNSHU_MCP_CONFIG"}


def scrub_line(line: str, limit: int = MAX_LINE) -> str:
    """One log line made safe to share: secrets redacted, payload fragments removed, cut."""
    line = log_rotation.redact(line.rstrip("\n"))
    line = _PAYLOAD[0].sub("input_value=[OMITTED]", line)
    for pat in _PAYLOAD[1:]:
        line = pat.sub(lambda m: f"{m.group(1)}[OMITTED]", line)
    return line[:limit]


def _read_lines(path: Path) -> list[str]:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", errors="replace") as fh:
                return fh.read().splitlines()
        return path.read_text(errors="replace").splitlines()
    except OSError:
        return []


def recent_errors(log: Path, limit: int = MAX_ERRORS) -> dict:
    """The last ``limit`` error-ish lines (live log, then newest archives) and their trace ids."""
    sources = [log, *log_rotation.archives(log)[:2]]
    picked: list[str] = []
    for src in sources:
        lines = [x for x in _read_lines(src) if _ERROR_LINE.search(x)]
        picked = lines[-limit:] + picked if src == log else picked + lines[-limit:]
        if len(picked) >= limit:
            break
    errors = [scrub_line(x) for x in picked[-limit:]]
    ids = list(dict.fromkeys(i for e in errors for i in _TRACE_ID.findall(e)))
    return {"lines": errors, "trace_ids": ids, "log": str(log)}


def _settings_rows() -> list[dict]:
    rows = []
    for r in settings.effective(("stable", "experimental", "internal")):
        if r["source"] == "default":
            continue  # only what the user changed matters
        value = r["value"]
        if r["name"] in _OPAQUE:
            value = "<set>"
        else:
            value = log_rotation.redact(f"{r['name']}={value}").split("=", 1)[1]
        rows.append({"name": r["name"], "value": value, "source": r["source"]})
    return rows


def _packages() -> dict:
    import importlib.metadata as md

    out: dict[str, str | None] = {}
    for name in (
        "mlx",
        "mlx-lm",
        "mlx-vlm",
        "mlx-audio",
        "llguidance",
        "transformers",
        "fastapi",
        "uvicorn",
        "openai",
    ):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def build(*, host: str = "127.0.0.1", port: int = 8000) -> dict:
    """The bundle as a JSON-serialisable dict."""
    import importlib
    from dataclasses import asdict

    from yunshu_engine.version import yunshu_version

    doctor = importlib.import_module("yunshu_cli.doctor")
    from .cache import _run as cache_status

    log = paths.log_dir() / "yunshu.log"
    try:
        checks = [
            asdict(c)
            for c in doctor.run_checks(settings.get("YUNSHU_MODEL"), host, port)
        ]
    except Exception as exc:  # noqa: BLE001 - a broken check must not stop the bundle
        checks = [{"name": "doctor", "status": "fail", "detail": str(exc), "fix": ""}]
    for c in checks:
        c["detail"] = scrub_line(c["detail"])
    try:
        caches = cache_status(apply=False)
    except Exception as exc:  # noqa: BLE001
        caches = [{"error": str(exc)}]
    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": yunshu_version(),
        "platform": {
            "system": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "packages": _packages(),
        "settings_changed": _settings_rows(),
        "paths": {
            "models": str(paths.models_dir()),
            "logs": str(paths.log_dir()),
            "launch_agent": str(paths.launch_agent_plist()),
            "user_config": str(settings.user_config_path()),
        },
        "doctor": checks,
        "caches": caches,
        "errors": recent_errors(log),
        "excluded": "prompts, completions, request bodies, weights; nothing is uploaded",
    }


def write(dest: Path, **kw) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(build(**kw), indent=2, ensure_ascii=False, default=str))
    return dest
