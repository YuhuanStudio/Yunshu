"""Console admin: settings writes, service status / restart, CORS origins.

- ``PATCH /v1/yunshu/config``          validated writes of any registered setting
- ``GET   /v1/yunshu/service``         launchd status of the ``yunshu service`` agent
- ``POST  /v1/yunshu/service/restart`` drain, then ``launchctl kickstart -k`` (launchd only)
- ``GET|PATCH /v1/yunshu/cors``        allowed CORS origins, validated, applied live

Every route needs the ``admin`` permission (the static token; with no token
configured the secure default denies it, see ``_check_permission``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from yunshu_control.audit_log import log_operation, resolve_actor
from yunshu_engine import paths, settings

from .models import _check_permission

logger = logging.getLogger(__name__)
router = APIRouter(tags=["yunshu-admin"])

_MAX_ORIGINS = 50
_HOST = re.compile(r"[A-Za-z0-9]([A-Za-z0-9.\-]*[A-Za-z0-9])?")
# Not writable through the API: the pointer to the config file itself.
_READ_ONLY = {"YUNSHU_CONFIG": "choose the config file with `yunshu serve --config`"}


def _err(status: int, code: str, message: str, **extra: Any) -> HTTPException:
    return HTTPException(
        status_code=status, detail={"code": code, "message": message, **extra}
    )


# ── CORS origins ───────────────────────────────────────────────────────

WILDCARD_WARNING = (
    "WARNING: '*' lets any website in any browser call this server. "
    "Credentials are never allowed with '*'. Use explicit origins "
    "unless this server is only reachable from this machine."
)


def validate_origins(origins: list[str]) -> tuple[list[str], list[str]]:
    """Normalise and check a list of origins. Returns ``(origins, warnings)``;
    raises a 422 listing every bad entry. Only ``scheme://host[:port]`` URL
    origins are accepted, or a lone ``*``."""
    if not origins:
        raise _err(422, "cors_empty", "give at least one origin, or null to reset")
    if len(origins) > _MAX_ORIGINS:
        raise _err(422, "cors_too_many", f"at most {_MAX_ORIGINS} origins")
    bad: dict[str, str] = {}
    out: list[str] = []
    for raw in origins:
        o = (raw or "").strip()
        if o == "*":
            out.append("*")
            continue
        try:
            u = urlsplit(o)
            port = u.port
        except ValueError:
            bad[raw] = "not a valid URL"
            continue
        host = u.hostname or ""
        if u.scheme not in ("http", "https") or not host:
            bad[raw] = "must look like http://host[:port] or https://host[:port]"
        elif u.path not in ("", "/") or u.query or u.fragment or "@" in u.netloc:
            bad[raw] = "an origin has no path, query, fragment or user info"
        elif ":" not in host and not _HOST.fullmatch(host):
            bad[raw] = "invalid host"
        else:
            shown = f"[{host}]" if ":" in host else host
            out.append(f"{u.scheme}://{shown}" + (f":{port}" if port else ""))
    if bad:
        raise _err(422, "cors_invalid_origin", "invalid origins", invalid=bad)
    out = list(dict.fromkeys(out))
    warnings: list[str] = []
    if "*" in out:
        if out != ["*"]:
            raise _err(
                422, "cors_wildcard_mixed", "'*' cannot be combined with other origins"
            )
        warnings.append(WILDCARD_WARNING)
    return out, warnings


def _cors_view(request: Request | None = None) -> dict:
    from ..middleware.live_cors import credentials_allowed, current_origins

    _, source = settings.raw("YUNSHU_CORS_ORIGINS")
    origins = current_origins()
    view: dict[str, Any] = {
        "object": "yunshu.cors",
        "origins": origins,
        "wildcard": origins == ["*"],
        "credentials": credentials_allowed(origins),
        "source": source,
        "default": settings.REGISTRY["YUNSHU_CORS_ORIGINS"].default,
        "applies": "live",
        "warnings": [],
    }
    if view["wildcard"]:
        view["warnings"].append(WILDCARD_WARNING)
    if source in ("env", "cli"):
        view["warnings"].append(
            f"YUNSHU_CORS_ORIGINS is set by the {source}: it overrides the config "
            "file, so edits are saved but do not change the running value."
        )
    origin = request.headers.get("origin") if request is not None else None
    if origin:
        view["request_origin"] = origin
        view["request_origin_allowed"] = (
            view["wildcard"] or origin.rstrip("/") in origins
        )
    return view


@router.get("/yunshu/cors")
async def get_cors(request: Request) -> dict:
    _check_permission(request, "admin")
    return _cors_view(request)


class CorsPatch(BaseModel):
    origins: list[str] | None = Field(
        default=None, description="allowed origins; null resets to the default"
    )
    allow_any_origin: bool = Field(
        default=False, description="must be true to accept '*'"
    )


@router.patch("/yunshu/cors")
async def patch_cors(body: CorsPatch, request: Request) -> dict:
    _check_permission(request, "admin")
    actor = resolve_actor(request)
    warnings: list[str] = []
    value: str | None = None
    if body.origins is not None:
        origins, warnings = validate_origins(body.origins)
        if origins == ["*"] and not body.allow_any_origin:
            raise _err(
                400,
                "cors_wildcard_unconfirmed",
                "'*' needs allow_any_origin=true",
                warnings=warnings,
            )
        value = ",".join(origins)
    target = settings.active_config_path()
    settings.write_config_values({"YUNSHU_CORS_ORIGINS": value}, target)
    log_operation("cors_update", value or "<default>", "success", actor=actor)
    view = _cors_view(request)
    view["warnings"] = list(dict.fromkeys(warnings + view["warnings"]))
    view["saved_to"] = str(target)
    return view


# ── PATCH /v1/yunshu/config ────────────────────────────────────────────


class ConfigPatch(BaseModel):
    settings: dict[str, Any] = Field(
        description="NAME -> new value; null resets the setting to its default"
    )
    confirm_experimental: bool = False
    dry_run: bool = False


def _validate_one(name: str, value: Any) -> Any:
    s = settings.REGISTRY[name]
    if value is None:
        return None
    if s.type == "bool" and not isinstance(value, (bool, str)):
        raise settings.SettingError(f"{name}: expected a boolean")
    if s.type in ("int", "float", "gb") and isinstance(value, bool):
        raise settings.SettingError(f"{name}: expected a number")
    if s.type in ("str", "path", "enum") and not isinstance(value, str):
        raise settings.SettingError(f"{name}: expected a string")
    if s.secret and isinstance(value, str) and not value.strip():
        raise settings.SettingError(
            f"{name}: a secret cannot be empty (send null to remove it)"
        )
    text = settings._to_text(value)
    settings._parse(s, text)  # type, choices, minimum
    if name == "YUNSHU_CORS_ORIGINS":
        origins = ["*"] if text.strip() == "*" else text.split(",")
        return ",".join(validate_origins(origins)[0])
    return value


def _message(exc: HTTPException | Exception) -> str:
    if isinstance(exc, HTTPException) and isinstance(exc.detail, dict):
        return str(exc.detail.get("message", exc.detail))
    return str(exc)


@router.patch("/yunshu/config")
async def patch_config(body: ConfigPatch, request: Request) -> dict:
    """Validate, persist and report when each change takes effect. All or
    nothing: one bad value writes none. Environment / CLI values still win at
    runtime and are reported as ``overridden``."""
    _check_permission(request, "admin")
    actor = resolve_actor(request)
    if not body.settings:
        raise _err(422, "empty", "settings is empty")
    errors: dict[str, str] = {}
    clean: dict[str, Any] = {}
    experimental: list[str] = []
    for name, value in body.settings.items():
        if name not in settings.REGISTRY:
            hint = settings.close_matches(name)
            errors[name] = "unknown setting" + (
                f" (did you mean {', '.join(hint)}?)" if hint else ""
            )
            continue
        if name in _READ_ONLY:
            errors[name] = f"read-only here: {_READ_ONLY[name]}"
            continue
        try:
            clean[name] = _validate_one(name, value)
        except (HTTPException, settings.SettingError) as exc:
            errors[name] = _message(exc)
            continue
        if settings.REGISTRY[name].stability != "stable":
            experimental.append(name)
    if errors:
        raise _err(422, "invalid_settings", "no setting was changed", errors=errors)
    if experimental and not body.confirm_experimental:
        raise _err(
            400,
            "experimental_unconfirmed",
            "experimental / internal settings are temporary and may be removed; "
            "send confirm_experimental=true to change them",
            settings=experimental,
        )
    target = settings.active_config_path()
    if not body.dry_run:
        settings.write_config_values(clean, target)
    results: dict[str, dict] = {}
    for name, value in clean.items():
        s = settings.REGISTRY[name]
        source = settings.raw(name)[1]
        if source in ("env", "cli"):
            status = "overridden"
        else:
            status = {
                "live": "applied",
                "reload": "needs_reload",
                "restart": "needs_restart",
            }[s.applies]
        row: dict[str, Any] = {
            "status": status,
            "applies": s.applies,
            "source": source,
            "reset": value is None,
        }
        if not s.secret and not body.dry_run:
            row["value"] = settings.get(name)
        if status == "overridden":
            row["note"] = f"saved, but the {source} sets {name} and wins at runtime"
        if name == "YUNSHU_AUTH_TOKEN":
            row["note"] = "the next request must present the new token"
        results[name] = row
    if not body.dry_run:
        log_operation("config_update", ",".join(sorted(clean)), "success", actor=actor)
    need_restart = sorted(
        n for n, r in results.items() if r["status"] == "needs_restart"
    )
    return {
        "object": "yunshu.config.patch",
        "dry_run": body.dry_run,
        "saved_to": None if body.dry_run else str(target),
        "results": results,
        "restart_required": bool(need_restart),
        "restart_for": need_restart,
        "reload_required": any(r["status"] == "needs_reload" for r in results.values()),
        "restart": _restart_hint() if need_restart else None,
    }


# ── service ────────────────────────────────────────────────────────────


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["launchctl", *args], capture_output=True, text=True, timeout=10
    )


def _target() -> str:
    return f"gui/{os.getuid()}/{paths.SERVICE_LABEL}"


def _manual_command() -> str:
    """How the operator restarts this process by hand when launchd does not own it."""
    argv = [a for a in sys.argv if a]
    if "serve" in argv:
        return "stop this process, then start it again: " + " ".join(
            shlex.quote(a) for a in argv
        )
    return "stop this process, then start it again with `yunshu serve ...`"


def _service_state() -> dict:
    from yunshu_cli import service as svc  # log_file / parse_print only

    plist = paths.launch_agent_plist()
    info: dict[str, Any] = {
        "label": paths.SERVICE_LABEL,
        "plist": str(plist),
        "installed": plist.exists(),
        "loaded": False,
        "pid": None,
        "state": None,
        "last_exit_code": None,
        "log": str(svc.log_file()),
    }
    try:
        r = _launchctl("print", _target())
    except (OSError, subprocess.SubprocessError):
        return info
    info["loaded"] = r.returncode == 0
    if info["loaded"]:
        parsed = svc.parse_print(r.stdout)
        info["pid"] = parsed.get("pid")
        info["state"] = parsed.get("state")
        info["last_exit_code"] = parsed.get("last_exit_code")
    return info


def _under_launchd(state: dict) -> bool:
    """This process is the launchd job's process (not a terminal-started one)."""
    return bool(state["loaded"] and state["pid"] == os.getpid())


def _restart_hint() -> dict:
    state = _service_state()
    under = _under_launchd(state)
    return {
        "available": under,
        "endpoint": "POST /v1/yunshu/service/restart",
        "cli": "yunshu service restart" if state["installed"] else None,
        "manual": None if under else _manual_command(),
    }


def _uptime_s() -> float:
    from .. import main as _main

    return round(time.monotonic() - _main._startup_time, 1)


@router.get("/yunshu/service")
async def get_service(request: Request) -> dict:
    _check_permission(request, "admin")
    from yunshu_engine.version import yunshu_version

    state = await asyncio.to_thread(_service_state)
    under = _under_launchd(state)
    return {
        "object": "yunshu.service",
        **state,
        "under_launchd": under,
        "restart_available": under,
        "pid_self": os.getpid(),
        "version": yunshu_version(),
        "uptime_s": _uptime_s(),
        "cli": {
            "status": "yunshu service status",
            "restart": "yunshu service restart",
            "install": "yunshu service install --model <model>",
        },
        "restart_note": None
        if under
        else "Not running under the yunshu launchd service: " + _manual_command(),
    }


_restart_tasks: set[asyncio.Task] = set()


def _active_requests() -> int:
    from .. import main as _main

    return int(_main._active_requests)


async def _drain_then_kickstart(timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while _active_requests() > 0 and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    # launchd then sends SIGTERM; the lifespan drains anything that arrived since.
    r = await asyncio.to_thread(_launchctl, "kickstart", "-k", _target())
    if r.returncode != 0:
        logger.error("launchctl kickstart failed: %s", r.stderr.strip())


class RestartBody(BaseModel):
    confirm: bool = Field(default=False, description="must be true")


@router.post("/yunshu/service/restart", status_code=202)
async def restart_service(body: RestartBody, request: Request) -> dict:
    _check_permission(request, "admin")
    actor = resolve_actor(request)
    state = await asyncio.to_thread(_service_state)
    if not _under_launchd(state):
        raise _err(
            409,
            "not_under_launchd",
            "This server is not running under the yunshu launchd service, so it "
            "cannot restart itself. " + _manual_command(),
            manual=_manual_command(),
            cli="yunshu service restart" if state["installed"] else None,
        )
    if not body.confirm:
        raise _err(400, "restart_unconfirmed", "send confirm=true to restart")
    timeout = float(settings.get("YUNSHU_DRAIN_TIMEOUT"))
    active = _active_requests()
    log_operation("service_restart", paths.SERVICE_LABEL, "success", actor=actor)
    task = asyncio.create_task(_drain_then_kickstart(timeout))
    _restart_tasks.add(task)
    task.add_done_callback(_restart_tasks.discard)
    return {
        "object": "yunshu.service.restart",
        "restarting": True,
        "active_requests": active,
        "drain_timeout_s": timeout,
        "cli": "yunshu service restart",
    }
