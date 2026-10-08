"""Realtime client secrets: ``POST /v1/realtime/client_secrets`` (GA), ``/realtime/sessions`` and
``/realtime/transcription_sessions`` (beta). The minted ``ek_`` value authenticates the WebSocket."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .. import realtime_secrets as rs
from .models import _check_permission
from .realtime import SessionConfig

router = APIRouter(tags=["realtime"])


def _error(
    status: int, message: str, param: str | None = None, code: str | None = None
):
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": code,
            }
        },
    )


async def _body(request: Request) -> dict:
    raw = await request.body()
    if not raw.strip():
        return {}
    try:
        import json

        data = json.loads(raw)
    except ValueError:
        raise rs.SecretError("Request body is not valid JSON") from None
    if not isinstance(data, dict):
        raise rs.SecretError("Request body must be a JSON object")
    return data


def _default_model() -> str:
    try:
        from ..engine import get_engine

        eng = get_engine()
        name = getattr(eng, "model_name", None) if eng is not None else None
        if isinstance(name, str) and name:
            return name
    except Exception:
        pass
    return "default"


def _effective(client: dict, kind: str) -> tuple[dict, dict]:
    """(config to store, full effective session in internal shape) for a client session body."""
    cfg = rs.internal_config(client)
    model = (
        client.get("model")
        if isinstance(client.get("model"), str) and client.get("model")
        else _default_model()
    )
    sc = SessionConfig()
    sc.update({k: v for k, v in cfg.items() if k != "model"})
    full = sc.to_dict()
    full.update({"model": model, "tool_choice": sc.tool_choice, "voice": sc.voice})
    if (
        cfg.get("input_audio_transcription") is not None
        or "input_audio_transcription" in cfg
    ):
        full["input_audio_transcription"] = cfg.get("input_audio_transcription")
    cfg = dict(cfg)
    if client.get("model"):
        cfg["model"] = model
    if kind == "transcription":
        full.setdefault("input_audio_transcription", None)
    return cfg, full


def _kind(session: dict) -> str:
    t = session.get("type", "realtime")
    if t not in ("realtime", "transcription"):
        raise rs.SecretError(
            "session.type must be 'realtime' or 'transcription'", "session.type"
        )
    return t


@router.post("/realtime/client_secrets")
async def create_client_secret(request: Request) -> Any:
    _check_permission(request, "can_infer")
    try:
        body = await _body(request)
        session = body.get("session") or {}
        if not isinstance(session, dict):
            raise rs.SecretError("session must be an object", "session")
        kind = _kind(session)
        ttl = rs.ttl_from(body.get("expires_after"))
        cfg, full = _effective(session, kind)
    except rs.SecretError as exc:
        return _error(400, exc.message, exc.param)
    sec = rs.mint(cfg, kind, ttl)
    return {
        "value": sec.value,
        "expires_at": sec.expires_at,
        "session": rs.ga_session(sec, full),
    }


async def _beta(request: Request, kind: str) -> Any:
    _check_permission(request, "can_infer")
    try:
        body = await _body(request)
        cfg, full = _effective(body, kind)
        ttl = rs.DEFAULT_TTL_S
    except rs.SecretError as exc:
        return _error(400, exc.message, exc.param)
    sec = rs.mint(cfg, kind, ttl)
    return rs.beta_session(sec, full)


@router.post("/realtime/sessions")
async def create_session(request: Request) -> Any:
    return await _beta(request, "realtime")


@router.post("/realtime/transcription_sessions")
async def create_transcription_session(request: Request) -> Any:
    return await _beta(request, "transcription")
