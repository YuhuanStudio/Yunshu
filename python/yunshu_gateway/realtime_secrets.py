"""Ephemeral Realtime client secrets (``POST /v1/realtime/client_secrets`` and the beta session routes).

A secret (``ek_...``) is minted by an authenticated caller together with a session configuration.
The WebSocket accepts it as the bearer token until it expires, and the connection starts with that
configuration applied. A secret can open several sessions until it expires. In memory only: a
restart invalidates every secret (they live minutes).
"""

from __future__ import annotations

import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field

from . import realtime_ga

DEFAULT_TTL_S = 600
MIN_TTL_S = 10
MAX_TTL_S = 7200
MAX_LIVE = 1000


class SecretError(Exception):
    def __init__(self, message: str, param: str | None = None):
        super().__init__(message)
        self.message = message
        self.param = param


@dataclass
class Secret:
    value: str
    expires_at: int
    session_id: str
    kind: str  # "realtime" | "transcription"
    config: dict = field(default_factory=dict)  # internal (beta-shaped) session fields


_lock = threading.Lock()
_secrets: dict[str, Secret] = {}


def ttl_from(expires_after: object) -> int:
    if expires_after is None:
        return DEFAULT_TTL_S
    if not isinstance(expires_after, dict):
        raise SecretError("expires_after must be an object", "expires_after")
    anchor = expires_after.get("anchor", "created_at")
    if anchor != "created_at":
        raise SecretError(
            "expires_after.anchor must be 'created_at'", "expires_after.anchor"
        )
    seconds = expires_after.get("seconds", DEFAULT_TTL_S)
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, int)
        or not MIN_TTL_S <= seconds <= MAX_TTL_S
    ):
        raise SecretError(
            f"expires_after.seconds must be an integer between {MIN_TTL_S} and {MAX_TTL_S}",
            "expires_after.seconds",
        )
    return seconds


def mint(config: dict, kind: str, ttl: int, *, now: float | None = None) -> Secret:
    t = time.time() if now is None else now
    sec = Secret(
        value="ek_" + secrets.token_urlsafe(32),
        expires_at=int(t) + ttl,
        session_id=f"sess_{uuid.uuid4().hex[:24]}",
        kind=kind,
        config=dict(config),
    )
    with _lock:
        for k in [k for k, v in _secrets.items() if v.expires_at <= t]:
            del _secrets[k]
        if len(_secrets) >= MAX_LIVE:
            oldest = min(_secrets.values(), key=lambda v: v.expires_at)
            del _secrets[oldest.value]
        _secrets[sec.value] = sec
    return sec


def lookup(value: str | None, *, now: float | None = None) -> Secret | None:
    if not value or not value.startswith("ek_"):
        return None
    t = time.time() if now is None else now
    with _lock:
        sec = _secrets.get(value)
        if sec is None:
            return None
        if sec.expires_at <= t:
            del _secrets[value]
            return None
        return sec


def reset() -> None:
    with _lock:
        _secrets.clear()


# -- session shapes ------------------------------------------------------------------------


def internal_config(body: dict) -> dict:
    """Client session (GA or beta shaped) -> the internal fields SessionConfig.update accepts."""
    cfg = realtime_ga._session_from_ga(body)
    if not body.get("model"):
        cfg.pop("model", None)
    tr = ((body.get("audio") or {}).get("input") or {}).get("transcription")
    if tr is not None:
        cfg["input_audio_transcription"] = tr
    return cfg


def _td_ga(td):
    if not isinstance(td, dict):
        return td
    td = {k: v for k, v in td.items() if k != "barge_in_min_ms"}
    td.setdefault("create_response", True)
    td.setdefault("interrupt_response", True)
    td.setdefault("idle_timeout_ms", None)
    return td


def ga_session(sec: Secret, cfg_full: dict) -> dict:
    """The effective session object for the GA client_secrets response (SDK-validated shape)."""
    if sec.kind == "transcription":
        return {
            "type": "transcription",
            "id": sec.session_id,
            "object": "realtime.transcription_session",
            "expires_at": sec.expires_at,
            "audio": {
                "input": {
                    "format": realtime_ga._fmt_to_ga(
                        cfg_full.get("input_audio_format")
                    ),
                    "transcription": cfg_full.get("input_audio_transcription"),
                    "noise_reduction": cfg_full.get("input_audio_noise_reduction"),
                    "turn_detection": _td_ga(cfg_full.get("turn_detection")),
                }
            },
        }
    out = realtime_ga._session_to_ga({**cfg_full, "id": sec.session_id})
    out["expires_at"] = sec.expires_at
    return out


def beta_session(sec: Secret, cfg_full: dict) -> dict:
    client_secret = {"value": sec.value, "expires_at": sec.expires_at}
    if sec.kind == "transcription":
        return {
            "id": sec.session_id,
            "object": "realtime.transcription_session",
            "modalities": cfg_full.get("modalities") or ["text"],
            "input_audio_format": cfg_full.get("input_audio_format"),
            "input_audio_transcription": cfg_full.get("input_audio_transcription"),
            "turn_detection": cfg_full.get("turn_detection"),
            "client_secret": client_secret,
        }
    return {
        "id": sec.session_id,
        "object": "realtime.session",
        "model": cfg_full.get("model"),
        "expires_at": sec.expires_at,
        "modalities": cfg_full.get("modalities"),
        "instructions": cfg_full.get("instructions", ""),
        "voice": cfg_full.get("voice"),
        "input_audio_format": cfg_full.get("input_audio_format"),
        "output_audio_format": cfg_full.get("output_audio_format"),
        "input_audio_transcription": cfg_full.get("input_audio_transcription"),
        "turn_detection": cfg_full.get("turn_detection"),
        "tools": cfg_full.get("tools", []),
        "tool_choice": cfg_full.get("tool_choice", "auto"),
        "temperature": cfg_full.get("temperature"),
        "max_response_output_tokens": cfg_full.get("max_response_output_tokens"),
        "client_secret": client_secret,
    }
