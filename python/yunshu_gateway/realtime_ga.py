"""OpenAI Realtime GA <-> internal (beta-shaped) event adapter.

The Realtime session engine speaks the beta schema (flat session, ``response.text.delta``,
``modalities``). OpenAI's GA schema (no ``OpenAI-Beta: realtime=v1`` header; what the
``openai`` SDK's ``client.realtime.connect`` speaks) renames events and nests the session
under ``audio.input`` / ``audio.output``. This module translates at the socket boundary so
one engine serves both dialects:

* ``from_ga``  -- client event (GA) -> internal event
* ``to_ga``    -- internal event -> list of GA server events

Pure functions, no I/O: covered by tests/unit/test_realtime_ga.py.
"""

from __future__ import annotations

import copy
from typing import Any

_FMT_TO_INTERNAL = {
    "audio/pcm": "pcm16",
    "audio/pcmu": "g711_ulaw",
    "audio/pcma": "g711_alaw",
}
_FMT_TO_GA = {v: k for k, v in _FMT_TO_INTERNAL.items()}

_RENAMES = {
    "response.text.delta": "response.output_text.delta",
    "response.text.done": "response.output_text.done",
    "response.audio.delta": "response.output_audio.delta",
    "response.audio.done": "response.output_audio.done",
    "response.audio_transcript.delta": "response.output_audio_transcript.delta",
    "response.audio_transcript.done": "response.output_audio_transcript.done",
}


def wants_beta(headers: Any) -> bool:
    """True when the client asked for the beta protocol (``OpenAI-Beta: realtime=v1``)."""
    return "realtime=v1" in (headers.get("openai-beta", "") or "").replace(" ", "")


def _fmt_to_internal(fmt: Any) -> str | None:
    if isinstance(fmt, dict):
        return _FMT_TO_INTERNAL.get(fmt.get("type"))
    if isinstance(fmt, str):
        return fmt
    return None


def _fmt_to_ga(name: str | None) -> dict:
    t = _FMT_TO_GA.get(name or "pcm16", "audio/pcm")
    return {"type": t, "rate": 24000} if t == "audio/pcm" else {"type": t}


# ── client -> internal ──


def _session_from_ga(s: dict) -> dict:
    out: dict = {}
    for k, v in s.items():
        if k in ("type", "object", "id", "expires_at", "audio", "tracing", "include"):
            continue
        if k == "output_modalities":
            out["modalities"] = v
        elif k == "max_output_tokens":
            out["max_response_output_tokens"] = v
        else:
            out[k] = v
    audio = s.get("audio") or {}
    inp = audio.get("input") or {}
    outp = audio.get("output") or {}
    if "format" in inp and _fmt_to_internal(inp["format"]):
        out["input_audio_format"] = _fmt_to_internal(inp["format"])
    if "format" in outp and _fmt_to_internal(outp["format"]):
        out["output_audio_format"] = _fmt_to_internal(outp["format"])
    if "voice" in outp:
        out["voice"] = outp["voice"]
    if "turn_detection" in inp:
        td = inp["turn_detection"]
        if isinstance(td, dict):
            td = dict(td)
            # semantic_vad has no separate implementation here: serve it as server_vad.
            if td.get("type") == "semantic_vad":
                td["type"] = "server_vad"
            for k in (
                "create_response",
                "interrupt_response",
                "idle_timeout_ms",
                "eagerness",
            ):
                td.pop(k, None)
        out["turn_detection"] = td
    return out


def _response_from_ga(r: dict) -> dict:
    skip = ("audio", "output_modalities", "max_output_tokens", "input")
    out = {k: v for k, v in r.items() if k not in skip}
    if "output_modalities" in r:
        out["modalities"] = r["output_modalities"]
    if "max_output_tokens" in r:
        out["max_response_output_tokens"] = r["max_output_tokens"]
    outp = (r.get("audio") or {}).get("output") or {}
    if "voice" in outp:
        out["voice"] = outp["voice"]
    if _fmt_to_internal(outp.get("format")):
        out["output_audio_format"] = _fmt_to_internal(outp["format"])
    return out


def from_ga(event: dict) -> dict:
    """Translate one client event from GA to the internal schema."""
    t = event.get("type")
    if t == "session.update" and isinstance(event.get("session"), dict):
        return {**event, "session": _session_from_ga(event["session"])}
    if t == "response.create" and isinstance(event.get("response"), dict):
        return {**event, "response": _response_from_ga(event["response"])}
    return event


# ── internal -> GA ──


def _session_to_ga(s: dict) -> dict:
    td = s.get("turn_detection")
    if isinstance(td, dict):
        td = {k: v for k, v in td.items() if k != "barge_in_min_ms"}
        td.setdefault("create_response", True)
        td.setdefault("interrupt_response", True)
    return {
        "type": "realtime",
        "object": "realtime.session",
        "id": s.get("id") or "sess_yunshu",
        "model": s.get("model"),
        "output_modalities": ["audio"]
        if "audio" in (s.get("modalities") or [])
        else ["text"],
        "instructions": s.get("instructions", ""),
        "tools": s.get("tools", []),
        "tool_choice": s.get("tool_choice", "auto"),
        "max_output_tokens": s.get("max_response_output_tokens"),
        "audio": {
            "input": {
                "format": _fmt_to_ga(s.get("input_audio_format")),
                "transcription": s.get("input_audio_transcription"),
                "noise_reduction": None,
                "turn_detection": td,
            },
            "output": {
                "format": _fmt_to_ga(s.get("output_audio_format")),
                "voice": s.get("voice"),
                "speed": 1.0,
            },
        },
    }


def _content_to_ga(content: list, role: str | None) -> list:
    out = []
    for part in content or []:
        if not isinstance(part, dict):
            out.append(part)
            continue
        p = dict(part)
        kind = p.get("type")
        if kind == "text":
            p["type"] = "output_text" if role == "assistant" else "input_text"
        elif kind == "audio":
            p["type"] = "output_audio" if role == "assistant" else "input_audio"
        out.append(p)
    return out


def _item_to_ga(item: dict) -> dict:
    item = dict(item)
    if "content" in item:
        item["content"] = _content_to_ga(item["content"], item.get("role"))
    item.setdefault("object", "realtime.item")
    return item


def _response_to_ga(resp: dict) -> dict:
    resp = copy.copy(resp)
    modalities = resp.pop("modalities", None)
    if modalities is not None:
        resp["output_modalities"] = ["audio"] if "audio" in modalities else ["text"]
    if isinstance(resp.get("output"), list):
        resp["output"] = [_item_to_ga(i) for i in resp["output"]]
    return resp


def to_ga(event: dict) -> list[dict]:
    """Translate one server event from the internal schema to zero or more GA events."""
    t = event.get("type", "")
    if t == "conversation.created":
        return []  # not part of GA
    if t in ("session.created", "session.updated"):
        return [{**event, "session": _session_to_ga(event.get("session") or {})}]
    if t == "conversation.item.created":
        item = _item_to_ga(event.get("item") or {})
        base = {"previous_item_id": event.get("previous_item_id"), "item": item}
        eid = event.get("event_id", "")
        return [
            {**base, "type": "conversation.item.added", "event_id": eid},
            {**base, "type": "conversation.item.done", "event_id": eid + "d"},
        ]
    if t in ("response.created", "response.done"):
        return [{**event, "response": _response_to_ga(event.get("response") or {})}]
    if t in ("response.output_item.added", "response.output_item.done"):
        return [{**event, "item": _item_to_ga(event.get("item") or {})}]
    if t in ("response.content_part.added", "response.content_part.done"):
        part = dict(event.get("part") or {})
        if part.get("type") == "text":
            part["type"] = "output_text"
        elif part.get("type") == "audio":
            part["type"] = "output_audio"
        return [{**event, "part": part}]
    if t in _RENAMES:
        return [{**event, "type": _RENAMES[t]}]
    return [event]
