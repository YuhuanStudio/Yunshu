"""Request-time resolution of sampling defaults (precedence and sources: ``yunshu_engine.sampling_defaults``)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from yunshu_engine.sampling_defaults import (  # noqa: F401
    _CACHE,
    _GC_KEYS,
    API_DEFAULTS,
    default_mode,
    defaults_by_mode,
    effective_defaults,
    fill_sampling,
    read_generation_config,
)

logger = logging.getLogger(__name__)


def model_path_for(model: str | None) -> str | None:
    """The directory of the model a request will be served by, without loading anything."""
    try:
        from . import engine as engine_mod
        from .engine import get_model_manager

        manager = get_model_manager()
        if manager is not None and model:
            entry = manager.get_entry(model)
            if entry is not None and getattr(entry, "model_path", None):
                return _as_dir(entry.model_path)
        eng = getattr(
            engine_mod, "_engine", None
        )  # not get_engine(): no lease side effects
        if eng is not None:
            for attr in ("_model_path", "model_path", "model_name"):
                v = getattr(eng, attr, None)
                if isinstance(v, str) and v:
                    d = _as_dir(v)
                    if d:
                        return d
        from yunshu_engine import settings

        v = settings.get("YUNSHU_MODEL")
        if v:
            return _as_dir(str(v))
    except Exception:
        logger.debug("model path lookup failed", exc_info=True)
    return None


def _as_dir(p: str) -> str | None:
    if Path(p).is_dir():
        return p
    try:
        from mlx_lm.utils import hf_repo_to_path

        d = Path(str(hf_repo_to_path(p)))
        return str(d) if d.is_dir() else None
    except Exception:
        return None


def request_mode(req: Any, model_path: str | None) -> tuple[str, str]:
    """(mode, why): whether THIS request runs with thinking, resolved as request handling does.

    Order: Anthropic ``thinking`` (absent = off) / ``enable_thinking`` (top level, then ``chat_template_kwargs``);
    a constrained output (JSON schema / grammar) runs without thinking; Gemma 4 defaults off;
    otherwise the chat template's own default."""
    et = None
    if hasattr(req, "thinking"):  # Anthropic: absent thinking = disabled
        try:
            from .routers.anthropic import _thinking_switches

            et = _thinking_switches(req)[0]
        except Exception:
            et = None
    if et is None:
        et = getattr(req, "enable_thinking", None)
    if et is None:
        ctk = getattr(req, "chat_template_kwargs", None)
        if isinstance(ctk, dict) and isinstance(ctk.get("enable_thinking"), bool):
            et = ctk["enable_thinking"]
    if et is None:
        # reasoning_effort none / minimal means no reasoning (output_limits.fold_effort)
        from yunshu_engine.output_limits import fold_effort

        et = fold_effort(None, getattr(req, "reasoning_effort", None))[0]
    if isinstance(et, bool):
        return ("thinking" if et else "non_thinking"), "request"
    rf = getattr(req, "response_format", None)
    if (
        (isinstance(rf, dict) and rf.get("type") in ("json_schema", "json_object"))
        or getattr(req, "guided_json", None)
        or getattr(req, "grammar", None)
    ):
        return "non_thinking", "constrained"
    return default_mode(model_path), "model_default"


def resolve_sampling(req: Any, model: str | None = None) -> None:
    """Fill every sampling field of a request model that the request left out (``None``)."""
    fields = [f for f in _GC_KEYS if hasattr(req, f)]
    values = {f: getattr(req, f) for f in fields}
    if all(v is not None for v in values.values()):
        _publish(
            {f: {"value": v, "source": "request"} for f, v in values.items()},
            False,
            None,
        )
        return
    path = model_path_for(model or getattr(req, "model", None))
    mode, why = request_mode(req, path)
    report = fill_sampling(values, path, mode)
    for f, v in values.items():
        setattr(req, f, v)
    _publish(report, True, {"mode": mode, "mode_from": why})


def _publish(
    report: dict[str, dict[str, Any]], filled: bool, mode: dict | None
) -> None:
    try:
        from yunshu_engine.request_tracker import current_request_info

        info: Any = current_request_info.get()
        if info is not None and (filled or getattr(info, "sampling", None) is None):
            info.sampling = {
                "temperature": report["temperature"]["value"],
                "top_p": report["top_p"]["value"],
                "top_k": report["top_k"]["value"],
                **{
                    k: report[k]["value"]
                    for k in (
                        "min_p",
                        "repetition_penalty",
                        "presence_penalty",
                        "frequency_penalty",
                    )
                    if k in report
                },
                "source": {k: v["source"] for k, v in report.items()},
                **(mode or {}),
            }
    except Exception:
        logger.debug("sampling report not published", exc_info=True)
