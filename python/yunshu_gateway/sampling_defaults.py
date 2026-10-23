"""Request-time resolution of sampling defaults (precedence and sources: ``yunshu_engine.sampling_defaults``)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from yunshu_engine.sampling_defaults import (  # noqa: F401
    _CACHE,
    _GC_KEYS,
    API_DEFAULTS,
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
        eng = getattr(engine_mod, "_engine", None)  # not get_engine(): no lease side effects
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


def resolve_sampling(req: Any, model: str | None = None) -> None:
    """Fill every sampling field of a request model that the request left out (``None``)."""
    fields = [f for f in _GC_KEYS if hasattr(req, f)]
    values = {f: getattr(req, f) for f in fields}
    if all(v is not None for v in values.values()):
        _publish(
            {f: {"value": v, "source": "request"} for f, v in values.items()}, False
        )
        return
    report = fill_sampling(values, model_path_for(model or getattr(req, "model", None)))
    for f, v in values.items():
        setattr(req, f, v)
    _publish(report, True)


def _publish(report: dict[str, dict[str, Any]], filled: bool) -> None:
    try:
        from yunshu_engine.request_tracker import current_request_info

        info = current_request_info.get()
        if info is not None and (filled or getattr(info, "sampling", None) is None):
            info.sampling = {
                "temperature": report["temperature"]["value"],
                "top_p": report["top_p"]["value"],
                "top_k": report["top_k"]["value"],
                "source": {k: v["source"] for k, v in report.items()},
            }
    except Exception:
        logger.debug("sampling report not published", exc_info=True)
