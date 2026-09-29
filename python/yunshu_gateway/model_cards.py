"""Cards for the models this server serves (multi-model registry or the single engine)."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from yunshu_engine.model_card import ModelCard, build_model_card

from .engine import get_display_model_id, get_engine, get_model_manager

logger = logging.getLogger(__name__)


def _safe_card(path: str, **kw) -> ModelCard:
    """A card even when the checkpoint directory is unreadable (never breaks /v1/models)."""
    try:
        return build_model_card(path, **kw)
    except Exception:
        logger.debug("model card derivation failed for %s", path, exc_info=True)
        mid = kw.get("model_id") or Path(path).name
        card = ModelCard(id=mid, display_name=mid.rsplit("/", 1)[-1], kind="chat")
        card.created = kw.get("created") or int(time.time())
        card.state = {"loaded": kw.get("loaded", False), "status": "unknown"}
        return card


def entry_card(entry) -> ModelCard:
    created = int(entry.load_time) if entry.load_time > 0 else 0
    return _safe_card(
        entry.model_path,
        model_id=entry.model_id,
        model_type_name=entry.model_type.name if entry.model_type else None,
        loaded=entry.is_loaded,
        loading=entry.is_loading,
        pinned=entry.is_pinned,
        load_error=entry.load_error,
        estimated_bytes=entry.estimated_bytes,
        created=created,
        engine=entry.engine,
    )


def engine_card(model_id: str | None = None) -> ModelCard | None:
    """The single loaded engine's card, or None when nothing is loaded."""
    engine = get_engine()
    if not (engine and engine.is_loaded):
        return None
    mid = model_id or get_display_model_id() or engine.model_name
    load_time = getattr(engine, "_load_time", None) or getattr(
        engine, "load_time", None
    )
    path = (
        getattr(engine, "_model_path", None)
        or getattr(engine, "model_path", None)
        or mid
    )
    return _safe_card(
        str(path),
        model_id=mid,
        loaded=True,
        created=int(load_time) if load_time else int(time.time()),
        engine=engine,
    )


def all_cards() -> list[ModelCard]:
    manager = get_model_manager()
    if manager is not None:
        return [entry_card(e) for e in manager.list_entries()]
    card = engine_card()
    return [card] if card else []


def find_card(model_id: str) -> ModelCard | None:
    """Resolve ``model_id`` the way inference does; None when unknown."""
    manager = get_model_manager()
    if manager is not None:
        entry = manager.get_entry(manager.resolve_model_id(model_id) or model_id)
        return entry_card(entry) if entry is not None else None
    engine = get_engine()
    if not (engine and engine.is_loaded):
        return None
    served = get_display_model_id() or getattr(engine, "model_name", None)
    resolver = getattr(engine, "resolve_model_id", None)
    known = (callable(resolver) and bool(resolver(model_id))) or model_id in (
        served,
        str(served).rsplit("/", 1)[-1],
    )
    return engine_card(model_id if known else None) if known else None
