"""Yunshu Gateway Engine — bridge to the L4 Engine + ModelManager.

Supports three modes:
1. Single-engine mode (legacy): set_engine() / init_engine()
2. Multi-model mode: uses ModelManager for LRU eviction and multi-model routing
3. BatchedEngine mode: uses BatchedEngine with EngineCore backend (oMLX pattern)

The gateway routers use get_engine_for_model() which works in all modes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from yunshu_engine.batched_engine import BatchedEngine as Engine
from yunshu_engine.engine import EngineConfig
from yunshu_engine.model_manager import ModelManager

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_model_manager: ModelManager | None = None


def get_engine() -> Engine | None:
    """Get the legacy single engine (backward compatible)."""
    return _engine


def init_engine(config: EngineConfig | None = None) -> Engine:
    """Create a single engine instance."""
    global _engine
    _engine = Engine(config)
    return _engine


def set_engine(engine: Engine) -> None:
    """Set the single engine (used by tests)."""
    global _engine
    _engine = engine


def get_model_manager() -> ModelManager | None:
    """Get the model manager (multi-model mode)."""
    return _model_manager


def init_model_manager(
    max_memory_bytes: int | None = None,
    models_dir: str | None = None,
) -> ModelManager:
    """Initialize the model manager and auto-discover models from models_dir."""
    global _model_manager
    _model_manager = ModelManager(max_memory_bytes=max_memory_bytes)

    if models_dir:
        _discover_models(models_dir)

    return _model_manager


def _discover_models(models_dir: str) -> None:
    """Scan a directory for model subdirectories and register them."""
    if _model_manager is None:
        return

    models_path = Path(models_dir)
    if not models_path.exists():
        return

    for subdir in sorted(models_path.iterdir()):
        if not subdir.is_dir():
            continue

        has_config = (subdir / "config.json").exists()
        has_weights = any(subdir.glob("*.safetensors"))
        has_nested_weights = any(subdir.rglob("*.safetensors"))
        has_model_index = (subdir / "model_index.json").exists()
        if not (has_config or has_weights or has_nested_weights or has_model_index):
            continue

        model_id = subdir.name
        raw_bytes = sum(f.stat().st_size for f in subdir.rglob("*.safetensors"))
        # 1.8x to cover quantization overhead, KV cache, framework allocations
        estimated = int(raw_bytes * 1.8)

        _model_manager.register_model(
            model_id=model_id,
            model_path=str(subdir),
            estimated_bytes=estimated,
        )


async def get_engine_for_model(model_id: str) -> Engine:
    """Get an engine for the specified model.

    Resolution order (oMLX pattern):
    1. If ModelManager is active, resolve through it (supports aliases)
    2. Fall back to single engine with resolve_model_id()

    The returned engine may be an Engine (legacy) or BatchedEngine.
    Both expose generate(), generate_stream(), chat(), stream_chat().
    """
    if _model_manager is not None:
        # Try model manager resolution
        entry = _model_manager.get_entry(model_id)
        if entry is not None:
            engine = await _model_manager.get_engine(model_id)
            await _ensure_engine_started(engine)
            return engine

        # Case-insensitive + prefix stripping fallback
        lower = model_id.lower()
        for entry in _model_manager.list_entries():
            if entry.model_id.lower() == lower:
                engine = await _model_manager.get_engine(entry.model_id)
                await _ensure_engine_started(engine)
                return engine
            # Strip provider prefix
            if '/' in model_id:
                stripped = model_id.rsplit('/', 1)[-1]
                if entry.model_id.lower() == stripped.lower():
                    engine = await _model_manager.get_engine(entry.model_id)
                    await _ensure_engine_started(engine)
                    return engine

        raise KeyError(f"Model '{model_id}' not found in model manager")

    # Single-engine fallback
    if _engine is not None and _engine.resolve_model_id(model_id):
        if not _engine.is_running:
            await _engine.start()
        return _engine

    raise KeyError(f"Model '{model_id}' not loaded")


async def _ensure_engine_started(engine) -> None:
    """Ensure an engine is started, handling both Engine and BatchedEngine."""
    if hasattr(engine, 'is_loaded') and not engine.is_loaded:
        if hasattr(engine, 'start'):
            await engine.start()
    elif hasattr(engine, '_running') and not engine._running:
        if hasattr(engine, 'start'):
            await engine.start()
    elif hasattr(engine, 'is_running') and callable(engine.is_running):
        if not engine.is_running():
            await engine.start()


__all__ = [
    "Engine",
    "EngineConfig",
    "ModelManager",
    "get_engine",
    "get_engine_for_model",
    "get_model_manager",
    "init_engine",
    "init_model_manager",
    "set_engine",
]
