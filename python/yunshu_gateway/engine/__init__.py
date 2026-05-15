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
from yunshu_engine.types import EngineConfig
from yunshu_engine.model_manager import ModelManager

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_model_manager: ModelManager | None = None
_dp_router = None  # DataParallelRouter for multi-replica routing


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
    """Scan a directory for model subdirectories and register them.

    Uses yunshu_engine.model_discovery for modality-aware detection
    (LLM, VLM, TTS, ASR, ImageGen) with estimated size calculation.
    Falls back to simple directory scan if model_discovery fails.
    """
    if _model_manager is None:
        return

    models_path = Path(models_dir)
    if not models_path.exists():
        return

    # Primary: use model_discovery module (oMLX pattern — modality-aware)
    try:
        from yunshu_engine.model_discovery import discover_models
        discovered = discover_models(models_path)
        for mid, info in discovered.items():
            _model_manager.register_model(
                model_id=mid,
                model_path=info.model_path,
                estimated_bytes=info.estimated_size,
            )
        if discovered:
            logger.info(
                "Auto-discovered %d models from %s (via model_discovery)",
                len(discovered), models_path,
            )
        return
    except Exception:
        logger.debug("model_discovery failed, falling back to simple scan", exc_info=True)

    # Fallback: simple directory scan
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

    Resolution order:
    1. If DataParallelRouter is active and model matches, route via DP
    2. If ModelManager is active, resolve through it (supports aliases)
    3. Fall back to single engine with resolve_model_id()

    The returned engine may be an Engine (legacy) or BatchedEngine.
    Both expose generate(), generate_stream(), chat(), stream_chat().
    """
    # Data-parallel routing: if DP is configured, select a replica node
    # The DPRouterMiddleware handles request lifecycle (start/end).
    # Here we use the node already selected by the middleware (stored in
    # per-request state) or fall back to selecting one ourselves.
    if _dp_router is not None and _dp_router.num_nodes > 0:
        node_id = None
        # Check if middleware already selected a node for this request
        try:
            # Best-effort: try to get request-scoped node from context
            from ..dp_middleware import get_dp_load_balancer
            lb = get_dp_load_balancer()
            if lb is not None:
                node_id = lb.select_node(model_id=model_id)
        except Exception:
            pass

        if node_id is None:
            node_id = _dp_router.select_node()

        if node_id is not None:
            _dp_router.record_request_start(node_id)
            # In single-process mode, all DP nodes share the same engine
            # In multi-process mode, the node_id maps to a remote engine
            # For now, route to the local engine and track load
            engine = _engine or (_model_manager.get_engine(model_id) if _model_manager else None)
            if engine is not None:
                await _ensure_engine_started(engine)
                return engine

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


def init_data_parallel(strategy: str = "least_loaded"):
    """Initialize DataParallelRouter for multi-replica routing.

    When enabled, the gateway can distribute requests across multiple
    engine instances running the same model (data parallelism).
    Requires nodes to be registered via add_dp_node().
    """
    global _dp_router
    from yunshu_mesh.data_parallel import DataParallelRouter
    _dp_router = DataParallelRouter(strategy=strategy)
    logger.info(f"DataParallelRouter initialized: strategy={strategy}")
    return _dp_router


def get_dp_router():
    """Get the DataParallelRouter (or None if not initialized)."""
    return _dp_router


def add_dp_node(node_id: str, rank: int = 0) -> None:
    """Register a data-parallel engine replica node."""
    if _dp_router is None:
        init_data_parallel()
    _dp_router.add_node(node_id, rank)
    logger.info(f"DP node registered: {node_id} rank={rank}")


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
    "init_data_parallel",
    "get_dp_router",
    "add_dp_node",
]
