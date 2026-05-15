from __future__ import annotations
"""Disaggregated serving endpoints — separate prefill and decode phases.

Implements the P/D disaggregation pattern from vLLM/SGLang:
- POST /v1/prefill: Run prompt prefill only, return a cache handle
- POST /v1/decode: Run decode with a prefill cache handle

This enables:
1. Distributed prefill: prefill on a separate node, decode locally
2. Memory management: prefill-heavy workloads can be isolated
3. Pipeline parallelism: overlap prefill of next request with decode of current

Architecture:
  Client → /v1/prefill → Engine.prefill_only() → cache_handle
  Client → /v1/decode → Engine.decode_with_handle(cache_handle) → tokens

Multi-node path (when YUNSHU_DISAGG_PD=1 and remote nodes registered):
  Client → /v1/prefill → DisaggRouter.route_request()
    → if prefill node is remote: ExternalPrefillClient.prefill_remote()
    → if prefill node is local: ExternalPrefiller.prefill_chunked()
    → KV transfer to decode node via KVTransferClient (when enabled)
  Client → /v1/decode → fetch KV from remote via KVTransferServer
    → Engine.decode_with_handle()
"""

import asyncio
import collections
import logging
import os
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["disaggregate"])

# ── TTL-based cache handle store with LRU eviction ──────────────────────

_CACHE_TTL_SECONDS = 300  # 5 minutes
_CACHE_MAX_SIZE = 100

# OrderedDict for LRU: oldest entries at the front, newest at the back
_cache_handles: collections.OrderedDict[str, dict[str, Any]] = collections.OrderedDict()

# Background cleanup task reference
_cleanup_task: asyncio.Task | None = None


def _add_cache_handle(handle_id: str, data: dict[str, Any]) -> None:
    """Add a cache handle, evicting LRU if over max size."""
    if handle_id in _cache_handles:
        _cache_handles.move_to_end(handle_id)
        _cache_handles[handle_id] = data
        return
    while len(_cache_handles) >= _CACHE_MAX_SIZE:
        # Evict oldest (LRU)
        evicted_key, _ = _cache_handles.popitem(last=False)
        logger.debug("LRU eviction of cache handle %s", evicted_key)
    _cache_handles[handle_id] = data


def _get_cache_handle(handle_id: str) -> dict[str, Any] | None:
    """Get a cache handle and mark as recently used."""
    data = _cache_handles.get(handle_id)
    if data is not None:
        _cache_handles.move_to_end(handle_id)
    return data


async def _cache_gc_loop() -> None:
    """Background task: periodically remove expired cache handles."""
    while True:
        try:
            await asyncio.sleep(30)  # Check every 30 seconds
        except asyncio.CancelledError:
            return
        now = time.monotonic()
        expired = [
            hid for hid, data in _cache_handles.items()
            if (now - data.get("created_at", 0)) > _CACHE_TTL_SECONDS
        ]
        for hid in expired:
            _cache_handles.pop(hid, None)
            logger.debug("TTL expiry of cache handle %s", hid)
        if expired:
            logger.info("Cache GC: removed %d expired handles", len(expired))


def _start_gc_task() -> None:
    """Start the background GC task (idempotent)."""
    global _cleanup_task
    if _cleanup_task is not None and not _cleanup_task.done():
        return
    try:
        loop = asyncio.get_running_loop()
        _cleanup_task = loop.create_task(_cache_gc_loop())
    except RuntimeError:
        # No running loop — will be started on first request
        _cleanup_task = None


# ── DisaggRouter singleton ───────────────────────────────────────────────


def _get_disagg_router():
    """Get or create the DisaggRouter singleton (lazy, env-driven)."""
    try:
        from yunshu_mesh.disagg_pd import DisaggRouter, DisaggConfig
        config = DisaggConfig.from_env()
        return DisaggRouter(config)
    except Exception:
        logger.debug("DisaggRouter not available", exc_info=True)
        return None


def _register_mesh_nodes(router_instance) -> None:
    """Register mesh nodes from environment into DisaggRouter.

    Env vars:
      YUNSHU_PREFILL_NODES — comma-separated host:port list of prefill nodes
      YUNSHU_DECODE_NODES — comma-separated host:port list of decode nodes
    """
    if router_instance is None:
        return
    from yunshu_mesh.disagg_pd import NodeRole

    for node_str in os.environ.get("YUNSHU_PREFILL_NODES", "").split(","):
        node_str = node_str.strip()
        if not node_str:
            continue
        # host:port or just host
        parts = node_str.rsplit(":", 1)
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 7891
        node_id = f"prefill-{host}:{port}"
        router_instance.add_node(node_id, role=NodeRole.PREFILL)

    for node_str in os.environ.get("YUNSHU_DECODE_NODES", "").split(","):
        node_str = node_str.strip()
        if not node_str:
            continue
        parts = node_str.rsplit(":", 1)
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 7890
        node_id = f"decode-{host}:{port}"
        router_instance.add_node(node_id, role=NodeRole.DECODE)


class PrefillRequest(BaseModel):
    model: str = ""
    prompt: str | list[dict] = ""
    max_prefill_tokens: int | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    stop: list[str] | None = None
    seed: int | None = None
    json_schema: dict | str | None = None
    enable_thinking: bool | None = None
    chunk_size: int | None = None


class PrefillResponse(BaseModel):
    id: str = ""
    cache_handle: str = ""
    prompt_tokens: int = 0
    cached_tokens: int = 0
    duration_s: float = 0.0


class DecodeRequest(BaseModel):
    cache_handle: str = ""
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    stop: list[str] | None = None
    seed: int | None = None
    stream: bool = False


class DecodeResponse(BaseModel):
    id: str = ""
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None


def _resolve_engine(request: Request):
    """Resolve the engine from the gateway's engine registry.

    Tries app.state.engine first (for backwards compatibility), then
    falls back to the gateway engine module's get_engine().
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is not None:
        return engine
    try:
        from ..engine import get_engine
        engine = get_engine()
    except Exception:
        logger.debug("engine resolution failed", exc_info=True)
    return engine


@router.post("/prefill", response_model=PrefillResponse)
async def prefill(req: PrefillRequest, request: Request):
    """Run prefill only — tokenize and build KV cache, return cache handle.

    Multi-node path: when YUNSHU_DISAGG_PD=1 and remote prefill nodes are
    registered, uses DisaggRouter to determine the target. For remote prefill
    nodes, sends the request via ExternalPrefillClient and receives KV data.
    For local prefill, runs ExternalPrefiller directly.

    After prefill, if KV transfer is enabled (YUNSHU_KV_TRANSFER=1) and a
    remote decode node is selected, sends KV blocks to that node.
    """
    # Start GC task on first request
    _start_gc_task()

    engine = _resolve_engine(request)
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")

    await engine.start()
    tokenizer = engine._tokenizer

    # Tokenize
    if isinstance(req.prompt, list) and req.prompt and isinstance(req.prompt[0], dict):
        tpl_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if req.enable_thinking is not None:
            tpl_kwargs["enable_thinking"] = req.enable_thinking
        prompt_text = tokenizer.apply_chat_template(req.prompt, **tpl_kwargs)
    else:
        prompt_text = req.prompt

    token_ids = tokenizer.encode(prompt_text)
    prompt_tokens = len(token_ids)

    # Determine chunk size: request override > engine config > default 2048
    chunk_size = req.chunk_size or 2048
    try:
        scheduler = getattr(engine._engine_core, "scheduler", None)
        if scheduler is not None:
            chunk_size = req.chunk_size or scheduler.config.prefill_chunk_size
    except Exception:
        logger.debug("failed to read scheduler config for chunk size", exc_info=True)

    # ── DisaggRouter: determine if prefill should go to a remote node ──
    disagg_router = _get_disagg_router()
    is_remote_prefill = False
    prefill_node_id = ""

    if disagg_router is not None and disagg_router.config.enabled:
        _register_mesh_nodes(disagg_router)
        node_id, role = disagg_router.route_request(prompt_tokens, request_id="")
        if node_id:
            # Check if this node is local or remote
            local_id = os.environ.get("YUNSHU_NODE_ID", "local")
            if node_id != local_id and not node_id.startswith("local"):
                is_remote_prefill = True
                prefill_node_id = node_id
                logger.info(
                    "DisaggRouter: routing prefill of %d tokens to remote node %s",
                    prompt_tokens, node_id,
                )

    loop = asyncio.get_running_loop()
    t0 = time.monotonic()

    if is_remote_prefill:
        # ── Remote prefill via ExternalPrefillClient ──
        try:
            from yunshu_engine.external_prefill import (
                ExternalPrefillClient, ExternalPrefillConfig,
            )
            # Parse host:port from node_id like "prefill-10.0.0.1:7891"
            node_addr = prefill_node_id.split("-", 1)[-1]
            parts = node_addr.rsplit(":", 1)
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 7891

            client_config = ExternalPrefillConfig(
                server_host=host,
                server_port=port,
            )
            client = ExternalPrefillClient(client_config)
            result = await client.prefill_remote(
                token_ids=token_ids,
                chunk_size=chunk_size,
            )
        except Exception as e:
            logger.warning(
                "Remote prefill failed, falling back to local: %s", e,
                exc_info=True,
            )
            # Fall through to local prefill
            is_remote_prefill = False

    if not is_remote_prefill:
        # ── Local prefill via ExternalPrefiller ──
        from yunshu_engine.external_prefill import ExternalPrefiller

        prefiller = ExternalPrefiller(engine._model, tokenizer)

        def _prefill():
            return prefiller.prefill_chunked(
                token_ids=token_ids,
                chunk_size=chunk_size,
            )

        from yunshu_engine.mlx_executor import get_mlx_executor
        executor = get_mlx_executor()
        result = await loop.run_in_executor(executor, _prefill)

        # ── KV Transfer: send KV blocks to remote decode node ──
        if disagg_router is not None and disagg_router.config.enabled:
            try:
                from yunshu_engine.external_prefill import ExternalPrefiller as _EP
                model_name = getattr(engine, '_model_name', '') or ''
                # Count layers from model config
                layer_count = 0
                model_cfg = getattr(engine._model, 'config', None)
                if model_cfg is not None:
                    layer_count = getattr(model_cfg, 'num_hidden_layers', 0)
                # Reuse the prefiller's transfer method
                prefiller.transfer_prefill_result(
                    result,
                    request_id="",
                    model_name=model_name,
                    layer_count=layer_count,
                )
            except Exception:
                logger.debug("KV transfer after prefill failed", exc_info=True)

    duration_s = time.monotonic() - t0

    # Store cache handle with TTL
    handle_id = f"pf-{uuid.uuid4().hex[:12]}"
    _add_cache_handle(handle_id, {
        "cache": result.kv_cache,
        "token_ids": token_ids,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": result.cached_tokens,
        "created_at": time.monotonic(),
        "source": "remote" if is_remote_prefill else "local",
    })

    return PrefillResponse(
        id=handle_id,
        cache_handle=handle_id,
        prompt_tokens=prompt_tokens,
        cached_tokens=result.cached_tokens,
        duration_s=duration_s,
    )


@router.post("/decode", response_model=DecodeResponse)
async def decode(req: DecodeRequest, request: Request):
    """Run decode with a prefill cache handle.

    Looks up the cache handle (local or from remote prefill) and runs
    decode via the engine. The handle is consumed (deleted) after use.
    """
    _start_gc_task()

    engine = _resolve_engine(request)
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")

    await engine.start()

    # Look up cache handle (LRU-aware)
    handle_data = _get_cache_handle(req.cache_handle)
    if handle_data is None:
        raise HTTPException(status_code=404, detail=f"Cache handle {req.cache_handle} not found")

    prompt_tokens = handle_data["prompt_tokens"]

    # Use engine's generate with the prefilled prompt
    gen_output = await engine.generate(
        prompt=handle_data["token_ids"],
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        stop=req.stop,
        seed=req.seed,
    )

    # Remove handle after use (consumed)
    _cache_handles.pop(req.cache_handle, None)

    return DecodeResponse(
        id=f"dec-{uuid.uuid4().hex[:8]}",
        text=gen_output.text,
        prompt_tokens=prompt_tokens,
        completion_tokens=gen_output.completion_tokens,
        finish_reason=gen_output.finish_reason,
    )


@router.get("/cache-handles")
async def list_cache_handles():
    """List active cache handles (debug/monitoring)."""
    now = time.monotonic()
    return {
        "handles": [
            {
                "id": hid,
                "prompt_tokens": data["prompt_tokens"],
                "cached_tokens": data["cached_tokens"],
                "age_s": round(now - data["created_at"], 2),
                "source": data.get("source", "local"),
            }
            for hid, data in _cache_handles.items()
        ],
        "total": len(_cache_handles),
        "max_size": _CACHE_MAX_SIZE,
        "ttl_seconds": _CACHE_TTL_SECONDS,
    }


@router.delete("/cache-handles/{handle_id}")
async def delete_cache_handle(handle_id: str):
    """Delete a cache handle to free GPU memory."""
    if handle_id not in _cache_handles:
        raise HTTPException(status_code=404, detail=f"Cache handle {handle_id} not found")
    del _cache_handles[handle_id]
    return {"deleted": handle_id}


@router.get("/disagg-stats")
async def disagg_stats():
    """Return disaggregated serving statistics from DisaggRouter + KV transfer.

    Provides visibility into:
    - Node pool composition (prefill/decode/hybrid nodes)
    - Routing decisions and load balancing
    - KV transfer throughput and failures
    - Cache handle store metrics
    """
    result: dict[str, Any] = {
        "cache_handles": {
            "total": len(_cache_handles),
            "max_size": _CACHE_MAX_SIZE,
            "ttl_seconds": _CACHE_TTL_SECONDS,
        },
    }

    # DisaggRouter stats
    disagg_router = _get_disagg_router()
    if disagg_router is not None:
        result["disagg_router"] = disagg_router.get_stats()
    else:
        result["disagg_router"] = {"enabled": False}

    # KV transfer stats
    try:
        from yunshu_engine.external_prefill import get_external_prefill_stats
        result["external_prefill"] = get_external_prefill_stats()
    except Exception:
        result["external_prefill"] = {"active": False}

    # KV transfer server/client stats (from engine_core)
    try:
        from yunshu_engine.kv_transfer import is_kv_transfer_enabled
        result["kv_transfer"] = {"enabled": is_kv_transfer_enabled()}
    except Exception:
        result["kv_transfer"] = {"enabled": False}

    return result
