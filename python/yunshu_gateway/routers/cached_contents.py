"""Explicit context-cache endpoints (Google Gemini-style CachedContent).

The third prompt-caching paradigm (see explicit_cache.py): the client explicitly
WRITES a cache (POST content + TTL → a named handle) and READS it by referencing
the handle in later chat/messages requests via `"cached_content": "<name>"`.

  POST   /v1/cachedContents          create  (warms the KV prefix cache)
  GET    /v1/cachedContents          list
  GET    /v1/cachedContents/{id}     get
  PATCH  /v1/cachedContents/{id}     update TTL
  DELETE /v1/cachedContents/{id}     delete

On create we run one max_tokens=1 prefill of the content so the underlying
KVPrefixCache is already hot for the first read.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..engine import get_engine_for_model
from ..explicit_cache import get_store
from .models import _check_model_access, _check_permission

logger = logging.getLogger(__name__)
router = APIRouter(tags=["cached_contents"])


def _owns(entry, request: Request) -> bool:
    """Per-key ownership for cached-content handles.

    Permissive: when no owner was stamped or the caller is the stamped owner,
    allow. Prevents one tenant from reading/deleting/extending another tenant's
    cached content by guessing its name."""
    from yunshu_control.audit_log import resolve_actor

    owner = getattr(entry, "owner", None)
    if not owner or owner == "anonymous":
        return True
    return resolve_actor(request) == owner


class CreateCachedContentRequest(BaseModel):
    model: str
    # Accept either OpenAI-style `messages`, or Gemini-style `system_instruction`
    # / `contents` (plain text or message list). All normalise to a message list.
    messages: list[dict] | None = None
    system_instruction: str | None = None
    contents: str | list[dict] | None = None
    ttl_seconds: float | None = Field(default=None, ge=1.0, le=86400.0)
    display_name: str = ""


def _normalise_messages(req: CreateCachedContentRequest) -> list[dict]:
    if req.messages:
        return req.messages
    msgs: list[dict] = []
    if req.system_instruction:
        msgs.append({"role": "system", "content": req.system_instruction})
    if isinstance(req.contents, str):
        msgs.append({"role": "user", "content": req.contents})
    elif isinstance(req.contents, list):
        msgs.extend(req.contents)
    return msgs


@router.post("/cachedContents")
async def create_cached_content(req: CreateCachedContentRequest, request: Request):
    # This endpoint runs a real prefill on req.model — it MUST enforce the
    # same auth as every other model-serving route (otherwise any authenticated
    # key, even one with no model access, could prefill any model).
    _check_permission(request, "can_infer")
    _check_model_access(request, req.model)
    messages = _normalise_messages(req)
    if not messages:
        raise HTTPException(
            status_code=400,
            detail="cached content requires messages / contents / system_instruction",
        )
    try:
        engine = await get_engine_for_model(req.model)
    except Exception as e:
        # SECURITY: log internals server-side, return a generic message
        # (the raw exception leaks model-loader paths / config internals).
        logger.warning("cached_content engine load failed for %r: %s", req.model, e)
        raise HTTPException(status_code=400, detail="model unavailable") from None
    # WARM: one prefill so the KVPrefixCache is hot; also gives the token count.
    token_count = 0
    try:
        out = await engine.chat(
            messages=messages, max_tokens=1, temperature=0.0, enable_thinking=False
        )
        token_count = int(
            getattr(out, "prompt_tokens", 0)
            or (out.get("prompt_tokens", 0) if isinstance(out, dict) else 0)
        )
    except Exception:
        logger.warning(
            "cached content warm prefill failed (handle still created)", exc_info=True
        )
    from yunshu_control.audit_log import resolve_actor

    entry = get_store().create(
        model=req.model,
        messages=messages,
        token_count=token_count,
        ttl_seconds=req.ttl_seconds,
        display_name=req.display_name,
        owner=resolve_actor(request),
    )
    return entry.to_api()


@router.get("/cachedContents")
async def list_cached_contents(request: Request):
    _check_permission(request, "can_infer")
    # Only the caller's own handles — otherwise this leaks every
    # tenant's handle names/models/token counts.
    return {
        "cachedContents": [e.to_api() for e in get_store().list() if _owns(e, request)]
    }


def _full_name(cid: str) -> str:
    return cid if cid.startswith("cachedContents/") else f"cachedContents/{cid}"


@router.get("/cachedContents/{cid}")
async def get_cached_content(cid: str, request: Request):
    _check_permission(request, "can_infer")
    e = get_store().get(_full_name(cid))
    if e is None or not _owns(e, request):
        # 404 (not 403) so a non-owner can't even confirm the handle exists.
        raise HTTPException(
            status_code=404, detail="cached content not found or expired"
        )
    return e.to_api()


class UpdateTTLRequest(BaseModel):
    ttl_seconds: float = Field(ge=1.0, le=86400.0)


@router.patch("/cachedContents/{cid}")
async def update_cached_content(cid: str, body: UpdateTTLRequest, request: Request):
    _check_permission(request, "can_infer")
    # Ownership check BEFORE mutating TTL.
    _existing = get_store().get(_full_name(cid))
    if _existing is None or not _owns(_existing, request):
        raise HTTPException(
            status_code=404, detail="cached content not found or expired"
        )
    e = get_store().update_ttl(_full_name(cid), body.ttl_seconds)
    if e is None:
        raise HTTPException(
            status_code=404, detail="cached content not found or expired"
        )
    return e.to_api()


@router.delete("/cachedContents/{cid}")
async def delete_cached_content(cid: str, request: Request):
    _check_permission(request, "can_infer")
    # Ownership check BEFORE delete.
    _existing = get_store().get(_full_name(cid))
    if _existing is None or not _owns(_existing, request):
        raise HTTPException(status_code=404, detail="cached content not found")
    ok = get_store().delete(_full_name(cid))
    if not ok:
        raise HTTPException(status_code=404, detail="cached content not found")
    return {"deleted": True, "name": _full_name(cid)}
