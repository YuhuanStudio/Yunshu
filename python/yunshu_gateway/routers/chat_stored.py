"""Stored chat completions: ``GET/POST/DELETE /v1/chat/completions/{id}`` and ``/messages``."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..chat_store import ChatStoreError, check_metadata, get_store
from .models import _check_permission

router = APIRouter(tags=["chat"])


def _err(exc: ChatStoreError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status,
        content={
            "error": {
                "message": exc.message,
                "type": "invalid_request_error",
                "param": exc.param,
                "code": exc.code,
            }
        },
    )


def _page_args(request: Request) -> dict:
    q = request.query_params
    try:
        limit = int(q.get("limit", 20))
    except ValueError:
        raise ChatStoreError(
            400, "limit must be an integer", "invalid_type", "limit"
        ) from None
    if not 1 <= limit <= 100:
        raise ChatStoreError(
            400, "limit must be between 1 and 100", "invalid_value", "limit"
        )
    order = q.get("order", "asc")
    if order not in ("asc", "desc"):
        raise ChatStoreError(
            400, "order must be 'asc' or 'desc'", "invalid_value", "order"
        )
    return {"limit": limit, "order": order, "after": q.get("after")}


@router.get("/chat/completions")
async def list_chat_completions(request: Request) -> Any:
    _check_permission(request, "can_infer")
    try:
        args = _page_args(request)
        meta = {
            k[len("metadata[") : -1]: v
            for k, v in request.query_params.items()
            if k.startswith("metadata[") and k.endswith("]")
        }
        return await asyncio.to_thread(
            get_store().list,
            model=request.query_params.get("model"),
            metadata=meta or None,
            **args,
        )
    except ChatStoreError as exc:
        return _err(exc)


@router.get("/chat/completions/{completion_id}")
async def get_chat_completion(completion_id: str, request: Request) -> Any:
    _check_permission(request, "can_infer")
    try:
        return await asyncio.to_thread(get_store().get, completion_id)
    except ChatStoreError as exc:
        return _err(exc)


@router.get("/chat/completions/{completion_id}/messages")
async def get_chat_completion_messages(completion_id: str, request: Request) -> Any:
    _check_permission(request, "can_infer")
    try:
        return await asyncio.to_thread(
            get_store().list_messages, completion_id, **_page_args(request)
        )
    except ChatStoreError as exc:
        return _err(exc)


@router.post("/chat/completions/{completion_id}")
async def update_chat_completion(completion_id: str, request: Request) -> Any:
    _check_permission(request, "can_infer")
    try:
        try:
            body = await request.json()
        except ValueError:
            raise ChatStoreError(
                400, "Request body is not valid JSON", "invalid_json"
            ) from None
        if not isinstance(body, dict) or "metadata" not in body:
            raise ChatStoreError(
                400, "metadata is required", "missing_required_parameter", "metadata"
            )
        check_metadata(body["metadata"])
        return await asyncio.to_thread(
            get_store().update_metadata, completion_id, body["metadata"]
        )
    except ChatStoreError as exc:
        return _err(exc)


@router.delete("/chat/completions/{completion_id}")
async def delete_chat_completion(completion_id: str, request: Request) -> Any:
    _check_permission(request, "can_infer")
    try:
        await asyncio.to_thread(get_store().delete, completion_id)
    except ChatStoreError as exc:
        return _err(exc)
    return {"id": completion_id, "object": "chat.completion.deleted", "deleted": True}
