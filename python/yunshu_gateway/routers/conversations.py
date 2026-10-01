"""OpenAI Conversations API: ``/v1/conversations`` and ``/v1/conversations/{id}/items``."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from yunshu_kv.disk_budget import is_enospc

from ..conversations_store import (
    MAX_CREATE_ITEMS,
    ConversationError,
    get_store,
)
from .models import _check_permission

router = APIRouter(tags=["conversations"])


def error_response(
    status: int, message: str, code: str | None = None, param: str | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": "server_error" if status >= 500 else "invalid_request_error",
                "param": param,
                "code": code,
            }
        },
    )


async def _io(fn, *args, **kwargs):
    """Run one whole store operation (its own process-lock RMW) off the event loop. A full
    disk is a 507 the client can act on, not an opaque 500; the store removes its temp file."""
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except OSError as exc:
        if not is_enospc(exc):
            raise
        raise ConversationError(
            507,
            "The server's disk is full, the change was not saved; free space and retry",
            "insufficient_storage",
        ) from exc


def _err(exc: ConversationError) -> JSONResponse:
    return error_response(exc.status, exc.message, exc.code, exc.param)


async def _body(request: Request) -> dict:
    raw = await request.body()
    if not raw.strip():
        return {}
    try:
        import json

        data = json.loads(raw)
    except ValueError:
        raise ConversationError(
            400, "Request body is not valid JSON", "invalid_json"
        ) from None
    if not isinstance(data, dict):
        raise ConversationError(
            400, "Request body must be a JSON object", "invalid_type"
        )
    return data


def _int_param(request: Request, name: str, default: int, lo: int, hi: int) -> int:
    raw = request.query_params.get(name)
    if raw is None:
        return default
    try:
        v = int(raw)
    except ValueError:
        raise ConversationError(
            400, f"{name} must be an integer", "invalid_type", name
        ) from None
    if not lo <= v <= hi:
        raise ConversationError(
            400, f"{name} must be between {lo} and {hi}", "invalid_value", name
        )
    return v


@router.post("/conversations")
async def create_conversation(request: Request):
    _check_permission(request, "can_infer")
    try:
        body = await _body(request)
        return await _io(get_store().create, body.get("items"), body.get("metadata"))
    except ConversationError as exc:
        return _err(exc)


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, request: Request):
    _check_permission(request, "can_infer")
    try:
        return await _io(get_store().get, conversation_id)
    except ConversationError as exc:
        return _err(exc)


@router.post("/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, request: Request):
    _check_permission(request, "can_infer")
    try:
        body = await _body(request)
        if "metadata" not in body:
            raise ConversationError(
                400, "metadata is required", "missing_required_parameter", "metadata"
            )
        return await _io(get_store().update_metadata, conversation_id, body["metadata"])
    except ConversationError as exc:
        return _err(exc)


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, request: Request):
    _check_permission(request, "can_infer")
    try:
        await _io(get_store().delete, conversation_id)
    except ConversationError as exc:
        return _err(exc)
    return {"id": conversation_id, "object": "conversation.deleted", "deleted": True}


@router.post("/conversations/{conversation_id}/items")
async def create_items(conversation_id: str, request: Request):
    _check_permission(request, "can_infer")
    try:
        body = await _body(request)
        items = body.get("items")
        if not isinstance(items, list):
            raise ConversationError(
                400, "items is required", "missing_required_parameter", "items"
            )
        added = await _io(
            get_store().add_items, conversation_id, items, cap=MAX_CREATE_ITEMS
        )
    except ConversationError as exc:
        return _err(exc)
    return {
        "object": "list",
        "data": added,
        "first_id": added[0]["id"] if added else None,
        "last_id": added[-1]["id"] if added else None,
        "has_more": False,
    }


@router.get("/conversations/{conversation_id}/items")
async def list_items(conversation_id: str, request: Request):
    _check_permission(request, "can_infer")
    try:
        limit = _int_param(request, "limit", 20, 1, 100)
        order = request.query_params.get("order", "desc")
        if order not in ("asc", "desc"):
            raise ConversationError(
                400, "order must be 'asc' or 'desc'", "invalid_value", "order"
            )
        return await _io(
            get_store().list_items,
            conversation_id,
            limit=limit,
            order=order,
            after=request.query_params.get("after"),
        )
    except ConversationError as exc:
        return _err(exc)


@router.get("/conversations/{conversation_id}/items/{item_id}")
async def get_item(conversation_id: str, item_id: str, request: Request) -> Any:
    _check_permission(request, "can_infer")
    try:
        return await _io(get_store().get_item, conversation_id, item_id)
    except ConversationError as exc:
        return _err(exc)


@router.delete("/conversations/{conversation_id}/items/{item_id}")
async def delete_item(conversation_id: str, item_id: str, request: Request):
    _check_permission(request, "can_infer")
    try:
        return await _io(get_store().delete_item, conversation_id, item_id)
    except ConversationError as exc:
        return _err(exc)
