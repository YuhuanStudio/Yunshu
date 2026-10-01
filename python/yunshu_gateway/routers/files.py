"""Files API: OpenAI and Anthropic (beta files-api-2025-04-14) on the same /v1/files paths.

The Anthropic SDKs always send ``anthropic-version``; the OpenAI SDK never
does, so that header picks the schema. One local store backs both.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from yunshu_kv.disk_budget import is_enospc

from ..files_store import FileStore, FileStoreError, get_store
from .models import _check_permission

router = APIRouter(tags=["files"])

OPENAI_PURPOSES = ("assistants", "batch", "fine-tune", "vision", "user_data", "evals")
_CHUNK = 1 << 20


def is_anthropic(request: Request) -> bool:
    return "anthropic-version" in request.headers


def error_response(
    request: Request,
    status: int,
    message: str,
    code: str | None = None,
    param: str | None = None,
    anthropic: bool | None = None,
) -> JSONResponse:
    if is_anthropic(request) if anthropic is None else anthropic:
        etype = {
            400: "invalid_request_error",
            401: "authentication_error",
            403: "permission_error",
            404: "not_found_error",
            409: "invalid_request_error",
            413: "request_too_large",
        }.get(status, "api_error")
        return JSONResponse(
            {"type": "error", "error": {"type": etype, "message": message}},
            status_code=status,
        )
    etype = "not_found_error" if status == 404 else "invalid_request_error"
    if status >= 500:
        etype = "server_error"
    return JSONResponse(
        {"error": {"message": message, "type": etype, "param": param, "code": code}},
        status_code=status,
    )


def auth(request: Request, anthropic: bool | None = None) -> JSONResponse | None:
    try:
        _check_permission(request, "can_infer")
    except HTTPException as exc:
        return error_response(
            request, exc.status_code, str(exc.detail), anthropic=anthropic
        )
    return None


def iso(ts: int | float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat().replace("+00:00", "Z")


def openai_file(m: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": m["id"],
        "object": "file",
        "bytes": m["bytes"],
        "created_at": m["created_at"],
        "expires_at": m.get("expires_at"),
        "filename": m["filename"],
        "purpose": m["purpose"],
        "status": "processed",
        "status_details": None,
    }


def anthropic_file(m: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": m["id"],
        "type": "file",
        "filename": m["filename"],
        "mime_type": m["mime_type"],
        "size_bytes": m["bytes"],
        "created_at": iso(m["created_at"]),
        "downloadable": bool(m.get("downloadable")),
    }


def paginate_ids(
    items: list[dict], limit: int, after: str | None = None, before: str | None = None
) -> tuple[list[dict], bool]:
    """Cursor paging over an already ordered list. ``after`` = items following
    the cursor in list order, ``before`` = items preceding it."""
    ids = [x["id"] for x in items]
    if after is not None:
        if after not in ids:
            return [], False
        items = items[ids.index(after) + 1 :]
    elif before is not None:
        if before not in ids:
            return [], False
        items = items[: ids.index(before)]
        page = items[-limit:]
        return page, len(items) > limit
    return items[:limit], len(items) > limit


def _int_param(
    request: Request, name: str, default: int, lo: int, hi: int
) -> int | JSONResponse:
    raw = request.query_params.get(name)
    if raw is None:
        return default
    try:
        v = int(raw)
    except ValueError:
        return error_response(request, 400, f"'{name}' must be an integer", param=name)
    if not lo <= v <= hi:
        return error_response(
            request, 400, f"'{name}' must be between {lo} and {hi}", param=name
        )
    return v


async def _read_upload(store: FileStore, upload: Any) -> bytes:
    buf = bytearray()
    while True:
        chunk = await upload.read(_CHUNK)
        if not chunk:
            break
        buf += chunk
        if len(buf) > store.max_bytes:
            raise FileStoreError(
                413, f"File exceeds the {store.max_bytes} byte limit", "file_too_large"
            )
    return bytes(buf)


@router.post("/files")
async def create_file(request: Request):
    if (r := auth(request)) is not None:
        return r
    anth = is_anthropic(request)
    store = get_store()
    try:
        form = await request.form()
    except Exception:
        return error_response(request, 400, "Expected a multipart/form-data body")
    upload = form.get("file")
    if upload is None or isinstance(upload, str):
        return error_response(
            request, 400, "'file' is required (multipart file part)", param="file"
        )
    purpose = form.get("purpose")
    expires_after = None
    if not anth:
        if purpose not in OPENAI_PURPOSES:
            return error_response(
                request,
                400,
                f"Invalid value for 'purpose': must be one of {', '.join(OPENAI_PURPOSES)}",
                "invalid_value",
                "purpose",
            )
        ea = _parse_expires(form)
        if isinstance(ea, JSONResponse):
            return ea
        expires_after = ea
    try:
        data = await _read_upload(store, upload)
        if not data and not anth:
            return error_response(
                request, 400, "The uploaded file is empty", param="file"
            )
        meta = await asyncio.to_thread(
            store.put,
            data,
            upload.filename or "file",
            purpose or "user_data",
            upload.content_type,
            expires_after,
        )
    except FileStoreError as exc:
        return error_response(request, exc.status, exc.message, exc.code)
    except OSError as exc:
        if not is_enospc(exc):
            raise
        # nothing was kept: the store removes its temp file and the half-written blob
        return error_response(
            request,
            507,
            "The server's disk is full, the file was not stored; free space and retry",
            "insufficient_storage",
        )
    return JSONResponse(anthropic_file(meta) if anth else openai_file(meta))


def _parse_expires(form) -> int | None | JSONResponse:
    anchor = form.get("expires_after[anchor]")
    seconds = form.get("expires_after[seconds]")
    raw = form.get("expires_after")
    if raw and isinstance(raw, str):
        import json

        try:
            obj = json.loads(raw)
            anchor, seconds = obj.get("anchor"), obj.get("seconds")
        except (ValueError, AttributeError):
            pass
    if anchor is None and seconds is None:
        return None
    try:
        secs = int(seconds)
    except (TypeError, ValueError):
        secs = -1
    if anchor != "created_at" or not 3600 <= secs <= 2592000:
        return JSONResponse(
            {
                "error": {
                    "message": "expires_after must be {anchor: 'created_at', seconds: 3600..2592000}",
                    "type": "invalid_request_error",
                    "param": "expires_after",
                    "code": "invalid_value",
                }
            },
            status_code=400,
        )
    return secs


@router.get("/files")
async def list_files(request: Request):
    if (r := auth(request)) is not None:
        return r
    anth = is_anthropic(request)
    store = get_store()
    if anth:
        limit = _int_param(request, "limit", 20, 1, 1000)
    else:
        limit = _int_param(request, "limit", 10000, 1, 10000)
    if isinstance(limit, JSONResponse):
        return limit
    q = request.query_params
    items = store.list(None if anth else q.get("purpose"))
    if anth:
        page, more = paginate_ids(items, limit, q.get("after_id"), q.get("before_id"))
        data = [anthropic_file(m) for m in page]
        return {
            "data": data,
            "first_id": data[0]["id"] if data else None,
            "last_id": data[-1]["id"] if data else None,
            "has_more": more,
        }
    order = q.get("order", "desc")
    if order not in ("asc", "desc"):
        return error_response(
            request, 400, "'order' must be 'asc' or 'desc'", param="order"
        )
    if order == "asc":
        items = items[::-1]
    after = q.get("after")
    if after:
        from ..files_store import normalize_id

        after = normalize_id(after)
    page, more = paginate_ids(items, limit, after)
    data = [openai_file(m) for m in page]
    return {
        "object": "list",
        "data": data,
        "has_more": more,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
    }


@router.get("/files/{file_id}")
async def retrieve_file(file_id: str, request: Request):
    if (r := auth(request)) is not None:
        return r
    try:
        meta = get_store().get_meta(file_id)
    except FileStoreError as exc:
        return error_response(request, exc.status, exc.message, exc.code)
    return anthropic_file(meta) if is_anthropic(request) else openai_file(meta)


@router.delete("/files/{file_id}")
async def delete_file(file_id: str, request: Request):
    if (r := auth(request)) is not None:
        return r
    try:
        meta = get_store().get_meta(file_id)
        get_store().delete(file_id)
    except FileStoreError as exc:
        return error_response(request, exc.status, exc.message, exc.code)
    if is_anthropic(request):
        return {"id": meta["id"], "type": "file_deleted"}
    return {"id": meta["id"], "object": "file", "deleted": True}


@router.get("/files/{file_id}/content")
async def file_content(file_id: str, request: Request):
    if (r := auth(request)) is not None:
        return r
    store = get_store()
    try:
        meta = store.get_meta(file_id)
    except FileStoreError as exc:
        return error_response(request, exc.status, exc.message, exc.code)
    if is_anthropic(request) and not meta.get("downloadable"):
        return error_response(
            request,
            403,
            "Files you uploaded cannot be downloaded; only files created by tools can.",
        )
    return FileResponse(
        store.blob_path(meta["id"]),
        media_type=meta["mime_type"],
        filename=meta["filename"],
        content_disposition_type="attachment",
    )
