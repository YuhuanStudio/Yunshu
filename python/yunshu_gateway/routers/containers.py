"""OpenAI ``/v1/containers``: the sandboxes behind ``code_interpreter`` and ``shell``, and their files.

A container here is a Yunshu code-execution container (a directory run under srt, see
:mod:`yunshu_gateway.sandbox`): the same one the Anthropic ``container`` parameter names. Off unless
``YUNSHU_CODE_EXECUTION`` is on (``400`` naming it); on without srt it is ``503``. Mapping to the
hosted API: ``expires_after`` (anchor ``last_active_at``) is the idle lifetime in minutes, capped
by ``YUNSHU_CODE_EXECUTION_TTL_MIN``; ``memory_limit`` is a ceiling under
``YUNSHU_CODE_EXECUTION_MEMORY_MB``; ``network_policy`` can only narrow
``YUNSHU_CODE_EXECUTION_NETWORK_ALLOW``. ``/mnt/data/<path>`` is the container's working directory.
"""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from yunshu_engine import settings

from ..files_store import FileStoreError, get_store
from ..sandbox import (
    Container,
    ContainerError,
    SandboxUnavailableError,
    get_executor,
    get_manager,
)
from ..sandbox import files as cfiles
from ..sandbox.containers import MEMORY_LIMITS, owner_ok
from ..sandbox.srt import host_allowed, parse_allowlist
from .files import _int_param, _read_upload, auth, paginate_ids

router = APIRouter(tags=["containers"])

PREFIX = "cntr"
DEFAULT_MEMORY = "1g"


def _err(
    status: int,
    message: str,
    code: str | None = None,
    param: str | None = None,
    etype: str | None = None,
) -> JSONResponse:
    et = etype or (
        "not_found_error"
        if status == 404
        else "server_error"
        if status >= 500
        else "invalid_request_error"
    )
    return JSONResponse(
        {"error": {"message": message, "type": et, "param": param, "code": code}},
        status_code=status,
    )


def _actor(request: Request) -> tuple[str, bool]:
    from .responses import _is_admin, _resolve_owner

    return _resolve_owner(request), _is_admin(request)


def gate(request: Request) -> JSONResponse | None:
    """Auth, the on/off setting and the srt requirement (never an unsandboxed run)."""
    if (r := auth(request, anthropic=False)) is not None:
        return r
    if not settings.get("YUNSHU_CODE_EXECUTION"):
        return _err(
            400,
            "Containers are disabled on this server. Set YUNSHU_CODE_EXECUTION=1 "
            "(code runs inside the sandbox-runtime, srt).",
            "code_execution_disabled",
        )
    try:
        get_executor().srt()
    except SandboxUnavailableError as e:
        return _err(503, str(e), "sandbox_unavailable")
    return None


def effective_label(c: Container) -> str:
    cap = get_executor().memory_bytes(c) // 2**20
    best = DEFAULT_MEMORY
    for label, mb in MEMORY_LIMITS.items():
        if mb <= cap:
            best = label
    return best


def container_object(c: Container) -> dict[str, Any]:
    allow = get_executor().network_allow(c)
    net: dict[str, Any] = (
        {"type": "allowlist", "allowed_domains": allow}
        if allow
        else {"type": "disabled"}
    )
    return {
        "id": c.id,
        "object": "container",
        "created_at": int(c.created_at),
        "name": c.name or "",
        "status": "running",
        "last_active_at": int(c.last_active),
        "expires_after": {
            "anchor": "last_active_at",
            "minutes": max(1, int(round(c.ttl_s / 60))),
        },
        "memory_limit": effective_label(c),
        "network_policy": net,
    }


def file_object(c: Container, e: cfiles.Entry) -> dict[str, Any]:
    return cfiles.record(c, e)


def _load(request: Request, cid: str) -> Container | JSONResponse:
    """The container (and refresh its idle timer), or the 404 the hosted API gives."""
    mgr = get_manager()
    try:
        c = mgr.get(cid)
    except ContainerError as e:
        if e.code == "invalid" or not cid.startswith(PREFIX + "_"):
            return _err(404, f"Container '{cid}' not found.", "container_not_found")
        return _err(
            404, f"Container '{cid}' not found or expired.", "container_not_found"
        )
    actor, admin = _actor(request)
    if not cid.startswith(PREFIX + "_") or not owner_ok(c, actor, admin):
        return _err(404, f"Container '{cid}' not found.", "container_not_found")
    return c


def _no_content() -> Response:
    return Response(status_code=204)


# ── containers ───────────────────────────────────────────────────────────────
@router.post("/containers")
async def create_container(request: Request):
    if (r := gate(request)) is not None:
        return r
    try:
        body = await request.json()
    except ValueError:
        return _err(400, "Request body must be JSON.", "invalid_json")
    if not isinstance(body, dict):
        return _err(400, "Request body must be a JSON object.")
    name = body.get("name")
    if not isinstance(name, str) or not name.strip() or len(name) > 256:
        return _err(
            400,
            "'name' is required (a non-empty string).",
            "missing_required_parameter",
            "name",
        )
    if body.get("skills"):
        return _err(
            400,
            "Container skills are not supported.",
            "unsupported_parameter",
            "skills",
        )
    server_ttl = float(settings.get("YUNSHU_CODE_EXECUTION_TTL_MIN")) * 60.0
    ttl_s = server_ttl
    ea = body.get("expires_after")
    if ea is not None:
        if (
            not isinstance(ea, dict)
            or ea.get("anchor") != "last_active_at"
            or not isinstance(ea.get("minutes"), int)
            or isinstance(ea.get("minutes"), bool)
            or ea["minutes"] < 1
        ):
            return _err(
                400,
                "'expires_after' must be {anchor: 'last_active_at', minutes: <int >= 1>}.",
                "invalid_value",
                "expires_after",
            )
        ttl_s = min(server_ttl, ea["minutes"] * 60.0)
    ml = body.get("memory_limit")
    if ml is not None and ml not in MEMORY_LIMITS:
        return _err(
            400,
            f"'memory_limit' must be one of {', '.join(MEMORY_LIMITS)}.",
            "invalid_value",
            "memory_limit",
        )
    server_net = parse_allowlist(settings.get("YUNSHU_CODE_EXECUTION_NETWORK_ALLOW"))
    net_allow: list[str] | None = None
    np_ = body.get("network_policy")
    if np_ is not None:
        if not isinstance(np_, dict) or np_.get("type") not in (
            "disabled",
            "allowlist",
        ):
            return _err(
                400,
                "'network_policy.type' must be 'disabled' or 'allowlist'.",
                "invalid_value",
                "network_policy",
            )
        if np_["type"] == "disabled":
            net_allow = []
        else:
            if np_.get("domain_secrets"):
                return _err(
                    400,
                    "domain_secrets are not supported.",
                    "unsupported_parameter",
                    "network_policy.domain_secrets",
                )
            domains = np_.get("allowed_domains")
            if not isinstance(domains, list) or not all(
                isinstance(d, str) and d.strip() for d in domains
            ):
                return _err(
                    400,
                    "'network_policy.allowed_domains' must be a list of host names.",
                    "invalid_value",
                    "network_policy.allowed_domains",
                )
            bad = [d for d in domains if not host_allowed(d, server_net)]
            if bad:
                return _err(
                    400,
                    f"Domain(s) {', '.join(bad)} are not allowed on this server; "
                    "the operator lists reachable hosts in YUNSHU_CODE_EXECUTION_NETWORK_ALLOW.",
                    "domain_not_allowed",
                    "network_policy.allowed_domains",
                )
            net_allow = [d.strip() for d in domains]
    file_ids = body.get("file_ids") or []
    if not isinstance(file_ids, list) or not all(isinstance(f, str) for f in file_ids):
        return _err(
            400, "'file_ids' must be a list of file ids.", "invalid_value", "file_ids"
        )
    ex = get_executor()
    actor, _ = _actor(request)
    mgr = get_manager()
    c = mgr.create(
        prefix=PREFIX,
        name=name.strip(),
        ttl_s=ttl_s,
        memory_limit=ml,
        network_allow=net_allow,
        owner=actor,
    )
    try:
        for fid in file_ids:
            await asyncio.to_thread(ex.stage_file, c, fid)
    except FileStoreError as exc:
        mgr.delete(c.id)
        return _err(
            404 if exc.status == 404 else 400, exc.message, exc.code, "file_ids"
        )
    return JSONResponse(container_object(mgr.get(c.id, touch=False)))


@router.get("/containers")
async def list_containers(request: Request):
    if (r := gate(request)) is not None:
        return r
    limit = _int_param(request, "limit", 20, 1, 100)
    if isinstance(limit, JSONResponse):
        return limit
    q = request.query_params
    order = q.get("order", "desc")
    if order not in ("asc", "desc"):
        return _err(400, "'order' must be 'asc' or 'desc'", "invalid_value", "order")
    actor, admin = _actor(request)
    items = [
        c
        for c in get_manager().list(PREFIX)
        if owner_ok(c, actor, admin) and (not q.get("name") or c.name == q["name"])
    ]
    if order == "desc":
        items.reverse()
    rows = [container_object(c) for c in items]
    after = q.get("after")
    if after and after not in {r["id"] for r in rows}:
        return _err(
            400,
            f"Container '{after}' (after) does not exist.",
            "invalid_value",
            "after",
        )
    page, more = paginate_ids(rows, limit, after)
    return {
        "object": "list",
        "data": page,
        "first_id": page[0]["id"] if page else None,
        "last_id": page[-1]["id"] if page else None,
        "has_more": more,
    }


@router.get("/containers/{container_id}")
async def retrieve_container(container_id: str, request: Request):
    if (r := gate(request)) is not None:
        return r
    c = _load(request, container_id)
    if isinstance(c, JSONResponse):
        return c
    return container_object(c)


@router.delete("/containers/{container_id}")
async def delete_container(container_id: str, request: Request):
    if (r := gate(request)) is not None:
        return r
    c = _load(request, container_id)
    if isinstance(c, JSONResponse):
        return c
    get_manager().delete(c.id)
    return _no_content()


# ── container files ──────────────────────────────────────────────────────────
@router.post("/containers/{container_id}/files")
async def create_container_file(container_id: str, request: Request):
    if (r := gate(request)) is not None:
        return r
    c = _load(request, container_id)
    if isinstance(c, JSONResponse):
        return c
    ctype = request.headers.get("content-type", "")
    data: bytes
    fname: str | None
    if "multipart/form-data" in ctype:
        try:
            form = await request.form()
        except Exception:  # noqa: BLE001
            return _err(400, "Expected a multipart/form-data body.")
        upload = form.get("file")
        if upload is not None and not isinstance(upload, str):
            try:
                data = await _read_upload(get_store(), upload)
            except FileStoreError as exc:
                return _err(exc.status, exc.message, exc.code)
            fname = (
                form.get("file_path")
                if isinstance(form.get("file_path"), str)
                else None
            )
            fname = fname or upload.filename
        else:
            fid = form.get("file_id")
            if not isinstance(fid, str):
                return _err(
                    400,
                    "Provide a 'file' (multipart file part) or a 'file_id'.",
                    "missing_required_parameter",
                    "file",
                )
            return await _copy_from_store(c, fid)
    else:
        try:
            body = await request.json()
        except ValueError:
            return _err(400, "Provide a multipart 'file' or JSON {\"file_id\": ...}.")
        fid = body.get("file_id") if isinstance(body, dict) else None
        if not isinstance(fid, str):
            return _err(
                400, "'file_id' is required.", "missing_required_parameter", "file_id"
            )
        return await _copy_from_store(c, fid)
    rel = cfiles.clean_relpath(fname)
    if rel is None:
        return _err(400, "The file name is not valid.", "invalid_value", "file")
    return await _put(c, rel, data)


async def _copy_from_store(c: Container, fid: str):
    try:
        store = get_store()
        meta = store.get_meta(fid)
        data = await asyncio.to_thread(store.read, fid)
    except FileStoreError as exc:
        return _err(404 if exc.status == 404 else 400, exc.message, exc.code, "file_id")
    rel = cfiles.clean_relpath(meta.get("filename") or fid)
    if rel is None:
        return _err(400, "The file name is not valid.", "invalid_value", "file_id")
    return await _put(c, rel, data)


async def _put(c: Container, rel: str, data: bytes):
    try:
        await asyncio.to_thread(cfiles.write_file, c, rel, data)
    except cfiles.FileOpError as e:
        return _err(413 if e.code == "too_large" else 400, e.message, "invalid_value")
    get_manager().add_user_file(c.id, rel)
    c = get_manager().get(c.id)
    ent = next((x for x in cfiles.scan(c) if x.rel == rel), None)
    if ent is None:
        return _err(500, "The file was not stored.")
    return JSONResponse(file_object(c, ent))


@router.get("/containers/{container_id}/files")
async def list_container_files(container_id: str, request: Request):
    if (r := gate(request)) is not None:
        return r
    c = _load(request, container_id)
    if isinstance(c, JSONResponse):
        return c
    limit = _int_param(request, "limit", 20, 1, 100)
    if isinstance(limit, JSONResponse):
        return limit
    order = request.query_params.get("order", "desc")
    if order not in ("asc", "desc"):
        return _err(400, "'order' must be 'asc' or 'desc'", "invalid_value", "order")
    ents = await asyncio.to_thread(cfiles.scan, c)
    if order == "desc":
        ents = ents[::-1]
    rows = [file_object(c, e) for e in ents]
    after = request.query_params.get("after")
    if after and after not in {r["id"] for r in rows}:
        return _err(
            400, f"File '{after}' (after) does not exist.", "invalid_value", "after"
        )
    page, more = paginate_ids(rows, limit, after)
    return {
        "object": "list",
        "data": page,
        "first_id": page[0]["id"] if page else None,
        "last_id": page[-1]["id"] if page else None,
        "has_more": more,
    }


def _find(c: Container, file_id: str) -> cfiles.Entry | JSONResponse:
    e = cfiles.find(c, file_id)
    if e is None:
        return _err(404, f"File '{file_id}' not found.", "file_not_found")
    return e


@router.get("/containers/{container_id}/files/{file_id}")
async def retrieve_container_file(container_id: str, file_id: str, request: Request):
    if (r := gate(request)) is not None:
        return r
    c = _load(request, container_id)
    if isinstance(c, JSONResponse):
        return c
    e = await asyncio.to_thread(_find, c, file_id)
    if isinstance(e, JSONResponse):
        return e
    return file_object(c, e)


@router.get("/containers/{container_id}/files/{file_id}/content")
async def container_file_content(container_id: str, file_id: str, request: Request):
    if (r := gate(request)) is not None:
        return r
    c = _load(request, container_id)
    if isinstance(c, JSONResponse):
        return c
    e = await asyncio.to_thread(_find, c, file_id)
    if isinstance(e, JSONResponse):
        return e
    try:
        data = await asyncio.to_thread(cfiles.read_file, c, e.rel)
    except cfiles.FileOpError as exc:
        return _err(
            404 if exc.code == "not_found" else 400, exc.message, "file_not_found"
        )
    base = e.rel.rsplit("/", 1)[-1]
    return Response(
        data,
        media_type=cfiles.mime_for(e.rel),
        headers={"Content-Disposition": "attachment; filename*=UTF-8''" + quote(base)},
    )


@router.delete("/containers/{container_id}/files/{file_id}")
async def delete_container_file(container_id: str, file_id: str, request: Request):
    if (r := gate(request)) is not None:
        return r
    c = _load(request, container_id)
    if isinstance(c, JSONResponse):
        return c
    e = await asyncio.to_thread(_find, c, file_id)
    if isinstance(e, JSONResponse):
        return e
    try:
        await asyncio.to_thread(cfiles.delete_file, c, e.rel)
    except cfiles.FileOpError as exc:
        return _err(404, exc.message, "file_not_found")
    get_manager().drop_user_file(c.id, e.rel)
    return _no_content()
