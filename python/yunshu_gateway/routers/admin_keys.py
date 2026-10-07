"""Admin API for the local API-key store (``admin`` scope; the YUNSHU_AUTH_TOKEN is an admin key)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from .. import api_keys
from .models import _check_permission

router = APIRouter(tags=["yunshu-keys"])

_FIELDS = ("requests", "prompt_tokens", "completion_tokens", "cached_tokens", "errors")


class KeyCreate(BaseModel):
    name: str
    scopes: list[str] | None = None
    quotas: dict[str, int | None] | None = None
    expires: float | None = None  # epoch seconds


def _admin(request: Request) -> api_keys.KeyStore:
    _check_permission(request, "admin")
    return api_keys.get_store()


def _get(store: api_keys.KeyStore, key_id: str) -> api_keys.KeyRecord:
    rec = store._keys.get(key_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"API key '{key_id}' not found")
    return rec


@router.get("/yunshu/keys")
async def list_keys(request: Request) -> dict:
    store = _admin(request)
    return {"object": "list", "data": store.list_keys()}


@router.post("/yunshu/keys", status_code=201)
async def create_key(body: KeyCreate, request: Request) -> dict:
    store = _admin(request)
    try:
        rec, secret = store.create(body.name, body.scopes, body.quotas, body.expires)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # The secret is shown exactly once; only its hash is stored.
    return {**store.public(rec), "secret": secret}


@router.patch("/yunshu/keys/{key_id}")
async def patch_key(key_id: str, request: Request, body: dict[str, Any]) -> dict:
    store = _admin(request)
    _get(store, key_id)
    unknown = set(body) - {"name", "enabled", "scopes", "quotas", "expires"}
    if unknown:
        raise HTTPException(
            status_code=400, detail=f"unknown field {sorted(unknown)[0]!r}"
        )
    try:
        rec = store.update(key_id, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return store.public(rec)


@router.delete("/yunshu/keys/{key_id}")
async def delete_key(key_id: str, request: Request) -> dict:
    store = _admin(request)
    _get(store, key_id)
    store.delete(key_id)
    return {"id": key_id, "object": "api_key", "deleted": True}


@router.post("/yunshu/keys/{key_id}/rotate")
async def rotate_key(key_id: str, request: Request) -> dict:
    store = _admin(request)
    _get(store, key_id)
    rec, secret = store.rotate(key_id)
    return {**store.public(rec), "secret": secret}


@router.get("/yunshu/usage")
async def usage(
    request: Request,
    key: str | None = None,
    since: str | None = Query(None, description="YYYY-MM-DD or a day count like 7d"),
    group: str = Query("day", pattern="^(day|key|total)$"),
) -> dict:
    store = _admin(request)
    since_day = _since(since)
    rows = store.usage(key, since_day)
    if group == "day":
        data: list[dict] = rows
    else:
        agg: dict[str, dict] = {}
        for r in rows:
            k = r["key"] if group == "key" else "total"
            a = agg.setdefault(
                k, {"key": k if group == "key" else None, **{f: 0 for f in _FIELDS}}
            )
            if group == "key":
                a["name"] = r["name"]
            for f in _FIELDS:
                a[f] += r[f]
        data = list(agg.values())
    return {"object": "list", "group": group, "since": since_day, "data": data}


def _since(since: str | None) -> str | None:
    if not since:
        return None
    try:
        if since.endswith("d") and since[:-1].isdigit():
            day = datetime.now(UTC) - timedelta(days=int(since[:-1]))
            return day.strftime("%Y-%m-%d")
        return datetime.strptime(since, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="since must be YYYY-MM-DD or like '7d'"
        ) from exc
