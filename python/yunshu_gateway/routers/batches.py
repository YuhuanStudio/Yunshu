"""Batch APIs: OpenAI ``/v1/batches`` and Anthropic ``/v1/messages/batches``.

A single background worker drains queued batches one request at a time by
calling this same server over loopback HTTP, so auth, model routing, prefix
cache and tools behave exactly as on the normal routes and interactive traffic
is interleaved (the engine itself serializes GPU work). Batch state lives in
the file store, so unfinished batches resume after a restart.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import secrets
import time
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from yunshu_engine import settings

from ..files_store import FileStore, FileStoreError, get_store, new_id
from .files import auth, error_response, iso, paginate_ids

logger = logging.getLogger(__name__)
router = APIRouter(tags=["batches"])

OPENAI_ENDPOINTS = (
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/responses",
)
MAX_OPENAI_LINES = 50_000
MAX_ANTHROPIC_REQUESTS = 100_000
WINDOW_SECONDS = 86_400
ACTIVE = ("validating", "in_progress", "cancelling")
_CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _default_client(base_url: str, headers: dict[str, str]) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=base_url, headers=headers, timeout=None)


class BatchRunner:
    def __init__(self) -> None:
        self.client_factory = _default_client
        self.autostart = True
        self.yield_seconds = 0.05
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake: asyncio.Event | None = None

    def ensure_started(self) -> None:
        if not self.autostart:
            return
        loop = asyncio.get_running_loop()
        if self._task is not None and not self._task.done() and self._loop is loop:
            if self._wake is not None:
                self._wake.set()
            return
        self._loop = loop
        self._wake = asyncio.Event()
        self._task = loop.create_task(self._run(), name="yunshu-batch-runner")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def _run(self) -> None:
        while True:
            try:
                bid = self.next_batch(get_store())
                if bid is None:
                    self._wake.clear()
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(self._wake.wait(), 30)
                    continue
                await self.process(bid)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("batch runner error")
                await asyncio.sleep(1)

    @staticmethod
    def next_batch(store: FileStore) -> str | None:
        active = [
            r
            for api in ("openai", "anthropic")
            for r in store.list_batches(api)
            if r["status"] in ACTIVE
        ]
        if not active:
            return None
        return min(active, key=lambda r: (r["created_at"], r["id"]))["id"]

    async def process_all(self) -> None:
        """Drain every queued batch (used by tests and on demand)."""
        store = get_store()
        while (bid := self.next_batch(store)) is not None:
            await self.process(bid)

    # -- one batch ---------------------------------------------------------
    async def process(self, bid: str) -> None:
        store = get_store()
        rec = store.load_batch(bid)
        if rec is None or rec["status"] not in ACTIVE:
            return
        lines = store.read_lines(bid, "in")
        done = {
            ln["custom_id"] for k in ("out", "err") for ln in store.read_lines(bid, k)
        }
        if rec["status"] != "cancelling":
            rec = (
                store.update_batch(
                    bid,
                    status="in_progress",
                    in_progress_at=rec.get("in_progress_at") or int(time.time()),
                )
                or rec
            )
        headers = self._headers(rec)
        end = "completed"
        async with self.client_factory(rec["base_url"], headers) as client:
            for line in lines:
                if line["custom_id"] in done:
                    continue
                cur = store.load_batch(bid)
                if cur is None:
                    return  # deleted mid-run
                rec = cur
                if rec["status"] == "cancelling":
                    end = "cancelled"
                    break
                if time.time() >= rec["expires_at"]:
                    end = "expired"
                    break
                await self._run_one(store, rec, client, line)
                await asyncio.sleep(self.yield_seconds)
        self._finalize(store, rec, lines, end)

    @staticmethod
    def _headers(rec: dict[str, Any]) -> dict[str, str]:
        h: dict[str, str] = {}
        token = settings.get("YUNSHU_AUTH_TOKEN")
        if token:
            h["Authorization"] = f"Bearer {token}"
            h["x-api-key"] = str(token)
        if rec["api"] == "anthropic":
            h["anthropic-version"] = "2023-06-01"
        return h

    async def _run_one(
        self, store: FileStore, rec: dict, client: httpx.AsyncClient, line: dict
    ) -> None:
        cid = line["custom_id"]
        body = dict(line["body"])
        body["stream"] = False
        status = 0
        request_id = None
        payload: Any = None
        exc_msg = None
        try:
            resp = await client.post(line["url"], json=body)
            status = resp.status_code
            request_id = resp.headers.get("x-request-id")
            try:
                payload = resp.json()
            except ValueError:
                payload = {"error": {"message": resp.text[:2000]}}
        except Exception as exc:  # transport failure
            exc_msg = f"{type(exc).__name__}: {exc}"
        request_id = request_id or f"req_{secrets.token_hex(12)}"
        ok = exc_msg is None and 200 <= status < 300
        c = rec["counts"]
        if rec["api"] == "openai":
            out = {
                "id": new_id("batch_req"),
                "custom_id": cid,
                "response": None
                if exc_msg
                else {"status_code": status, "request_id": request_id, "body": payload},
                "error": {"code": "request_failed", "message": exc_msg}
                if exc_msg
                else None,
            }
        else:
            if ok:
                result = {"type": "succeeded", "message": payload}
            else:
                err = (
                    payload
                    if isinstance(payload, dict) and payload.get("type") == "error"
                    else None
                )
                if err is None:
                    detail = (
                        payload.get("error", payload)
                        if isinstance(payload, dict)
                        else payload
                    )
                    msg = (
                        detail.get("message")
                        if isinstance(detail, dict)
                        else str(detail)
                    )
                    err = {
                        "type": "error",
                        "error": {
                            "type": "api_error"
                            if status >= 500 or exc_msg
                            else "invalid_request_error",
                            "message": exc_msg or msg or f"HTTP {status}",
                        },
                    }
                result = {"type": "errored", "error": err}
            out = {"custom_id": cid, "result": result}
        store.append_line(
            rec["id"], "out" if (ok or rec["api"] == "anthropic") else "err", out
        )
        c["succeeded" if ok else "errored"] += 1
        store.update_batch(rec["id"], counts=c)

    def _finalize(
        self, store: FileStore, rec: dict, lines: list[dict], end: str
    ) -> None:
        now = int(time.time())
        c = rec["counts"]
        if rec["api"] == "anthropic":
            done = {ln["custom_id"] for ln in store.read_lines(rec["id"], "out")}
            kind = {"cancelled": "canceled", "expired": "expired"}.get(end)
            for ln in lines:
                if ln["custom_id"] not in done and kind:
                    store.append_line(
                        rec["id"],
                        "out",
                        {"custom_id": ln["custom_id"], "result": {"type": kind}},
                    )
                    c[kind] += 1
        else:
            rec["finalizing_at"] = now
            store.update_batch(rec["id"], status="finalizing", finalizing_at=now)
            for kind, key in (("out", "output_file_id"), ("err", "error_file_id")):
                data = store.batch_file(rec["id"], kind)
                if data.exists() and data.stat().st_size and not rec.get(key):
                    rec[key] = store.put(
                        data.read_bytes(),
                        f"{rec['id']}_{'output' if kind == 'out' else 'error'}.jsonl",
                        "batch_output" if kind == "out" else "batch_error",
                        "application/x-jsonl",
                    )["id"]
        ts_key = {
            "completed": "completed_at",
            "cancelled": "cancelled_at",
            "expired": "expired_at",
        }[end]
        store.update_batch(
            rec["id"],
            counts=c,
            status=end,
            **{ts_key: now},
            **(
                {
                    k: rec[k]
                    for k in ("output_file_id", "error_file_id", "finalizing_at")
                    if k in rec
                }
            ),
        )


runner = BatchRunner()


def start_runner() -> None:
    """Called from the gateway lifespan: resume unfinished batches."""
    try:
        if BatchRunner.next_batch(get_store()) is not None:
            runner.ensure_started()
    except Exception:
        logger.warning("could not resume batches", exc_info=True)


async def stop_runner() -> None:
    await runner.stop()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _base(request: Request) -> str:
    return f"{request.url.scheme}://{request.url.netloc}"


def _new_record(api: str, prefix: str, request: Request, total: int) -> dict[str, Any]:
    now = int(time.time())
    return {
        "id": new_id(prefix),
        "api": api,
        "status": "validating",
        "created_at": now,
        "expires_at": now + WINDOW_SECONDS,
        "base_url": _base(request),
        "in_progress_at": None,
        "finalizing_at": None,
        "completed_at": None,
        "failed_at": None,
        "expired_at": None,
        "cancelling_at": None,
        "cancelled_at": None,
        "errors": None,
        "output_file_id": None,
        "error_file_id": None,
        "metadata": None,
        "total": total,
        "counts": {"succeeded": 0, "errored": 0, "canceled": 0, "expired": 0},
    }


def _write_input(store: FileStore, rec: dict, lines: list[dict]) -> None:
    p = store.batch_file(rec["id"], "in")
    p.write_bytes(
        b"".join(json.dumps(x, separators=(",", ":")).encode() + b"\n" for x in lines)
    )


async def _json_body(request: Request, anthropic: bool):
    try:
        body = await request.json()
    except Exception:
        return None, error_response(
            request, 400, "Request body must be valid JSON", anthropic=anthropic
        )
    if not isinstance(body, dict):
        return None, error_response(
            request, 400, "Request body must be a JSON object", anthropic=anthropic
        )
    return body, None


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def openai_batch(rec: dict[str, Any]) -> dict[str, Any]:
    c = rec["counts"]
    return {
        "id": rec["id"],
        "object": "batch",
        "endpoint": rec["endpoint"],
        "errors": rec["errors"],
        "input_file_id": rec["input_file_id"],
        "completion_window": "24h",
        "status": rec["status"],
        "output_file_id": rec["output_file_id"],
        "error_file_id": rec["error_file_id"],
        "created_at": rec["created_at"],
        "in_progress_at": rec["in_progress_at"],
        "expires_at": rec["expires_at"],
        "finalizing_at": rec["finalizing_at"],
        "completed_at": rec["completed_at"],
        "failed_at": rec["failed_at"],
        "expired_at": rec["expired_at"],
        "cancelling_at": rec["cancelling_at"],
        "cancelled_at": rec["cancelled_at"],
        "request_counts": {
            "total": rec["total"],
            "completed": c["succeeded"],
            "failed": c["errored"],
        },
        "metadata": rec["metadata"],
    }


def _validate_openai_input(raw: bytes, endpoint: str) -> tuple[list[dict], list[dict]]:
    lines: list[dict] = []
    errors: list[dict] = []
    seen: set[str] = set()

    def bad(n: int, code: str, msg: str, param: str | None = None) -> None:
        if len(errors) < 20:
            errors.append({"code": code, "message": msg, "param": param, "line": n})

    n = 0
    for n, text in enumerate(raw.decode("utf-8", errors="replace").split("\n"), 1):
        if not text.strip():
            continue
        if len(lines) >= MAX_OPENAI_LINES:
            bad(
                n,
                "batch_request_limit_exceeded",
                f"Batches are limited to {MAX_OPENAI_LINES} requests",
            )
            break
        try:
            obj = json.loads(text)
        except ValueError:
            bad(n, "invalid_json_line", "This line is not parseable as valid JSON.")
            continue
        if not isinstance(obj, dict):
            bad(n, "invalid_request", "Each line must be a JSON object.")
            continue
        cid = obj.get("custom_id")
        if not isinstance(cid, str) or not cid:
            bad(n, "missing_custom_id", "Missing required 'custom_id'.", "custom_id")
            continue
        if cid in seen:
            bad(n, "duplicate_custom_id", f"Duplicate custom_id: {cid}", "custom_id")
            continue
        seen.add(cid)
        if obj.get("method") != "POST":
            bad(n, "invalid_method", "The 'method' must be POST.", "method")
        if obj.get("url") != endpoint:
            bad(
                n,
                "invalid_url",
                f"The 'url' must match the batch endpoint {endpoint}.",
                "url",
            )
        body = obj.get("body")
        if not isinstance(body, dict):
            bad(n, "invalid_request", "The 'body' must be a JSON object.", "body")
        elif body.get("stream"):
            bad(
                n,
                "invalid_request",
                "Streaming is not supported in batches.",
                "body.stream",
            )
        lines.append(
            {"custom_id": cid, "method": "POST", "url": endpoint, "body": body}
        )
    if not lines and not errors:
        bad(0, "empty_file", "The input file is empty.")
    return lines, errors


@router.post("/batches")
async def create_batch(request: Request):
    if (r := auth(request, False)) is not None:
        return r
    body, err = await _json_body(request, False)
    if err:
        return err
    store = get_store()
    fid, endpoint = body.get("input_file_id"), body.get("endpoint")
    if not isinstance(fid, str) or not fid:
        return error_response(
            request,
            400,
            "'input_file_id' is required",
            "missing_required_parameter",
            "input_file_id",
            False,
        )
    if endpoint not in OPENAI_ENDPOINTS:
        return error_response(
            request,
            400,
            f"'endpoint' must be one of {', '.join(OPENAI_ENDPOINTS)}",
            "invalid_value",
            "endpoint",
            False,
        )
    if body.get("completion_window") != "24h":
        return error_response(
            request,
            400,
            "'completion_window' must be '24h'",
            "invalid_value",
            "completion_window",
            False,
        )
    meta_in = body.get("metadata")
    if meta_in is not None and (
        not isinstance(meta_in, dict)
        or len(meta_in) > 16
        or any(not isinstance(v, str) for v in meta_in.values())
    ):
        return error_response(
            request,
            400,
            "'metadata' must be an object of up to 16 string values",
            "invalid_value",
            "metadata",
            False,
        )
    try:
        info = store.get_meta(fid)
        if info["purpose"] != "batch":
            return error_response(
                request,
                400,
                f"File '{fid}' has purpose '{info['purpose']}', expected 'batch'",
                "invalid_value",
                "input_file_id",
                False,
            )
        raw = store.read(fid)
    except FileStoreError as exc:
        return error_response(
            request, 400, exc.message, exc.code, "input_file_id", False
        )
    lines, errors = _validate_openai_input(raw, endpoint)
    rec = _new_record("openai", "batch", request, len(lines))
    rec.update(endpoint=endpoint, input_file_id=info["id"], metadata=meta_in)
    if errors:
        rec.update(
            status="failed",
            failed_at=int(time.time()),
            errors={"object": "list", "data": errors},
        )
        store.save_batch(rec)
        return openai_batch(rec)
    _write_input(store, rec, lines)
    store.save_batch(rec)
    runner.ensure_started()
    return openai_batch(rec)


def _openai_get(store: FileStore, bid: str) -> dict | None:
    try:
        rec = store.load_batch(normalize_batch(bid))
    except FileStoreError:
        return None
    return rec if rec and rec["api"] == "openai" else None


def normalize_batch(bid: str) -> str:
    return bid.replace("batch-", "batch_", 1) if bid.startswith("batch-") else bid


@router.get("/batches/{batch_id}")
async def retrieve_batch(batch_id: str, request: Request):
    if (r := auth(request, False)) is not None:
        return r
    rec = _openai_get(get_store(), batch_id)
    if rec is None:
        return error_response(
            request,
            404,
            f"No batch found with id '{batch_id}'.",
            "batch_not_found",
            anthropic=False,
        )
    return openai_batch(rec)


@router.post("/batches/{batch_id}/cancel")
async def cancel_batch(batch_id: str, request: Request):
    if (r := auth(request, False)) is not None:
        return r
    store = get_store()
    rec = _openai_get(store, batch_id)
    if rec is None:
        return error_response(
            request,
            404,
            f"No batch found with id '{batch_id}'.",
            "batch_not_found",
            anthropic=False,
        )
    if rec["status"] in ("validating", "in_progress"):
        rec.update(status="cancelling", cancelling_at=int(time.time()))
        store.save_batch(rec)
        runner.ensure_started()
    elif rec["status"] != "cancelling":
        return error_response(
            request,
            409,
            f"Cannot cancel a batch with status '{rec['status']}'.",
            "invalid_state",
            anthropic=False,
        )
    return openai_batch(rec)


@router.get("/batches")
async def list_batches(request: Request):
    if (r := auth(request, False)) is not None:
        return r
    try:
        limit = int(request.query_params.get("limit", 20))
        if not 1 <= limit <= 100:
            raise ValueError
    except ValueError:
        return error_response(
            request,
            400,
            "'limit' must be an integer between 1 and 100",
            param="limit",
            anthropic=False,
        )
    after = request.query_params.get("after")
    page, more = paginate_ids(
        get_store().list_batches("openai"),
        limit,
        normalize_batch(after) if after else None,
    )
    data = [openai_batch(x) for x in page]
    return {
        "object": "list",
        "data": data,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
        "has_more": more,
    }


# ---------------------------------------------------------------------------
# Anthropic Message Batches
# ---------------------------------------------------------------------------


def anthropic_batch(rec: dict[str, Any], request: Request) -> dict[str, Any]:
    c = rec["counts"]
    ended = rec["status"] not in ACTIVE
    end_ts = next(
        (
            rec[k]
            for k in ("completed_at", "cancelled_at", "expired_at", "failed_at")
            if rec.get(k)
        ),
        None,
    )
    processing = 0 if ended else max(0, rec["total"] - sum(c.values()))
    return {
        "id": rec["id"],
        "type": "message_batch",
        "processing_status": "ended"
        if ended
        else ("canceling" if rec["status"] == "cancelling" else "in_progress"),
        "request_counts": {
            "processing": processing,
            "succeeded": c["succeeded"],
            "errored": c["errored"],
            "canceled": c["canceled"],
            "expired": c["expired"],
        },
        "ended_at": iso(end_ts) if ended and end_ts else None,
        "created_at": iso(rec["created_at"]),
        "expires_at": iso(rec["expires_at"]),
        "cancel_initiated_at": iso(rec["cancelling_at"])
        if rec.get("cancelling_at")
        else None,
        "results_url": f"{_base(request)}/v1/messages/batches/{rec['id']}/results"
        if ended
        else None,
    }


def _anth_get(store: FileStore, bid: str) -> dict | None:
    try:
        rec = store.load_batch(bid)
    except FileStoreError:
        return None
    return rec if rec and rec["api"] == "anthropic" else None


def _aerr(request: Request, status: int, message: str) -> JSONResponse:
    return error_response(request, status, message, anthropic=True)


@router.post("/messages/batches")
async def create_message_batch(request: Request):
    if (r := auth(request, True)) is not None:
        return r
    body, err = await _json_body(request, True)
    if err:
        return err
    reqs = body.get("requests")
    if not isinstance(reqs, list) or not reqs:
        return _aerr(request, 400, "requests: must be a non-empty list")
    if len(reqs) > MAX_ANTHROPIC_REQUESTS:
        return _aerr(
            request,
            413,
            f"requests: a batch may hold at most {MAX_ANTHROPIC_REQUESTS} requests",
        )
    lines, seen = [], set()
    for i, item in enumerate(reqs):
        cid = item.get("custom_id") if isinstance(item, dict) else None
        params = item.get("params") if isinstance(item, dict) else None
        if not isinstance(cid, str) or not _CUSTOM_ID_RE.match(cid):
            return _aerr(
                request,
                400,
                f"requests.{i}.custom_id: must match ^[a-zA-Z0-9_-]{{1,64}}$",
            )
        if cid in seen:
            return _aerr(
                request, 400, f"requests.{i}.custom_id: duplicate custom_id '{cid}'"
            )
        seen.add(cid)
        if not isinstance(params, dict):
            return _aerr(request, 400, f"requests.{i}.params: Field required")
        for key in ("model", "max_tokens", "messages"):
            if key not in params:
                return _aerr(request, 400, f"requests.{i}.params.{key}: Field required")
        lines.append(
            {"custom_id": cid, "method": "POST", "url": "/v1/messages", "body": params}
        )
    store = get_store()
    rec = _new_record("anthropic", "msgbatch", request, len(lines))
    rec["status"] = "in_progress"
    _write_input(store, rec, lines)
    store.save_batch(rec)
    runner.ensure_started()
    return anthropic_batch(rec, request)


@router.get("/messages/batches")
async def list_message_batches(request: Request):
    if (r := auth(request, True)) is not None:
        return r
    q = request.query_params
    try:
        limit = int(q.get("limit", 20))
        if not 1 <= limit <= 1000:
            raise ValueError
    except ValueError:
        return _aerr(request, 400, "limit: must be an integer between 1 and 1000")
    page, more = paginate_ids(
        get_store().list_batches("anthropic"),
        limit,
        q.get("after_id"),
        q.get("before_id"),
    )
    data = [anthropic_batch(x, request) for x in page]
    return {
        "data": data,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
        "has_more": more,
    }


@router.get("/messages/batches/{batch_id}")
async def retrieve_message_batch(batch_id: str, request: Request):
    if (r := auth(request, True)) is not None:
        return r
    rec = _anth_get(get_store(), batch_id)
    if rec is None:
        return _aerr(request, 404, f"Message batch {batch_id} not found")
    return anthropic_batch(rec, request)


@router.post("/messages/batches/{batch_id}/cancel")
async def cancel_message_batch(batch_id: str, request: Request):
    if (r := auth(request, True)) is not None:
        return r
    store = get_store()
    rec = _anth_get(store, batch_id)
    if rec is None:
        return _aerr(request, 404, f"Message batch {batch_id} not found")
    if rec["status"] == "in_progress" or rec["status"] == "validating":
        rec.update(status="cancelling", cancelling_at=int(time.time()))
        store.save_batch(rec)
        runner.ensure_started()
    elif rec["status"] != "cancelling":
        return _aerr(request, 400, "Cannot cancel a batch that has already ended")
    return anthropic_batch(rec, request)


@router.delete("/messages/batches/{batch_id}")
async def delete_message_batch(batch_id: str, request: Request):
    if (r := auth(request, True)) is not None:
        return r
    store = get_store()
    rec = _anth_get(store, batch_id)
    if rec is None:
        return _aerr(request, 404, f"Message batch {batch_id} not found")
    if rec["status"] in ACTIVE:
        return _aerr(
            request,
            400,
            "Cannot delete a batch that is still processing; cancel it first",
        )
    store.delete_batch(rec["id"])
    return {"id": rec["id"], "type": "message_batch_deleted"}


@router.get("/messages/batches/{batch_id}/results")
async def message_batch_results(batch_id: str, request: Request):
    if (r := auth(request, True)) is not None:
        return r
    store = get_store()
    rec = _anth_get(store, batch_id)
    if rec is None:
        return _aerr(request, 404, f"Message batch {batch_id} not found")
    if rec["status"] in ACTIVE:
        return _aerr(
            request, 409, "Results are available only after the batch has ended"
        )
    p = store.batch_file(rec["id"], "out")
    if not p.exists():
        p.write_bytes(b"")
    return FileResponse(p, media_type="application/x-jsonl")
