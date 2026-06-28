from __future__ import annotations

"""Batch inference endpoint — process multiple requests efficiently.

Leverages Yunshu's continuous batching to handle multiple prompts
in a single batch with optimal GPU utilization.

Features:
- Configurable concurrency and batch timeout
- Progress tracking via in-memory store + status/results endpoints
- Partial success: individual item failures don't abort the batch
- CSV upload for bulk prompts and CSV download of results
- File size limit on CSV uploads
"""

import asyncio
import csv
import io
import logging
import os
import threading
import time
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

import contextlib

from yunshu_control.audit_log import log_operation, resolve_actor

router = APIRouter(tags=["batch"])

# Configurable limits
_BATCH_MAX_ITEMS = int(os.environ.get("YUNSHU_BATCH_MAX_ITEMS", "500"))
_BATCH_DEFAULT_TIMEOUT = float(os.environ.get("YUNSHU_BATCH_TIMEOUT", "300"))
_BATCH_MAX_CSV_SIZE = int(os.environ.get("YUNSHU_BATCH_MAX_CSV_SIZE", str(50 * 1024 * 1024)))  # 50 MB
_BATCH_STORE_TTL = int(os.environ.get("YUNSHU_BATCH_STORE_TTL", "3600"))  # 1 hour default
_BATCH_STORE_MAX_SIZE = int(os.environ.get("YUNSHU_BATCH_STORE_MAX_SIZE", "1000"))

# In-memory progress tracking
_batch_store: dict[str, dict] = {}
_batch_store_lock = threading.Lock()


def _csv_safe(cell):
    """Neutralize CSV formula injection. A cell whose value starts with a formula
    trigger (= + - @) or a leading tab/CR runs as a formula when the CSV is opened in a
    spreadsheet; prefix it with a single quote (OWASP neutralization). Non-strings pass
    through unchanged."""
    if isinstance(cell, str) and cell[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + cell
    return cell


def _mark_batch_terminal(batch_id: str, status: str) -> None:
    """Force a batch entry to a terminal status (under lock, .get()-guarded), so it
    can never be stuck at 'in_progress' (which _cleanup_batch_store refuses to evict)."""
    with _batch_store_lock:
        _info = _batch_store.get(batch_id)
        if _info is not None and _info.get("status") == "in_progress":
            _info["status"] = status
            _info.setdefault("finished_at", time.time())


def _cleanup_batch_store() -> None:
    """Remove expired batch entries and enforce max size.

    Called after each batch creation. TTL-based expiry prevents unbounded
    memory growth from accumulated batch results.
    """
    now = time.time()
    # a hard age cap for in_progress ZOMBIES. A batch's own timeout is ≤ the store
    # TTL, so an in_progress entry older than 2*TTL cannot still be legitimately running — it
    # was orphaned (e.g. the runner died before writing a terminal status). Without this,
    # such a zombie is never evictable (the in_progress guard below) and accumulates → the
    # store grows unbounded (the HIGH). Belt-and-suspenders to _mark_batch_terminal.
    _zombie_cutoff = _BATCH_STORE_TTL * 2
    with _batch_store_lock:
        _zombies = [
            bid for bid, info in _batch_store.items()
            if info.get("status") == "in_progress"
            and (now - info.get("started_at", 0)) > _zombie_cutoff
        ]
        for bid in _zombies:
            del _batch_store[bid]
        # NEVER evict a batch that is still running. This cleanup runs
        # at the start of every create_batch; a batch's own timeout can be as long as the
        # store TTL, so a concurrently-created batch could TTL/size-evict a still-running
        # batch's entry — then the running batch's unguarded _batch_store[batch_id][...]
        # writes (progress + final update) raise KeyError, silently dropping items or
        # 500-ing the whole batch after generation already succeeded.
        expired = [
            bid for bid, info in _batch_store.items()
            if info.get("status") != "in_progress"
            and (now - info.get("started_at", 0)) > _BATCH_STORE_TTL
        ]
        for bid in expired:
            del _batch_store[bid]

        # Enforce max size — evict oldest FINISHED batch first (never an in-progress one)
        while len(_batch_store) > _BATCH_STORE_MAX_SIZE:
            _evictable = [k for k, v in _batch_store.items() if v.get("status") != "in_progress"]
            if not _evictable:
                break  # all in-flight — let the store grow rather than corrupt a live batch
            oldest_id = min(_evictable, key=lambda k: _batch_store[k].get("started_at", 0))
            del _batch_store[oldest_id]


class BatchItem(BaseModel):
    custom_id: str
    method: str = "POST"
    url: str = "/v1/chat/completions"
    body: dict


class BatchRequest(BaseModel):
    requests: list[BatchItem]
    max_concurrent: int = Field(default=4, ge=1, le=64)
    timeout: float = Field(
        default=_BATCH_DEFAULT_TIMEOUT,
        ge=1.0,
        le=3600.0,
        description="Seconds before the entire batch is aborted",
    )
    model: str | None = Field(
        default=None,
        description=(
            "Optional default model id propagated to every item whose body "
            "does not specify its own 'model' field."
        ),
    )

    @model_validator(mode="after")
    def _propagate_default_model(self):
        """Fill in body.model for any item missing it from the top-level default."""
        if self.model:
            for item in self.requests:
                if not item.body.get("model"):
                    item.body["model"] = self.model
        return self


class BatchResponse(BaseModel):
    id: str
    object: str = "batch"
    status: str
    results: list[dict]


@router.post("/batch", response_model=None)
async def create_batch(req: BatchRequest, request: Request):
    """Execute a batch of inference requests concurrently.

    Supports configurable concurrency, batch timeout, and partial success.
    Progress is tracked in real-time so /batch/{id}/status reflects
    completed items even while the batch is running.
    """
    if not req.requests:
        raise HTTPException(status_code=400, detail="Batch cannot be empty")

    from .models import _check_model_access, _check_permission
    _check_permission(request, "can_infer")
    # Enforce per-key model isolation on EVERY model the
    # batch touches. create_batch only checked can_infer, so a key scoped to model A
    # could run inference/embeddings on any loaded model B via a batch item's
    # body["model"]. Every other model-serving route calls _check_model_access; the
    # batch wrapper (incl. the CSV-upload path that funnels here) skipped it.
    for _item in req.requests:
        _check_model_access(request, _item.body.get("model"))

    if len(req.requests) > _BATCH_MAX_ITEMS:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(req.requests)} exceeds limit of {_BATCH_MAX_ITEMS} (set YUNSHU_BATCH_MAX_ITEMS env var)",
        )

    batch_id = f"batch_{uuid.uuid4().hex[:24]}"
    total = len(req.requests)
    actor = resolve_actor(request)
    log_operation("batch_create", batch_id, "started", actor=actor, total=total)

    # Initialize progress tracking — counters are computed lazily from
    # per-item results to avoid race conditions with concurrent tasks.
    started_at = time.time()
    with _batch_store_lock:
        _batch_store[batch_id] = {
            "id": batch_id,
            "status": "in_progress",
            "total": total,
            "started_at": started_at,
            "results": [None] * total,
            "owner": actor,   # per-key ownership for status/results/csv IDOR
        }

    semaphore = asyncio.Semaphore(req.max_concurrent)
    processed = [None] * total

    async def _process_item(index: int, item: BatchItem) -> tuple[int, dict, str]:
        try:
            async with semaphore:
                result = await _execute_batch_item(item)
            status = "success"
        except Exception as exc:
            result = {"error": str(exc)}
            status = "error"

        # Store per-item result — counters are derived on demand in status()
        processed[index] = {
            "custom_id": req.requests[index].custom_id,
            "status": status,
            "response": result if status == "success" else None,
            "error": result.get("error") if status == "error" else None,
        }
        # Also update the shared store so status endpoint reflects progress.
        # Guard with .get(): the in-progress eviction guard should keep this entry
        # alive, but never let a store miss turn into a lost item / task crash.
        with _batch_store_lock:
            _info = _batch_store.get(batch_id)
            if _info is not None:
                _info["results"][index] = processed[index]

        return index, result, status

    tasks = [_process_item(i, item) for i, item in enumerate(req.requests)]
    task_handles = [asyncio.create_task(t) for t in tasks]

    try:
        done, pending = await asyncio.wait(task_handles, timeout=req.timeout)
    except asyncio.CancelledError:
        # Request cancelled (e.g., client disconnect) — propagate CancelledError.
        # LEAK fix: was just `task.cancel()` without await — cancellation
        # never completed, tasks kept running underlying generate() calls and
        # held GPU resources. Now gather to ensure cancellation lands.
        # Set the terminal status BEFORE the gather. The gather is itself a
        # cancellation point — under a re-delivered CancelledError (normal on client
        # disconnect) it would propagate out before the old post-gather status write ran,
        # leaving the entry stuck at "in_progress" FOREVER. _cleanup_batch_store then refuses
        # to ever evict an in_progress entry → the store grows unbounded (zombie leak).
        _mark_batch_terminal(batch_id, "cancelled")
        for task in task_handles:
            task.cancel()
        await asyncio.gather(*task_handles, return_exceptions=True)
        raise
    except TimeoutError as exc:
        # fix: timeout → 408 not 500 (matches OpenAI Batch contract).
        _mark_batch_terminal(batch_id, "timeout")  # before the gather (see above)
        for task in task_handles:
            task.cancel()
        await asyncio.gather(*task_handles, return_exceptions=True)
        raise HTTPException(status_code=408, detail=f"Batch timed out: {exc}") from None
    except Exception as exc:
        # LEAK fix: same gather-after-cancel pattern.
        _mark_batch_terminal(batch_id, "error")  # before the gather (see above)
        for task in task_handles:
            task.cancel()
        await asyncio.gather(*task_handles, return_exceptions=True)
        raise HTTPException(status_code=500, detail=f"Batch execution failed: {exc}") from None

    # Cancel pending (timed-out) tasks
    for task in pending:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    timed_out = len(pending)

    # Collect results from completed tasks
    # (processed[] is already filled by _process_item callbacks)
    for task in done:
        # consume any CancelledError / result exception
        with contextlib.suppress(asyncio.CancelledError, Exception):
            task.result()

    # Handle timed-out items
    for i in range(total):
        if processed[i] is None:
            processed[i] = {
                "custom_id": req.requests[i].custom_id,
                "status": "error",
                "error": "Timed out",
            }

    succeeded = sum(1 for p in processed if p.get("status") == "success")
    failed = sum(1 for p in processed if p.get("status") == "error")
    final_status = "completed" if failed == 0 else ("partial" if failed < total else "failed")

    finished_at = time.time()
    # Normalize to the OpenAI Batch status enum (: "partial"/"timeout"/"error"
    # are not valid OpenAI statuses — a batch with per-item failures is
    # "completed"; per-item errors surface via request_counts.failed).
    _oai_status = {"partial": "completed", "timeout": "expired", "error": "failed"}.get(
        final_status, final_status
    )
    # asyncio.wait() RETURNS (done, pending) on timeout — it never raises
    # TimeoutError, so the except TimeoutError branch above is dead and the "timeout"→
    # "expired" mapping was unreachable. Timed-out items were marked per-item errors and
    # folded into failed/partial, so a window-expired batch reported "completed"/"failed"
    # but never "expired". Derive expired directly from the pending count.
    if timed_out > 0:
        _oai_status = "expired"
    _ep = "/v1/chat/completions"
    try:
        if getattr(req, "requests", None):
            _ep = req.requests[0].url or _ep
    except Exception:
        pass
    response_data = {
        # OpenAI Batch object — canonical fields
        "id": batch_id,
        "object": "batch",
        "endpoint": _ep,
        "input_file_id": None,
        "completion_window": "24h",
        "status": _oai_status,
        "output_file_id": None,
        "error_file_id": None,
        "errors": None,
        "request_counts": {
            "total": total,
            "completed": succeeded,
            "failed": failed,
        },
        "created_at": int(started_at),
        "in_progress_at": int(started_at),
        "completed_at": int(finished_at) if _oai_status == "completed" else None,
        "failed_at": int(finished_at) if _oai_status == "failed" else None,
        "expired_at": int(finished_at) if _oai_status == "expired" else None,
        "metadata": None,
        # Yunshu-specific extras (clients ignore unknown keys)
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "timed_out": timed_out,
        "elapsed_s": round(finished_at - started_at, 2),
        "results": processed,
    }

    with _batch_store_lock:
        # this is the final write; the in-progress eviction guard keeps the
        # entry alive until here, but use .get()+setdefault so a raced/absent entry can
        # never 500 the request after generation already succeeded.
        _final = _batch_store.get(batch_id)
        if _final is None:
            _final = {"id": batch_id, "started_at": started_at, "owner": actor}
            _batch_store[batch_id] = _final
        _final.update({
            # store the OpenAI-normalized status (was the raw final_status), so a
            # later GET /batch/{id}/status returns the SAME status this POST returned —
            # previously POST said "completed" but the store kept "partial", so a poll
            # surfaced a non-OpenAI status the client never saw on create.
            "status": _oai_status,
            "detail_status": final_status,  # Yunshu-specific (partial/timeout/error) detail
            "timed_out": timed_out,
            "results": processed,
            "finished_at": time.time(),
        })

    # Evict expired/oversized batch entries to prevent memory leak
    _cleanup_batch_store()

    log_operation(
        "batch_complete", batch_id, "success" if final_status != "failed" else "failure",
        actor=actor, status=final_status, succeeded=succeeded, failed=failed,
    )
    return JSONResponse(response_data)


def _owns_batch(info: dict, request: Request) -> bool:
    """Per-key ownership for batch handles (IDOR fix). Permissive when
    no owner was stamped; otherwise the caller must match the creator. Prevents
    one tenant from reading another tenant's batch results/content by guessing
    the batch_id."""
    owner = info.get("owner")
    if not owner or owner == "anonymous":
        return True
    return resolve_actor(request) == owner


@router.get("/batch/{batch_id}/status")
async def get_batch_status(batch_id: str, request: Request):
    """Get progress/status of a previously submitted batch."""
    from .models import _check_permission
    _check_permission(request, "can_infer")
    with _batch_store_lock:
        info = _batch_store.get(batch_id)
        if info is None or not _owns_batch(info, request):
            raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found")
        # Snapshot results under lock to avoid partial reads
        results = list(info.get("results", []))
        status = info["status"]
        started_at = info["started_at"]
        batch_id_snap = info["id"]
        total = info["total"]
        finished_at = info.get("finished_at")
    # Derive counters from per-item results to avoid race conditions
    completed = sum(1 for r in results if r is not None)
    succeeded = sum(1 for r in results if r is not None and r.get("status") == "success")
    failed = sum(1 for r in results if r is not None and r.get("status") == "error")
    summary = {
        "id": batch_id_snap,
        "status": status,
        "total": total,
        "completed": completed,
        "succeeded": succeeded,
        "failed": failed,
    }
    if finished_at is not None:
        summary["elapsed_s"] = round(finished_at - started_at, 2)
    else:
        summary["elapsed_s"] = round(time.time() - started_at, 2)
    return JSONResponse(summary)


@router.get("/batch/{batch_id}/results")
async def get_batch_results(batch_id: str, request: Request):
    """Get full results of a completed batch."""
    from .models import _check_permission
    _check_permission(request, "can_infer")
    with _batch_store_lock:
        info = _batch_store.get(batch_id)
        if info is None or not _owns_batch(info, request):
            raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found")
        if info["status"] == "in_progress":
            raise HTTPException(status_code=409, detail="Batch is still in progress")
        # Snapshot under lock
        results = list(info.get("results", []))
        batch_id_snap = info["id"]
        status = info["status"]
        total = info["total"]
    # Derive counters from per-item results
    succeeded = sum(1 for r in results if r is not None and r.get("status") == "success")
    failed = sum(1 for r in results if r is not None and r.get("status") == "error")
    return JSONResponse({
        "id": batch_id_snap,
        "status": status,
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "results": results,
    })


@router.get("/batch/{batch_id}/results.csv")
async def download_batch_csv(batch_id: str, request: Request):
    """Download batch results as CSV file."""
    from .models import _check_permission
    _check_permission(request, "can_infer")
    with _batch_store_lock:
        info = _batch_store.get(batch_id)
        if info is None or not _owns_batch(info, request):
            raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found")
        if info["status"] == "in_progress":
            raise HTTPException(status_code=409, detail="Batch is still in progress")
        # Snapshot results under lock
        results = list(info.get("results", []))

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["custom_id", "status", "finish_reason", "content", "error",
                      "prompt_tokens", "completion_tokens", "total_tokens"])

    for r in results:
        if r is None:
            continue
        row = [r.get("custom_id", ""), r.get("status", "")]
        resp = r.get("response")
        if resp:
            choices = resp.get("choices", [])
            if choices:
                row.append(choices[0].get("finish_reason", ""))
                msg = choices[0].get("message", {})
                row.append(msg.get("content", ""))
            else:
                row.extend(["", ""])
            # Header order: ..., content, error, prompt_tokens, completion_tokens, total_tokens
            row.append("")  # no error
            usage = resp.get("usage", {})
            row.extend([
                usage.get("prompt_tokens", ""),
                usage.get("completion_tokens", ""),
                usage.get("total_tokens", ""),
            ])
        else:
            row.extend(["", ""])  # finish_reason, content
            row.append(r.get("error", ""))
            row.extend(["", "", ""])  # prompt_tokens, completion_tokens, total_tokens
        # CSV formula-injection hardening. Cells carry prompt-controlled model
        # output (content/error/custom_id); a value starting with = + - @ (or a leading
        # tab/CR) executes as a formula when the download is opened in Excel/Sheets. Prefix
        # such cells with a single quote (the OWASP-recommended neutralization).
        writer.writerow([_csv_safe(c) for c in row])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=batch_{batch_id}.csv"},
    )


@router.post("/batch/upload/csv", response_model=None)
async def upload_batch_csv(
    request: Request,
    file: UploadFile = File(...),
    # FastAPI requires explicit `Form(...)` for multipart-form fields next
    # to an UploadFile — without it the field defaults to "default" and the
    # caller's `-F "model=..."` is silently dropped, causing per-row
    # "Model 'default' not available" errors.
    model: str = Form("default"),
    max_tokens: int = Form(512),
    max_concurrent: int = Form(4),
):
    """Upload a CSV file with prompts and execute as a batch.

    Expected CSV columns: custom_id, prompt (or messages_json)
    Optional columns: max_tokens, temperature, system_prompt
    """
    from .models import _check_permission
    _check_permission(request, "can_infer")
    if not model or not model.strip():
        raise HTTPException(status_code=400, detail="model parameter is required and cannot be empty")

    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise HTTPException(status_code=400, detail="max_tokens must be a positive integer")
    max_tokens = min(max_tokens, 131072)

    # enforce the SAME upper bound the BatchRequest schema does (le=64). Without
    # it, a form value like max_concurrent=1000 passed validation here but raised a Pydantic
    # ValidationError later inside create_batch (past the FastAPI boundary → uncaught → 500
    # instead of a clean 400).
    if not isinstance(max_concurrent, int) or max_concurrent < 1 or max_concurrent > 64:
        raise HTTPException(status_code=400, detail="max_concurrent must be an integer in [1, 64]")

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    content = await file.read()
    if len(content) > _BATCH_MAX_CSV_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"CSV file too large: {len(content)} bytes exceeds limit of {_BATCH_MAX_CSV_SIZE} bytes (set YUNSHU_BATCH_MAX_CSV_SIZE env var)",
        )
    # a non-UTF-8 upload raised an uncaught UnicodeDecodeError → 500. Return a
    # clean 400 for the malformed-input case instead.
    try:
        text_content = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="CSV file must be UTF-8 encoded") from None
    # restval="" — a RAGGED/short row (fewer fields than the header) makes
    # DictReader fill the missing columns with None (its default restval), and the code
    # below does row.get(col, "").strip() → None.strip() → AttributeError, which the
    # (ValueError, TypeError) guards do NOT catch → an uncaught 500 that aborts the WHOLE
    # upload on a realistic CSV mistake (a trailing optional column omitted on some rows).
    # restval="" makes every missing cell an empty string, so .strip() is always safe and a
    # short row defaults its cells instead of crashing. (Same W966 class — non-UTF8/
    # max_concurrent already hardened; this ragged-row case was missed.)
    reader = csv.DictReader(io.StringIO(text_content), restval="")

    items: list[BatchItem] = []
    for row in reader:
        # `or` (not the .get default) so a missing OR empty custom_id gets a generated id
        # (with restval="" the key is present-but-empty, so a .get default would never fire).
        custom_id = row.get("custom_id") or str(uuid.uuid4().hex[:8])
        prompt = row.get("prompt", "")
        messages_json = row.get("messages_json", "")
        system_prompt = row.get("system_prompt", "")
        row_max_tokens_raw = row.get("max_tokens", "")
        try:
            row_max_tokens = int(row_max_tokens_raw) if row_max_tokens_raw.strip() else max_tokens
        except (ValueError, TypeError):
            row_max_tokens = max_tokens
        if row_max_tokens < 1:
            row_max_tokens = max_tokens
        row_max_tokens = min(row_max_tokens, 131072)
        row_temp_raw = row.get("temperature", "")
        try:
            row_temp = float(row_temp_raw) if row_temp_raw.strip() else 0.7
        except (ValueError, TypeError):
            row_temp = 0.7
        if row_temp < 0:
            row_temp = 0.7

        if messages_json:
            import json
            try:
                messages = json.loads(messages_json)
            except json.JSONDecodeError:
                messages = [{"role": "user", "content": prompt}]
        elif prompt:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
        else:
            continue

        items.append(BatchItem(
            custom_id=custom_id,
            url="/v1/chat/completions",
            body={
                "model": model,
                "messages": messages,
                "max_tokens": row_max_tokens,
                "temperature": row_temp,
            },
        ))

    if not items:
        raise HTTPException(status_code=400, detail="CSV contained no valid prompts")

    batch_req = BatchRequest(requests=items, max_concurrent=max_concurrent)
    return await create_batch(batch_req, request)


async def _execute_batch_item(item: BatchItem) -> dict:
    """Execute a single batch item."""
    body = item.body
    url = item.url

    if url == "/v1/chat/completions":
        return await _execute_chat_completion(body)
    elif url == "/v1/completions":
        return await _execute_completion(body)
    elif url == "/v1/embeddings":
        return await _execute_embedding(body)
    else:
        raise ValueError(f"Unsupported batch URL: {url}")


def _validate_logit_bias(body: dict) -> None:
    """The batch surface was the ONE generation router with
    NO logit_bias value check (chat/completions/responses/anthropic all validate).
    A NaN/Inf/out-of-range bias → all-NaN softmax → garbage output for that item.
    Raise ValueError (the batch per-item error contract) on a bad value so the
    offending row fails cleanly instead of silently producing junk."""
    lb = body.get("logit_bias")
    if not lb:
        return
    import math
    if not isinstance(lb, dict):
        raise ValueError("logit_bias: must be an object mapping token ids to biases")
    for k, v in lb.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            raise ValueError(f"logit_bias[{k}]: must be a finite number")
        if v < -100.0 or v > 100.0:
            raise ValueError(f"logit_bias[{k}]={v}: must be between -100 and 100")


def _validate_batch_sampling(body: dict) -> None:
    """The batch item `body` is a raw unvalidated dict, so every sampling param
    EXCEPT max_tokens/logit_bias was forwarded to the engine with NO range check — bypassing
    the Field(ge/le) bounds the chat/completions endpoints enforce. An out-of-range value
    (e.g. presence_penalty 1e18, min_p 5.0) produces degenerate/garbage output, and
    temperature<0 is silently coerced to greedy (a no-op, not the OpenAI 422). Validate to
    the SAME bounds the request models declare; raise ValueError (the batch per-item error
    contract) so a bad row fails cleanly instead of silently emitting junk."""
    import math

    def _rng(name: str, lo: float, hi: float, default: float) -> None:
        v = body.get(name, default)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or math.isnan(v) or math.isinf(v):
            raise ValueError(f"{name}: must be a finite number")
        if v < lo or v > hi:
            raise ValueError(f"{name}: must be in [{lo}, {hi}], got {v}")

    _rng("temperature", 0.0, 2.0, 0.7)
    _rng("top_p", 0.0, 1.0, 1.0)
    _rng("min_p", 0.0, 1.0, 0.0)
    _rng("repetition_penalty", 0.0, 2.0, 1.0)
    _rng("frequency_penalty", -2.0, 2.0, 0.0)
    _rng("presence_penalty", -2.0, 2.0, 0.0)
    _rng("xtc_probability", 0.0, 1.0, 0.0)
    _rng("xtc_threshold", 0.0, 0.5, 0.0)  # engine hard-requires [0, 0.5]
    _tk = body.get("top_k", 0)
    if not isinstance(_tk, int) or isinstance(_tk, bool) or _tk < 0:
        raise ValueError("top_k: must be a non-negative integer")
    _n = body.get("n", 1)
    if not isinstance(_n, int) or isinstance(_n, bool) or _n < 1:
        raise ValueError("n: must be a positive integer")
    _seed = body.get("seed")
    if _seed is not None and (not isinstance(_seed, int) or isinstance(_seed, bool)
                              or _seed < -(2**63) or _seed >= 2**63):
        raise ValueError("seed: must be within the 64-bit signed integer range")


async def _execute_chat_completion(body: dict) -> dict:
    """Execute a chat completion request."""
    from yunshu_engine.batched_engine import BatchedEngine

    from ..engine import get_engine, get_engine_for_model

    model = body.get("model", "")
    if not model or not model.strip():
        raise ValueError("model: field is required and cannot be empty")

    messages = body.get("messages", [])
    if not messages:
        raise ValueError("messages: field is required and cannot be empty")

    max_tokens = body.get("max_tokens", 512)
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens: must be a positive integer")
    if max_tokens > 131072:
        raise ValueError("max_tokens: must not exceed 131072")

    _validate_logit_bias(body)
    _validate_batch_sampling(body)

    temperature = body.get("temperature", 0.7)
    top_p = body.get("top_p", 1.0)
    stop = body.get("stop")

    engine = get_engine()
    is_batched = isinstance(engine, BatchedEngine) if engine else False

    if engine is None or not engine.is_loaded or not engine.resolve_model_id(model):
        try:
            engine = await get_engine_for_model(model)
            is_batched = isinstance(engine, BatchedEngine)
        except (KeyError, Exception):
            raise ValueError(f"Model '{model}' not available") from None

    if is_batched:
        result = await engine.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=body.get("top_k", 0),
            min_p=body.get("min_p", 0.0),
            seed=body.get("seed"),
            stop=stop,
            stop_token_ids=body.get("stop_token_ids"),
            enable_thinking=body.get("enable_thinking"),
            thinking_budget=body.get("thinking_budget"),
            repetition_penalty=body.get("repetition_penalty", 1.0),
            frequency_penalty=body.get("frequency_penalty", 0.0),
            presence_penalty=body.get("presence_penalty", 0.0),
            logit_bias=body.get("logit_bias"),
            priority=body.get("priority", 0),
            reasoning_effort=body.get("reasoning_effort"),
            spec_decode=body.get("spec_decode", False),
            xtc_probability=body.get("xtc_probability", 0.0),
            xtc_threshold=body.get("xtc_threshold", 0.0),
            logprobs=body.get("logprobs", False),
            top_logprobs=body.get("top_logprobs"),
            logits_processors=body.get("logits_processors"),
        )
        text = result.text
        prompt_tokens = result.prompt_tokens
        completion_tokens = result.completion_tokens
        finish_reason = result.finish_reason or "stop"
    else:
        # Non-batched engine: apply chat template to convert messages to string
        if isinstance(messages, list) and messages:
            tokenizer = getattr(engine, '_tokenizer', None)
            if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            else:
                # Last resort: concatenate message content
                prompt_text = "\n".join(
                    m.get("content", str(m)) for m in messages
                )
        else:
            prompt_text = str(messages)
        state = await engine.generate(
            prompt=prompt_text,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=body.get("top_k", 0),
            min_p=body.get("min_p", 0.0),
            seed=body.get("seed"),
            stop=stop,
            stop_token_ids=body.get("stop_token_ids"),
            enable_thinking=body.get("enable_thinking"),
            thinking_budget=body.get("thinking_budget"),
            repetition_penalty=body.get("repetition_penalty", 1.0),
            frequency_penalty=body.get("frequency_penalty", 0.0),
            presence_penalty=body.get("presence_penalty", 0.0),
            logit_bias=body.get("logit_bias"),
            priority=body.get("priority", 0),
            reasoning_effort=body.get("reasoning_effort"),
            spec_decode=body.get("spec_decode", False),
            xtc_probability=body.get("xtc_probability", 0.0),
            xtc_threshold=body.get("xtc_threshold", 0.0),
            logprobs=body.get("logprobs", False),
            top_logprobs=body.get("top_logprobs"),
            logits_processors=body.get("logits_processors"),
        )
        if isinstance(state, dict):
            text = state.get("text", "")
            prompt_tokens = state.get("prompt_tokens", 0)
            completion_tokens = state.get("completion_tokens", 0)
            finish_reason = state.get("finish_reason", "stop") or "stop"
        else:
            # CRITICAL fix: GenerationOutput uses text/prompt_tokens/
            # completion_tokens (batched_engine.py:84-87); prior code used
            # non-existent generated_text/prompt_token_count/
            # completion_token_count → AttributeError on every batch item.
            text = getattr(state, "text", None) or getattr(state, "generated_text", "")
            prompt_tokens = getattr(state, "prompt_tokens", None) or getattr(state, "prompt_token_count", 0)
            completion_tokens = getattr(state, "completion_tokens", None) or getattr(state, "completion_token_count", 0)
            finish_reason = getattr(state, "finish_reason", None) or "stop"

    import time as _time
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(_time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def _execute_completion(body: dict) -> dict:
    """Execute a text completion request."""
    from yunshu_engine.batched_engine import BatchedEngine

    from ..engine import get_engine, get_engine_for_model

    model = body.get("model", "")
    if not model or not model.strip():
        raise ValueError("model: field is required and cannot be empty")

    prompt = body.get("prompt", "")
    if isinstance(prompt, str) and not prompt.strip():
        raise ValueError("prompt: field is required and cannot be empty")
    if isinstance(prompt, list) and not prompt:
        raise ValueError("prompt: field is required and cannot be empty")

    max_tokens = body.get("max_tokens", 128)
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens: must be a positive integer")

    _validate_logit_bias(body)
    _validate_batch_sampling(body)

    temperature = body.get("temperature", 0.7)

    engine = get_engine()
    if engine is None or not engine.is_loaded or not engine.resolve_model_id(model):
        try:
            engine = await get_engine_for_model(model)
        except (KeyError, Exception):
            raise ValueError(f"Model '{model}' not available") from None

    # The HTTP /v1/completions route decodes token-id prompts
    # (list[int] / list[list[int]]) to text via _normalize_prompts, but this batch
    # executor passed body["prompt"] straight to engine.generate(). A token-id array
    # then hit _generate_fast's `else: text = str(prompt)` and was tokenized as the
    # LITERAL string "[785, 3489]" → silent garbage output. Decode token-id forms here.
    if isinstance(prompt, list) and prompt and not all(isinstance(p, str) for p in prompt):
        from .completions import _normalize_prompts
        _tok = getattr(engine, "_tokenizer", None)
        _norm = _normalize_prompts(prompt, _tok)
        prompt = _norm[0] if len(_norm) == 1 else "\n".join(_norm)

    is_batched = isinstance(engine, BatchedEngine)
    if is_batched:
        result = await engine.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=body.get("top_p", 1.0),
            top_k=body.get("top_k", 0),
            min_p=body.get("min_p", 0.0),
            seed=body.get("seed"),
            repetition_penalty=body.get("repetition_penalty", 1.0),
            frequency_penalty=body.get("frequency_penalty", 0.0),
            presence_penalty=body.get("presence_penalty", 0.0),
            stop=body.get("stop"),
            stop_token_ids=body.get("stop_token_ids"),
            logit_bias=body.get("logit_bias"),
            enable_thinking=body.get("enable_thinking"),
            thinking_budget=body.get("thinking_budget"),
            reasoning_effort=body.get("reasoning_effort"),
            spec_decode=body.get("spec_decode", False),
            xtc_probability=body.get("xtc_probability", 0.0),
            xtc_threshold=body.get("xtc_threshold", 0.0),
            logprobs=body.get("logprobs", False),
            top_logprobs=body.get("top_logprobs"),
            logits_processors=body.get("logits_processors"),
            priority=body.get("priority", 0),
        )
        text = result.text
        prompt_tokens = result.prompt_tokens
        completion_tokens = result.completion_tokens
        finish_reason = result.finish_reason or "stop"
    else:
        state = await engine.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=body.get("top_p", 1.0),
            top_k=body.get("top_k", 0),
            min_p=body.get("min_p", 0.0),
            seed=body.get("seed"),
            repetition_penalty=body.get("repetition_penalty", 1.0),
            frequency_penalty=body.get("frequency_penalty", 0.0),
            presence_penalty=body.get("presence_penalty", 0.0),
            stop=body.get("stop"),
            stop_token_ids=body.get("stop_token_ids"),
            logit_bias=body.get("logit_bias"),
            enable_thinking=body.get("enable_thinking"),
            thinking_budget=body.get("thinking_budget"),
            reasoning_effort=body.get("reasoning_effort"),
            spec_decode=body.get("spec_decode", False),
            xtc_probability=body.get("xtc_probability", 0.0),
            xtc_threshold=body.get("xtc_threshold", 0.0),
            logprobs=body.get("logprobs", False),
            top_logprobs=body.get("top_logprobs"),
            logits_processors=body.get("logits_processors"),
            priority=body.get("priority", 0),
        )
        if isinstance(state, dict):
            text = state.get("text", "")
            prompt_tokens = state.get("prompt_tokens", 0)
            completion_tokens = state.get("completion_tokens", 0)
            finish_reason = state.get("finish_reason", "stop") or "stop"
        else:
            # CRITICAL fix: GenerationOutput uses text/prompt_tokens/
            # completion_tokens (batched_engine.py:84-87); prior code used
            # non-existent generated_text/prompt_token_count/
            # completion_token_count → AttributeError on every batch item.
            text = getattr(state, "text", None) or getattr(state, "generated_text", "")
            prompt_tokens = getattr(state, "prompt_tokens", None) or getattr(state, "prompt_token_count", 0)
            completion_tokens = getattr(state, "completion_tokens", None) or getattr(state, "completion_token_count", 0)
            finish_reason = getattr(state, "finish_reason", None) or "stop"

    import time as _time
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:24]}",
        "object": "text_completion",
        "created": int(_time.time()),
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def _execute_embedding(body: dict) -> dict:
    """Execute an embedding request via the shared embeddings core.

    Auth was performed at batch-submission time (consistent with the chat /
    completion executors that call the engine directly), so we reuse
    embeddings._embed_and_format which skips per-request auth.
    """
    from .embeddings import EmbeddingRequest, _embed_and_format

    try:
        req = EmbeddingRequest(**body)
    except Exception as e:
        raise ValueError(f"Invalid embedding request body: {e}") from None
    return await _embed_and_format(req)
