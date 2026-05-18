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
import time
import uuid


from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["batch"])

# Configurable limits
_BATCH_MAX_ITEMS = int(os.environ.get("YUNSHU_BATCH_MAX_ITEMS", "500"))
_BATCH_DEFAULT_TIMEOUT = float(os.environ.get("YUNSHU_BATCH_TIMEOUT", "300"))
_BATCH_MAX_CSV_SIZE = int(os.environ.get("YUNSHU_BATCH_MAX_CSV_SIZE", str(50 * 1024 * 1024)))  # 50 MB
_BATCH_STORE_TTL = int(os.environ.get("YUNSHU_BATCH_STORE_TTL", "3600"))  # 1 hour default
_BATCH_STORE_MAX_SIZE = int(os.environ.get("YUNSHU_BATCH_STORE_MAX_SIZE", "1000"))

# In-memory progress tracking
_batch_store: dict[str, dict] = {}


def _cleanup_batch_store() -> None:
    """Remove expired batch entries and enforce max size.

    Called after each batch creation. TTL-based expiry prevents unbounded
    memory growth from accumulated batch results.
    """
    now = time.time()
    expired = [
        bid for bid, info in _batch_store.items()
        if (now - info.get("started_at", 0)) > _BATCH_STORE_TTL
    ]
    for bid in expired:
        del _batch_store[bid]

    # Enforce max size — evict oldest first
    while len(_batch_store) > _BATCH_STORE_MAX_SIZE:
        oldest_id = min(_batch_store, key=lambda k: _batch_store[k].get("started_at", 0))
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


class BatchResponse(BaseModel):
    id: str
    object: str = "batch"
    status: str
    results: list[dict]


@router.post("/batch", response_model=None)
async def create_batch(req: BatchRequest):
    """Execute a batch of inference requests concurrently.

    Supports configurable concurrency, batch timeout, and partial success.
    Progress is tracked in real-time so /batch/{id}/status reflects
    completed items even while the batch is running.
    """
    if not req.requests:
        raise HTTPException(status_code=400, detail="Batch cannot be empty")

    if len(req.requests) > _BATCH_MAX_ITEMS:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(req.requests)} exceeds limit of {_BATCH_MAX_ITEMS} (set YUNSHU_BATCH_MAX_ITEMS env var)",
        )

    batch_id = f"batch_{uuid.uuid4().hex[:24]}"
    total = len(req.requests)

    # Initialize progress tracking
    _batch_store[batch_id] = {
        "id": batch_id,
        "status": "in_progress",
        "total": total,
        "completed": 0,
        "succeeded": 0,
        "failed": 0,
        "started_at": time.time(),
        "results": [None] * total,
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

        # Update progress in real-time
        processed[index] = {
            "custom_id": req.requests[index].custom_id,
            "status": status,
            "response": result if status == "success" else None,
            "error": result.get("error") if status == "error" else None,
        }
        _batch_store[batch_id]["completed"] += 1
        if status == "success":
            _batch_store[batch_id]["succeeded"] += 1
        else:
            _batch_store[batch_id]["failed"] += 1

        return index, result, status

    tasks = [_process_item(i, item) for i, item in enumerate(req.requests)]
    task_handles = [asyncio.create_task(t) for t in tasks]

    try:
        done, pending = await asyncio.wait(task_handles, timeout=req.timeout)
    except asyncio.CancelledError:
        # Request cancelled (e.g., client disconnect) — propagate CancelledError
        for task in task_handles:
            task.cancel()
        _batch_store[batch_id]["status"] = "cancelled"
        raise
    except Exception as exc:
        # Cancel everything before raising
        for task in task_handles:
            task.cancel()
        _batch_store[batch_id]["status"] = "error"
        raise HTTPException(status_code=500, detail=f"Batch execution failed: {exc}")

    # Cancel pending (timed-out) tasks
    for task in pending:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    timed_out = len(pending)

    # Collect results from completed tasks
    # (processed[] is already filled by _process_item callbacks)
    for task in done:
        try:
            task.result()  # consume any CancelledError
        except (asyncio.CancelledError, Exception):
            pass

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

    response_data = {
        "id": batch_id,
        "object": "batch",
        "status": final_status,
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "timed_out": timed_out,
        "elapsed_s": round(time.time() - _batch_store[batch_id]["started_at"], 2),
        "results": processed,
    }

    _batch_store[batch_id].update({
        "status": final_status,
        "completed": total,
        "succeeded": succeeded,
        "failed": failed,
        "timed_out": timed_out,
        "results": processed,
        "finished_at": time.time(),
    })

    # Evict expired/oversized batch entries to prevent memory leak
    _cleanup_batch_store()

    return JSONResponse(response_data)


@router.get("/batch/{batch_id}/status")
async def get_batch_status(batch_id: str):
    """Get progress/status of a previously submitted batch."""
    info = _batch_store.get(batch_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found")
    summary = {
        "id": info["id"],
        "status": info["status"],
        "total": info["total"],
        "completed": info["completed"],
        "succeeded": info["succeeded"],
        "failed": info["failed"],
    }
    if "finished_at" in info:
        summary["elapsed_s"] = round(info["finished_at"] - info["started_at"], 2)
    else:
        summary["elapsed_s"] = round(time.time() - info["started_at"], 2)
    return JSONResponse(summary)


@router.get("/batch/{batch_id}/results")
async def get_batch_results(batch_id: str):
    """Get full results of a completed batch."""
    info = _batch_store.get(batch_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found")
    if info["status"] == "in_progress":
        raise HTTPException(status_code=409, detail="Batch is still in progress")
    return JSONResponse({
        "id": info["id"],
        "status": info["status"],
        "total": info["total"],
        "succeeded": info["succeeded"],
        "failed": info["failed"],
        "results": info["results"],
    })


@router.get("/batch/{batch_id}/results.csv")
async def download_batch_csv(batch_id: str):
    """Download batch results as CSV file."""
    info = _batch_store.get(batch_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found")
    if info["status"] == "in_progress":
        raise HTTPException(status_code=409, detail="Batch is still in progress")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["custom_id", "status", "finish_reason", "content", "error",
                      "prompt_tokens", "completion_tokens", "total_tokens"])

    for r in info["results"]:
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
        writer.writerow(row)

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=batch_{batch_id}.csv"},
    )


@router.post("/batch/upload/csv", response_model=None)
async def upload_batch_csv(
    file: UploadFile = File(...),
    model: str = "default",
    max_tokens: int = 512,
    max_concurrent: int = 4,
):
    """Upload a CSV file with prompts and execute as a batch.

    Expected CSV columns: custom_id, prompt (or messages_json)
    Optional columns: max_tokens, temperature, system_prompt
    """
    if not model or not model.strip():
        raise HTTPException(status_code=400, detail="model parameter is required and cannot be empty")

    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise HTTPException(status_code=400, detail="max_tokens must be a positive integer")

    if not isinstance(max_concurrent, int) or max_concurrent < 1:
        raise HTTPException(status_code=400, detail="max_concurrent must be a positive integer")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    content = await file.read()
    if len(content) > _BATCH_MAX_CSV_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"CSV file too large: {len(content)} bytes exceeds limit of {_BATCH_MAX_CSV_SIZE} bytes (set YUNSHU_BATCH_MAX_CSV_SIZE env var)",
        )
    text_content = content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text_content))

    items: list[BatchItem] = []
    for row in reader:
        custom_id = row.get("custom_id", str(uuid.uuid4().hex[:8]))
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
    return await create_batch(batch_req)


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
            raise ValueError(f"Model '{model}' not available")

    if is_batched:
        result = await engine.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top,
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
        text = state.generated_text
        prompt_tokens = state.prompt_token_count
        completion_tokens = state.completion_token_count
        finish_reason = state.finish_reason or "stop"

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
    from ..engine import get_engine, get_engine_for_model
    from yunshu_engine.batched_engine import BatchedEngine

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

    temperature = body.get("temperature", 0.7)

    engine = get_engine()
    if engine is None or not engine.is_loaded or not engine.resolve_model_id(model):
        try:
            engine = await get_engine_for_model(model)
        except (KeyError, Exception):
            raise ValueError(f"Model '{model}' not available")

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
        text = state.generated_text
        prompt_tokens = state.prompt_token_count
        completion_tokens = state.completion_token_count
        finish_reason = state.finish_reason or "stop"

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
    """Execute an embedding request."""
    raise ValueError("Batch embedding not yet supported")
