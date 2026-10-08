"""Cancellable Evals jobs use the application's ordinary chat request path.

Credentials remain in memory, never in persisted runs. A restart fails interrupted
runs explicitly rather than silently replaying requests with different credentials.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import time
from typing import cast

import httpx
from jsonschema import Draft202012Validator

from .conversations_store import ConversationError
from .evals_grading import lexical, lookup, render
from .evals_store import EvalStore, get_store, new_id, stored_completions
from .files_store import text_of

_tasks: dict[str, asyncio.Task] = {}


def chat_messages(messages: list) -> list[dict]:
    """Convert SDK EvalItem content blocks to normal Chat Completions messages."""
    if not isinstance(messages, list):
        raise ConversationError(
            400, "Input trajectory must be an array of messages", "invalid_type"
        )
    result = []
    for message in messages:
        if not isinstance(message, dict):
            raise ConversationError(
                400, "Input messages must be objects", "invalid_type"
            )
        message = dict(message)
        content = message.get("content")
        if isinstance(content, dict):
            content = [content]
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    part = {"type": "text", "text": part}
                elif isinstance(part, dict):
                    part = dict(part)
                    if part.get("type") in ("input_text", "output_text"):
                        part["type"] = "text"
                    elif part.get("type") == "input_image":
                        image = {"url": part["image_url"]}
                        if "detail" in part:
                            image["detail"] = part["detail"]
                        part = {"type": "image_url", "image_url": image}
                parts.append(part)
            message["content"] = parts
        result.append(message)
    return result


def sample_inputs(messages: list) -> list[dict]:
    # The SDK's output SampleInput.content is a string even for media inputs.
    return [
        {
            "role": m["role"],
            "content": m.get("content")
            if isinstance(m.get("content"), str)
            else json.dumps(m.get("content") or "", ensure_ascii=False),
        }
        for m in messages
    ]


def source_rows(source: dict, config: dict) -> list[dict]:
    kind = source.get("type")
    if kind == "file_content":
        rows = source["content"]
    elif kind == "file_id":
        try:
            rows = [
                json.loads(line)
                for line in text_of(source["id"]).splitlines()
                if line.strip()
            ]
        except ValueError as exc:
            raise ConversationError(
                400, "File must contain valid JSONL", "invalid_value"
            ) from exc
    elif kind == "stored_completions":
        records = stored_completions()
        rows = []
        for stored in records:
            c = stored["completion"]
            if source.get("model") and c.get("model") != source["model"]:
                continue
            if any(
                (c.get("metadata") or {}).get(k) != v
                for filters in (
                    config.get("metadata") or {},
                    source.get("metadata") or {},
                )
                for k, v in filters.items()
            ):
                continue
            if (
                source.get("created_after") is not None
                and c["created"] < source["created_after"]
            ):
                continue
            if (
                source.get("created_before") is not None
                and c["created"] > source["created_before"]
            ):
                continue
            messages = stored["messages"]
            rows.append(
                {
                    "item": {
                        "input": sample_inputs(messages),
                        "input_trajectory": messages,
                        "metadata": c.get("metadata") or {},
                        **c,
                    },
                    "sample": {
                        "model": c.get("model") or "",
                        "usage": usage_of(c),
                        "input": sample_inputs(messages),
                        "finish_reason": c["choices"][0].get("finish_reason") or "stop",
                        "output_text": c["choices"][0]["message"].get("content") or "",
                        "output": [c["choices"][0]["message"]],
                    },
                }
            )
        if source.get("limit") is not None:
            rows = rows[: source["limit"]]
    else:
        raise ConversationError(
            400,
            "Unsupported data source",
            "unsupported_value",
            "data_source.source.type",
        )
    if not isinstance(rows, list) or len(rows) > 10000:
        raise ConversationError(
            400, "Data source must contain at most 10000 rows", "invalid_value"
        )
    try:
        json.dumps(rows, allow_nan=False)
    except ValueError as exc:
        raise ConversationError(
            400, "Source rows must contain finite JSON values", "invalid_value"
        ) from exc
    schema = config.get("item_schema")
    validator = Draft202012Validator(schema) if schema else None
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("item"), dict)
            or ("sample" in row and not isinstance(row["sample"], dict))
        ):
            raise ConversationError(
                400,
                "Each source row needs an item object and optional sample object",
                "invalid_type",
            )
        if validator:
            error = next(validator.iter_errors(row["item"]), None)
            if error:
                raise ConversationError(
                    400,
                    f"Item does not match item_schema: {error.message}",
                    "invalid_value",
                )
    return rows


def usage_of(payload: dict) -> dict:
    u = payload.get("usage") or {}
    return dict(
        prompt_tokens=u.get("prompt_tokens", 0),
        completion_tokens=u.get("completion_tokens", 0),
        total_tokens=u.get("total_tokens", 0),
        cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
    )


async def completion(
    client, model: str, messages: list, params: dict, schema=None
) -> dict:
    params = {k: v for k, v in params.items() if v is not None}
    if "max_completions_tokens" in params:
        params["max_completion_tokens"] = params.pop("max_completions_tokens")
    body = {
        "model": model,
        "messages": chat_messages(messages),
        **params,
        "stream": False,
    }
    if schema:
        body.setdefault("enable_thinking", False)
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "eval_grade", "strict": True, "schema": schema},
        }
    response = await client.post("/v1/chat/completions", json=body)
    response.raise_for_status()
    return cast(dict, response.json())


def add_usage(rec: dict, model: str, payload: dict):
    entries = rec["per_model_usage"]
    entry = next((u for u in entries if u["model_name"] == model), None)
    if entry is None:
        entry = dict(model_name=model, invocation_count=0, **usage_of({}))
        entries.append(entry)
    entry["invocation_count"] += 1
    for k, v in usage_of(payload).items():
        entry[k] += v


async def process(store: EvalStore, rid: str, app, headers: dict):
    rec = store.get(rid)
    try:
        store.change(rid, status="in_progress")
        rec["status"] = "in_progress"
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://yunshu.local",
            headers=headers,
            timeout=None,
        ) as client:
            for index, row in enumerate(rec["_rows"]):
                ds = rec["data_source"]
                context = {"item": row["item"], "sample": row.get("sample") or {}}
                messages = []
                params = ds.get("sampling_params") or {}
                sample = dict(
                    error=None,
                    finish_reason="stop",
                    input=[],
                    output=[],
                    model=ds.get("model") or "",
                    max_completion_tokens=params.get("max_completion_tokens", 0),
                    seed=params.get("seed", 0),
                    temperature=params.get("temperature", 1),
                    top_p=params.get("top_p", 1),
                    usage=usage_of({}),
                )
                item = dict(
                    id=new_id("evalitem"),
                    object="eval.run.output_item",
                    created_at=int(time.time()),
                    eval_id=rec["eval_id"],
                    run_id=rid,
                    datasource_item_id=index,
                    datasource_item=row["item"],
                    sample=sample,
                    results=[],
                    status="pass",
                )
                try:
                    if ds["type"] == "completions" and ds.get("model"):
                        im = ds["input_messages"]
                        messages = (
                            render(im["template"], context)
                            if im["type"] == "template"
                            else lookup(im["item_reference"], context)
                        )
                        payload = await completion(
                            client, ds["model"], messages, params
                        )
                        add_usage(rec, ds["model"], payload)
                        choice = payload["choices"][0]
                        context["sample"] = {
                            "output_text": choice["message"].get("content") or "",
                            "output": [choice["message"]],
                            "tool_calls": choice["message"].get("tool_calls") or [],
                        }
                        sample.update(
                            input=sample_inputs(messages),
                            output=context["sample"]["output"],
                            usage=usage_of(payload),
                            finish_reason=choice.get("finish_reason") or "stop",
                        )
                    else:
                        sample.update(
                            {
                                k: context["sample"][k]
                                for k in ("model", "usage", "input", "finish_reason")
                                if k in context["sample"]
                            }
                        )
                        sample["output"] = context["sample"].get("output") or [
                            {
                                "role": "assistant",
                                "content": context["sample"].get("output_text", ""),
                            }
                        ]
                    for criterion in rec["_criteria"]:
                        kind = criterion["type"]
                        if kind in ("string_check", "text_similarity"):
                            result = lexical(criterion, context)
                        else:
                            g = render(criterion, context)
                            prop = (
                                {"type": "string", "enum": g["labels"]}
                                if kind == "label_model"
                                else {
                                    "type": "number",
                                    "minimum": g.get("range", [0, 1])[0],
                                    "maximum": g.get("range", [0, 1])[1],
                                }
                            )
                            key = "label" if kind == "label_model" else "score"
                            schema = dict(
                                type="object",
                                properties={key: prop},
                                required=[key],
                                additionalProperties=False,
                            )
                            prompts = [
                                *g["input"],
                                {
                                    "role": "system",
                                    "content": f"Return a JSON object with {key} matching this schema: {json.dumps(schema)}",
                                },
                            ]
                            payload = await completion(
                                client,
                                g["model"],
                                prompts,
                                g.get("sampling_params")
                                or {"temperature": 0, "max_completion_tokens": 128},
                                schema,
                            )
                            add_usage(rec, g["model"], payload)
                            value = json.loads(
                                payload["choices"][0]["message"]["content"]
                            )
                            Draft202012Validator(schema).validate(value)
                            score = (
                                float(value[key] in g["passing_labels"])
                                if kind == "label_model"
                                else float(value[key])
                            )
                            if not math.isfinite(score):
                                raise ValueError("Grader returned a non-finite score")
                            result = dict(
                                name=g["name"],
                                type=kind,
                                score=score,
                                passed=score >= g.get("pass_threshold", 1),
                                sample=value,
                            )
                        item["results"].append(result)
                    item["status"] = (
                        "pass" if all(r["passed"] for r in item["results"]) else "fail"
                    )
                except Exception as exc:
                    item["status"] = "error"
                    sample["error"] = dict(
                        code="evaluation_error", message=str(exc)[:1000]
                    )
                rec["_outputs"].append(item)
                counts = rec["result_counts"]
                counts["total"] += 1
                counts[
                    {"pass": "passed", "fail": "failed", "error": "errored"}[
                        item["status"]
                    ]
                ] += 1
                for result in item["results"]:
                    summary = next(
                        s
                        for s in rec["per_testing_criteria_results"]
                        if s["testing_criteria"] == result["name"]
                    )
                    summary["passed" if result["passed"] else "failed"] += 1
                store.save(rec)
                await asyncio.sleep(0.01)
        store.change(rid, status="completed")
    except asyncio.CancelledError:
        with contextlib.suppress(ConversationError):
            store.change(rid, status="canceled")
        raise
    except Exception as exc:
        with contextlib.suppress(ConversationError):
            store.change(
                rid,
                status="failed",
                error=dict(code="run_error", message=str(exc)[:1000]),
            )
    finally:
        _tasks.pop(rid, None)


def start(store: EvalStore, rid: str, app, headers: dict):
    _tasks[rid] = asyncio.create_task(
        process(store, rid, app, headers), name=f"yunshu-eval-{rid}"
    )


async def cancel(rid: str):
    task = _tasks.pop(rid, None)
    if task:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def recover():
    store = get_store()
    for rec in store.rows("evalrun"):
        if rec["status"] in ("queued", "in_progress"):
            store.change(
                rec["id"],
                status="failed",
                error=dict(
                    code="interrupted",
                    message="Server restarted during evaluation; create a new run to retry.",
                ),
            )


async def stop():
    for rid in list(_tasks):
        await cancel(rid)
