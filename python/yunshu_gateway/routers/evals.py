"""OpenAI Evals CRUD, background runs and output-item routes."""

from __future__ import annotations

import functools
import json
import math
import time
from collections.abc import Iterable
from typing import NoReturn

from fastapi import APIRouter, Request
from jsonschema import Draft202012Validator, SchemaError
from openai.types.eval_create_params import EvalCreateParams
from openai.types.evals.run_create_params import RunCreateParams
from pydantic import TypeAdapter, ValidationError

from .. import evals_runner
from ..conversations_store import ConversationError, check_metadata
from ..evals_grading import METRICS
from ..evals_store import get_store, new_id, page, public
from ..files_store import FileStoreError
from .conversations import _body, _err, _int_param, _io
from .models import _check_permission

router = APIRouter(tags=["evals"])


def endpoint(fn):
    @functools.wraps(fn)
    async def wrapped(request: Request, *args, **kwargs):
        _check_permission(request, "can_infer")
        try:
            return await fn(request, *args, **kwargs)
        except ConversationError as exc:
            return _err(exc)
        except FileStoreError as exc:
            return _err(ConversationError(exc.status, exc.message, exc.code))

    return wrapped


def invalid(message: str, param=None) -> NoReturn:
    raise ConversationError(400, message, "invalid_value", param)


def sdk_body(body: dict, sdk_type) -> dict:
    try:
        json.dumps(body, allow_nan=False)
        adapter = TypeAdapter(sdk_type)
        value = adapter.validate_python(body, strict=True)

        def exhaust(v):
            if isinstance(v, dict):
                for child in v.values():
                    exhaust(child)
            elif isinstance(v, Iterable) and not isinstance(v, (str, bytes)):
                for child in v:
                    exhaust(child)

        exhaust(value)  # SDK Iterable fields validate lazily; never persist iterators.
        return body
    except (ValidationError, ValueError) as exc:
        invalid(str(exc), "body")


def validate_eval(body: dict):
    config = body.get("data_source_config")
    if not isinstance(config, dict) or config.get("type") not in (
        "custom",
        "stored_completions",
        "logs",
    ):
        invalid(
            "data_source_config must be custom, logs or stored_completions",
            "data_source_config",
        )
    if config["type"] == "custom":
        if not isinstance(config.get("item_schema"), dict):
            invalid("custom data source requires item_schema")
        try:
            Draft202012Validator.check_schema(config["item_schema"])
        except SchemaError as exc:
            invalid(str(exc))
    criteria = body.get("testing_criteria")
    if not isinstance(criteria, list) or not criteria:
        invalid("testing_criteria must be a nonempty array")
    names = set()
    for g in criteria:
        if not isinstance(g, dict) or g.get("type") not in (
            "string_check",
            "text_similarity",
            "score_model",
            "label_model",
        ):
            invalid(
                "Supported graders: string_check, text_similarity, score_model, label_model"
            )
        if not isinstance(g.get("name"), str) or not g["name"] or g["name"] in names:
            invalid("Grader names must be nonempty and unique")
        names.add(g["name"])
        kind = g["type"]
        if kind in ("string_check", "text_similarity"):
            if not all(isinstance(g.get(k), str) for k in ("input", "reference")):
                invalid("Lexical graders require string input and reference")
            if kind == "string_check" and g.get("operation") not in (
                "eq",
                "ne",
                "like",
                "ilike",
            ):
                invalid("Unsupported string_check operation")
            if kind == "text_similarity" and (
                g.get("evaluation_metric") not in METRICS or "pass_threshold" not in g
            ):
                invalid(
                    "text_similarity requires a supported metric and pass_threshold"
                )
        else:
            if (
                not isinstance(g.get("model"), str)
                or not g["model"]
                or not isinstance(g.get("input"), list)
                or not g["input"]
            ):
                invalid("Model graders require model and input messages")
            if kind == "label_model":
                labels, passing = g.get("labels"), g.get("passing_labels")
                if (
                    not isinstance(labels, list)
                    or not labels
                    or not all(isinstance(s, str) for s in labels)
                    or not isinstance(passing, list)
                    or not all(s in labels for s in passing)
                ):
                    invalid("passing_labels must be a subset of nonempty labels")
            else:
                bounds = g.get("range", [0, 1])
                if (
                    not isinstance(bounds, list)
                    or len(bounds) != 2
                    or not all(
                        isinstance(n, (int, float))
                        and not isinstance(n, bool)
                        and math.isfinite(n)
                        for n in bounds
                    )
                    or bounds[0] >= bounds[1]
                ):
                    invalid("score_model range must be two increasing finite numbers")
        if "pass_threshold" in g and (
            not isinstance(g["pass_threshold"], (int, float))
            or isinstance(g["pass_threshold"], bool)
            or not math.isfinite(g["pass_threshold"])
        ):
            invalid("pass_threshold must be finite")
    check_metadata(body.get("metadata"))
    if "name" in body and not isinstance(body["name"], str):
        invalid("name must be a string")


def query(request: Request, default="asc") -> dict:
    order = request.query_params.get("order", default)
    if order not in ("asc", "desc"):
        invalid("order must be asc or desc", "order")
    return dict(
        limit=_int_param(request, "limit", 20, 1, 100),
        order=order,
        after=request.query_params.get("after"),
    )


@router.post("/evals")
@endpoint
async def create_eval(request: Request):
    body = sdk_body(await _body(request), EvalCreateParams)
    validate_eval(body)
    return await _io(
        get_store().create_eval,
        {
            k: body[k]
            for k in ("name", "metadata", "data_source_config", "testing_criteria")
            if k in body
        },
    )


@router.get("/evals")
@endpoint
async def list_evals(request: Request):
    rows = await _io(get_store().rows, "eval")
    order_by = request.query_params.get("order_by", "created_at")
    if order_by not in ("created_at", "updated_at"):
        invalid("order_by must be created_at or updated_at", "order_by")
    rows.sort(key=lambda r: (r.get(order_by, r["created_at"]), r["id"]))
    return page(rows, **query(request, "desc"))


@router.get("/evals/{eval_id}")
@endpoint
async def get_eval(request: Request, eval_id: str):
    return public(await _io(get_store().get_eval, eval_id))


@router.post("/evals/{eval_id}")
@endpoint
async def update_eval(request: Request, eval_id: str):
    body = await _body(request)
    changes: dict = {}
    if "metadata" in body:
        changes["metadata"] = check_metadata(body["metadata"])
    if "name" in body:
        if not isinstance(body["name"], str):
            invalid("name must be a string")
        changes["name"] = body["name"]
    await _io(get_store().get_eval, eval_id)
    changes["updated_at"] = int(time.time())
    return public(await _io(get_store().change, eval_id, **changes))


@router.delete("/evals/{eval_id}")
@endpoint
async def delete_eval(request: Request, eval_id: str):
    store = get_store()
    children = await _io(store.delete_eval, eval_id)
    for rid in children:
        await evals_runner.cancel(rid)
    return dict(object="eval.deleted", deleted=True, eval_id=eval_id)


@router.post("/evals/{eval_id}/runs")
@endpoint
async def create_run(request: Request, eval_id: str):
    store = get_store()
    ev = await _io(store.get_eval, eval_id)
    body = sdk_body(await _body(request), RunCreateParams)
    ds = body.get("data_source")
    if (
        not isinstance(ds, dict)
        or ds.get("type") not in ("jsonl", "completions")
        or not isinstance(ds.get("source"), dict)
    ):
        invalid("Supported run data sources: jsonl, completions", "data_source")
    if ds["type"] == "completions" and ds.get("model"):
        im = ds.get("input_messages")
        if not isinstance(im, dict) or im.get("type") not in (
            "template",
            "item_reference",
        ):
            invalid("Sampling requires input_messages")
        if im["type"] == "template" and not isinstance(im.get("template"), list):
            invalid("template must contain messages")
        if im["type"] == "item_reference" and not isinstance(
            im.get("item_reference"), str
        ):
            invalid("item_reference must be a string")
    limit = ds["source"].get("limit")
    if limit is not None and not 1 <= limit <= 10000:
        invalid("stored_completions limit must be 1–10000")
    if not isinstance(ds.get("sampling_params", {}), dict):
        invalid("sampling_params must be an object")
    rows = await _io(evals_runner.source_rows, ds["source"], ev["_config"])
    meta = check_metadata(body.get("metadata"))
    if "name" in body and not isinstance(body["name"], str):
        invalid("name must be a string")
    rec = dict(
        id=new_id("evalrun"),
        object="eval.run",
        created_at=int(time.time()),
        eval_id=eval_id,
        data_source=ds,
        name=body.get("name") or "",
        metadata=meta,
        model=ds.get("model") or "",
        status="queued",
        error=None,
        report_url="",
        per_model_usage=[],
        per_testing_criteria_results=[
            dict(testing_criteria=g["name"], passed=0, failed=0)
            for g in ev["testing_criteria"]
        ],
        result_counts=dict(total=0, passed=0, failed=0, errored=0),
        _rows=rows,
        _criteria=ev["testing_criteria"],
        _outputs=[],
    )
    await _io(store.create_run, rec)
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() in ("authorization", "x-api-key")
    }
    evals_runner.start(store, rec["id"], request.app, headers)
    return public(rec)


@router.get("/evals/{eval_id}/runs")
@endpoint
async def list_runs(request: Request, eval_id: str):
    store = get_store()
    await _io(store.get_eval, eval_id)
    rows = [r for r in await _io(store.rows, "evalrun") if r["eval_id"] == eval_id]
    status = request.query_params.get("status")
    if status:
        if status not in ("queued", "in_progress", "completed", "canceled", "failed"):
            invalid("Invalid run status")
        rows = [r for r in rows if r["status"] == status]
    return page(rows, **query(request))


@router.get("/evals/{eval_id}/runs/{run_id}")
@endpoint
async def get_run(request: Request, eval_id: str, run_id: str):
    return public(await _io(get_store().run, eval_id, run_id))


@router.post("/evals/{eval_id}/runs/{run_id}/cancel")
@endpoint
async def cancel_run(request: Request, eval_id: str, run_id: str):
    store = get_store()
    rec = await _io(store.run, eval_id, run_id)
    if rec["status"] in ("queued", "in_progress"):
        await evals_runner.cancel(run_id)
        rec = await _io(store.change, run_id, status="canceled")
    return public(rec)


@router.delete("/evals/{eval_id}/runs/{run_id}")
@endpoint
async def delete_run(request: Request, eval_id: str, run_id: str):
    store = get_store()
    await _io(store.run, eval_id, run_id)
    await evals_runner.cancel(run_id)
    await _io(store.delete, run_id)
    return dict(object="eval.run.deleted", deleted=True, run_id=run_id)


@router.get("/evals/{eval_id}/runs/{run_id}/output_items")
@endpoint
async def list_output_items(request: Request, eval_id: str, run_id: str):
    rec = await _io(get_store().run, eval_id, run_id)
    rows = rec["_outputs"]
    status = request.query_params.get("status")
    if status:
        if status not in ("pass", "fail"):
            invalid("status must be pass or fail")
        rows = [r for r in rows if r["status"] == status]
    return page(rows, **query(request))


@router.get("/evals/{eval_id}/runs/{run_id}/output_items/{output_item_id}")
@endpoint
async def get_output_item(
    request: Request, eval_id: str, run_id: str, output_item_id: str
):
    rec = await _io(get_store().run, eval_id, run_id)
    for row in rec["_outputs"]:
        if row["id"] == output_item_id:
            return row
    raise ConversationError(404, "Output item not found", "not_found")
