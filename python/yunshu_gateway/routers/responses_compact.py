"""``POST /v1/responses/compact``: compact a conversation into kept user messages + one opaque item."""

from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..responses_context import ContextError, compact_endpoint, error_json
from .models import _check_permission

router = APIRouter(tags=["responses"])


@router.post("/responses/compact")
async def compact_response(request: Request):
    _check_permission(request, "can_infer")
    try:
        try:
            body = json.loads(await request.body() or b"{}")
        except ValueError:
            raise ContextError(
                400, "Request body is not valid JSON", "invalid_json"
            ) from None
        if not isinstance(body, dict):
            raise ContextError(
                400, "Request body must be a JSON object", "invalid_type"
            )
        return JSONResponse(await compact_endpoint(request, body))
    except ContextError as exc:
        return error_json(exc)
