"""Text-API WebSocket endpoints (see yunshu_gateway/ws_transport.py)."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket

from yunshu_engine import settings
from yunshu_gateway.token_compare import tokens_equal

from ..ws_transport import Connection

router = APIRouter()


async def _admit(ws: WebSocket) -> bool:
    """Origin and bearer checks *before* the upgrade, so a rejected client gets
    an HTTP 403 handshake failure (what the OpenAI SDK and browsers expect)."""
    origin = ws.headers.get("origin", "")
    if origin:
        allowed_cfg = settings.get("YUNSHU_CORS_ORIGINS")
        if allowed_cfg != "*":
            allowed = {
                o.strip().rstrip("/") for o in allowed_cfg.split(",") if o.strip()
            }
            if origin.rstrip("/") not in allowed:
                await ws.close(code=1008)
                return False
    token = settings.get("YUNSHU_AUTH_TOKEN")
    if token and not settings.get_bool("YUNSHU_AUTH_DISABLED"):
        got = ws.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not got:
            got = ws.headers.get("x-api-key", "") or (
                ws.query_params.get("token") or ""
            )
        if not (got and tokens_equal(got, token)):
            await ws.close(code=1008)
            return False
    return True


@router.websocket("/v1/stream")
async def stream_socket(ws: WebSocket):
    """Multiplexed streaming for chat.completions / completions / responses / messages."""
    if not await _admit(ws):
        return
    await ws.accept()
    await Connection(ws, "yunshu").run()


@router.websocket("/v1/responses")
async def responses_socket(ws: WebSocket):
    """OpenAI Responses WebSocket mode: response.create in, response.* events out."""
    if not await _admit(ws):
        return
    await ws.accept()
    await Connection(ws, "responses").run()
