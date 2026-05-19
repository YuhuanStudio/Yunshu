from __future__ import annotations
"""Generation cancellation endpoint — POST /v1/cancel.

Both endpoints require authentication when YUNSHU_AUTH_TOKEN is set.
Without auth, any client could cancel arbitrary in-progress generations
or enumerate active request metadata.
"""

import os

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(tags=["cancel"])


class CancelRequest(BaseModel):
    request_id: str | None = None
    cancel_all: bool = False


def _check_auth(request: Request) -> None:
    """Verify auth token when YUNSHU_AUTH_TOKEN is configured.

    Also accepts valid RBAC API keys (ys_ prefixed) set by TenantAuthMiddleware.

    Raises HTTPException 401 if auth is required but missing/invalid.
    """
    # Check RBAC key first (set by TenantAuthMiddleware for ys_-prefixed keys)
    if hasattr(request, "state"):
        rbac_key = getattr(request.state, "rbac_key", None)
        if isinstance(rbac_key, str) and rbac_key:
            return  # Valid RBAC key — already authenticated by middleware

        # Check tenant attribute (set by TenantAuthMiddleware for static tokens)
        tenant = getattr(request.state, "tenant", None)
        if isinstance(tenant, str) and tenant:
            return  # Authenticated via static token through middleware

    auth_token = os.environ.get("YUNSHU_AUTH_TOKEN")
    if not auth_token:
        return  # No auth configured
    if os.environ.get("YUNSHU_AUTH_DISABLED", "").lower() in ("true", "1", "yes"):
        return  # Auth explicitly disabled

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = auth[7:]
    import hmac
    if not hmac.compare_digest(token, auth_token):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/v1/cancel")
async def cancel_generation(req: CancelRequest, request: Request):
    """Cancel an in-progress generation or all active generations.

    Pass `request_id` to cancel a specific generation.
    Pass `cancel_all: true` to cancel all active generations.
    """
    _check_auth(request)

    from yunshu_engine.request_tracker import get_request_tracker

    tracker = get_request_tracker()

    if req.cancel_all:
        count = tracker.cancel_all()
        return {"status": "cancelled", "count": count}

    if req.request_id:
        found = tracker.cancel(req.request_id)
        if found:
            return {"status": "cancelled", "request_id": req.request_id}
        raise HTTPException(status_code=404, detail=f"Request '{req.request_id}' not found or already completed")

    raise HTTPException(status_code=400, detail="Provide either request_id or cancel_all=true")


@router.get("/v1/active-generations")
async def list_active_generations(request: Request):
    """List all currently active (in-progress) generations."""
    _check_auth(request)

    from yunshu_engine.request_tracker import get_request_tracker
    tracker = get_request_tracker()
    return {"active": tracker.list_active(), "count": tracker.active_count}
