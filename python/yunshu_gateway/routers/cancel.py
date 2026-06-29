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

from yunshu_control.audit_log import log_operation, resolve_actor


class CancelRequest(BaseModel):
    request_id: str | None = None
    cancel_all: bool = False


def _check_auth(request: Request) -> None:
    """Verify auth token when YUNSHU_AUTH_TOKEN is configured.

    Single-consumer model: the simplified TenantAuthMiddleware stamps
    ``request.state.role = "owner"`` on every request it admits. Honor that
    (plus the legacy ``rbac_key`` / ``tenant`` attributes, kept for backward
    compat with tests that set them directly).

    Raises HTTPException 401 if auth is required but missing/invalid.
    """
    # The simplified middleware has already authenticated the request and
    # stamped request.state.role — trust that.
    if hasattr(request, "state"):
        if getattr(request.state, "role", None):
            return
        # Legacy compat: tests / older shims may set rbac_key or tenant directly.
        if getattr(request.state, "rbac_key", None) is not None:
            return
        if getattr(request.state, "tenant", None) is not None:
            return

    auth_token = os.environ.get("YUNSHU_AUTH_TOKEN")
    if not auth_token:
        # Deny by default when no auth is configured (secure default).
        # Explicit opt-in via YUNSHU_AUTH_DISABLED=true is required to
        # skip authentication.
        if os.environ.get("YUNSHU_AUTH_DISABLED", "").lower() not in (
            "true",
            "1",
            "yes",
        ):
            raise HTTPException(
                status_code=401,
                detail="No authentication configured. Set YUNSHU_AUTH_TOKEN or YUNSHU_AUTH_DISABLED=true.",
            )
        return
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


@router.post("/cancel")
async def cancel_generation(req: CancelRequest, request: Request):
    """Cancel an in-progress generation or all active generations.

    Pass `request_id` to cancel a specific generation.
    Pass `cancel_all: true` to cancel all active generations.
    """
    actor = resolve_actor(request)
    _check_auth(request)

    from yunshu_engine.request_tracker import get_request_tracker

    tracker = get_request_tracker()

    # Admin gate: cancel_all and arbitrary request_id cancellation are powerful,
    # so gate them on the canonical request.state.role stamped by the auth
    # middleware. Single-consumer model: the simplified TenantAuthMiddleware
    # stamps role="owner" on every admitted request (or "admin"/"system" for
    # static-token holders), so the single owner can always cancel_all and
    # cancel by id. Auth-disabled dev mode is also treated as admin.
    _auth_disabled = os.environ.get("YUNSHU_AUTH_DISABLED", "").lower() in (
        "true",
        "1",
        "yes",
    )
    _role_str = str(getattr(request.state, "role", "") or "")
    is_admin = (
        _auth_disabled
        or _role_str.lower() in ("admin", "system", "owner")
        or _role_str.upper().endswith("ADMIN")
    )

    if req.cancel_all:
        if not is_admin:
            raise HTTPException(
                status_code=403,
                detail="cancel_all requires admin role (use specific request_id to cancel your own request)",
            )
        count = tracker.cancel_all()

        # Also cancel dedup shadow requests that are not registered with
        # the tracker but tracked in engine_core._dedup_shadows.
        shadow_count = 0
        try:
            from ..engine import get_engine, get_model_manager

            engines_to_check = []
            manager = get_model_manager()
            if manager is not None:
                for entry in manager.list_entries():
                    if entry.is_loaded and hasattr(entry, "engine"):
                        engines_to_check.append(entry.engine)
            else:
                engine = get_engine()
                if engine is not None:
                    engines_to_check.append(engine)

            for eng in engines_to_check:
                core = getattr(eng, "_engine_core", None)
                if core is None:
                    continue
                for shadow_id in list(getattr(core, "_dedup_shadows", {}).keys()):
                    # Only count a REAL cancellation. Dedup shadows are
                    # engine-internal and not tracker-registered, so cancel() returns
                    # False (no-op) — the old unconditional += inflated the reported
                    # count. Shadows still terminate via their primary's fan-out (the
                    # primaries ARE cancelled by the main cancel_all above).
                    if tracker.cancel(shadow_id):
                        shadow_count += 1
        except Exception:
            pass  # Best-effort: dedup shadows may not exist in all modes

        total = count + shadow_count
        log_operation("cancel_all", "all_requests", "success", actor=actor, count=total)
        return {"status": "cancelled", "count": total}

    if req.request_id:
        # Scope per-request cancellation by ownership. Admin
        # can cancel anyone; others can only cancel their own requests.
        if not is_admin and hasattr(tracker, "get_owner"):
            owner = tracker.get_owner(req.request_id)
            if owner and owner != actor:
                log_operation(
                    "cancel_request",
                    req.request_id,
                    "failure",
                    actor=actor,
                    detail=f"not_owner (req owned by {owner})",
                )
                raise HTTPException(
                    status_code=403,
                    detail="cannot cancel another user's request",
                )
        found = tracker.cancel(req.request_id)
        if found:
            log_operation("cancel_request", req.request_id, "success", actor=actor)
            return {"status": "cancelled", "request_id": req.request_id}
        log_operation(
            "cancel_request", req.request_id, "failure", actor=actor, detail="not_found"
        )
        raise HTTPException(
            status_code=404,
            detail=f"Request '{req.request_id}' not found or already completed",
        )

    raise HTTPException(
        status_code=400, detail="Provide either request_id or cancel_all=true"
    )


@router.get("/active-generations")
async def list_active_generations(request: Request):
    """List currently active (in-progress) generations — the caller's own unless
    admin (was leaking every tenant's live request_id + model)."""
    _check_auth(request)
    actor = resolve_actor(request)
    _auth_disabled = os.environ.get("YUNSHU_AUTH_DISABLED", "").lower() in (
        "true",
        "1",
        "yes",
    )
    _role_str = str(getattr(request.state, "role", "") or "")
    is_admin = (
        _auth_disabled
        or _role_str.lower() in ("admin", "system", "owner")
        or _role_str.upper().endswith("ADMIN")
    )

    from yunshu_engine.request_tracker import get_request_tracker

    tracker = get_request_tracker()
    active = tracker.list_active()
    # Don't disclose other tenants' in-flight requests. Mirror the per-request
    # cancel ownership semantics: deny only entries owned by a DIFFERENT actor
    # (owner None = best-effort/unattributed → visible, matching cancel).
    if not is_admin and hasattr(tracker, "get_owner"):
        active = [
            a
            for a in active
            if (lambda o: o is None or o == actor)(
                tracker.get_owner(a.get("request_id"))
            )
        ]
    # Standardize to OpenAI list envelope `{object:"list", data:[...]}` to match
    # the rest of the gateway. Legacy `active`/`count` fields are preserved for
    # backwards-compat with existing clients/dashboards.
    return {
        "object": "list",
        "data": active,
        "active": active,
        "count": len(active) if not is_admin else tracker.active_count,
    }
