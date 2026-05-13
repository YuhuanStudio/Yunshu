"""Generation cancellation endpoint — POST /v1/cancel."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["cancel"])


class CancelRequest(BaseModel):
    request_id: str | None = None
    cancel_all: bool = False


@router.post("/v1/cancel")
async def cancel_generation(req: CancelRequest):
    """Cancel an in-progress generation or all active generations.

    Pass `request_id` to cancel a specific generation.
    Pass `cancel_all: true` to cancel all active generations.
    """
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
async def list_active_generations():
    """List all currently active (in-progress) generations."""
    from yunshu_engine.request_tracker import get_request_tracker
    tracker = get_request_tracker()
    return {"active": tracker.list_active(), "count": tracker.active_count}
