"""Per-request token-usage accounting context.

The tenant/RBAC `tokens_per_minute` quota was settable/persisted/displayed but NEVER
enforced: the admission call `tenant.check_and_record()` passed no `tokens` arg → the
running `_token_count` only ever grew by 0, so the TPM check never tripped (a FREE tenant
configured 50k TPM got unlimited tokens). The fix needs the per-request token total
attributed back to the tenant, but the totals are produced deep in the routers (the bare
`_record_metrics` choke point, no `request`) while the tenant lives on `request.state`.

This module bridges that gap with a per-request mutable accumulator box, propagated via a
ContextVar. Verified empirically (Starlette 1.2.0) to propagate from the auth middleware's
`dispatch` into the endpoint AND into a StreamingResponse generator's drain, so a single
`record_billed_tokens()` call in each `_record_metrics` reaches the box the middleware
reads at request settle (inline for non-stream; in the post-drain background task for
streaming, via the SAME box object also stashed on `request.state`).
"""
from __future__ import annotations

import contextlib
import contextvars

# A mutable single-element list ([total_tokens]); None when no request scope is active
# (e.g. auth disabled, or a background/non-HTTP caller).
_billed_box: contextvars.ContextVar[list[int] | None] = contextvars.ContextVar(
    "yunshu_billed_tokens_box", default=None
)


def new_billing_box() -> list[int]:
    """Start a fresh per-request accumulator and bind it to the current context.

    Called once at the top of the auth middleware's dispatch. Returns the box so the
    caller can also stash it on ``request.state`` (the streaming background task reads it
    there, since it may run in a different context than the generator that filled it)."""
    box = [0]
    _billed_box.set(box)
    return box


def record_billed_tokens(n: int) -> None:
    """Accumulate ``n`` tokens against the current request's box (no-op outside a
    request scope). Called from every ``_record_metrics`` choke point."""
    box = _billed_box.get()
    if box is not None and n:
        with contextlib.suppress(TypeError, ValueError):
            box[0] += int(n)
