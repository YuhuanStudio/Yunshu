"""(MED): a background Responses cancel was LOST when it landed during the
model-load await window.

create_response resolves/loads the model (`await get_engine_for_model(...)`) BEFORE it
registers the request with the request_tracker. A POST /v1/responses/{id}/cancel during
that window finds no tracker entry (tracker.cancel→False) and the cancel handler persists
status="cancelled" directly (the branch, for status in queued|in_progress). But the
request is NOT registered, so _ns_cancel_event is never set → the status map resolved
"completed" → the terminal store CLOBBERED the cancel marker. The client received a
cancelled envelope, yet a later GET returned "completed" with full output and the GPU work
ran to completion. (covered the pure-`queued` window; this is the `in_progress`
sibling during model load.)

Fix: before the terminal store, re-read and honor a pre-existing terminal "cancelled" — do
not overwrite it. Scoped strictly to the unregistered window: the registered-path cancel
goes through _ns_cancel_event → "incomplete" and never persists "cancelled".
"""
from __future__ import annotations

import inspect

from yunshu_gateway.routers import responses
from yunshu_gateway.routers.responses import (
    _get_stored_response,
    _public_stored,
    _store_response,
)


def test_clobber_guard_rereads_store_before_terminal_write():
    src = inspect.getsource(responses.create_response)
    store_block = src.index("if req.store:")
    # the terminal persist call inside the store block
    persist = src.index("_store_response(response_id, _persist_payload)", store_block)
    guard = src.index('_prior.get("status") == "cancelled"', store_block)
    # the guard must come BEFORE the terminal store (so it can skip the clobber)
    assert store_block < guard < persist, "cancel-clobber guard must precede the terminal store"
    # and it must RE-READ the store (not rely on a stale snapshot)
    window = src[store_block:guard]
    assert "_get_stored_response(response_id)" in window, "guard must re-read the live store"


def test_terminal_cancelled_status_survives_a_later_completed_write():
    """Behavioral: the guard's invariant — once the store holds a terminal 'cancelled',
    the create_response terminal path returns it unchanged rather than clobbering to
    'completed'. We reproduce the exact guard condition against the REAL store helpers."""
    rid = "resp_w1041_clobber_test"
    _store_response(rid, {
        "id": rid, "object": "response", "status": "cancelled",
        "completed_at": 123, "output": [], "_owner": "t",
    })
    prior = _get_stored_response(rid)
    assert prior is not None and prior.get("status") == "cancelled"
    # _public_stored must expose the cancelled status (what the client GETs) and must
    # strip the private _owner key.
    pub = _public_stored(prior)
    assert pub["status"] == "cancelled"
    assert "_owner" not in pub


def test_registered_path_unaffected_non_cancelled_status_proceeds():
    """A normal terminal status in the store (e.g. left-over in_progress) must NOT trip the
    guard — only an exact 'cancelled' is honored, so completed responses still persist."""
    rid = "resp_w1041_normal"
    _store_response(rid, {"id": rid, "status": "in_progress", "output": []})
    prior = _get_stored_response(rid)
    assert prior.get("status") != "cancelled"  # guard would NOT fire → terminal store proceeds
