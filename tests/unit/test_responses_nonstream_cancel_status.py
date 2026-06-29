"""a grounded Responses-API lifecycle hunt found the surface largely correct
(cross-tenant IDOR holds across retrieve/cancel/chain; chaining order/roles/tool-items/
cycle-bounds sound). The one HIGH: the NON-streaming/background path derived the response
status purely from last_finish_reason, but the engine reports finish_reason="stop" on
cancel (not a distinct reason) — so a POST /v1/responses/{id}/cancel on a background
response stored + polled it as "completed". The STREAMING path already checks
cancel_event.is_set() FIRST and reports "incomplete"; that fix was never propagated
to the non-stream path. Now the non-stream path checks _ns_cancel_event first → "incomplete".
"""

from __future__ import annotations

import inspect

from yunshu_gateway.routers import responses


def test_nonstream_status_checks_cancel_event_first():
    src = inspect.getsource(responses.create_response)
    # the non-stream status derivation must consult the cancel event BEFORE falling back
    # to the finish_reason map (which never sees a cancel reason).
    i = src.index("_response_status = (")
    window = src[max(0, i - 600) : i]
    assert "_ns_cancel_event" in window and ".is_set()" in window, (
        "non-stream status derivation must check _ns_cancel_event.is_set() before the "
        "finish_reason map"
    )
    # and on a set cancel event it resolves to incomplete (mirrors the streaming fix)
    assert "if _ns_cancel_event is not None and _ns_cancel_event.is_set():" in src
    j = src.index("if _ns_cancel_event is not None and _ns_cancel_event.is_set():")
    assert '"incomplete"' in src[j : j + 120]


def test_streaming_path_still_has_cancel_first_check_parity():
    # parity guard: the streaming path's cancel-first check must remain
    src = inspect.getsource(responses)
    assert "_cancel_evt is not None and _cancel_evt.is_set()" in src
