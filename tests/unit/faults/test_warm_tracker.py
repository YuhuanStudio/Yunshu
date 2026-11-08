"""Every request the harness's warm-up sends has left the request tracker by the time its reply is complete."""

from __future__ import annotations

import pytest

from .harness import (
    DIALECTS,
    FakeBatchedEngine,
    Script,
    asgi_call,
    request_for,
    running_app,
)


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("dialect", list(DIALECTS))
async def test_a_finished_request_is_gone_from_the_tracker(
    make_app, audit, dialect, stream
):
    eng = FakeBatchedEngine(Script.ok("w"))
    app = make_app(eng)
    path, body = request_for(dialect, stream)
    async with running_app(app, warm=False):
        a = audit(eng)
        r = await asgi_call(app, "POST", path, body)
        assert r.status == 200
        left = [(g.request_id, round(g.elapsed_s, 3)) for g in a.tracker.all_active()]
        assert a.tracker.active_count == a.active0, left
