"""Responses cancel-during-queued, Gemma-4 system turn boundary, explicit
cache per-owner eviction.

(HIGH): a cancel for a BACKGROUND Responses generation arriving during the `queued`
  window (before _runner registered with the request tracker) was silently dropped — the
  request ran to completion. Persist a cancel marker + have _runner bail before generating.
the Gemma-4 adapter left prev_role unchanged after a system message, so
  [user, system, user] merged the two user turns into one (silent structure loss).
the explicit context cache enforced max_entries GLOBALLY, so one tenant could evict
  another tenant's still-valid handles. Enforce the cap per-owner.
"""

from __future__ import annotations

import inspect


def test_cancel_persists_and_runner_bails():
    from yunshu_gateway.routers import responses

    src = inspect.getsource(responses)
    # cancel endpoint persists status=cancelled when the tracker had no entry
    assert 'stored.get("status") in ("queued", "in_progress")' in src
    assert '_cancel_payload["status"] = "cancelled"' in src
    # _runner bails on a persisted cancel before generating
    runner_region = src[
        src.index("async def _runner") : src.index("async def _runner") + 900
    ]
    assert 'cur.get("status") == "cancelled"' in runner_region
    assert "return" in runner_region


def test_gemma4_system_is_turn_boundary():
    from yunshu_engine.message_adapter import Gemma4MessageAdapter

    out = Gemma4MessageAdapter().adapt(
        [
            {"role": "user", "content": "first"},
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "second"},
        ]
    )
    roles = [m["role"] for m in out]
    assert roles.count("user") == 2, f"user turns merged across system: {roles}"
    # the second user turn is its own message, not concatenated into the first
    assert any(m["content"] == "second" for m in out)


def test_explicit_cache_per_owner_eviction():
    from yunshu_gateway.explicit_cache import ExplicitContextCache

    c = ExplicitContextCache(max_entries=3)
    for _ in range(3):
        c.create("m", [{"role": "user", "content": "a"}], 1, owner="A")
    for _ in range(6):
        c.create("m", [{"role": "user", "content": "b"}], 1, owner="B")
    owners = [e.owner for e in c._entries.values()]
    # tenant A's 3 valid handles survive a flood from tenant B (no cross-tenant eviction)
    assert owners.count("A") == 3
    assert owners.count("B") <= 4  # B's own bucket is bounded (lazy cap)
