"""previous_response_id: unknown id is a 404, long chains are not silently truncated."""

from __future__ import annotations

import pytest

from yunshu_gateway.responses_context import ContextError
from yunshu_gateway.routers import responses as resp_mod


def _req(prev):
    return resp_mod.ResponsesRequest(model="m", input="hi", previous_response_id=prev)


def _store(monkeypatch, store):
    monkeypatch.setattr(resp_mod, "_get_stored_response", lambda i: store.get(i))
    monkeypatch.setattr(resp_mod, "_owns_stored", lambda request, p: True)


def test_unknown_previous_response_id_is_404(monkeypatch):
    _store(monkeypatch, {})
    with pytest.raises(ContextError) as ei:
        resp_mod._apply_previous_response_chain(_req("resp_nope"), [], None)
    assert ei.value.status == 404
    assert ei.value.code == "previous_response_not_found"


def test_long_chain_replays_all_hops(monkeypatch):
    n = 40
    store = {
        f"r{i}": {
            "previous_response_id": f"r{i - 1}" if i else None,
            "_input_messages": [{"role": "user", "content": f"u{i}"}],
            "output": [],
        }
        for i in range(n)
    }
    _store(monkeypatch, store)
    out = resp_mod._apply_previous_response_chain(_req(f"r{n - 1}"), [], None)
    assert [m["content"] for m in out] == [f"u{i}" for i in range(n)]


def test_chain_over_cap_errors_instead_of_truncating(monkeypatch):
    monkeypatch.setattr(resp_mod, "_CHAIN_MAX_HOPS", 3)
    store = {
        f"r{i}": {
            "previous_response_id": f"r{i - 1}" if i else None,
            "_input_messages": [],
            "output": [],
        }
        for i in range(6)
    }
    _store(monkeypatch, store)
    with pytest.raises(ContextError) as ei:
        resp_mod._apply_previous_response_chain(_req("r5"), [], None)
    assert ei.value.status == 400
