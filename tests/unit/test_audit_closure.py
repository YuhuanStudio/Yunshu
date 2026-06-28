"""the endpoint audit-closure tracker stays honest (no drift)."""
from __future__ import annotations

import importlib.util
import pathlib

_SPEC = importlib.util.spec_from_file_location(
    "audit_closure",
    pathlib.Path(__file__).resolve().parent.parent.parent / "scripts" / "audit_closure.py",
)
audit_closure = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit_closure)


def test_discovers_all_endpoints():
    rows = audit_closure.discover()
    # every gateway router endpoint is found (sanity floor; grows as routes are added)
    assert len(rows) >= 100
    # each row is well-formed
    for stem, method, path, _fn in rows:
        assert stem and method and path.startswith("/")


def test_no_stale_status_entries():
    """A STATUS entry with no matching endpoint means the tracker has drifted."""
    rows = audit_closure.discover()
    keys = {f"{s}:{m}:{p}" for s, m, p, _ in rows}
    stale = set(audit_closure.STATUS) - keys
    assert not stale, f"STATUS has entries for removed endpoints: {sorted(stale)}"


def test_inference_surface_is_closed():
    """The default-serving inference endpoints must all be clean/hardened (not todo)."""
    rows = {f"{s}:{m}:{p}": (s, m, p) for s, m, p, _ in audit_closure.discover()}
    must_be_closed = [
        "chat:POST:/chat/completions",
        "completions:POST:/completions",
        "responses:POST:/responses",
        "anthropic:POST:/messages",
        "embeddings:POST:/embeddings",
        "scoring:POST:/pooling",
        "scoring:POST:/score",
        "scoring:POST:/rerank",
        "scoring:POST:/classify",
    ]
    for key in must_be_closed:
        assert key in rows, f"expected endpoint missing: {key}"
        verdict, _ = audit_closure.STATUS.get(key, ("todo", ""))
        assert verdict in ("clean", "hardened"), f"{key} regressed to {verdict}"
