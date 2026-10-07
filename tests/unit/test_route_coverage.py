"""Route coverage gate: every route the gateway registers has a real-server check or a reasoned
exemption (scripts/research/route_checks.py). Adding a route without one fails here.

The checks themselves run against real servers (`scripts/dev/m3sweep --only routes`); this file
only guards the registry, so it needs no model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))

import route_checks as rc  # noqa: E402


def enumerate_routes(app) -> set[str]:
    """Every (method, path) the app serves, websockets included, as 'METHOD /path'.

    FastAPI 0.13x keeps included routers as lazy `_IncludedRouter` nodes, older versions flatten
    them; both are walked. HEAD (implied by GET) and OPTIONS are not separate routes."""

    def walk(routes):
        for r in routes:
            if hasattr(r, "effective_candidates"):
                yield from walk(r.effective_candidates())
                continue
            orig = getattr(r, "original_route", r)
            path = getattr(r, "path", "") or getattr(orig, "path", "")
            if "WebSocket" in type(orig).__name__:
                yield "WS " + path
            else:
                for m in getattr(orig, "methods", None) or ():
                    if m not in ("HEAD", "OPTIONS"):
                        yield f"{m} {path}"

    return set(walk(app.routes))


@pytest.fixture(scope="module")
def app_routes():
    from yunshu_gateway.main import create_app

    return enumerate_routes(create_app())


def test_enumeration_sees_http_and_websocket_routes(app_routes):
    assert len(app_routes) > 80
    assert "POST /v1/chat/completions" in app_routes
    assert "WS /v1/realtime" in app_routes
    assert "GET /openapi.json" in app_routes


def test_every_route_has_a_real_check_or_an_exemption(app_routes):
    known = rc.served_routes() | set(rc.EXEMPT)
    missing = sorted(app_routes - known)
    assert not missing, (
        "routes without a SERVED real-server check (a model with the capability, a successful, validated answer; "
        "an absent-capability or error-path check does not count). Add a served @check in scripts/research/route_checks*.py, "
        f"or an EXEMPT entry with a reason): {missing}"
    )


def test_registry_has_no_stale_route(app_routes):
    stale = sorted((rc.checked_routes() | set(rc.EXEMPT)) - app_routes)
    assert not stale, f"registry names routes the app does not register: {stale}"


def test_exemptions_are_reasoned_and_not_also_served():
    for route, why in rc.EXEMPT.items():
        assert len(why.split()) >= 4, f"exemption {route} needs a real reason"
        assert route not in rc.served_routes(), (
            f"{route} has a served check and is exempt"
        )


def test_checks_are_well_formed():
    assert rc.REGISTRY
    for name, chk in rc.REGISTRY.items():
        assert chk.name == name and callable(chk.fn)
        assert chk.needs in (
            "main",
            "multi",
            "tts",
            "asr",
            "ocr",
            "image",
            "embed",
            "embed2",
            "rerank",
            "classifier",
            "translate",
            "omni",
            "cascade",
            "native",
        )
        assert chk.routes, f"{name} covers no route"
        for r in chk.routes:
            method, _, path = r.partition(" ")
            assert method in ("GET", "POST", "DELETE", "PUT", "PATCH", "WS"), r
            assert path.startswith("/"), r


def test_exemption_list_stays_tiny():
    """Every exemption is reviewed by the lead; a long list means the gate is being talked around."""
    assert len(rc.EXEMPT) <= 8, sorted(rc.EXEMPT)


def test_every_check_declares_served_and_unserved_ones_are_error_paths():
    for name, chk in rc.REGISTRY.items():
        assert isinstance(chk.served, bool), name
    unserved = {n for n, c in rc.REGISTRY.items() if not c.served}
    assert {"audio_absent", "images_absent", "omni_absent", "embeddings"} <= unserved
    assert rc.REGISTRY["tts_served"].served and rc.REGISTRY["files"].served
