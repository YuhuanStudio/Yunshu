"""Every endpoint the installed openai / anthropic SDKs know is implemented, or declared planned /
not applicable with a reason (scripts/dev/api_coverage_na.json). Decisions was missed because
nobody walked the SDKs; this test is that walk."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/dev"))
import api_coverage as ac  # noqa: E402

pytest.importorskip("openai")
pytest.importorskip("anthropic")


@pytest.fixture(scope="module")
def world():
    endpoints = []
    for sdk in ac.SDKS:
        eps, _, _ = ac.walk_sdk(sdk)
        endpoints += eps
    return endpoints, ac.implemented_routes(), ac.load_declarations()


def test_every_sdk_endpoint_is_implemented_or_declared(world):
    endpoints, routes, decls = world
    rows, _ = ac.classify(endpoints, routes, decls)
    bad = [e.key for e, state, _ in rows if state == "UNCLASSIFIED"]
    assert not bad, (
        "SDK endpoints that are neither served nor declared in scripts/dev/api_coverage_na.json "
        "(add a route, or declare not_applicable / planned with a reason):\n  "
        + "\n  ".join(bad)
    )


def test_declarations_are_not_stale(world):
    endpoints, routes, decls = world
    _, used = ac.classify(endpoints, routes, decls)
    unused = [d["match"] for i, d in enumerate(decls) if i not in used]
    assert not unused, f"declarations that match no SDK endpoint any more: {unused}"
    served_but_declared = ac.contradictions(endpoints, routes, decls)
    assert not served_but_declared, (
        "implemented routes still declared planned / not applicable: "
        f"{[(e.key, m) for e, m in served_but_declared]}"
    )


def test_declarations_carry_real_reasons():
    for d in ac.load_declarations():
        assert d["status"] in ("not_applicable", "planned")
        assert len(d["reason"]) >= 40, d["match"]


def test_the_walk_finds_decisions_and_the_endpoints_we_serve(world):
    endpoints, routes, _ = world
    by_key = {e.key: e for e in endpoints}
    for key in (
        "openai POST /decisions",
        "openai POST /responses",
        "openai GET /responses/{}/input_items",
        "openai WS /responses",
        "openai WS /realtime",
        "anthropic POST /v1/messages",
        "anthropic GET /v1/messages/batches/{}/results",
    ):
        assert key in by_key, f"the SDK walk no longer finds {key}"
        assert ac.served(by_key[key], routes), f"{key} is not served"
    assert len(endpoints) > 400  # the walk covers the beta and admin surfaces too


def test_an_unknown_endpoint_is_flagged():
    new = ac.Endpoint("openai", "POST", "/brand_new_resource")
    rows, _ = ac.classify([new], set(), ac.load_declarations())
    assert rows[0][1] == "UNCLASSIFIED"
    served = ac.Endpoint("openai", "POST", "/decisions")
    rows, _ = ac.classify([served], {"POST /v1/decisions"}, [])
    assert rows[0][1] == "implemented"
