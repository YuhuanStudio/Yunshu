"""console_contracts: the response models behind the console's /yunshu/* routes."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from yunshu_gateway import console_contracts as cc


def _entry(**kw):
    base = {
        "model": "m",
        "id": "e1",
        "namespace": "ns",
        "tokens": 10,
        "bytes_logical": None,
        "bytes_physical": None,
        "tier": "ram",
        "last_hit": None,
        "hits": 0,
    }
    return base | kw


def test_cache_entry_allows_null_sizes_and_optional_device():
    e = cc.CacheEntry(**_entry())
    assert e.device is None and e.bytes_physical is None
    assert cc.CacheEntry(**_entry(device="gpu0", bytes_physical=5)).bytes_physical == 5


def test_cache_entry_sizes_are_nullable_but_the_keys_are_required():
    bad = _entry()
    del bad["bytes_physical"]
    with pytest.raises(ValidationError):
        cc.CacheEntry(**bad)


def test_cache_entry_requires_its_identity_fields():
    bad = _entry()
    del bad["tokens"]
    with pytest.raises(ValidationError):
        cc.CacheEntry(**bad)


def test_bytes_physical_documents_that_it_is_attributed_not_allocator_bytes():
    prop = cc.CacheEntry.model_json_schema()["properties"]["bytes_physical"]
    assert "not allocator footprint" in prop["description"]


def test_cache_view_nests_entries_and_events():
    event = {
        "model": "m",
        "id": 1,
        "t": 1.5,
        "action": "evict",
        "entry_id": "e1",
        "tokens": None,
        "tier": "ram",
        "reason": "pressure",
        "request_id": None,
    }
    view = cc.CacheView(
        object="list",
        data=[_entry()],
        count=1,
        events=[event],
        event_capacity=100,
        models=["m"],
        scope="process",
    )
    assert view.events[0].action == "evict"
    assert view.model_dump()["data"][0]["id"] == "e1"


def test_request_metadata_only_requires_the_request_identity():
    nulls = dict.fromkeys(
        (
            "t",
            "model",
            "path",
            "status",
            "prompt_tokens",
            "completion_tokens",
            "cached_tokens",
            "prefill_tps",
            "decode_tps",
            "ttft_ms",
            "latency",
            "energy",
            "speculative",
            "cache",
        )
    )
    row = cc.RequestMetadata(request_id="r1", stream=False, **nulls)
    assert row.structured_output is None and row.reasons is None
    with pytest.raises(ValidationError):
        cc.RequestMetadata(stream=False, **nulls)  # no request_id


def test_history_page_cursor_may_be_null():
    page = cc.HistoryPage(
        object="list", enabled=False, data=[], count=0, next_cursor=None
    )
    assert page.model_dump()["next_cursor"] is None


def test_model_impact_is_a_nested_contract():
    impact = cc.ModelImpact(
        object="model_impact",
        model="m",
        unload={"in_flight_policy": "wait", "waits": True, "interrupts": False},
        load={
            "would_evict": ["old"],
            "blocked": False,
            "advisory": True,
            "post_load_pressure": "low",
        },
    )
    assert impact.load.would_evict == ["old"]
    with pytest.raises(ValidationError):
        cc.ModelImpact(object="x", model="m", unload={}, load={})


def test_spec_aggregate_depths_allow_unknown_acceptance_rate():
    agg = cc.SpecAggregate(
        object="spec",
        scope="process",
        position_basis="draft",
        data=[
            {
                "mode": "mtp",
                "num_drafts": 2,
                "num_draft_tokens": 8,
                "num_accepted_tokens": 5,
                "per_depth": [
                    {
                        "position": 0,
                        "drafted": 2,
                        "accepted": 2,
                        "acceptance_rate": 1.0,
                    },
                    {
                        "position": 1,
                        "drafted": 0,
                        "accepted": 0,
                        "acceptance_rate": None,
                    },
                ],
            }
        ],
    )
    assert agg.data[0].per_depth[1].acceptance_rate is None


def test_the_console_routes_publish_these_models():
    """The routes the console reads declare these contracts as their response model."""
    from yunshu_gateway.routers.yunshu import router

    declared = {
        route.path: route.response_model
        for route in router.routes
        if getattr(route, "response_model", None) is not None
    }
    assert declared["/yunshu/requests/history"] is cc.HistoryPage
    assert declared["/yunshu/spec-decode"] is cc.SpecAggregate
    assert declared["/yunshu/bundle/manifest"] is cc.BundleManifest
    assert declared["/yunshu/models/impact"] is cc.ModelImpact
    assert declared["/yunshu/bundle"] is cc.DiagnosticsBundle
    assert declared["/yunshu/cache"] is cc.CacheView
    assert declared["/yunshu/cache/clear"] is cc.CacheClear
