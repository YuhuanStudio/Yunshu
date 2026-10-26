"""metadata_projection.extras: the whitelist between serve-log rows and the console."""

from __future__ import annotations

import math

from yunshu_gateway.metadata_projection import extras

BLOCKS = ("cache", "reasons", "speculative", "structured_output", "latency")


def test_empty_row_projects_to_null_blocks_and_no_energy():
    out = extras({})
    for key in BLOCKS:
        assert out[key] is None
    assert out["energy"] is None


def test_non_dict_blocks_are_dropped_not_trusted():
    out = extras({k: "free text" for k in BLOCKS} | {"energy": ["x"]})
    for key in BLOCKS:
        assert out[key] is None
    assert out["energy"] is None


def test_unknown_keys_and_free_text_never_pass_through():
    raw = {
        "prompt": "secret user text",
        "cache": {
            "tier": "ram",
            "cached_tokens": 128,
            "reload_ms": 1.5,
            "path": "/Users/me/.yunshu/cache/apc",
            "note": "free text with spaces",
        },
        "reasons": {"cache": "hit", "spec": "this is a sentence, not a label"},
    }
    out = extras(raw)
    assert out["cache"] == {"tier": "ram", "cached_tokens": 128, "reload_ms": 1.5}
    assert out["reasons"] == {"cache": "hit", "spec": None}
    assert "secret" not in repr(out) and "/Users" not in repr(out)


def test_cache_device_is_kept_only_when_present_and_label_shaped():
    assert "device" not in extras({"cache": {"tier": "ssd"}})["cache"]
    assert extras({"cache": {"device": "apple/m5"}})["cache"]["device"] == "apple/m5"
    assert extras({"cache": {"device": "has space"}})["cache"]["device"] is None


def test_numbers_must_be_finite_real_numbers():
    spec = {
        "mode": "mtp",
        "drafted": 10,
        "accepted": True,  # bool is not a count
        "acceptance_rate": float("nan"),
        "rounds": "3",
    }
    out = extras({"speculative": spec})["speculative"]
    assert out["mode"] == "mtp"
    assert out["drafted"] == 10
    assert out["accepted"] is None
    assert out["acceptance_rate"] is None
    assert out["rounds"] is None
    assert not any(isinstance(v, float) and math.isnan(v) for v in out.values())


def test_per_depth_rows_are_filtered_and_capped():
    rows = [
        {"position": i, "drafted": 4, "accepted": 2, "acceptance_rate": 0.5, "x": 1}
        for i in range(200)
    ]
    rows.insert(0, "not a row")
    depth = extras({"speculative": {"per_depth": rows}})["speculative"]["per_depth"]
    # the cap applies to the first 128 raw rows, before non-dict rows are dropped
    assert len(depth) == 127
    assert set(depth[0]) == {"position", "drafted", "accepted", "acceptance_rate"}
    assert (
        extras({"speculative": {"per_depth": "oops"}})["speculative"]["per_depth"] == []
    )


def test_copy_block_only_appears_when_the_row_has_one():
    assert "copy" not in extras({"speculative": {}})["speculative"]
    out = extras({"speculative": {"copy": {"rounds": 3, "tokens": 40, "extra": 9}}})
    assert out["speculative"]["copy"] == {"rounds": 3, "tokens": 40}


def test_structured_output_flags_are_strict_booleans():
    out = extras(
        {
            "structured_output": {
                "requested": True,
                "enforced": 1,  # truthy but not True
                "engine": "llguidance",
                "grammar_backend": "x y",
            }
        }
    )["structured_output"]
    assert out["requested"] is True and out["enforced"] is False
    assert out["engine"] == "llguidance"
    assert out["grammar_backend"] is None


def test_latency_keeps_only_known_stage_names():
    latency = {
        "clock": "monotonic",
        "flush_boundary": "sse",
        "milestones_ms": {"prefill_start": 12.5, "mystery": 1.0},
        "durations_ms": {"prefill": 80, "template_tokenize": "slow"},
    }
    out = extras({"latency": latency})["latency"]
    assert out["milestones_ms"] == {"prefill_start": 12.5}
    assert out["durations_ms"] == {"prefill": 80, "template_tokenize": None}
    assert out["clock"] == "monotonic"
    assert extras({"latency": {"milestones_ms": "x"}})["latency"]["milestones_ms"] == {}


def test_energy_receipts_whitelist_reasons_and_flags():
    raw = {
        "energy": {
            "schema": "v1",
            "method": "powermetrics",
            "includes_other_processes": True,
            "overlapping_requests_share_host_energy": "yes",
            "prefill": {
                "state": "ok",
                "reason": "phase timing unavailable",
                "joules": 12.0,
                "ignored": 1,
            },
            "decode": {"state": "partial", "reason": "arbitrary text"},
        }
    }
    energy = extras(raw)["energy"]
    assert energy["includes_other_processes"] is True
    assert energy["overlapping_requests_share_host_energy"] is False
    assert energy["prefill"]["reason"] == "phase timing unavailable"
    assert energy["prefill"]["joules"] == 12.0
    assert "ignored" not in energy["prefill"]
    assert energy["decode"]["reason"] is None
    assert energy["decode"]["joules"] is None
