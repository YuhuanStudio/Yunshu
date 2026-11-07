"""Attribution schema: required fields, explicit unavailability, fail-closed MISSING."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_attribution as fa  # noqa: E402


def _full_yunshu_row() -> dict:
    vals = {f: 1.0 for f in fa.required_layer1("yunshu")}
    vals["t_prefill_chunks"] = [[0.0, 0.1, 64]]
    vals["t_rounds"] = [0.2]
    vals["tokens_per_round"] = [2]
    vals["rows_per_round"] = [1]
    vals["chunk_sizes"] = [64]
    vals["host_gaps_ms"] = [0.5]
    vals["engine_log_path"] = "/x/serve.log"
    return {"attr": vals}


def _yunshu_profile() -> dict:
    stages = {}
    for field in fa.required_layer2("yunshu"):
        if field.startswith("stages."):
            node = stages
            parts = field.split(".")[1:]
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = 1.0
    stages.setdefault("decode_round", {})
    return {"stages": stages, "host_sync_count": 3, "op_census": {"matmul": 4}}


def test_every_engine_has_a_schema_and_unknown_engine_fails():
    for eng in fa.ENGINE_OVERRIDES:
        assert fa.engine_attr(eng).engine == eng
    with pytest.raises(KeyError):
        fa.engine_attr("not-an-engine")


def test_yunshu_full_row_passes():
    cov = fa.check_request("yunshu", _full_yunshu_row())
    assert "MISSING" not in cov.values()
    res = fa.check_cell("yunshu", [_full_yunshu_row()], _yunshu_profile())
    assert res["ok"], res["missing"]


def test_missing_required_field_fails_closed():
    row = _full_yunshu_row()
    del row["attr"]["t_first_token"]
    res = fa.check_cell("yunshu", [row], _yunshu_profile())
    assert not res["ok"]
    assert "t_first_token" in res["missing"]


def test_none_value_counts_as_missing():
    row = _full_yunshu_row()
    row["attr"]["t_admit"] = None
    assert fa.check_request("yunshu", row)["t_admit"] == "MISSING"


def test_no_rows_is_failure_not_success():
    res = fa.check_cell("yunshu", [], _yunshu_profile())
    assert not res["ok"]


def test_profile_missing_fails_for_yunshu():
    res = fa.check_cell("yunshu", [_full_yunshu_row()], None)
    assert not res["ok"]
    assert any(m.startswith("profile:") for m in res["missing"])


def test_competitor_server_stages_need_explicit_reason():
    # Client-side fields only, server-side ones are explicit unavailability.
    row = {
        "attr": {
            "t_receive": 1.0,
            "t_first_token": 1.2,
            "t_last_token": 1.9,
            "t_response_end": 2.0,
            "cache_hit_tokens": 0,
            "engine_log_path": None,
        }
    }
    cov = fa.check_request("tf-1.2.1", row)
    assert cov["t_receive"] == "present"
    assert cov["t_template"].startswith("unavailable (")
    assert cov["t_first_token"] == "present"
    assert cov["engine_log_path"] == "MISSING"


def test_competitor_row_with_no_reason_is_missing():
    # A field the engine does expose (not in its unavailable map) and that is absent is MISSING.
    cov = fa.check_request("yunshu", {"attr": {}})
    assert cov["t_receive"] == "MISSING"


def test_row_level_reason_is_accepted_for_optional_engine_field():
    row = {
        "attr": {},
        "attr_unavailable": {"t_admit": "queue not instrumented in this build"},
    }
    cov = fa.check_request("mlx-lm-server", row)
    assert cov["t_admit"].startswith("unavailable (")


def test_competitor_profile_is_stats_as_is():
    assert fa.required_layer2("omlx-0.7.3") == ["competitor_stats"]
    assert (
        fa.check_profile("omlx-0.7.3", {"competitor_stats": ["TIMING x"]})[
            "competitor_stats"
        ]
        == "present"
    )
    assert fa.check_profile("omlx-0.7.3", {})["competitor_stats"] == "MISSING"


def test_llama_cpp_marks_spec_counters_unavailable():
    # Optional spec counters are not checked for a non-spec engine, the reason lives in the schema.
    cov = fa.check_request("llama-cpp-b12061", {"attr": {}})
    assert cov["drafted"].startswith("unavailable (")
