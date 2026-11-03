"""Yunshu layer-1 attribution mapping (tfbench.attribution_fields + fullcmp_attribution).

Server stages come from x_yunshu.latency.milestones_ms (ms since gateway arrival) and the round
totals from x_yunshu.speculative. Fields Yunshu does not expose are explicit unavailable reasons,
never MISSING. CPU only: no server, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_attribution as fa  # noqa: E402
import tfbench  # noqa: E402


@pytest.fixture
def fullcmp_env(monkeypatch):
    monkeypatch.setenv("TFB_FULLCMP", "1")
    monkeypatch.setenv("TFB_ATTR_ENGINE", "yunshu")


class _Srv:
    engine = "yunshu"
    log = Path("/x/serve.log")


def _row(xy_latency=None, xy_spec=None, t_receive=1000.0):
    return {
        "_attr_client": {
            "t_receive": t_receive,
            "t_first_token": t_receive + 0.5,
            "t_last_token": t_receive + 2.0,
            "t_response_end": t_receive + 2.1,
            "cache_hit_tokens": 0,
        },
        "_xy_speculative": xy_spec,
        "_xy_latency": xy_latency,
    }


def test_yunshu_milestones_map_to_layer1_fields(fullcmp_env):
    lat = {
        "milestones_ms": {
            "gateway_receive": 0.0,
            "engine_admit": 12.5,
            "template_end": 3.0,
            "apc_end": 20.0,
            "prefill_start": 21.0,
        }
    }
    out = tfbench.attribution_fields(
        _Srv(), _row(lat, {"mode": "dflash", "drafted": 10, "accepted": 6, "rounds": 4})
    )
    attr = out["attr"]
    assert attr["t_template"] == pytest.approx(1000.0 + 0.003)
    assert attr["t_admit"] == pytest.approx(1000.0 + 0.0125)
    assert attr["t_cache_lookup"] == pytest.approx(1000.0 + 0.020)
    assert attr["rounds"] == 4 and attr["drafted"] == 10 and attr["accepted"] == 6
    assert attr["engine_log_path"] == "/x/serve.log"
    # Required fields are present or explicitly unavailable: nothing is MISSING.
    cov = fa.check_request("yunshu", out)
    assert "MISSING" not in cov.values(), cov


def test_yunshu_no_speculative_rounds_is_reasoned_not_missing(fullcmp_env):
    out = tfbench.attribution_fields(
        _Srv(),
        _row(
            {
                "milestones_ms": {
                    "template_end": 1.0,
                    "engine_admit": 2.0,
                    "apc_end": 3.0,
                }
            },
            None,
        ),
    )
    cov = fa.check_request("yunshu", out)
    assert cov["rounds"].startswith("unavailable ("), cov["rounds"]
    assert "MISSING" not in cov.values(), cov


def test_missing_milestone_gets_a_per_row_reason(fullcmp_env):
    # No APC lookup for this request (cache disabled): t_cache_lookup needs a reason, not a silence.
    out = tfbench.attribution_fields(
        _Srv(),
        _row({"milestones_ms": {"template_end": 1.0, "engine_admit": 2.0}}, None),
    )
    assert out["attr_unavailable"].get("t_cache_lookup"), out["attr_unavailable"]
    cov = fa.check_request("yunshu", out)
    assert "MISSING" not in cov.values(), cov


def test_copy_rounds_recorded_from_speculative(fullcmp_env):
    spec = {
        "mode": "dflash",
        "rounds": 5,
        "drafted": 9,
        "accepted": 4,
        "copy": {"rounds": 2, "tokens": 6},
    }
    out = tfbench.attribution_fields(_Srv(), _row({"milestones_ms": {}}, spec))
    assert out["attr"]["copy_rounds"] == 2


def test_no_fullcmp_mode_leaves_schema_untouched(monkeypatch):
    monkeypatch.delenv("TFB_FULLCMP", raising=False)
    out = tfbench.attribution_fields(_Srv(), _row(None, None))
    assert "t_template" not in out["attr"]
    assert out["attr_unavailable"] == {}
