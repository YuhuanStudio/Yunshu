"""Per-engine Yunshu pins: dev0 ('yunshu' / 'yunshu-main') and the Phase B preview ('yunshu-pb2').
A cell validates against its own engine's pin only. Pure: reads the pin files, no GPU, no server.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_provenance as fp  # noqa: E402
import fullcmp_run as frun  # noqa: E402


def _dev0_pin():
    return json.loads(fp.PIN_PATH.read_text())


def test_default_and_dev0_ids_resolve_to_the_dev0_pin():
    assert fp.pin_path_for() == fp.PIN_PATH
    assert fp.pin_path_for("yunshu") == fp.PIN_PATH
    assert fp.pin_path_for("yunshu-main") == fp.PIN_PATH


def test_phase_b_preview_has_its_own_pin():
    assert fp.pin_path_for("yunshu-pb2") == fp.PB2_PIN_PATH
    assert fp.PB2_PIN_PATH != fp.PIN_PATH


def test_unknown_yunshu_like_engine_fails_closed():
    with pytest.raises(fp.ProvenanceError):
        fp.pin_path_for("yunshu-pb9")


def test_pb2_cell_validates_against_the_pb2_tree_and_venv():
    pin = json.loads(fp.PB2_PIN_PATH.read_text())
    env = {
        "TFB_YUNSHU_BIN": pin["launch"]["harness_env"]["TFB_YUNSHU_BIN"],
        "TFB_YUNSHU_SRC": pin["launch"]["harness_env"]["TFB_YUNSHU_SRC"],
    }
    got = fp.require_yunshu_pin(env, engine="yunshu-pb2")
    assert got["engine"] == "yunshu-pb2"
    assert got["git_sha"] == "4f99270cae4068019cfd80d02a177646fdfdb558"


def test_dev0_tree_is_refused_for_the_pb2_engine():
    dev0 = _dev0_pin()
    env = {
        "TFB_YUNSHU_BIN": dev0["launch"]["harness_env"]["TFB_YUNSHU_BIN"],
        "TFB_YUNSHU_SRC": dev0["launch"]["harness_env"]["TFB_YUNSHU_SRC"],
    }
    with pytest.raises(fp.ProvenanceError):
        fp.require_yunshu_pin(env, engine="yunshu-pb2")


def test_dev0_cell_still_validates_against_dev0():
    dev0 = _dev0_pin()
    env = {
        "TFB_YUNSHU_BIN": dev0["launch"]["harness_env"]["TFB_YUNSHU_BIN"],
        "TFB_YUNSHU_SRC": dev0["launch"]["harness_env"]["TFB_YUNSHU_SRC"],
    }
    got = fp.require_yunshu_pin(env)
    assert got["engine"] == "yunshu" and got["git_sha"] == dev0["sha"]


def test_engine_pin_routes_the_env_per_engine():
    dev0 = _dev0_pin()
    assert frun.engine_pin("yunshu", dev0) is dev0
    pb2 = frun.engine_pin("yunshu-pb2", dev0)
    assert "wt-fullcmp-yunshu-pb2" in pb2["launch"]["harness_env"]["TFB_YUNSHU_SRC"]
    assert "wt-fullcmp-yunshu/" not in pb2["launch"]["harness_env"]["TFB_YUNSHU_SRC"]


def test_cell_env_carries_the_engines_own_tree():
    dev0 = _dev0_pin()
    dev0_env = frun.cell_env("yunshu", dev0, group="g")
    pb2_env = frun.cell_env("yunshu-pb2", dev0, group="g")
    assert any(x.startswith("TFB_YUNSHU_SRC=") and "wt-fullcmp-yunshu/" in x for x in dev0_env)
    assert any(x.startswith("TFB_YUNSHU_SRC=") and "wt-fullcmp-yunshu-pb2" in x for x in pb2_env)
    assert not any("wt-fullcmp-yunshu-pb2" in x for x in dev0_env)
