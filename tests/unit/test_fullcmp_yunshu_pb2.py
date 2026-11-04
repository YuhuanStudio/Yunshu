"""The Yunshu Phase B preview engine (yunshu-pb2, phaseb2-main 4f99270, unmerged) is modelled exactly on
dev0 ('yunshu-main'): same launch spec, same manifest cells, pins that resolve, evidence through the
Yunshu entries (fullcmp_engaged.EVIDENCE_ALIAS). Pure: reads the census, the pin files and the engaged
evidence table; builds launch specs; no server, no GPU, no gpuq.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import bench_engines as be  # noqa: E402
import fullcmp_attribution as fa  # noqa: E402
import fullcmp_audit_manifest as fam  # noqa: E402
import fullcmp_engaged as fe  # noqa: E402
import fullcmp_manifest as fm  # noqa: E402
import fullcmp_provenance as fp  # noqa: E402
import fullcmp_run as frun  # noqa: E402

PB2 = "yunshu-pb2"
DEV0 = "yunshu-main"  # the bench id of the dev0 row (manifest id 'yunshu')


def test_bench_engine_is_the_yunshu_kind_with_the_same_mode_and_checkpoint():
    pb2, dev0 = be.ENGINES[PB2], be.ENGINES[DEV0]
    assert pb2.kind == dev0.kind == "yunshu"
    assert pb2.expected_mode == dev0.expected_mode == "dflash"
    assert pb2.checkpoint == dev0.checkpoint and pb2.drafter == dev0.drafter
    assert pb2.product == "Yunshu Phase B preview (unmerged, 4f99270)"


def test_manifest_id_maps_both_ways():
    assert be.tfbench_engine(PB2) == PB2
    assert be.manifest_id(PB2) == PB2


def test_launch_spec_is_identical_to_dev0_for_the_same_home_and_env(tmp_path):
    env = {}
    home = str(tmp_path / "home")
    a = be.build_launch(DEV0, 18990, home, env)
    b = be.build_launch(PB2, 18990, home, env)
    assert a.cmd == b.cmd
    assert a.env == b.env
    assert a.files == b.files
    assert a.links == b.links


def test_manifest_has_the_engine_and_its_cells():
    assert PB2 in fm.ENGINE_IDS
    assert fm.ENGINES[PB2]["caps"] == fm.ENGINES["yunshu"]["caps"]
    rows, _meta = fm.build_rows(engines=[PB2])
    cells = [r for r in rows if r["engine"] == PB2 and r.get("batch") in ("B1", "B2")]
    assert cells, "the preview has no B1/B2 manifest cells"
    assert {r["engine"] for r in cells} == {PB2}


def test_pb2_pin_resolves_with_the_census_and_is_its_own_tree():
    import json

    census = json.loads(frun.RUN_DIR.joinpath("census.json").read_text())
    pin = json.loads(fp.PIN_PATH.read_text())
    pins = fam.build_pins(census, pin)
    p = pins[PB2]
    assert p["version"] == "0.1.7.dev0"
    assert p["release_date"] and p["binary_sha256"]
    assert p["install_path"] and "fullcmp-pb2" in p["install_path"]
    assert fam.WEIGHT_GROUP[PB2] == "qwen3.8-27b-4bit-jundot"
    assert fam.census_weights(census)[fam.WEIGHT_GROUP[PB2]] is not None


def test_evidence_alias_maps_the_preview_to_the_yunshu_entries():
    table = fe.load_table()
    assert table is not None
    assert fe.evidence_key(PB2) == "yunshu"
    assert fe.is_yunshu(PB2) and fe.is_yunshu(DEV0) and fe.is_yunshu("yunshu")
    assert fe.mode_entry(table, PB2, "dflash") == fe.mode_entry(table, "yunshu", "dflash")
    assert fe.mode_entry(table, PB2, "dflash") is not None


def test_attribution_schema_and_profile_rows_follow_the_yunshu_shape():
    assert fa.engine_attr(PB2).layer2 == "stages"
    assert fa.engine_attr(PB2).unavailable == fa.engine_attr("yunshu").unavailable


def test_preview_has_its_own_pin_file_and_the_dev0_tree_is_refused_for_it():
    assert frun.engine_pin(PB2, {})["sha"] == "4f99270cae4068019cfd80d02a177646fdfdb558"
    assert frun.engine_pin("yunshu", {}) is not None
    assert fp.pin_path_for(PB2) != fp.pin_path_for("yunshu")


def test_both_yunshu_pins_record_their_package_versions_and_differences_are_footnoted():
    import json as _json

    dev0 = _json.loads(fp.PIN_PATH.read_text())
    pb2 = _json.loads(fp.PB2_PIN_PATH.read_text())
    dev0_pkgs = dev0["packages"] if isinstance(dev0["packages"], dict) else _json.loads(dev0["packages"].replace("'", '"'))
    assert pb2["packages"]
    stack = ("mlx", "mlx-lm", "mlx-vlm")  # the MLX stack the cells run on
    differs = [k for k in stack if dev0_pkgs.get(k) != pb2["packages"].get(k)]
    if differs:
        assert pb2.get("package_note"), "a version difference between the two venvs needs a footnote"
    assert not differs, differs


def test_package_versions_are_read_from_each_pinned_venv_and_match():
    pb2 = json.loads(fp.PB2_PIN_PATH.read_text())
    dev0 = json.loads(fp.PIN_PATH.read_text())
    got_pb2 = fp.pinned_package_versions(pb2["python_executable"])
    got_dev0 = fp.pinned_package_versions(dev0["python_executable"])
    assert got_pb2 == got_dev0 == {"mlx": "0.32.3", "mlx-lm": "0.32.2", "mlx-vlm": "0.7.8"}
