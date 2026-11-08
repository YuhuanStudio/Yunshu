"""The pb2-generic candidate (yunshu-next, detached tree 89740a60f) is modelled exactly on yunshu-pb2:
own pin, own tree and venv, same checkpoint and flags, evidence read through the Yunshu entries, and a
small re-run manifest (NEXT_CELLS) whose ids are the yunshu cells of the real run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import bench_engines as be  # noqa: E402
import fullcmp_engaged as fe  # noqa: E402
import fullcmp_manifest as fm  # noqa: E402
import fullcmp_provenance as fp  # noqa: E402
import fullcmp_run as frun  # noqa: E402

NEXT = "yunshu-next"


def test_engine_is_pb2_with_its_own_tree():
    n, p = be.ENGINES[NEXT], be.ENGINES["yunshu-pb2"]
    assert n.kind == p.kind == "yunshu"
    assert (n.expected_mode, n.checkpoint, n.drafter) == (
        p.expected_mode,
        p.checkpoint,
        p.drafter,
    )
    assert be.MANIFEST_ENGINES[NEXT] == NEXT
    assert be.manifest_id(NEXT) == NEXT


def test_pin_path_and_fail_closed_ids():
    assert fp.pin_path_for(NEXT) == fp.NEXT_PIN_PATH != fp.PB2_PIN_PATH
    assert fp.census_key_for(NEXT) == "yunshu"
    with pytest.raises(fp.ProvenanceError):
        fp.pin_path_for("yunshu-next2")


@pytest.mark.skipif(
    not fp.NEXT_PIN_PATH.is_file(), reason="pin file not on this machine"
)
def test_pin_is_the_detached_tree_and_the_pb2_tree_is_refused():
    pin = json.loads(fp.NEXT_PIN_PATH.read_text())
    assert pin["sha"].startswith("89740a60f")
    assert "wt-fullcmp-next-engine" in pin["worktree"] and pin["venv"].endswith(
        "venv-fullcmp-next"
    )
    env = pin["launch"]["harness_env"]
    got = fp.require_yunshu_pin(
        {
            "TFB_YUNSHU_BIN": env["TFB_YUNSHU_BIN"],
            "TFB_YUNSHU_SRC": env["TFB_YUNSHU_SRC"],
        },
        engine=NEXT,
    )
    assert got["git_sha"] == pin["sha"]
    pb2 = json.loads(fp.PB2_PIN_PATH.read_text())["launch"]["harness_env"]
    with pytest.raises(fp.ProvenanceError):
        fp.require_yunshu_pin(
            {
                "TFB_YUNSHU_BIN": pb2["TFB_YUNSHU_BIN"],
                "TFB_YUNSHU_SRC": pb2["TFB_YUNSHU_SRC"],
            },
            engine=NEXT,
        )


def test_run_pins_and_evidence_alias():
    assert (
        set(frun.yunshu_pins()) == {"yunshu-pb2", "yunshu-next"}
        if fp.NEXT_PIN_PATH.is_file()
        else True
    )
    assert fe.EVIDENCE_ALIAS[NEXT] == "yunshu"


def test_next_cells_are_the_requested_24_unverified_yunshu_cells():
    assert len(fm.NEXT_CELLS) == len(set(fm.NEXT_CELLS)) == 24
    rows = fm.next_cell_rows()
    assert [r["id"] for r in rows] == [t.replace("@", NEXT) for t in fm.NEXT_CELLS]
    assert {r["engine"] for r in rows} == {NEXT} and all(
        r["status"] == fm.UNV and r["reps"] == 2 for r in rows
    )
    # every id is the real run's yunshu cell id with the engine swapped
    yun = {
        r["id"]
        for r in fm.build_rows()[0]
        if r["engine"] == "yunshu" and r["status"] == fm.UNV
    }
    assert {t.replace("@", "yunshu") for t in fm.NEXT_CELLS} <= yun
    assert NEXT in fm.EXTRA_ENGINES
