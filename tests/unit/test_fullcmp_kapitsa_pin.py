"""Kapitsa 0.2.31 runs from a read-only copy pinned by sha256, launched with auto-update off."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import bench_engines as be  # noqa: E402
import fullcmp_manifest as fm  # noqa: E402


def test_verify_pinned_binary_accepts_the_pin_and_refuses_anything_else(tmp_path):
    f = tmp_path / "kapitsa"
    f.write_bytes(b"binary-a")
    be.verify_pinned_binary(str(f), hashlib.sha256(b"binary-a").hexdigest())
    f.write_bytes(b"binary-b")  # replaced under the same name (a self-update)
    with pytest.raises(be.PinMismatch, match="is not the pin"):
        be.verify_pinned_binary(str(f), hashlib.sha256(b"binary-a").hexdigest())
    with pytest.raises(be.PinMismatch, match="missing"):
        be.verify_pinned_binary(str(tmp_path / "nope"), "0" * 64)


def test_launch_refuses_a_binary_that_is_not_the_pin(monkeypatch, tmp_path):
    monkeypatch.setattr(be, "KAPITSA_BIN_0231", str(tmp_path / "kapitsa"))
    (tmp_path / "kapitsa").write_bytes(b"x")
    with pytest.raises(be.PinMismatch):
        be.build_launch("kapitsa-0.2.31", 18990, tmp_path, {})


@pytest.mark.skipif(
    not Path(be.KAPITSA_BIN_0231).is_file(), reason="pinned copy not on this machine"
)
def test_launch_of_the_pinned_copy_turns_auto_update_off():
    launch = be.build_launch("kapitsa-0.2.31", 18990, "/tmp/h", {"CI": "false"})
    assert launch.cmd[0] == be.KAPITSA_BIN_0231 and "--private" in launch.cmd
    assert launch.env["KAPITSA_NO_AUTO_UPDATE"] == "1" and launch.env["CI"] == "true"
    # 0.2.26's launch is unchanged (history rows)
    old = be.build_launch("kapitsa-0.2.26", 18990, "/tmp/h", {})
    assert old.cmd[0] == be.KAPITSA_BIN_0226 and "KAPITSA_NO_AUTO_UPDATE" not in old.env


def test_pin_and_install_path_agree():
    import fullcmp_audit_manifest as fam

    assert fam.INSTALL["kapitsa-0.2.31"] == be.KAPITSA_BIN_0231
    assert len(be.KAPITSA_PIN_0231) == 64
    assert be.SUPERSEDED_BY == {"kapitsa-0.2.26": "kapitsa-0.2.31"}
    assert fm.ENGINES["kapitsa-0.2.26"]["superseded_by"] == "kapitsa-0.2.31"


def test_the_rerun_engine_mirrors_the_old_rows_and_is_outside_the_plan_total():
    rows, _ = fm.build_rows()
    import io

    fm.apply_cuts(rows, out=io.StringIO())
    old = {
        r["id"].replace("kapitsa-0.2.26", "@"): r
        for r in rows
        if r["engine"] == "kapitsa-0.2.26"
    }
    new = {
        r["id"].replace("kapitsa-0.2.31", "@"): r
        for r in rows
        if r["engine"] == "kapitsa-0.2.31"
    }
    assert new and set(new) == set(old)
    assert all(new[k]["status"] == old[k]["status"] for k in new)
    assert "kapitsa-0.2.31" in fm.EXTRA_ENGINES
