"""The comparison must fail if isolated HOME hides the requested drafter."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "tfbench", Path(__file__).resolve().parents[2] / "scripts/research/tfbench.py"
)
tfbench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tfbench)


def test_yunshu_passes_drafter_explicitly():
    original = {}
    mode, env = tfbench.spec_request("yunshu", original)
    assert mode == "dflash"
    assert env["YUNSHU_VLM_DRAFT"] == tfbench.D
    assert original == {}


@pytest.mark.parametrize(
    "override,mode", [("mtp", "mtp"), ("off", "off"), ("/draft", "dflash")]
)
def test_explicit_ab_arm_is_preserved(override, mode):
    actual, env = tfbench.spec_request("yunshu", {"YUNSHU_VLM_DRAFT": override})
    assert actual == mode
    assert env["YUNSHU_VLM_DRAFT"] == override


@pytest.mark.parametrize(
    "log",
    [
        "",
        "Speculative decoding: dflash (automatic)",
        "Speculative decoding: dflash (automatic)\nVLM batch runner: apc=off draft=mtp block=6",
    ],
)
def test_dflash_fails_closed_on_missing_or_fallback_runner(tmp_path, log):
    server = object.__new__(tfbench.Srv)
    server.engine = "yunshu"
    server.requested_spec_mode = "dflash"
    server.log = tmp_path / "server.log"
    server.log.write_text(log)
    with pytest.raises(RuntimeError, match="requested spec=dflash"):
        server.verify_spec_mode()


def test_records_actual_runner_mode(tmp_path):
    server = object.__new__(tfbench.Srv)
    server.engine = "yunshu"
    server.requested_spec_mode = "dflash"
    server.log = tmp_path / "server.log"
    server.log.write_text("VLM batch runner: apc=32.0GiB draft=dflash block=8")
    server.verify_spec_mode()
    assert server.engaged_spec_mode == "dflash"
