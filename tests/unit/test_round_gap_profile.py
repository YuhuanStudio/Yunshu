"""Timeline diagnostics must not replay live cache kernels or accept missing evidence."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from verify.stages import round_profile_evidence

from yunshu_engine import gpu_window


def test_timeline_never_captures_or_replays_live_kernels(monkeypatch):
    def forbidden(*args):
        pytest.fail("timeline attempted a destructive kernel replay")

    lib = SimpleNamespace(
        metalrt_profile=forbidden,
        metalrt_cap_begin=forbidden,
        metalrt_begin=lambda: None,
    )
    monkeypatch.setitem(gpu_window._STATE, "lib", lib)
    monkeypatch.setitem(gpu_window._STATE, "n", gpu_window.START - 2)
    monkeypatch.setitem(gpu_window._STATE, "marks", None)
    monkeypatch.setattr(gpu_window, "_log_kernel_profile", forbidden)
    gpu_window.tick(capture_kernels=False)
    gpu_window.tick(capture_kernels=False)
    assert gpu_window._STATE["marks"]


@pytest.mark.parametrize("mode", ["gaps", "stages"])
def test_missing_profile_evidence_fails_closed(mode):
    with pytest.raises(ValueError):
        round_profile_evidence("server ready; complete true", mode)


def test_stage_census_is_required():
    r = {"rounds": 40, "stages": {"decode_round": {"commit": 1.2}}}
    with pytest.raises(ValueError):
        round_profile_evidence("YUNSHU_LAYER2 " + json.dumps(r), "stages")
    r["dispatches_per_round"] = {"commit": 48}
    assert round_profile_evidence("YUNSHU_LAYER2 " + json.dumps(r), "stages")


def test_gap_and_host_numbers_remain_separate():
    text = (
        "GPU idle by host segment (ms/round, gaps > 20 us): {'mtp.commit': 0.1}\n"
        "Host time by segment (ms/round): {'mtp.commit': 0.7}"
    )
    r = round_profile_evidence(text, "gaps")
    assert r["idle_ms_per_round"] == [{"mtp.commit": 0.1}]
    assert r["host_ms_per_round"] == [{"mtp.commit": 0.7}]


def test_stage_builds_separate_nonreplay_timeline_and_census_cells(tmp_path):
    from verify.stages import stage_round_profile

    class StopError(Exception):
        pass

    cells = []

    class Executor:
        def run_cells(self, items):
            cells.extend(items)
            raise StopError

    ctx = SimpleNamespace(
        suite={"ctx": [1024], "kinds": ["prose"], "decode_tokens": 768},
        arm_env=lambda arm: {"YUNSHU_VLM_DRAFT": "mtp"},
        tree=lambda arm: SimpleNamespace(path=Path(__file__).resolve().parents[2]),
        run=SimpleNamespace(path=tmp_path),
        py="python",
        model="/model",
        mem_gb=64,
        exe=Executor(),
        observe=False,
    )
    with pytest.raises(StopError):
        stage_round_profile(ctx)
    assert len(cells) == 2
    for cell, mode in zip(cells, ["0", "1"], strict=True):
        argv = cell.argv
        assert "YUNSHU_DEBUG_TREE_PROFILE=" + mode in argv
        assert "YUNSHU_VLM_DRAFT=mtp" in argv
        assert "--decode-tokens" in argv and "768" in argv
        assert cell.quiet


def test_plain_decode_timeline_is_host_gap_evidence():
    """The draft-off arm times the single-row plain path, whose segments are named plain.*."""
    text = (
        "GPU idle by host segment (ms/round, gaps > 20 us): {'plain.readback': 0.1}\n"
        "Host time by segment (ms/round): {'plain.graph': 0.7}"
    )
    r = round_profile_evidence(text, "gaps")
    assert r["idle_ms_per_round"] == [{"plain.readback": 0.1}]


def test_round_profile_device_m3_routes_cells_to_laptop(tmp_path):
    from verify.stages import stage_round_profile

    class StopError(Exception):
        pass

    cells = []

    class Executor:
        def run_cells(self, items):
            cells.extend(items)
            raise StopError

    ctx = SimpleNamespace(
        suite={"ctx": [1024], "kinds": ["prose"], "decode_tokens": 768, "device": "m3"},
        arm_env=lambda arm: {"YUNSHU_VLM_DRAFT": "off"},
        tree=lambda arm: SimpleNamespace(path=Path(__file__).resolve().parents[2]),
        run=SimpleNamespace(path=tmp_path),
        py="python",
        model="/model",
        mem_gb=14,
        exe=Executor(),
        observe=False,
    )
    with pytest.raises(StopError):
        stage_round_profile(ctx)
    assert len(cells) == 2
    for cell in cells:
        assert cell.device == "m3"
        assert not cell.quiet  # gpuq refuses --quiet off the M5 lane
        assert cell.timeout_min == 30
        assert (
            "DYLD_INSERT_LIBRARIES=/Volumes/P5Plus/yunshu-build/metalrt/libmetalrt_trace2.dylib"
            in cell.argv
        )
        assert cell.argv[1] == "TFB_WORK=/Volumes/P5Plus/yunshu-build/tfnew"
