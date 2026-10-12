"""CPU-only checks for the refsync Stage 2 planner (scripts/research/refsync_jobs.py)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import bench_engines as be  # noqa: E402
import bench_snapshot as bs  # noqa: E402
import pytest  # noqa: E402
import refsync_jobs as rj  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_snapshot_globals():
    saved = (bs.LABEL_PREFIX, bs.BUILD, dict(bs._STARTUP_S))
    yield
    bs.LABEL_PREFIX, bs.BUILD = saved[0], saved[1]
    bs._STARTUP_S.clear()
    bs._STARTUP_S.update(saved[2])


def test_engines_registered_and_launchable(tmp_path):
    rj.configure()
    for spec in rj.SETS.values():
        for e in spec["engines"]:
            assert e in be.ENGINES
            launch = be.build_launch(e, 18990, tmp_path, {}, 140000, 1)
            assert launch.cmd and launch.env["HOME"] == str(tmp_path)
            assert be.ENGINES[e].expected_mode in ("dflash", "mtp")
            assert be.engine_version(e, str(ROOT / "python"))[0] != ""


def test_cells_are_ten_prompts_and_labelled(tmp_path):
    rj.configure()
    jobs = rj.cells("27b", rj.SETS["27b"]["engines"], 1, tmp_path)
    assert len(jobs) == 3 * len(rj.SETS["27b"]["engines"])
    for j in jobs:
        assert j.label.startswith("refsync-")
        files = j.argv[j.argv.index("--prompt-files") + 1].split(",")
        assert len(files) == 10
        assert all(f.endswith(".txt") for f in files)
        assert j.argv[j.argv.index("--decode-tokens") + 1] == "1024"


def test_flash_next_cells_reserve_memory(tmp_path):
    rj.configure()
    jobs = rj.cells("fn", rj.SETS["fn"]["engines"], 1, tmp_path)
    assert len(jobs) == 2 * 3
    assert {j.mem_gb for j in jobs} == {rj.FN_MEM_GB}
    assert {j.group for j in jobs} == {"f1p", "f8p"}


def test_detect_mode_new_engines():
    assert (
        be.detect_mode(
            "fn-omlx-latest",
            "Speculative backend selected for x: Lightning MTP (model_type=qwen4_exp, active)",
        )
        == "mtp"
    )
    assert be.detect_mode("fn-omlx-latest", "nothing") is None
    log = "common_speculative_impl_draft_dflash: adding speculative implementation 'draft-dflash'"
    assert be.detect_mode("llamacpp-dflash", log) == "dflash"
    assert be.detect_mode("llamacpp-dflash", "plain") is None
    mtp = "I common_speculative_init_result: loading draft model '/m/MTP/mtp-Qwen3.8-27B-Q4_0.gguf'"
    assert be.detect_mode("llamacpp-mtp", mtp) == "mtp"
    assert be.detect_mode("llamacpp-mtp", "plain") is None
    tf = (
        "info: Qwen3.8-27B: 7 streams of 140000 tokens, DFlash2 drafts for one of them\n"
        "[tensorfold] serving Qwen3.8-27B-MLX-4bit at http://127.0.0.1:18990/v1 (drafts: on; context: 140000)"
    )
    assert be.detect_mode("tf-native", tf) == "dflash"
    assert be.detect_mode("tf-native", "drafts: off") is None
