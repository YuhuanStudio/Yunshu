"""yv multimodal cells can serve media with spec on (2026-11-09).

Before: the stage forced YUNSHU_VLM_DRAFT=off over the arm env, so no run ever checked
speculative decode on image requests.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from verify.stages import _multimodal_env  # noqa: E402


def test_default_pins_drafting_off():
    env = _multimodal_env({}, "/x")
    assert env["YUNSHU_VLM_DRAFT"] == "off" and env["YUNSHU_MEDIA_DIR"] == "/x"


def test_arm_env_can_restore_the_default_drafter():
    env = _multimodal_env({"YUNSHU_VLM_DRAFT": "default"}, "/x")
    assert "YUNSHU_VLM_DRAFT" not in env


def test_arm_env_overrides_pins_but_not_the_scratch_dir():
    env = _multimodal_env({"YUNSHU_VLM_DRAFT": "mtp", "YUNSHU_MEDIA_DIR": "/y"}, "/x")
    assert env["YUNSHU_VLM_DRAFT"] == "mtp" and env["YUNSHU_MEDIA_DIR"] == "/x"


def test_zh_kinds_are_always_passed_to_tfbench():
    """2026-11-09: the zh suite dropped --only-kind (its kinds were 'all' kinds) and ran
    tfbench's default prose/code, so every Chinese cell was missing."""
    from verify.stages import _kind_args

    assert _kind_args(["zhcn", "zhtw"], ["zhcn", "zhtw"]) == [
        "--only-kind",
        "zhcn",
        "--only-kind",
        "zhtw",
    ]
    assert _kind_args(["prose", "code"], ["prose", "code"]) == []
    assert _kind_args(
        ["prose", "code", "zhcn", "zhtw"], ["prose", "code", "zhcn", "zhtw"]
    )[-2:] == [
        "--only-kind",
        "zhtw",
    ]
