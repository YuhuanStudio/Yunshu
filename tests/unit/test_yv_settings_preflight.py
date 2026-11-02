"""yv checks every YUNSHU_* value on the CPU before a GPU job (2026-11-02: a bad enum value
and the 27B drafter forced on Flash-Next both failed only at server start on the GPU)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from verify import runner  # noqa: E402
from verify.core import InfraError  # noqa: E402


def test_bad_setting_values_fail_before_the_gpu():
    with pytest.raises(InfraError, match="YUNSHU_PLE_STORAGE"):
        runner.check_settings_env(ROOT, {"YUNSHU_PLE_STORAGE": "resident"})
    with pytest.raises(InfraError, match="not a Yunshu setting"):
        runner.check_settings_env(ROOT, {"YUNSHU_PLE_STORAGEE": "memory"})
    runner.check_settings_env(ROOT, {"YUNSHU_PLE_STORAGE": "memory", "MLX_X": "1"})


def test_the_27b_drafter_only_for_a_target_it_fits(tmp_path, monkeypatch):
    import tfbench

    drafter = tmp_path / "drafter"
    drafter.mkdir()
    (drafter / "config.json").write_text(json.dumps({"hidden_size": 5120}))
    monkeypatch.setattr(tfbench, "D", str(drafter))
    big = tmp_path / "27b"
    big.mkdir()
    (big / "config.json").write_text(json.dumps({"text_config": {"hidden_size": 5120}}))
    fn = tmp_path / "fn"
    fn.mkdir()
    (fn / "config.json").write_text(json.dumps({"hidden_size": 2560}))
    assert tfbench.default_drafter(str(big)) == str(drafter)
    assert tfbench.default_drafter(str(fn)) == "mtp"
    mode, env = tfbench.spec_request("yunshu", {}, str(fn))
    assert mode == "mtp" and env["YUNSHU_VLM_DRAFT"] == "mtp"
