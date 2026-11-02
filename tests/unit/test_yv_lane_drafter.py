"""yv's lane_bandwidth picks its drafter by the target, as tfbench's default_drafter does: the 27B
DFlash2 drafter only for a target of its hidden size; any other target has no drafter directory
to probe, so the stage fails closed with that reason (2026-11-02: the 27B drafter was hard-coded)."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from verify import stages  # noqa: E402


def _model(root, hidden, nested=False):
    root.mkdir(parents=True, exist_ok=True)
    cfg = (
        {"text_config": {"hidden_size": hidden}} if nested else {"hidden_size": hidden}
    )
    (root / "config.json").write_text(json.dumps(cfg))
    return str(root)


def test_picker_matches_tfbench_default_drafter(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "tfbench_lane", ROOT / "scripts/research/tfbench.py"
    )
    tfbench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tfbench)
    drafter = _model(tmp_path / "drafter", 5120)
    monkeypatch.setattr(tfbench, "D", drafter)
    monkeypatch.setattr(stages, "DRAFTER_27B", drafter)
    big = _model(tmp_path / "27b", 5120, nested=True)
    small = _model(tmp_path / "fn", 2560)
    for model in (big, small, None):
        assert stages.default_drafter(model) == tfbench.default_drafter(model)
    assert stages.default_drafter(big) == drafter
    assert stages.default_drafter(small) == "mtp"


def test_lane_stage_refuses_a_target_without_a_drafter_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "DRAFTER_27B", _model(tmp_path / "drafter", 5120))
    small = _model(tmp_path / "fn", 2560)
    ctx = SimpleNamespace(
        env={},
        model=small,
        model_name="fn",
        run=SimpleNamespace(append=lambda *a, **k: None),
        exe=SimpleNamespace(
            jobs=[], run_cells=None
        ),  # no run_cells: no GPU job may start
        cand=SimpleNamespace(path=tmp_path / "cand"),
        suite={},
    )
    res = stages.stage_lane_bandwidth(ctx)
    assert not res.passed
    assert "MTP head" in " ".join(res.reasons)
