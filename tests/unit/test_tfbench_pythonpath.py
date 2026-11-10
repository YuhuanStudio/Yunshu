"""tfbench's Yunshu source path fails closed instead of silently serving main (2026-11-10)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
try:
    from scripts.research import tfbench
except ImportError:  # pragma: no cover - run from another root
    import tfbench


def test_a_checkout_root_resolves_to_its_python_dir(tmp_path):
    (tmp_path / "python" / "yunshu_engine").mkdir(parents=True)
    assert tfbench.yunshu_pythonpath(str(tmp_path)) == str(tmp_path / "python")
    assert tfbench.yunshu_pythonpath(str(tmp_path / "python")) == str(
        tmp_path / "python"
    )


def test_a_path_without_yunshu_engine_is_refused(tmp_path):
    """phaseb_conc / supersede_probe passed tree roots as TFB_YUNSHU_SRC; PYTHONPATH then held no
    yunshu_engine and the server imported the venv's editable install, so both A/B arms ran main."""
    with pytest.raises(RuntimeError, match="holds no yunshu_engine"):
        tfbench.yunshu_pythonpath(str(tmp_path))
