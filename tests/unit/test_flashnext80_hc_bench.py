import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
import flashnext80_hc_bench as hb  # noqa: E402

mx = pytest.importorskip("mlx.core")


def test_diff_summary():
    a = mx.array([1.0, 2.0, 3.0, 4.0])
    b = mx.array([1.0, 2.0, 3.0, 4.5])
    d = hb.summarize_diff(a, b)
    assert d["exact_fraction"] == 0.75 and d["max_abs"] == 0.5
    assert d["max_rel_to_max"] == pytest.approx(0.125)


def test_vendored_modules_import():
    from yunshu_engine.kernels.tf_flash_next import hc

    assert callable(hc.hc_norm) and callable(hc.hc_project)
