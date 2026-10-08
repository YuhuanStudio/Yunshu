from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
import decisions_probe as dp  # noqa: E402


def test_diff_helpers():
    assert dp.flat([[1.0, 2.0], [3.0]]) == [1.0, 2.0, 3.0]
    assert dp.max_abs_diff([1.0, 2.0], [1.0, 2.5]) == 0.5
    assert dp.max_abs_diff([], []) == 0.0
