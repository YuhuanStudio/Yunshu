from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

import omni_apc_probe as p  # noqa: E402


def test_png_fixture_is_a_png_and_digest_is_stable():
    b = p.png(8)
    assert b[:8] == b"\x89PNG\r\n\x1a\n"
    import numpy as np

    assert p.digest(np.ones((2, 3))) == p.digest(np.ones((2, 3)))
    assert p.digest(np.ones((2, 3))) != p.digest(np.zeros((2, 3)))
