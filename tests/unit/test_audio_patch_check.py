"""scripts/research/audio_patch_check.py helpers (CPU)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
import audio_patch_check as apc  # noqa: E402


def test_cjk_share():
    assert apc.cjk_share("hello world") == 0.0
    assert apc.cjk_share("你好吗") == 1.0
    assert apc.cjk_share("ab你好") == 0.5
    assert apc.cjk_share("123 ...") == 0.0
