import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
import flashnext80_profile as fp  # noqa: E402


def test_summarize_and_toks():
    s = fp.summarize([1.0, 2.0, 9.0])
    assert s["median_ms"] == 2.0 and s["min_ms"] == 1.0 and s["n"] == 3
    assert fp.tok_s(25.0) == 40.0
    with pytest.raises(ValueError):
        fp.summarize([])
