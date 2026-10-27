import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
import startup_probe as sp  # noqa: E402

LOG = """import time:       120 |        120 | _io
import time:      2000 |      52000 | mlx.core
import time:       300 |       8000 | yunshu_gateway.main
server noise
import time:       bad | line
"""


def test_top_imports_sorts_by_cumulative_ms_and_skips_noise():
    top = sp.top_imports(LOG, 2)
    assert [r["module"] for r in top] == ["mlx.core", "yunshu_gateway.main"]
    assert top[0]["cum_ms"] == 52.0


def test_rss_of_a_dead_pid_is_zero():
    assert sp.rss_mb(2**22 + 12345) == 0.0
