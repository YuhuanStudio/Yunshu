"""The S4 pressure sampler parses host swap / paging counters."""

import importlib.util
import sys
from pathlib import Path


def test_parse_vm():
    scripts = Path(__file__).resolve().parents[2] / "scripts/research"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "seg", scripts / "deploy27b_segment.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(scripts))
    row = mod.parse_vm(
        "total = 10240.00M  used = 8568.94M  free = 1671.06M  (encrypted)",
        "Pageouts:   125723.\nSwapins:   5.\nSwapouts:   109915793.\n",
        "System-wide memory free percentage: 93%\n",
    )
    assert row["swap_used_mb"] == 8568.94
    assert row["swapouts"] == 109915793 and row["swapins"] == 5
    assert row["pageouts"] == 125723 and row["free_pct"] == 93
