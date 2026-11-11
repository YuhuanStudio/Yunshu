"""ci-local counts skipped tests per pass and fails above the committed baseline."""

import importlib.util
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "ci_skips", _ROOT / "scripts/dev/ci_skips.py"
)
ci_skips = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci_skips)

LOG = """\
SKIPPED [2] tests/unit/test_a.py:19: could not import 'gguf'
SKIPPED [1] tests/unit/test_b.py:7: no census on this machine
14712 passed, 3 skipped, 1 xfailed in 172.54s (0:02:52)
7 passed, 5 skipped, 15157 deselected in 24.31s
"""


def test_parse_counts_summary_and_reasons():
    total, reasons = ci_skips.parse(LOG)
    assert total == 8
    assert reasons["could not import 'gguf'"] == 2


def test_judge_fails_above_baseline_and_without_baseline():
    assert ci_skips.judge("plain", 8, {"plain": 8})[0]
    assert not ci_skips.judge("plain", 9, {"plain": 8})[0]
    assert not ci_skips.judge("plain", 1, {})[0]


def test_committed_baseline_covers_both_passes():
    data = json.loads((_ROOT / "scripts/dev/ci_skip_baseline.json").read_text())
    assert set(data) >= {"plain", "paravirtual"}
