import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
import load_peak_trace as lp  # noqa: E402


def test_step_table_attributes_a_transient_to_the_mark_that_ends_its_step():
    g = 1_000_000_000
    samples = [
        {"t": 0.1, "active": 2 * g, "cache": 0, "peak": 2 * g},
        {"t": 0.2, "active": 20 * g, "cache": 0, "peak": 20 * g},
        {"t": 0.3, "active": 16 * g, "cache": 0, "peak": 20 * g},
        {"t": 0.6, "active": 16 * g, "cache": g, "peak": 20 * g},
    ]
    marks = [{"t": 0.35, "msg": "loaded"}, {"t": 0.7, "msg": "ready"}]
    rows = lp.step_table(samples, marks)
    assert rows[0]["max_active_gb"] == 20.0 and rows[0]["active_gb"] == 16.0
    assert rows[1]["max_active_gb"] == 16.0 and rows[1]["cache_gb"] == 1.0
