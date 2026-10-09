import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "media_cold_profile", ROOT / "scripts/research/media_cold_profile.py"
)
prof = importlib.util.module_from_spec(spec)
sys.modules["media_cold_profile"] = prof
spec.loader.exec_module(prof)


def test_timers_accumulate_and_missing_targets_are_recorded():
    t = prof.Timers()
    targets = (
        ("math", "sqrt", "sqrt"),
        ("math", "nope", "nope"),
        ("no_such_module_x", "f", "gone"),
    )
    import math

    original = math.sqrt
    try:
        done, missing = prof.install(t, targets)
        assert done == ["sqrt"] and missing == ["nope", "gone"]
        assert math.sqrt(4) == 2 and math.sqrt(9) == 3
        assert t.calls["sqrt"] == 2 and t.snapshot_ms()["sqrt"] >= 0
    finally:
        math.sqrt = original
    assert prof.summarize([{"a": 1.0}, {"a": 3.0}, {"a": 2.0}]) == {"a": 2.0}
