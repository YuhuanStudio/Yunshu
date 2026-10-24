import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "phase_probe",
    Path(__file__).resolve().parents[2] / "scripts/research/phase_probe.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_first_seen_and_order():
    seen = mod.first_seen(
        [(0.1, "queued"), (0.3, "queued"), (0.5, "prefill"), (9.0, "decode")]
    )
    assert seen == {"queued": 0.1, "prefill": 0.5, "decode": 9.0}
    assert mod.order_ok(seen)
    assert not mod.order_ok({"decode": 1.0, "prefill": 2.0})


def test_missing_prefill_flags_queued_straight_to_decode():
    assert mod.missing_prefill({"queued": 0.1, "decode": 9.0}, 0.0, 9.0)
    assert not mod.missing_prefill(
        {"queued": 0.1, "prefill": 1.0, "decode": 9.0}, 0.0, 9.0
    )
    assert not mod.missing_prefill(
        {"decode": 0.5}, 0.0, 0.6
    )  # a fast first token needs no prefill row
