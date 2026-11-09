import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "apc_restart_probe",
    Path(__file__).resolve().parents[2] / "scripts/research/apc_restart_probe.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_helpers():
    assert m.cached_of({"usage": {"prompt_tokens_details": {"cached_tokens": 5}}}) == 5
    assert m.cached_of({}) == 0
    assert len(m.make_prompt(100).split()) == 100
