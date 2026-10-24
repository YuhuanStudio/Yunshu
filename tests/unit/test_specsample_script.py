import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "specsample", Path(__file__).resolve().parents[2] / "scripts/research/specsample.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_body_and_rate():
    b = m.sampling_body("x", "qwen_think", 10)
    assert b["top_k"] == 20 and b["temperature"] == 0.6 and b["max_tokens"] == 10
    assert "temperature" not in m.sampling_body("x", "agent_default", 10)
    assert m.decode_rate(101, 1.0, 3.0) == 50
    assert m.decode_rate(1, 1.0, 3.0) is None
