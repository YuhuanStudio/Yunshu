import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "round3_ab", Path(__file__).resolve().parents[2] / "scripts/research/round3_ab.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_arms():
    assert m.arm_env("off") == {"YUNSHU_ROUND_DRIVER": "0"}
    assert m.arm_env("routed") == {"YUNSHU_ROUND_DRIVER": "1"}
    assert "32768" in m.bench_args("32k")


def test_server_wrapper_and_8k_cell():
    compile(m.SERVE, "serve", "exec")
    assert "8192" in m.bench_args("8k")
